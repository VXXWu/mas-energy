"""MASLegalBench benchmark adapter.

MASLegalBench (arxiv 2509.24922) has 950 GDPR compliance MCQs from 15 real
enforcement cases. Tasks follow the IRAC legal reasoning framework (Issue,
Rule, Application, Conclusion), providing a structurally obvious decomposition
where Centralized topology should benefit.

We wrap BM25 retrieval over case documents as a tool call, matching the
FanOutQA pattern: the agent gets a `legal_search` tool that returns ranked
chunks from the relevant case document.

Dataset structure (HuggingFace Arrow):
  - type="question": JSON with question, options, correct_answer
  - type="legal framework"|"background"|"entity"|"relation"|"inferred alignment":
    corpus chunks for BM25 retrieval, keyed by source (case ID)
"""

import json
import math
import random
import re
import warnings
from collections import defaultdict

from config import SGLANG_CONTEXT_LENGTH

# ─────────────────────────────────────────────────────────
# Tool schema
# ─────────────────────────────────────────────────────────

MASLEGALBENCH_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "legal_search",
            "description": (
                "Search the GDPR enforcement case document for relevant "
                "sections. Returns the most relevant passages from the case "
                "ranked by relevance to the query. Use this to find facts, "
                "legal rules, commissioner findings, or penalty details."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": (
                            "Search query describing what you are looking for "
                            "in the case document (e.g., 'Article 32 security "
                            "measures', 'data breach notification timeline', "
                            "'penalty assessment factors')"
                        ),
                    },
                },
                "required": ["query"],
            },
        },
    },
]

# ─────────────────────────────────────────────────────────
# BM25 retrieval over case documents
# ─────────────────────────────────────────────────────────

PARAM_K1 = 1.5
PARAM_B = 0.75
EPSILON = 0.25
CHARS_PER_TOKEN = 4
MAX_SEARCH_CHARS = (SGLANG_CONTEXT_LENGTH // 4) * CHARS_PER_TOKEN


def _tokenize(text):
    return str(text).lower().split()


class SimpleBM25:
    """Lightweight BM25 for case document retrieval."""

    def __init__(self, corpus):
        self.corpus = corpus
        self.corpus_size = len(corpus)
        self.avgdl = 0
        self.doc_freqs = []
        self.idf = {}
        self.doc_len = []
        self._build(corpus)

    def _build(self, corpus):
        nd = {}
        total_len = 0
        for doc in corpus:
            tokens = _tokenize(doc)
            self.doc_len.append(len(tokens))
            total_len += len(tokens)
            freqs = {}
            for t in tokens:
                freqs[t] = freqs.get(t, 0) + 1
            self.doc_freqs.append(freqs)
            for t in freqs:
                nd[t] = nd.get(t, 0) + 1

        self.avgdl = total_len / max(self.corpus_size, 1)
        idf_sum = 0
        negative = []
        for word, freq in nd.items():
            val = math.log(self.corpus_size - freq + 0.5) - math.log(freq + 0.5)
            self.idf[word] = val
            idf_sum += val
            if val < 0:
                negative.append(word)
        avg_idf = idf_sum / max(len(self.idf), 1)
        eps = EPSILON * avg_idf
        for w in negative:
            self.idf[w] = eps

    def search(self, query, top_k=5):
        query_tokens = _tokenize(query)
        scores = []
        for i in range(self.corpus_size):
            score = 0
            for t in query_tokens:
                if t not in self.doc_freqs[i]:
                    continue
                tf = self.doc_freqs[i][t]
                score += (
                    self.idf.get(t, 0) * tf * (PARAM_K1 + 1)
                    / (tf + PARAM_K1 * (1 - PARAM_B + PARAM_B * self.doc_len[i] / self.avgdl))
                )
            scores.append((score, i))
        scores.sort(key=lambda x: -x[0])
        results = []
        total_chars = 0
        for score, idx in scores[:top_k * 2]:
            chunk = self.corpus[idx]
            if total_chars + len(chunk) > MAX_SEARCH_CHARS:
                if results:
                    break
            results.append(chunk)
            total_chars += len(chunk)
            if len(results) >= top_k:
                break
        return results


# ─────────────────────────────────────────────────────────
# Executor
# ─────────────────────────────────────────────────────────

class MASLegalBenchExecutor:
    """Executes legal_search over a case's document chunks."""

    def __init__(self, bm25_index, top_k=5):
        self._index = bm25_index
        self._top_k = top_k

    def __call__(self, tool_name, args):
        if tool_name == "legal_search":
            return self._search(args)
        return f"Unknown tool: {tool_name}"

    def _search(self, args):
        query = args.get("query", "")
        if not query:
            return "Query not provided."
        results = self._index.search(query, top_k=self._top_k)
        if not results:
            return "No relevant passages found."
        parts = []
        for i, chunk in enumerate(results):
            clean = chunk.replace("\\", "").strip()
            parts.append(f"<passage id=\"{i+1}\">\n{clean}\n</passage>")
        return "\n".join(parts)


# ─────────────────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────────────────

def _normalize_answer(text):
    """Extract answer from model output. Handles JSON, plain text, etc."""
    if not text:
        return ""
    text = text.strip()

    # Try JSON extraction (matches their parse_string.py pattern)
    for pattern in [
        r"\{.*?\}",
        r"```json\s*(.*?)```",
    ]:
        m = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        if m:
            try:
                obj = json.loads(m.group(0) if "{" in m.group(0) else m.group(1))
                ans = obj.get("answer", "")
                if ans:
                    return str(ans).strip()
            except (json.JSONDecodeError, IndexError):
                pass

    # Plain text: look for Yes/No/A/B/C/D
    # Check last line first (most likely location of final answer)
    last_line = text.strip().split("\n")[-1].strip()
    last_upper = last_line.upper()

    # Exact single-token answer
    for candidate in ["YES", "NO", "A", "B", "C", "D"]:
        if last_upper == candidate or last_upper == candidate + ".":
            return candidate.capitalize() if candidate in ("YES", "NO") else candidate

    # "The answer is X" pattern
    m = re.search(r"(?:answer|choice|option)\s+(?:is|:)\s*([A-D]|Yes|No)", text, re.IGNORECASE)
    if m:
        ans = m.group(1).upper()
        return ans.capitalize() if ans in ("YES", "NO") else ans

    # Last resort: rightmost A/B/C/D/Yes/No in last line
    for candidate in reversed(["A", "B", "C", "D"]):
        if re.search(rf"\b{candidate}\b", last_upper):
            return candidate
    for candidate in ["YES", "NO"]:
        if candidate in last_upper:
            return candidate.capitalize()

    return text.strip()


def evaluate_answer(task, final_answer):
    """Evaluate model answer against ground truth.

    Returns (correct: bool, answer_extracted: str).
    """
    correct_answer = str(task["correct_answer"]).strip()
    extracted = _normalize_answer(final_answer)

    # Normalize for comparison
    correct_norm = correct_answer.upper().strip()
    extracted_norm = extracted.upper().strip()

    is_correct = correct_norm == extracted_norm
    return is_correct, extracted


# ─────────────────────────────────────────────────────────
# Benchmark class
# ─────────────────────────────────────────────────────────

class MASLegalBenchmark:
    """Adapter for MASLegalBench GDPR compliance benchmark.

    Follows the same four-method pattern as FanOutQABenchmark:
        load_tasks, get_tools, make_executor, evaluate
    """

    def __init__(self, data_dir=None):
        """Load dataset and build per-case BM25 indices.

        Args:
            data_dir: path to the MASLegalBench/dataset directory.
                      If None, tries standard locations.
        """
        from datasets import load_from_disk

        if data_dir is None:
            import os
            candidates = [
                os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "..", "..", "MASLegalBench", "dataset"),
                "/atlas2/u/{}/mas_project/MASLegalBench/dataset".format(
                    os.environ.get("USER", "")),
            ]
            for c in candidates:
                if os.path.isdir(c):
                    data_dir = c
                    break
            if data_dir is None:
                raise FileNotFoundError(
                    "MASLegalBench dataset not found. Clone it: "
                    "git clone https://github.com/HKUST-KnowComp/MASLegalBench"
                )

        ds = load_from_disk(data_dir)["train"]

        # Parse questions
        self._questions = []
        for item in ds:
            if item["type"] == "question":
                q = json.loads(item["content"])
                q["source"] = item["source"]
                self._questions.append(q)

        # Build per-case corpus and BM25 indices
        # Use all non-question chunk types as retrieval corpus
        self._corpus = defaultdict(list)
        for item in ds:
            if item["type"] != "question":
                self._corpus[item["source"]].append(item["content"])

        self._bm25_indices = {}
        for source, chunks in self._corpus.items():
            if chunks:
                self._bm25_indices[source] = SimpleBM25(chunks)

    def load_tasks(self, n_tasks=None, seed=42):
        """Load GDPR compliance questions.

        Each task dict has:
            id, question, options, correct_answer, source (case ID)
        """
        tasks = []
        for i, q in enumerate(self._questions):
            options_str = " | ".join(
                f"{k}: {v}" for k, v in q["options"].items()
            )
            task_question = (
                f"{q['question']}\n\nOptions: {options_str}\n\n"
                f"Answer with one of: {', '.join(q['options'].keys())}"
            )
            tasks.append({
                "id": f"legal_{q['source'][:20]}_{i}",
                "question": task_question,
                "raw_question": q["question"],
                "options": q["options"],
                "correct_answer": q["correct_answer"],
                "source": q["source"],
                "n_corpus_chunks": len(self._corpus.get(q["source"], [])),
            })

        if n_tasks and n_tasks < len(tasks):
            rng = random.Random(seed)
            rng.shuffle(tasks)
            tasks = tasks[:n_tasks]

        return tasks

    def get_tools(self):
        """Return the legal_search tool schema."""
        return MASLEGALBENCH_TOOLS

    def make_executor(self, task=None):
        """Create a fresh executor for the task's case document.

        Returns (executor, cleanup_fn). Cleanup is a no-op.
        """
        source = task["source"] if task else ""
        index = self._bm25_indices.get(source)
        if index is None:
            warnings.warn(f"No corpus for case {source}; search will return nothing")
            index = SimpleBM25([])
        executor = MASLegalBenchExecutor(index, top_k=5)
        return executor, lambda: None
