"""QAMPARI benchmark adapter.

QAMPARI (ACL 2023) has list-answer questions where each answer entity is
found in a different Wikipedia paragraph. This creates a genuine breadth
bottleneck: more search = more answers found. MAS topologies that
parallelize search should outperform single-agent on recall.

Tools:
  - search(query) -> BM25-ranked passages from a per-task corpus
    (gold proof texts + distractors from other questions)

Evaluation: automated string-matching F1 over predicted entity lists,
following QAMPARI's official evaluation protocol (normalize_answer +
alias matching). No LLM judge.
"""

import json
import logging
import random
import re
import string
from pathlib import Path

log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────
# Tool schema
# ─────────────────────────────────────────────────────────

QAMPARI_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search",
            "description": (
                "Search Wikipedia for passages related to the query. "
                "Returns the most relevant passages ranked by relevance. "
                "Each passage is a paragraph from a Wikipedia article."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query to find relevant Wikipedia passages",
                    },
                },
                "required": ["query"],
            },
        },
    },
]


# ─────────────────────────────────────────────────────────
# Evaluation (QAMPARI official protocol)
# ─────────────────────────────────────────────────────────

def normalize_answer(s: str) -> str:
    """Normalize answer string following QAMPARI's SQuAD-style normalization."""
    s = s.lower()
    # Remove articles
    s = re.sub(r'\b(a|an|the)\b', ' ', s)
    # Remove punctuation
    s = ''.join(ch for ch in s if ch not in string.punctuation)
    # Fix whitespace
    s = ' '.join(s.split())
    return s


def build_alias_map(answer_list: list[dict]) -> dict[str, str]:
    """Build normalized_alias -> canonical_answer_text map."""
    alias_map = {}
    for ans in answer_list:
        canonical = ans["answer_text"]
        all_names = [canonical] + ans.get("aliases", [])
        for name in all_names:
            norm = normalize_answer(name)
            if norm:
                alias_map[norm] = canonical
    return alias_map


def parse_answer_list(text: str) -> list[str]:
    """Extract individual answers from model's free-text response.

    Handles formats like:
    - "1. Answer One\n2. Answer Two\n3. Answer Three"
    - "Answer One, Answer Two, Answer Three"
    - "- Answer One\n- Answer Two"
    - Plain prose mentioning answers

    Following ALCE (Gao et al. 2023): primary parsing is comma-separated
    after stripping trailing punctuation. Numbered/bullet lists also handled
    since agentic models often use them.
    """
    if not text:
        return []

    # Strip tool call XML that may leak into responses
    text = re.sub(r'</?tool_call>|</?function[^>]*>|</?parameter[^>]*>', '', text)
    text = text.strip()
    if not text:
        return []

    # Try numbered list first (most structured)
    numbered = re.findall(r'^\s*\d+[\.\)]\s*(.+?)$', text, re.MULTILINE)
    if numbered:
        return [a.strip() for a in numbered if a.strip()]

    # Try bullet list
    bulleted = re.findall(r'^\s*[-*•]\s*(.+?)$', text, re.MULTILINE)
    if bulleted:
        return [a.strip() for a in bulleted if a.strip()]

    # ALCE-style: comma-separated on potentially multiple lines
    # Join all lines, strip trailing punct, split on commas
    flat = ' '.join(text.split('\n')).strip().rstrip('.').rstrip(',')
    parts = [p.strip() for p in flat.split(',') if p.strip()]
    if len(parts) >= 2:
        return parts

    # Fallback: return the whole text as a single answer
    return [text.strip()]


def clean_predicted_answer(s: str) -> str:
    """Strip markdown formatting and parenthetical info from model output."""
    s = s.replace("**", "").replace("*", "")
    # Remove parenthetical birth/death years, descriptions: "(1885-1935)", "(manga)"
    s = re.sub(r'\s*\([^)]*\)\s*', ' ', s)
    # Remove trailing colons (from "Answer:" style output)
    s = re.sub(r'\s*:\s*$', '', s)
    return s.strip()


def _substring_recall(gold_answers: list[dict], full_text: str) -> tuple[int, int]:
    """Count how many gold answers appear as substrings in the full text.

    Uses word-boundary matching after normalization, same approach as
    FanOutQA's answer_in_text. This catches correct answers embedded
    in prose that the list parser can't extract.
    """
    norm_text = normalize_answer(full_text)
    found = set()
    for ans in gold_answers:
        all_names = [ans["answer_text"]] + ans.get("aliases", [])
        for name in all_names:
            norm_name = normalize_answer(name)
            if not norm_name:
                continue
            if re.search(rf'\b{re.escape(norm_name)}\b', norm_text):
                found.add(ans["answer_text"])
                break
    return len(found), len(gold_answers)


def evaluate_qampari(gold_answers: list[dict], predicted_text: str) -> dict:
    """Evaluate predicted answers against gold using QAMPARI F1.

    Returns dict with:
        f1, precision, recall: strict list-based matching (QAMPARI official)
        recall_substr: substring-based recall over full text (lenient)
        correct: f1 >= 0.5
    """
    if not predicted_text:
        return {
            "correct": False, "f1": 0.0, "precision": 0.0, "recall": 0.0,
            "recall_substr": 0.0, "num_predicted": 0, "num_gold": len(gold_answers),
            "num_correct": 0, "num_correct_substr": 0,
        }

    alias_map = build_alias_map(gold_answers)
    predictions = parse_answer_list(predicted_text)

    matched_gold = set()
    num_correct = 0

    for pred in predictions:
        cleaned = clean_predicted_answer(pred)
        norm_pred = normalize_answer(cleaned)
        if norm_pred in alias_map:
            canonical = alias_map[norm_pred]
            if canonical not in matched_gold:
                matched_gold.add(canonical)
                num_correct += 1

    num_predicted = len(predictions)
    num_gold = len(gold_answers)

    precision = num_correct / num_predicted if num_predicted > 0 else 0.0
    recall = num_correct / num_gold if num_gold > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    # Substring-based recall: how many gold entities appear anywhere in text
    n_substr, _ = _substring_recall(gold_answers, predicted_text)
    recall_substr = n_substr / num_gold if num_gold > 0 else 0.0

    return {
        "correct": f1 >= 0.5,
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "recall_substr": recall_substr,
        "num_predicted": num_predicted,
        "num_gold": num_gold,
        "num_correct": num_correct,
        "num_correct_substr": n_substr,
    }


# ─────────────────────────────────────────────────────────
# Per-task corpus and BM25 search
# ─────────────────────────────────────────────────────────

class QampariCorpus:
    """Per-task searchable corpus built from proof texts + distractors."""

    def __init__(self, passages: list[dict]):
        """passages: list of {text, source_url, is_gold}"""
        self.passages = passages
        self._bm25 = None
        self._tokenized = None

    def _ensure_index(self):
        if self._bm25 is not None:
            return
        from rank_bm25 import BM25Plus
        self._tokenized = [p["text"].lower().split() for p in self.passages]
        self._bm25 = BM25Plus(self._tokenized)

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        self._ensure_index()
        import numpy as np
        tokens = query.lower().split()
        scores = self._bm25.get_scores(tokens)
        top_idxs = np.argsort(scores)[::-1][:top_k]
        results = []
        for idx in top_idxs:
            if scores[idx] <= 0:
                break
            p = self.passages[idx]
            results.append({
                "text": p["text"],
                "source": p.get("source_url", ""),
                "score": float(scores[idx]),
            })
        return results


def build_task_corpus(task: dict, distractor_pool: list[dict],
                      n_distractors: int = 150, seed: int = None) -> QampariCorpus:
    """Build a per-task corpus from gold proof texts + distractors.

    Gold proofs are the actual Wikipedia passages proving each answer.
    Distractors are proof texts from OTHER questions, making search non-trivial.
    """
    passages = []

    # Add gold proof texts
    for ans in task["answer_list"]:
        for proof in ans.get("proof", []):
            passages.append({
                "text": proof["proof_text"],
                "source_url": proof.get("found_in_url", ""),
                "is_gold": True,
            })

    # Add distractors
    rng = random.Random(seed)
    distractors = rng.sample(distractor_pool, min(n_distractors, len(distractor_pool)))
    for d in distractors:
        passages.append({
            "text": d["text"],
            "source_url": d.get("source_url", ""),
            "is_gold": False,
        })

    # Shuffle so gold passages aren't clustered at the top
    rng.shuffle(passages)
    return QampariCorpus(passages)


def collect_distractor_pool(all_tasks: list[dict], exclude_qid: str) -> list[dict]:
    """Collect proof texts from all tasks except the current one."""
    pool = []
    for task in all_tasks:
        if task["qid"] == exclude_qid:
            continue
        for ans in task["answer_list"]:
            for proof in ans.get("proof", []):
                pool.append({
                    "text": proof["proof_text"],
                    "source_url": proof.get("found_in_url", ""),
                })
    return pool


# ─────────────────────────────────────────────────────────
# Executor
# ─────────────────────────────────────────────────────────

class QampariExecutor:
    """Executes search tool calls against the per-task corpus."""

    def __init__(self, corpus: QampariCorpus):
        self.corpus = corpus

    def __call__(self, tool_name: str, args: dict) -> str:
        if tool_name == "search":
            return self._search(args)
        return f"Unknown tool: {tool_name}"

    def _search(self, args: dict) -> str:
        query = args.get("query", "")
        if not query:
            return "Error: query is required."
        results = self.corpus.search(query, top_k=5)
        if not results:
            return "No relevant passages found."
        parts = []
        for i, r in enumerate(results, 1):
            parts.append(f"[Result {i}] {r['text']}")
        return "\n\n".join(parts)


# ─────────────────────────────────────────────────────────
# Benchmark class
# ─────────────────────────────────────────────────────────

class QampariBenchmark:
    """Adapter for QAMPARI list-answer QA benchmark.

    Follows the four-method pattern:
        load_tasks, get_tools, make_executor, evaluate
    """

    def __init__(self, data_dir: str = None, n_distractors: int = 150):
        if data_dir is None:
            data_dir = str(Path(__file__).parent.parent / "data" / "qampari" / "qampari_data")
        self.data_dir = Path(data_dir)
        self.n_distractors = n_distractors
        self._all_tasks = None
        self._distractor_pools = {}

    def _load_all_tasks(self) -> list[dict]:
        if self._all_tasks is not None:
            return self._all_tasks

        dev_file = self.data_dir / "dev_data.jsonl"
        if not dev_file.exists():
            raise FileNotFoundError(
                f"QAMPARI dev data not found at {dev_file}. "
                f"Download from https://aggreg-qa.s3.amazonaws.com/qampari.zip"
            )
        with open(dev_file) as f:
            self._all_tasks = [json.loads(line) for line in f]
        return self._all_tasks

    def load_tasks(self, n_tasks: int = None, seed: int = 42) -> list[dict]:
        """Load QAMPARI dev questions.

        Each task dict has:
            qid, question (str), answer_list (list of answer dicts with
            answer_text, aliases, proof texts), question_text
        """
        all_tasks = self._load_all_tasks()

        tasks = []
        for q in all_tasks:
            # ALCE-style format instruction for agents that answer.
            # question_text is the raw question (used by orchestrator for
            # decomposition). question is the formatted version (used by
            # SAS, workers, synthesizer).
            raw_q = q["question_text"]
            formatted_q = (
                f"{raw_q}\n\n"
                f"Provide a list of accurate answers for the given question "
                f"using the search tool to find evidence. "
                f"Separate answers by commas."
            )
            tasks.append({
                "id": q["qid"],
                "qid": q["qid"],
                "question": formatted_q,
                "question_text": raw_q,
                "answer_list": q["answer_list"],
                "entities": q.get("entities", []),
                "num_answers": len(q["answer_list"]),
            })

        if n_tasks and n_tasks < len(tasks):
            rng = random.Random(seed)
            rng.shuffle(tasks)
            tasks = tasks[:n_tasks]

        return tasks

    def get_tools(self) -> list[dict]:
        """Return the search tool schema."""
        return QAMPARI_TOOLS

    def make_executor(self, task=None):
        """Create a QampariExecutor with a per-task searchable corpus.

        The corpus contains gold proof texts (the Wikipedia passages proving
        each answer) mixed with distractor passages from other questions.
        """
        all_tasks = self._load_all_tasks()

        # Build distractor pool (lazy, cached by qid)
        qid = task["qid"] if task else ""
        if qid not in self._distractor_pools:
            pool = collect_distractor_pool(all_tasks, qid)
            self._distractor_pools[qid] = pool

        corpus = build_task_corpus(
            task, self._distractor_pools[qid],
            n_distractors=self.n_distractors,
            seed=hash(qid) % (2**31),
        )
        executor = QampariExecutor(corpus)
        return executor, lambda: None

    def evaluate(self, task: dict, recorder, final_answer: str = "") -> dict:
        """Evaluate the agent's answer using QAMPARI F1.

        Returns dict with correct (bool, F1>=0.5), f1, precision, recall.
        """
        return evaluate_qampari(task["answer_list"], final_answer)
