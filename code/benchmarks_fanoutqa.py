"""FanOutQA open-book benchmark adapter.

FanOutQA (ACL 2024) has fan-out structure: questions decompose into N
independent Wikipedia lookups. The open-book variant gives the agent a
single `search` tool that retrieves BM25+-ranked chunks from Wikipedia.

Matches the paper's protocol (run_openbook.py):
  - Single tool: search(query) takes a Wikipedia article title
  - If found: page is chunked (1024 chars), BM25+ ranked against the
    original question, and top chunks returned up to half the context
  - If not found: returns similar article titles as suggestions
  - Evaluation: loose accuracy (proportion of reference answer components
    found in the model's final text via word-boundary matching after
    normalization)
"""

import logging
import random
import warnings

import fanoutqa
from fanoutqa.retrieval import Corpus, chunk_text
from fanoutqa.wiki import wiki_search, wiki_content

from config import SGLANG_CONTEXT_LENGTH

log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────
# Tool schema (OpenAI function calling format)
# ─────────────────────────────────────────────────────────

FANOUTQA_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search",
            "description": (
                "Search Wikipedia for an article with the given title, "
                "and get its content. If no such article is found, "
                "return similar article names."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Wikipedia article title to look up",
                    },
                },
                "required": ["query"],
            },
        },
    },
]


# ─────────────────────────────────────────────────────────
# BM25+ retrieval (matches paper's WikipediaKani.search)
# ─────────────────────────────────────────────────────────

# Approximate chars-per-token for budget calculation.
# The paper uses engine.message_len() with the exact tokenizer;
# we approximate since SGLang doesn't expose a tokenizer API.
CHARS_PER_TOKEN = 4

# Paper uses max_context_size // 2 as the per-search token budget.
MAX_SEARCH_CHARS = (SGLANG_CONTEXT_LENGTH // 2) * CHARS_PER_TOKEN

# Chunk size in chars, matching paper's Corpus(doc_len=1024)
CHUNK_SIZE = 1024


def _build_search_result(evidence, question):
    """Chunk a page and return BM25+-ranked fragments matching the paper's
    XML format, capped at MAX_SEARCH_CHARS.

    Matches WikipediaKani.search() from the paper's run_openbook.py:
    - Corpus([{"title": ..., "pageid": ...}], doc_len=1024)
    - Ranks chunks against self.question (the original top-level question)
    - Greedy packing until exceeding max_search_tokens
    """
    content = wiki_content(evidence)
    chunks = chunk_text(content, max_chunk_size=CHUNK_SIZE)

    # BM25+ rank chunks against the original question
    try:
        from fanoutqa.norm import normalize as _norm
        _norm("test")
        tokenize = lambda text: _norm(text).split(" ")
    except Exception:
        tokenize = lambda text: str(text).lower().split()
        warnings.warn(
            "spaCy unavailable; BM25+ using simple word splitting. "
            "Production runs on cluster should use spaCy.",
            stacklevel=2,
        )

    from rank_bm25 import BM25Plus
    import numpy as np

    tokenized_chunks = [tokenize(c) for c in chunks]
    index = BM25Plus(tokenized_chunks)
    scores = index.get_scores(tokenize(question))
    ranked_idxs = np.argsort(scores)[::-1]

    # Greedy packing into budget
    fragments = []
    total_chars = 0
    header = f"<document>\n<title>{evidence.title}</title>\n"
    footer = "</document>"

    for idx in ranked_idxs:
        fragment = f"<fragment>\n{chunks[idx]}\n</fragment>\n"
        # Check if adding this fragment exceeds budget
        candidate_len = len(header) + total_chars + len(fragment) + len(footer)
        if candidate_len > MAX_SEARCH_CHARS:
            break
        fragments.append(fragment)
        total_chars += len(fragment)

    return header + "".join(fragments) + footer


# ─────────────────────────────────────────────────────────
# Executor
# ─────────────────────────────────────────────────────────

class FanOutQAExecutor:
    """Executes the single `search` tool matching the paper's open-book
    protocol (WikipediaKani.search).

    The model provides a Wikipedia article title. If found, returns
    BM25+-ranked chunks. If not found, returns similar titles.
    """

    def __init__(self, question):
        self._question = question

    def __call__(self, tool_name, args):
        if tool_name == "search":
            return self._search(args)
        else:
            return f"Unknown tool: {tool_name}"

    def _search(self, args):
        query = args.get("query", "")
        if not query:
            return "Query not provided."

        # Title-based lookup: search Wikipedia, check for exact title match
        results = wiki_search(query)
        if not results:
            return f"No Wikipedia page found for '{query}'."

        exact_match = None
        for ev in results:
            if ev.title.lower() == query.lower():
                exact_match = ev
                break

        if exact_match is None:
            # No exact match -- return suggestions (matching paper's PageError path)
            suggestions = "\n".join(
                f"  search(query=\"{ev.title}\")" for ev in results[:5]
            )
            return (
                f"No page with that exact title exists. "
                f"Try one of these similar articles:\n{suggestions}"
            )

        # Found the page -- chunk, BM25+ rank, and return
        return _build_search_result(exact_match, self._question)


# ─────────────────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────────────────

_answer_in_text_fn = None


def _simple_normalize(text):
    """Fallback normalizer when spaCy is unavailable (e.g. Python 3.14).

    Matches fanoutqa.norm.normalize minus lemmatization.
    Production runs on the cluster use the real normalize via spaCy.
    """
    import re
    try:
        import ftfy
        text = ftfy.fix_text(str(text).lower())
    except ImportError:
        text = str(text).lower()
    text = re.sub(r"(\d+,)+\d+(\.\d+)?", lambda m: m[0].replace(",", ""), text)
    text = re.sub(r"[,.?!:;]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _make_answer_in_text():
    """Build answer_in_text, using fanoutqa's normalize if spaCy works,
    else falling back to _simple_normalize.

    Exact logic copy of fanoutqa.eval.string.answer_in_text, inlined
    to bypass the BLEURT hard dependency in fanoutqa.eval.__init__.
    """
    import itertools
    import re
    from collections import namedtuple

    AccuracyResult = namedtuple("AccuracyResult", "found score missing")

    try:
        from fanoutqa.norm import normalize
        normalize("test")
    except Exception:
        normalize = _simple_normalize
        warnings.warn(
            "spaCy unavailable; evaluation using simplified normalization "
            "(no lemmatization). Results may differ from official FanOutQA "
            "scores. Production runs on cluster should use spaCy.",
            stacklevel=2,
        )

    def answer_in_text(reference, candidate):
        if isinstance(reference, list):
            missing = []
            for a in reference:
                result = answer_in_text(a, candidate)
                missing.extend(result.missing)
            n_found = len(reference) - len(missing)
            return AccuracyResult(
                found=n_found == len(reference),
                score=n_found / len(reference),
                missing=missing,
            )
        elif isinstance(reference, dict):
            missing = []
            vals = itertools.chain(reference.keys(), reference.values())
            for a in vals:
                result = answer_in_text(a, candidate)
                missing.extend(result.missing)
            n_ref = len(reference) * 2
            n_found = n_ref - len(missing)
            return AccuracyResult(
                found=n_found == n_ref,
                score=n_found / n_ref,
                missing=missing,
            )
        else:
            if isinstance(reference, bool):
                reference = "yes" if reference else "no"
            norm_ans = normalize(reference)
            norm_cand = normalize(candidate)
            if not re.search(rf"\b{re.escape(norm_ans)}\b", norm_cand):
                return AccuracyResult(found=False, score=0, missing=[norm_ans])
        return AccuracyResult(found=True, score=1, missing=[])

    return answer_in_text


def _get_answer_in_text():
    global _answer_in_text_fn
    if _answer_in_text_fn is None:
        _answer_in_text_fn = _make_answer_in_text()
    return _answer_in_text_fn


def evaluate_answer(task, final_answer):
    """Evaluate a model's final text answer against FanOutQA ground truth.

    Returns (correct: bool, loose_accuracy: float).
    correct = strict accuracy per FanOutQA paper (all components found).
    """
    if not final_answer:
        return False, 0.0
    answer_in_text = _get_answer_in_text()
    result = answer_in_text(task["ground_truth_answer"], final_answer)
    return result.found, result.score


# ─────────────────────────────────────────────────────────
# Benchmark class
# ─────────────────────────────────────────────────────────

class FanOutQABenchmark:
    """Adapter for FanOutQA open-book benchmark.

    Follows the same four-method pattern as WorkBenchBenchmark:
        load_tasks, get_tools, make_executor, evaluate
    """

    def load_tasks(self, n_tasks=None, seed=42):
        """Load FanOutQA dev set (310 questions).

        Each task dict has:
            id, question, ground_truth_answer, decomposition,
            necessary_evidence, categories
        """
        questions = fanoutqa.load_dev()
        tasks = []
        for q in questions:
            tasks.append({
                "id": q.id,
                "question": q.question,
                "ground_truth_answer": q.answer,
                "decomposition": [
                    {"id": sq.id, "question": sq.question}
                    for sq in q.decomposition
                ],
                "necessary_evidence": [
                    {"title": e.title, "pageid": e.pageid}
                    for e in q.necessary_evidence
                ],
                "categories": q.categories,
            })

        if n_tasks and n_tasks < len(tasks):
            rng = random.Random(seed)
            rng.shuffle(tasks)
            tasks = tasks[:n_tasks]

        return tasks

    def get_tools(self):
        """Return the single search tool schema."""
        return FANOUTQA_TOOLS

    def make_executor(self, task=None):
        """Create a fresh FanOutQAExecutor with the task's question for
        BM25+ ranking.

        Returns (executor, cleanup_fn). Cleanup is a no-op.
        """
        question = task["question"] if task else ""
        executor = FanOutQAExecutor(question)
        return executor, lambda: None

    def evaluate(self, task, recorder, final_answer=""):
        """Answer-based evaluation using loose accuracy.

        Unlike WorkBench (state-based), FanOutQA evaluates the final
        text answer against the ground truth.

        Returns dict with correct (bool) and loose_accuracy (float).
        """
        correct, score = evaluate_answer(task, final_answer)
        return {"correct": correct, "loose_accuracy": score}
