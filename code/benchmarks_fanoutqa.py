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

import json
import logging
import os
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

def _shard_of(title, n_shards):
    """Stable shard assignment for an article title. md5, NOT hash() — Python's
    hash() is salted per process, which would re-shard between runs/agents."""
    import hashlib
    return int(hashlib.md5(title.strip().lower().encode()).hexdigest(), 16) % n_shards


class FanOutQAExecutor:
    """Executes the single `search` tool matching the paper's open-book
    protocol (WikipediaKani.search).

    The model provides a Wikipedia article title. If found, returns
    BM25+-ranked chunks. If not found, returns similar titles.

    MAS_SHARD_CORPUS=N (sharded-corpus ablation, 2026-07-13): partitions the
    retrievable corpus into N disjoint shards by md5(title) % N; agent i can
    only READ articles in shard i (identity via agent_context contextvar, set
    by react_loop for every topology; SAS/synthesizer → shard 0). Suggestions
    are filtered to the agent's shard too, so inaccessible titles are not
    leaked through the PageError path. Creates genuine information asymmetry:
    hop-2 lookups depend on hop-1 findings held by another agent — the
    condition-2a test of the communication-null boundary (§5.2). Unset/0 = off.
    """

    def __init__(self, question):
        self._question = question
        self._n_shards = int(os.environ.get("MAS_SHARD_CORPUS", "0") or 0)

    def _my_shard(self):
        from agent_context import current_agent_id, extract_agent_idx
        return extract_agent_idx(current_agent_id.get()) % self._n_shards

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

        shard = self._my_shard() if self._n_shards else None

        exact_match = None
        for ev in results:
            if ev.title.lower() == query.lower():
                exact_match = ev
                break

        if exact_match is not None and shard is not None \
                and _shard_of(exact_match.title, self._n_shards) != shard:
            return (
                f"The article '{exact_match.title}' exists but is NOT in your "
                f"index partition — you cannot read it. Another agent may have "
                f"access. Search for articles available in your own partition."
            )

        if exact_match is None:
            # No exact match -- return suggestions (matching paper's PageError path)
            sugg = results[:5]
            if shard is not None:
                sugg = [ev for ev in results
                        if _shard_of(ev.title, self._n_shards) == shard][:5]
                if not sugg:
                    return (f"No page with that exact title exists, and no "
                            f"similar articles are in your index partition.")
            suggestions = "\n".join(
                f"  search(query=\"{ev.title}\")" for ev in sugg
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


# ─────────────────────────────────────────────────────────
# Chained-shard variant (MAS_CHAINED_SHARDS=1)
# ─────────────────────────────────────────────────────────

# Pre-generated chain tasks (generate_chained_fanoutqa.py, seed 42).
# Persisted so cluster runs are reproducible without regeneration drift.
CHAINED_TASKS_FILE = os.environ.get(
    "MAS_CHAINED_TASKS_FILE",
    os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..", "data", "fanoutqa_chained", "chained_tasks_m3_n100_seed42.json",
    ),
)

UNLOCK_TOOL = {
    "type": "function",
    "function": {
        "name": "unlock_step",
        "description": (
            "Reveal the question for a locked step of this chained task. "
            "Provide the step number to unlock (2 or higher) and your answer "
            "to the PREVIOUS step as the key. Returns the step's question if "
            "the key is correct, otherwise an error."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "step": {
                    "type": "integer",
                    "description": "Step number to unlock (2 or higher)",
                },
                "answer": {
                    "type": "string",
                    "description": "Your answer to step (step - 1), the key",
                },
            },
            "required": ["step", "answer"],
        },
    },
}


class FanOutQAChainedExecutor(FanOutQAExecutor):
    """Executor for chained-shard tasks (MAS_CHAINED_SHARDS=1, 2026-07-16).

    Positive control for the R-null finding: unlike the sharded fan-out
    ablation (independent hops merged in one synthesis), hops here are
    SEQUENTIALLY dependent by construction. Hop j's question is locked
    behind hop j-1's answer via `unlock_step`, and hop j's evidence article
    lives in shard (j-1) mod N (enforced at generation time), so the agent
    who can read it is never the one holding the key. Progress beyond hop 1
    therefore requires inter-agent broadcast; predicted accuracy steps up
    with R until R ~ chain length, and R=1 is structurally capped at hop 1.

    `unlock_step` is a stateless key check available to every agent (only
    `search` is shard-restricted): the caller supplies the previous hop's
    answer, matched with the SAME normalization + word-boundary logic as
    evaluation (answer_in_text), so unlock strictness == eval strictness.
    Wrong keys get an error with no leakage of the locked question.
    """

    def __init__(self, task):
        super().__init__(task["question"])
        self._hops = task["hops"]

    def __call__(self, tool_name, args):
        if tool_name == "unlock_step":
            return self._unlock(args)
        return super().__call__(tool_name, args)

    def _unlock(self, args):
        try:
            step = int(args.get("step", 0))
        except (TypeError, ValueError):
            return "unlock_step error: 'step' must be an integer (2 or higher)."
        n = len(self._hops)
        if step == 1:
            return "Step 1 is already unlocked; its question is in the task."
        if step < 2 or step > n:
            return f"unlock_step error: step must be between 2 and {n}."
        key = str(args.get("answer", "") or "").strip()
        if not key:
            return ("unlock_step error: 'answer' must contain your answer to "
                    f"step {step - 1}.")
        prev_hop = self._hops[step - 2]
        answer_in_text = _get_answer_in_text()
        aliases = prev_hop.get("answer_aliases") or [prev_hop["answer"]]
        if any(answer_in_text(a, key).found for a in aliases):
            return (f"Correct key. Step {step} question: "
                    f"{self._hops[step - 1]['question']}")
        return (f"Wrong key: that is not the correct answer to step {step - 1}. "
                f"Step {step} remains locked.")


class FanOutQAChainedBenchmark:
    """Adapter for the chained-shard FanOutQA variant (MAS_CHAINED_SHARDS=1).

    Same four-method pattern as FanOutQABenchmark. Tasks come from the
    pre-generated JSON (CHAINED_TASKS_FILE); each task dict has:
        id, question (agent-visible prompt revealing only hop 1),
        ground_truth_answer (final hop's answer), n_hops,
        hops (per-hop question/answer/aliases/evidence/shard),
        shard_map (evidence title -> shard index)
    Evaluation is the standard FanOutQA answer matching on the final hop's
    answer, so extraction scripts work unchanged (correct + loose_accuracy).
    """

    def load_tasks(self, n_tasks=None, seed=42):
        """Load pre-generated chain tasks.

        Tasks are already in seeded generation order; slicing the first
        n_tasks keeps n=50 a byte-identical prefix of n=100 (same resume
        convention as the sharded n100 extension). `seed` is accepted for
        signature compatibility but unused.
        """
        with open(CHAINED_TASKS_FILE) as f:
            tasks = json.load(f)["tasks"]
        if n_tasks and n_tasks < len(tasks):
            tasks = tasks[:n_tasks]
        return tasks

    def get_tools(self):
        """search (shard-restricted via MAS_SHARD_CORPUS) + unlock_step."""
        return FANOUTQA_TOOLS + [UNLOCK_TOOL]

    def make_executor(self, task=None):
        executor = FanOutQAChainedExecutor(task)
        return executor, lambda: None

    def evaluate(self, task, recorder, final_answer=""):
        correct, score = evaluate_answer(task, final_answer)
        return {"correct": correct, "loose_accuracy": score}
