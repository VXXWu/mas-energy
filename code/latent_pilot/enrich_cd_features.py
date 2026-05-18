"""Enrich the CD jsonl with runtime-observable features derived from
the task's question text. These are the "initial conditions" features
that a deployable orchestrator could compute BEFORE running any topology.

Feature families:
  1. Text statistics (free): word/char counts, entity density proxies,
     reasoning-marker presence, cardinality-marker presence, wh-words.
  2. Task taxonomy from task_id structure (already done in
     analyze_collaboration_dividend.py).
  3. [Optional, --embed] sentence-transformer embedding (384-dim) — only if
     the package is installed. Adds semantic signal.

Input: collaboration_dividend.jsonl
Output: collaboration_dividend_enriched.jsonl (additional fields per row)

Usage:
    python enrich_cd_features.py \
        --in mas-energy/results/latent_pilot/collaboration_dividend.jsonl \
        --out mas-energy/results/latent_pilot/collaboration_dividend_enriched.jsonl \
        --qampari-data mas-energy/data/qampari/qampari_data/dev_data.jsonl \
        [--embed]
"""
import argparse
import json
import re
from pathlib import Path


# Reasoning markers: presence suggests the question asks for analysis/integration
REASONING_WORDS = {
    "why", "how", "explain", "analyze", "analyse", "compare", "contrast",
    "evaluate", "discuss", "because", "therefore", "however", "whereas",
    "although", "tradeoff", "trade-off", "implication", "implications",
    "reason", "reasons", "effect", "cause", "influence", "impact",
}

# Cardinality markers: presence suggests the question asks for a list/set
CARDINALITY_WORDS = {
    "list", "all", "every", "each", "which", "name", "identify", "enumerate",
    "what are the", "what were the", "who are the", "which of",
}

# Multi-hop markers: signals a multi-step reasoning / composition
MULTIHOP_WORDS = {
    "both", "also", "that also", "who then", "who also", "as well as",
    "and then", "then", "subsequently", "afterward", "followed by",
    "in turn", "which in turn", "that said", "given that",
}

# Wh-question words
WH_WORDS = {"what", "which", "who", "whom", "whose", "when", "where", "why", "how"}


def compute_text_features(question):
    """Return scalar features derived purely from question text.

    Feature design organized around Kim et al.'s parallelizability axis:
    - Parallelizable (Indep wins): list-lookup, factual, single-entity-specified
    - Reasoning (Decent wins): multi-hop, causal, integrative
    """
    q = (question or "").lower().strip()
    original = question or ""
    if not q:
        return {}
    n_chars = len(q)
    words = re.findall(r"[a-z]+", q)
    n_words = len(words)
    n_sentences = max(len(re.findall(r"[.!?]+", q)), 1)
    n_commas = q.count(",")
    n_conjunctions = sum(1 for w in words if w in {"and", "or", "but", "nor"})
    n_subord = sum(1 for w in words if w in {"because", "although", "since", "though", "while", "whereas"})

    n_reasoning_markers = sum(1 for word in REASONING_WORDS if word in q)
    n_cardinality_markers = sum(1 for phrase in CARDINALITY_WORDS if phrase in q)
    n_multihop_markers = sum(1 for phrase in MULTIHOP_WORDS if phrase in q)

    first_word = words[0] if words else ""
    primary_wh = first_word if first_word in WH_WORDS else "other"

    orig_words = re.findall(r"[A-Z][a-z]+", original)
    proper_nouns = len(orig_words)

    # NEW CREATIVE FEATURES:

    # Imperative vs interrogative: "List X" vs "What is X"
    is_imperative = 1 if any(q.startswith(p) for p in
                              ("list ", "name ", "identify ", "enumerate ",
                               "give ", "provide ")) else 0
    has_question_mark = 1 if "?" in original else 0

    # Entity position: where do proper nouns appear? Start-position proper
    # nouns indicate "X did what" (parallelizable); mid-position indicates
    # "what about X" (single lookup).
    # Split into first-third / mid / last-third by char position
    propn_first_third = 0
    propn_last_third = 0
    for m in re.finditer(r"\b[A-Z][a-z]+", original):
        pos_frac = m.start() / max(len(original), 1)
        if pos_frac < 0.33:
            propn_first_third += 1
        elif pos_frac >= 0.66:
            propn_last_third += 1

    # "The X" count: definite article + word sequences. Proxy for
    # "the thing you're asking about is specific" (parallelizable) vs
    # "the things that..." (compound).
    n_definite_refs = len(re.findall(r"\bthe\s+[a-z]+", q))

    # Template patterns (regex proxies):
    # "what ___ did X" → parallelizable list
    is_what_did = 1 if re.search(r"what (movies|films|books|songs|albums|novels|works|papers|operas|symphonies)\s+(did|have|has|were)", q) else 0
    # "which of" → selection from known set (parallelizable)
    is_which_of = 1 if "which of" in q else 0
    # "how did X" / "why did X" → reasoning
    is_why_how_did = 1 if re.search(r"(how|why) (did|does|do|was|were|is|are) ", q) else 0
    # "both X and Y" → multi-entity reasoning
    is_both_and = 1 if re.search(r"both\s+\w+\s+and\s+\w+", q) else 0
    # Contains year (specific factual temporal reference)
    has_year = 1 if re.search(r"\b(19|20)\d{2}\b", original) else 0

    # Count of conditional/relative clauses ("who", "which", "that", "where" as relatives)
    n_relative_clauses = sum(1 for w in words[1:] if w in {"who", "which", "that", "where"})

    # Length asymmetry: are words heavy tail or uniform? Complex vocabulary
    # tends to be in the tail; simple lookup questions use common words.
    from statistics import stdev
    word_lens = [len(w) for w in words]
    word_len_var = stdev(word_lens) if len(word_lens) > 1 else 0.0

    return {
        # Existing:
        "q_n_chars": n_chars,
        "q_n_words": n_words,
        "q_n_sentences": n_sentences,
        "q_n_commas": n_commas,
        "q_n_conjunctions": n_conjunctions,
        "q_n_subord_conjunctions": n_subord,
        "q_reasoning_markers": n_reasoning_markers,
        "q_cardinality_markers": n_cardinality_markers,
        "q_multihop_markers": n_multihop_markers,
        "q_primary_wh": primary_wh,
        "q_proper_nouns": proper_nouns,
        "q_proper_noun_density": proper_nouns / max(n_words, 1),
        "q_mean_word_len": sum(len(w) for w in words) / max(n_words, 1),
        # New — creative / Kim-et-al-inspired:
        "q_is_imperative": is_imperative,
        "q_has_question_mark": has_question_mark,
        "q_propn_first_third": propn_first_third,
        "q_propn_last_third": propn_last_third,
        "q_n_definite_refs": n_definite_refs,
        "q_is_what_did": is_what_did,
        "q_is_which_of": is_which_of,
        "q_is_why_how_did": is_why_how_did,
        "q_is_both_and": is_both_and,
        "q_has_year": has_year,
        "q_n_relative_clauses": n_relative_clauses,
        "q_word_len_var": word_len_var,
    }


def load_qampari_questions(path):
    """task_id → question_text + gold answer count."""
    qs = {}
    for line in open(path):
        r = json.loads(line)
        qid = r.get("qid") or r.get("id")
        qtext = r.get("question_text") or r.get("question")
        if qid and qtext:
            qs[qid] = {
                "question_text": qtext,
                "gold_answer_count": len(r.get("answer_list", [])),
            }
    return qs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path", type=str, required=True)
    ap.add_argument("--out", dest="out_path", type=str, required=True)
    ap.add_argument("--qampari-data", type=str, default=None,
                    help="QAMPARI dev_data.jsonl for question lookup.")
    ap.add_argument("--embed", action="store_true",
                    help="Also compute sentence-transformer embeddings (slow).")
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.in_path)]

    qampari_qs = {}
    if args.qampari_data and Path(args.qampari_data).exists():
        qampari_qs = load_qampari_questions(args.qampari_data)
        print(f"Loaded {len(qampari_qs)} QAMPARI questions from {args.qampari_data}")

    # Optional embeddings
    embedder = None
    if args.embed:
        try:
            from sentence_transformers import SentenceTransformer
            embedder = SentenceTransformer("all-MiniLM-L6-v2")
            print(f"Loaded embedding model all-MiniLM-L6-v2")
        except ImportError:
            print("sentence-transformers not installed; skipping --embed.")

    enriched_counts = {"qampari": 0, "other": 0, "missing_question": 0}
    for r in rows:
        bench = r.get("benchmark", "")
        tid = r.get("task_id", "")
        question = None
        if bench == "qampari" and tid in qampari_qs:
            question = qampari_qs[tid]["question_text"]
            r["gold_answer_count"] = qampari_qs[tid]["gold_answer_count"]

        if question:
            r["question_text"] = question
            r["text_features"] = compute_text_features(question)
            if embedder:
                r["question_embedding"] = embedder.encode(question).tolist()
            enriched_counts["qampari"] += 1
        else:
            r["text_features"] = {}
            enriched_counts["missing_question"] += 1

    Path(args.out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    print(f"\nEnriched {len(rows)} rows:")
    for k, v in enriched_counts.items():
        print(f"  {k}: {v}")
    print(f"Output: {args.out_path}")


if __name__ == "__main__":
    main()
