"""Measure cross-agent tool-call duplicate rate from pulled jsonls.

Requires jsonl rows to include `text_phase1_tool_calls` (list per agent)
— added by agentic_latentmas.py metadata_out pathway. Runs produced before
that change will skip with a notice.

Decides whether tool-result caching is worth engineering:
  - Exact-match duplicate rate < 20%: caching won't help much; skip.
  - 20-40%: modest; worth if implementation is trivial (~0.5 day).
  - > 40%: worth engineering tool+prefix cache.

Also computes Jaccard on query sets (fuzzier semantic-match proxy).
"""
import json
import sys
from collections import Counter
from pathlib import Path

RESULTS = Path("mas-energy/results/latent_pilot")


def normalize_call(c):
    """Canonicalize a tool_call dict to a comparable string."""
    name = c.get("name", "")
    args = c.get("args") or {}
    # QAMPARI: the only arg is "query"; normalize whitespace + case
    if "query" in args:
        q = args["query"].strip().lower()
        return f"{name}(query={q!r})"
    return f"{name}({sorted(args.items())!r})"


def analyze_file(path):
    rows = [json.loads(l) for l in open(path)]
    has_calls = [r for r in rows if r.get("text_phase1_tool_calls") is not None]
    if not has_calls:
        return None

    exact_match_pairs = 0
    total_pairs = 0
    jaccards = []
    per_task_stats = []

    for r in has_calls:
        per_agent = r["text_phase1_tool_calls"]
        if not per_agent or len(per_agent) < 2:
            continue
        norm_sets = [set(normalize_call(c) for c in calls) for calls in per_agent]
        # Pairwise
        n_agents = len(norm_sets)
        pair_dups, pair_total = 0, 0
        pair_jaccs = []
        for i in range(n_agents):
            for j in range(i + 1, n_agents):
                if norm_sets[i] and norm_sets[j]:
                    inter = norm_sets[i] & norm_sets[j]
                    union = norm_sets[i] | norm_sets[j]
                    jac = len(inter) / len(union) if union else 0.0
                    pair_jaccs.append(jac)
                    jaccards.append(jac)
                # Per-call duplicate match: for each call in agent i, does
                # any agent j>i have the same normalized call?
                for c in norm_sets[i]:
                    pair_total += 1
                    if c in norm_sets[j]:
                        pair_dups += 1
        exact_match_pairs += pair_dups
        total_pairs += pair_total
        n_calls_per_agent = [len(s) for s in norm_sets]
        per_task_stats.append({
            "task_id": r.get("task_id"),
            "n_calls_per_agent": n_calls_per_agent,
            "mean_jaccard": sum(pair_jaccs) / len(pair_jaccs) if pair_jaccs else 0.0,
        })

    return {
        "n_tasks": len(per_task_stats),
        "exact_dup_rate": exact_match_pairs / max(total_pairs, 1),
        "mean_jaccard": sum(jaccards) / len(jaccards) if jaccards else 0.0,
        "per_task": per_task_stats,
    }


def main():
    target = sys.argv[1] if len(sys.argv) > 1 else None
    if target:
        paths = [RESULTS / target]
    else:
        paths = sorted(RESULTS.glob("eval_Qwen*.jsonl"))

    print(f"{'File':<70} {'n':>4} {'exact-dup':>10} {'jaccard':>9}")
    print("-" * 95)
    for p in paths:
        if not p.exists():
            continue
        result = analyze_file(p)
        if result is None:
            print(f"{p.name:<70} {'no tool-call metadata':>30}")
            continue
        print(f"{p.name:<70} {result['n_tasks']:>4} "
              f"{result['exact_dup_rate']:>9.1%}  {result['mean_jaccard']:>8.3f}")

    print()
    print("Interpretation:")
    print("  exact-dup < 20%: tool caching not worth the engineering")
    print("  20-40%: modest — worth if cache impl is trivial")
    print("  > 40%: worth engineering explicit tool-result cache + KV cache share")
    print()
    print("jaccard is a looser semantic-match proxy (query-set overlap).")
    print("If jaccard >> exact-dup, fuzzy matching would unlock more gains.")


if __name__ == "__main__":
    main()
