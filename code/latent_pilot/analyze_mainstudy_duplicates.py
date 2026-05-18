"""Cross-agent tool-call duplicate analysis on main-study transcript data.

Reads call_records from main-study jsonls where SAVE_TRANSCRIPTS was True
(jsonls containing `tool_call` fields with `arguments` payload). Computes
per-task cross-agent exact-match duplicate rate and Jaccard similarity.

Use this to compare duplicate rates across benchmarks once transcript
pilots exist for each. Currently only QAMPARI has transcript data at
`a5000_transcripts_qampari/`. For SWE-bench, BrowseComp, FanOutQA,
WorkBench, would need re-runs with SAVE_TRANSCRIPTS=True.

Usage:
    python analyze_mainstudy_duplicates.py <jsonl_path>
    python analyze_mainstudy_duplicates.py mas-energy/results/a5000_transcripts_qampari/*.jsonl
"""
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path


def normalize(tc):
    """Canonicalize a tool_call dict: (name, args-canonical-str)."""
    name = tc.get("name", "")
    args = tc.get("arguments", "")
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except Exception:
            return f"{name}({args.strip().lower()})"
    if isinstance(args, dict):
        # Use the first string arg value (typical for search/bash/etc.)
        for key in ("query", "command", "cmd", "input", "text"):
            if key in args and isinstance(args[key], str):
                return f"{name}({args[key].strip().lower()})"
        # Fallback: stringified dict
        return f"{name}({sorted(args.items())})"
    return f"{name}({str(args)[:80].lower()})"


def analyze_file(path):
    tasks = [json.loads(l) for l in open(path)]
    per_task_dup = []
    per_task_jacc = []
    per_task_counts = []

    for task in tasks:
        by_agent = defaultdict(list)
        for cr in task.get("call_records", []):
            tc = cr.get("tool_call")
            if tc and cr.get("call_type") == "tool_execution":
                aid = cr.get("agent_id", "?")
                by_agent[aid].append(normalize(tc))

        # Restrict to Phase 1 parallel agents (e.g., *_init for decentralized)
        phase1 = {aid: calls for aid, calls in by_agent.items()
                  if aid.endswith("_init") or "init" in aid}
        if len(phase1) < 2:
            # Fall back to all agents if no init-suffix pattern
            phase1 = by_agent
        if len(phase1) < 2:
            continue

        sets = [set(c) for c in phase1.values()]
        flat = [c for calls in phase1.values() for c in calls]
        total = len(flat)
        if total == 0:
            continue
        per_task_dup.append(1.0 - len(set(flat)) / total)
        per_task_counts.append(total)

        jaccs = []
        n = len(sets)
        for i in range(n):
            for j in range(i + 1, n):
                if sets[i] or sets[j]:
                    jaccs.append(len(sets[i] & sets[j]) / len(sets[i] | sets[j]))
        if jaccs:
            per_task_jacc.append(statistics.mean(jaccs))

    return {
        "path": str(path),
        "n_tasks_analyzed": len(per_task_dup),
        "mean_dup_rate": statistics.mean(per_task_dup) if per_task_dup else None,
        "median_dup_rate": statistics.median(per_task_dup) if per_task_dup else None,
        "mean_jaccard": statistics.mean(per_task_jacc) if per_task_jacc else None,
        "median_jaccard": statistics.median(per_task_jacc) if per_task_jacc else None,
        "mean_calls_per_task": statistics.mean(per_task_counts) if per_task_counts else None,
    }


def main():
    if len(sys.argv) < 2:
        print(__doc__.splitlines()[-2])
        return
    print(f"{'File':<72} {'n':>4} {'exact-dup':>10} {'jaccard':>9} {'calls':>6}")
    print("-" * 102)
    for p in sys.argv[1:]:
        path = Path(p)
        if not path.exists():
            print(f"{p}: MISSING")
            continue
        res = analyze_file(path)
        if res["n_tasks_analyzed"] == 0:
            print(f"{path.name:<72}  (no tool_call data — needs SAVE_TRANSCRIPTS=True)")
            continue
        print(f"{path.name:<72} {res['n_tasks_analyzed']:>4} "
              f"{res['mean_dup_rate']:>9.1%}  {res['mean_jaccard']:>8.3f} "
              f"{res['mean_calls_per_task']:>6.1f}")


if __name__ == "__main__":
    main()
