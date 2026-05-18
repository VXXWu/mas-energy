"""Collect training data for the topology orchestrator classifier.

Reads per-task jsonls from paired Independent-like and Decentralized-like
runs across benchmarks and extracts:

  Features (computable at runtime after Phase 1 only):
    - phase1_agreement           : Jaccard across N parallel agents
    - mean_tool_calls            : mean tool-call count per agent
    - tool_dup_rate              : 1 - unique/total tool calls across agents
    - response_length_var        : variance of Phase 1 response token-counts
    - phase1_energy              : J consumed up to this decision point
    - benchmark                  : categorical (for leave-one-out eval)

  Label:
    - debate_helps (bool)        : text_loose_acc > single_loose_acc + margin

Writes one consolidated training jsonl. Usage:
    python collect_orchestrator_data.py \
        --results-dir mas-energy/results/latent_pilot \
        --out-path orchestrator_train.jsonl \
        --margin 0.05
"""
import argparse
import json
import statistics
from pathlib import Path


def load_rows(path):
    if not path.exists():
        return []
    return [json.loads(l) for l in open(path)]


def infer_benchmark(filename):
    """Extract benchmark name from filename pattern
    eval_<model>_<bench>_agentic_k*_m*.jsonl"""
    name = filename.stem
    parts = name.split("_")
    for i, p in enumerate(parts):
        if p == "agentic" and i >= 1:
            return parts[i - 1]
    return "unknown"


def response_length_from_tool_calls(tool_calls_list):
    """Proxy for agent response length: total characters in tool call args
    across all agent's calls. If tool_calls is empty, returns 0."""
    if not tool_calls_list:
        return 0
    total = 0
    for call in tool_calls_list:
        args = call.get("args", {})
        if isinstance(args, dict):
            for v in args.values():
                if isinstance(v, str):
                    total += len(v)
    return total


def extract_features_label(row, margin=0.05):
    """Return (features_dict, label) or None if row lacks required fields."""
    single_la = row.get("single_loose_accuracy")
    text_la = row.get("text_loose_accuracy")
    if single_la is None or text_la is None:
        return None

    agreement = row.get("text_phase1_agreement")
    p1_calls = row.get("text_phase1_tool_calls")
    if agreement is None or p1_calls is None:
        return None

    # Feature 1: agreement
    # Feature 2: mean tool calls per agent
    calls_per_agent = [len(c) for c in p1_calls] if p1_calls else []
    mean_calls = statistics.mean(calls_per_agent) if calls_per_agent else 0.0

    # Feature 3: tool-call duplicate rate across agents
    def norm(c):
        name = c.get("name", "")
        args = c.get("args", {}) or {}
        q = args.get("query", "") if isinstance(args, dict) else ""
        return f"{name}({q.strip().lower()})" if isinstance(q, str) else f"{name}(?)"
    flat = [norm(c) for calls in (p1_calls or []) for c in calls]
    dup_rate = 1.0 - (len(set(flat)) / max(len(flat), 1)) if flat else 0.0

    # Feature 4: response length variance (per-agent)
    lens = [response_length_from_tool_calls(c) for c in (p1_calls or [])]
    length_var = statistics.variance(lens) if len(lens) > 1 else 0.0

    # Feature 5: phase1 energy (approximation: use single_energy_j as proxy
    # — reflects SAS cost, similar to Phase 1 per-agent cost in parallel MAS)
    phase1_energy = row.get("single_energy_j", 0.0)

    feats = {
        "agreement": float(agreement),
        "mean_calls": float(mean_calls),
        "dup_rate": float(dup_rate),
        "length_var": float(length_var),
        "phase1_energy": float(phase1_energy),
    }
    label = 1 if (text_la - single_la) > margin else 0
    return feats, label


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", type=str,
                    default="mas-energy/results/latent_pilot")
    ap.add_argument("--out-path", type=str, required=True)
    ap.add_argument("--margin", type=float, default=0.05,
                    help="debate_helps = (text_la - single_la) > margin")
    ap.add_argument("--pattern", type=str, default="eval_*_qampari_*.jsonl,eval_*_fanoutqa_*.jsonl",
                    help="Comma-sep glob patterns for eligible jsonls.")
    args = ap.parse_args()

    results_dir = Path(args.results_dir)
    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    patterns = [p.strip() for p in args.pattern.split(",")]
    paths = set()
    for p in patterns:
        for match in results_dir.glob(p):
            paths.add(match)
    paths = sorted(paths)
    print(f"Scanning {len(paths)} jsonls from {results_dir}...")

    out_rows = []
    per_bench_counts = {}
    for path in paths:
        bench = infer_benchmark(path)
        rows = load_rows(path)
        for r in rows:
            res = extract_features_label(r, margin=args.margin)
            if res is None:
                continue
            feats, label = res
            out_rows.append({
                "task_id": r.get("task_id"),
                "benchmark": bench,
                "source_file": path.name,
                "features": feats,
                "label": label,
                # Also save raw accuracies + energies for Pareto analysis
                "single_loose_accuracy": r["single_loose_accuracy"],
                "text_loose_accuracy": r["text_loose_accuracy"],
                "single_energy_j": r.get("single_energy_j", 0),
                "text_energy_j": r.get("text_energy_j", 0),
            })
            per_bench_counts[bench] = per_bench_counts.get(bench, 0) + 1

    # Write out
    if out_path.exists():
        out_path.unlink()
    with open(out_path, "w") as f:
        for r in out_rows:
            f.write(json.dumps(r) + "\n")

    # Summary
    print(f"Wrote {len(out_rows)} examples to {out_path}")
    print(f"By benchmark:")
    for b, n in sorted(per_bench_counts.items()):
        pos = sum(1 for r in out_rows if r["benchmark"] == b and r["label"] == 1)
        print(f"  {b:<15} n={n}  positive={pos} ({pos/max(n,1):.1%})")
    total_pos = sum(1 for r in out_rows if r["label"] == 1)
    print(f"Overall: {total_pos}/{len(out_rows)} positive "
          f"({total_pos/max(len(out_rows),1):.1%})")


if __name__ == "__main__":
    main()
