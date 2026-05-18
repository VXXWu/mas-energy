"""Per-phase joule decomposition of MAS coordination cost.

Reads main-study call_records and decomposes total MAS energy into:
  - phase1_independent: each agent's parallel ReAct (the "useful work" each
    agent does in isolation)
  - phase2_debate:      debate rounds (inter-agent communication overhead)
  - phase3_synthesis:   final aggregation
  - tool_execution:     tool-call execution (not LLM inference)
  - llm_inference:      LLM forward passes only (subset of phase1+2+3)

Coordination cost = phase2 + phase3 (everything that exists ONLY because
we're running MAS, not because we're doing the actual task).

Outputs per-task and aggregate decomposition for decentralized runs across
all main-study benchmarks. Lets us answer: "what fraction of MAS energy is
strictly inter-agent coordination, in joules?"

Usage:
    python analyze_coordination_cost.py
    python analyze_coordination_cost.py --benchmark qampari
"""
import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path
import re


PHASE_PATTERNS = [
    # agent_id regex → phase tag
    (re.compile(r".*_init$"),       "phase1_independent"),
    (re.compile(r".*_r\d+$"),       "phase2_debate"),
    (re.compile(r".*_round_\d+$"),  "phase2_debate"),
    (re.compile(r"^synthesizer$"),  "phase3_synthesis"),
    (re.compile(r"^orchestrator$"), "phase3_synthesis"),
    (re.compile(r"^sas_agent$"),    "phase1_independent"),  # SAS baseline
    (re.compile(r"^independent_\d+$"), "phase1_independent"),
    (re.compile(r"^worker_\d+$"),   "phase1_independent"),
]


def classify_phase(agent_id):
    for pat, label in PHASE_PATTERNS:
        if pat.match(agent_id or ""):
            return label
    return "other"


def decompose_task(call_records):
    """Return dict of per-phase energy + token counts for one task."""
    by_phase = defaultdict(lambda: {"energy": 0.0, "calls": 0, "tokens_in": 0, "tokens_out": 0})
    inference_energy = 0.0
    tool_energy = 0.0
    for cr in call_records:
        e = float(cr.get("gpu_dynamic_energy_joules", 0) or 0)
        ct = cr.get("call_type", "")
        agent = cr.get("agent_id", "")
        is_tool = (ct == "tool_execution")
        if is_tool:
            tool_energy += e
            phase = classify_phase(agent)
            by_phase[phase]["energy"] += e  # tool exec attributed to phase that called it
            by_phase[phase]["calls"] += 1
        else:
            inference_energy += e
            phase = classify_phase(agent)
            by_phase[phase]["energy"] += e
            by_phase[phase]["calls"] += 1
            by_phase[phase]["tokens_in"] += int(cr.get("prompt_tokens", 0) or 0)
            by_phase[phase]["tokens_out"] += int(cr.get("completion_tokens", 0) or 0)
    return {
        "by_phase": dict(by_phase),
        "inference_energy": inference_energy,
        "tool_energy": tool_energy,
        "total_energy": inference_energy + tool_energy,
    }


def find_decentralized_files(root):
    """Yield (path, benchmark) for main-study decentralized result jsonls."""
    benchmarks = ["qampari", "fanoutqa", "browsecomp", "workbench", "swebench"]
    for path in Path(root).rglob("*decentralized*.jsonl"):
        if ".bak" in path.name or ".preclean" in path.name:
            continue
        bench = "unknown"
        for b in benchmarks:
            if b in str(path).lower():
                bench = b
                break
        yield path, bench


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-root", type=str, default="mas-energy/results")
    ap.add_argument("--benchmark", type=str, default=None,
                    help="Only analyze this benchmark (qampari, fanoutqa, etc.)")
    args = ap.parse_args()

    files = list(find_decentralized_files(args.results_root))
    if args.benchmark:
        files = [(p, b) for p, b in files if b == args.benchmark]
    print(f"Found {len(files)} decentralized jsonls.")

    by_bench = defaultdict(list)
    for path, bench in files:
        try:
            rows = [json.loads(l) for l in open(path)]
        except Exception as e:
            print(f"  skip {path}: {e}")
            continue
        for r in rows:
            crs = r.get("call_records") or []
            if not crs:
                continue
            d = decompose_task(crs)
            d["task_id"] = r.get("task_id")
            d["benchmark"] = bench
            d["source"] = path.name
            by_bench[bench].append(d)

    if not any(by_bench.values()):
        print("No decompositions produced; check paths/data.")
        return

    # Per-benchmark + aggregate decomposition
    for bench in sorted(by_bench.keys()):
        rows = by_bench[bench]
        if not rows:
            continue
        print(f"\n{'='*78}")
        print(f"  {bench}  (n={len(rows)} task-runs across all decentralized configs)")
        print('=' * 78)

        # Aggregate
        total_e = statistics.mean(r["total_energy"] for r in rows)
        infer_e = statistics.mean(r["inference_energy"] for r in rows)
        tool_e = statistics.mean(r["tool_energy"] for r in rows)
        print(f"  Mean total energy / task: {total_e:>10.0f} J")
        print(f"    inference (LLM):        {infer_e:>10.0f} J  ({infer_e/max(total_e,1)*100:5.1f}%)")
        print(f"    tool execution:         {tool_e:>10.0f} J  ({tool_e/max(total_e,1)*100:5.1f}%)")

        # Per-phase
        phase_totals = defaultdict(list)
        phase_calls = defaultdict(list)
        phase_tokens_out = defaultdict(list)
        for r in rows:
            for phase, stats in r["by_phase"].items():
                phase_totals[phase].append(stats["energy"])
                phase_calls[phase].append(stats["calls"])
                phase_tokens_out[phase].append(stats["tokens_out"])

        print(f"\n  Per-phase breakdown (mean per task):")
        print(f"    {'phase':<25} {'energy (J)':>12} {'% of total':>12} "
              f"{'mean calls':>12} {'mean out-tok':>14}")
        # Sorted by phase order
        ordered = ["phase1_independent", "phase2_debate", "phase3_synthesis", "other"]
        for phase in ordered:
            if phase not in phase_totals:
                continue
            e = statistics.mean(phase_totals[phase])
            pct = e / max(total_e, 1) * 100
            calls = statistics.mean(phase_calls[phase])
            tok = statistics.mean(phase_tokens_out[phase])
            print(f"    {phase:<25} {e:>12.0f} {pct:>11.1f}% {calls:>12.1f} {tok:>14.0f}")

        # Coordination cost = phase2 + phase3
        coord = (statistics.mean(phase_totals.get("phase2_debate", [0])) +
                 statistics.mean(phase_totals.get("phase3_synthesis", [0])))
        useful = statistics.mean(phase_totals.get("phase1_independent", [0]))
        coord_pct = coord / max(total_e, 1) * 100
        print(f"\n  Coordination cost (Phase 2 + Phase 3):")
        print(f"    {coord:>10.0f} J  ({coord_pct:5.1f}% of total)")
        print(f"  Independent work (Phase 1):")
        print(f"    {useful:>10.0f} J  ({useful/max(total_e,1)*100:5.1f}% of total)")
        ratio = coord / max(useful, 1)
        print(f"  Coordination/independent ratio: {ratio:.2f}× "
              f"({'coordination DOMINATES' if ratio > 1 else 'independent dominates'})")

    # Overall summary
    print(f"\n{'='*78}")
    print(f"  OVERALL across all benchmarks")
    print('=' * 78)
    all_rows = [r for rows in by_bench.values() for r in rows]
    total_e_all = statistics.mean(r["total_energy"] for r in all_rows)
    print(f"  Mean total energy / task: {total_e_all:.0f} J  (n={len(all_rows)} task-runs)")

    overall_phase = defaultdict(list)
    for r in all_rows:
        for phase, stats in r["by_phase"].items():
            overall_phase[phase].append(stats["energy"])

    print(f"\n  Aggregate per-phase share of MAS energy:")
    for phase in ("phase1_independent", "phase2_debate", "phase3_synthesis", "other"):
        if phase not in overall_phase:
            continue
        e = statistics.mean(overall_phase[phase])
        print(f"    {phase:<25} {e:>10.0f} J  ({e/max(total_e_all,1)*100:5.1f}%)")

    coord = (statistics.mean(overall_phase.get("phase2_debate", [0])) +
             statistics.mean(overall_phase.get("phase3_synthesis", [0])))
    coord_pct = coord / max(total_e_all, 1) * 100
    print(f"\n  Coordination overhead = {coord_pct:.1f}% of total MAS energy")
    print(f"  i.e., {coord:.0f} J/task spent on inter-agent communication that wouldn't "
          f"exist if you ran independent agents and aggregated.")


if __name__ == "__main__":
    main()
