"""Diagnose F1 collapses in agentic_latentmas output.

Reads a per-task aggregate file produced by `agentic_latentmas.py` and
extracts whatever diagnostic signal is available:

  - Aggregate F1 / loose-accuracy / energy by condition.
  - Per-task divergence cases (text > 0, latent = 0) with answers if the
    file was written after 2026-05-16 (newer records include single_ans,
    text_ans, latent_ans, question fields).
  - For older files without answer captures, reports the failure modes
    detectable from metrics alone (e.g. zero tool-cache hits suggests
    workers aren't issuing valid search calls).

Usage:
    python -m latent_pilot.diagnose_agentic_run \\
        --jsonl mas-energy/results/latent_pilot/eval_Qwen_Qwen3-4B_qampari_agentic_k5_m20.jsonl \\
        --top-n 10
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", required=True, type=Path)
    ap.add_argument("--top-n", type=int, default=10,
                    help="How many divergence cases to print")
    ap.add_argument("--out", type=Path, default=None,
                    help="Optional JSON summary path")
    args = ap.parse_args()

    if not args.jsonl.exists():
        raise SystemExit(f"Not found: {args.jsonl}")

    records = []
    with open(args.jsonl) as f:
        for line in f:
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    if not records:
        raise SystemExit("No records in file.")

    has_answers = any("latent_ans" in r for r in records)

    # ---- Aggregates ----
    conditions = ("single", "text", "latent")
    f1 = {c: [] for c in conditions}
    loose = {c: [] for c in conditions}
    energy = {c: [] for c in conditions}
    for r in records:
        for c in conditions:
            if r.get(f"{c}_f1") is not None:
                f1[c].append(r[f"{c}_f1"])
            if r.get(f"{c}_loose_accuracy") is not None:
                loose[c].append(r[f"{c}_loose_accuracy"])
            if r.get(f"{c}_energy_j") is not None:
                energy[c].append(r[f"{c}_energy_j"])

    def mean(xs):
        return sum(xs) / len(xs) if xs else 0.0

    print(f"\nLoaded {len(records)} records from {args.jsonl.name}")
    print(f"Per-condition answer captures available: {has_answers}")
    print()
    print(f"{'condition':>10} {'n':>4} {'f1':>7} {'loose':>7} {'energy_J':>12}")
    print("-" * 50)
    for c in conditions:
        print(f"{c:>10} {len(f1[c]):>4} {mean(f1[c]):>7.3f} {mean(loose[c]):>7.3f} "
              f"{mean(energy[c]):>12.1f}")

    text_f1 = mean(f1["text"])
    latent_f1 = mean(f1["latent"])
    f1_delta = latent_f1 - text_f1
    print()
    print(f"  Δ F1 (latent − text): {f1_delta:+.4f}")
    print(f"  Δ loose acc:           {mean(loose['latent']) - mean(loose['text']):+.4f}")
    text_e = mean(energy["text"]) or 1.0
    latent_e = mean(energy["latent"])
    print(f"  Latent energy vs text: {100 * (latent_e - text_e) / text_e:+.1f}%")

    # ---- Failure-mode breakdown ----
    n_text_pos = sum(1 for r in records if (r.get("text_f1") or 0) > 0)
    n_latent_pos = sum(1 for r in records if (r.get("latent_f1") or 0) > 0)
    n_both_zero = sum(1 for r in records
                      if (r.get("text_f1") or 0) == 0 and (r.get("latent_f1") or 0) == 0)
    n_text_only = sum(1 for r in records
                      if (r.get("text_f1") or 0) > 0.05 and (r.get("latent_f1") or 0) == 0)
    n_latent_only = sum(1 for r in records
                        if (r.get("latent_f1") or 0) > 0.05 and (r.get("text_f1") or 0) == 0)

    print()
    print("Per-task pattern:")
    print(f"  text  scored:         {n_text_pos} / {len(records)}")
    print(f"  latent scored:        {n_latent_pos} / {len(records)}")
    print(f"  both zero:            {n_both_zero}")
    print(f"  text > 0.05, latent = 0 (regression cases): {n_text_only}")
    print(f"  latent > 0.05, text = 0 (recovery cases):   {n_latent_only}")

    # ---- Divergence examples ----
    summary = {
        "file": str(args.jsonl),
        "n_records": len(records),
        "has_answer_captures": has_answers,
        "means": {
            c: {"f1": mean(f1[c]), "loose": mean(loose[c]), "energy_j": mean(energy[c])}
            for c in conditions
        },
        "pattern": {
            "n_text_pos": n_text_pos, "n_latent_pos": n_latent_pos,
            "n_both_zero": n_both_zero, "n_text_only_wins": n_text_only,
            "n_latent_only_wins": n_latent_only,
        },
    }

    if has_answers:
        print()
        print(f"=== Top {args.top_n} divergence cases (text > 0.05, latent = 0) ===")
        divergences = sorted(
            [r for r in records
             if (r.get("text_f1") or 0) > 0.05 and (r.get("latent_f1") or 0) == 0],
            key=lambda r: -(r["text_f1"] or 0),
        )[:args.top_n]

        # Categorize latent answer failure modes
        failure_modes = Counter()
        examples = []
        for r in divergences:
            la = (r.get("latent_ans") or "").strip()
            ta = (r.get("text_ans") or "").strip()
            mode = _classify_failure(la, ta)
            failure_modes[mode] += 1
            examples.append({
                "task_id": r.get("task_id"),
                "text_f1": r.get("text_f1"),
                "latent_f1": r.get("latent_f1"),
                "failure_mode": mode,
                "question": (r.get("question") or "")[:200],
                "text_ans": ta[:300],
                "latent_ans": la[:300],
            })
            print(f"\n  task_id: {r.get('task_id')}")
            print(f"    text_f1={r['text_f1']:.2f}  latent_f1={r['latent_f1']:.2f}  mode={mode}")
            print(f"    Q:      {(r.get('question') or '')[:150]}")
            print(f"    text:   {ta[:200]}")
            print(f"    latent: {la[:200]}")

        print()
        print(f"=== Failure-mode distribution (n={len(divergences)} regression cases) ===")
        for mode, n in failure_modes.most_common():
            print(f"  {mode:>30}  {n}")
        summary["divergences"] = examples
        summary["failure_modes"] = dict(failure_modes)
    else:
        print()
        print("(Answer captures not available in this file — rerun with patched")
        print(" agentic_latentmas.py that includes single_ans/text_ans/latent_ans.)")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\nSaved summary: {args.out}")


_TOOL_PATTERN = re.compile(r'(tool_call|<tool_call>|search\(|"name"\s*:)', re.I)


def _classify_failure(latent_ans: str, text_ans: str) -> str:
    """Heuristic categorization of how the latent answer differs from text."""
    if not latent_ans:
        return "empty_response"
    if len(latent_ans) < 10:
        return "truncated_response"
    if _TOOL_PATTERN.search(latent_ans) and not _TOOL_PATTERN.search(text_ans or ""):
        return "leaked_tool_syntax"  # latent emitted raw tool-call text instead of final answer
    if not re.search(r"[,;]\s|\bor\b|\band\b", latent_ans) and \
       re.search(r"[,;]\s|\bor\b|\band\b", text_ans or ""):
        return "single_entity_vs_list"  # latent gave one entity, text gave a list
    if "I cannot" in latent_ans or "I don't" in latent_ans or "unable" in latent_ans.lower():
        return "refusal_or_abstain"
    if abs(len(latent_ans) - len(text_ans or "")) / max(1, len(text_ans or "")) > 2:
        return "length_mismatch_large"
    return "different_content"


if __name__ == "__main__":
    main()
