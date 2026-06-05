"""Visualize an actual MAS execution trace from the results folder.

For a given (benchmark, topology, k, rounds, task_id?, rep?), load the task
record and pretty-print its call_records as a structured trace showing:
  - call index, agent_id, call_type
  - prompt / completion tokens
  - GPU dynamic energy
  - per-call wall time + cumulative wall time
  - tool execution placeholders

Optionally dump to markdown for the paper appendix.

Usage:
    python analysis/visualize_trace.py --benchmark qampari --topology centralized --k 5 --rounds 2
    python analysis/visualize_trace.py --benchmark fanoutqa --topology decentralized --k 5
    python analysis/visualize_trace.py --benchmark workbench --topology independent --k 5 --task-id calendar_8
    python analysis/visualize_trace.py --all-topologies --benchmark qampari --md trace.md
"""
from __future__ import annotations
import argparse
import json
import os
import random
import re
import sys
from collections import defaultdict

RESULTS_BASE = "mas-energy/results"

# Map benchmark name → directory name (a5000_*)
BENCH_DIR = {
    "qampari": "a5000_qampari_v4",
    "fanoutqa": "a5000_fanoutqa_v4",
    "browsecomp_plus": "a5000_browsecomp_pilot",
    "browsecomp": "a5000_browsecomp_pilot",
    "workbench": "a5000_workbench_v2",
    "swebench": "a5000_swebench",
    # Transcript-logging pilot runs (records contain full request/response text)
    "qampari_transcripts": "a5000_transcripts_qampari",
}


# ----------------------------------------------------------------------
# Question text loaders (best-effort, cached at first use)
# ----------------------------------------------------------------------

_question_cache: dict = {}


def _load_qampari_questions() -> dict:
    out = {}
    path = "mas-energy/data/qampari/qampari_data/dev_data.jsonl"
    if not os.path.exists(path):
        return out
    for line in open(path):
        try:
            r = json.loads(line)
        except Exception:
            continue
        out[r["qid"]] = dict(
            question=r.get("question_text", ""),
            gold=", ".join(a.get("answer_text", "") for a in r.get("answer_list", [])[:8])
                  + (" ..." if len(r.get("answer_list", [])) > 8 else ""),
            n_gold=len(r.get("answer_list", [])),
        )
    return out


def _load_fanoutqa_questions() -> dict:
    try:
        import fanoutqa
        ds = fanoutqa.load_dev()
    except Exception:
        return {}
    out = {}
    for q in ds:
        gold = q.answer
        if isinstance(gold, dict):
            gold_str = "; ".join(f"{k}: {v}" for k, v in list(gold.items())[:6])
        else:
            gold_str = str(gold)[:300]
        out[q.id] = dict(question=q.question, gold=gold_str,
                         n_gold=len(gold) if isinstance(gold, (list, dict)) else 1)
    return out


def _load_swebench_questions() -> dict:
    try:
        from datasets import load_dataset
        ds = load_dataset("princeton-nlp/SWE-bench_Lite", split="test")
    except Exception:
        return {}
    out = {}
    for r in ds:
        out[r["instance_id"]] = dict(
            question=r.get("problem_statement", "")[:6000],
            gold=str(r.get("patch", ""))[:3000],
            n_gold=len(str(r.get("patch", "")).splitlines()),
        )
    return out


def get_question_meta(benchmark: str, task_id: str) -> dict:
    """Return dict(question, gold, n_gold) or empty dict if unavailable."""
    if benchmark not in _question_cache:
        loaders = {
            "qampari": _load_qampari_questions,
            "fanoutqa": _load_fanoutqa_questions,
            "swebench": _load_swebench_questions,
        }
        loader = loaders.get(benchmark)
        _question_cache[benchmark] = loader() if loader else {}
    return _question_cache[benchmark].get(str(task_id), {})


CONFIG_RE = re.compile(r"Qwen_Qwen3\.5-9B_([a-z]+)_k(\d+)(?:_R(\d+))?\.jsonl$")


def list_available_configs(benchmark: str) -> list:
    """Return list of (topology, k, rounds) tuples for files in this benchmark dir."""
    bench_dir = BENCH_DIR.get(benchmark, benchmark)
    base = os.path.join(RESULTS_BASE, bench_dir)
    if not os.path.isdir(base):
        return []
    out = []
    for fname in sorted(os.listdir(base)):
        m = CONFIG_RE.match(fname)
        if not m:
            continue
        topo = m.group(1)
        k = int(m.group(2))
        rounds = int(m.group(3)) if m.group(3) else 2
        out.append((topo, k, rounds))
    return out


def find_file(benchmark: str, topology: str, k: int, rounds: int) -> str:
    bench_dir = BENCH_DIR.get(benchmark, benchmark)
    base = os.path.join(RESULTS_BASE, bench_dir)
    suffix = "_R1" if rounds == 1 else ""
    fname = f"Qwen_Qwen3.5-9B_{topology}_k{k}{suffix}.jsonl"
    path = os.path.join(base, fname)
    if not os.path.exists(path):
        avail = list_available_configs(benchmark)
        if avail:
            avail_str = "\n  ".join(
                f"--topology {t} --k {k_} --rounds {r}" for t, k_, r in avail
            )
            raise FileNotFoundError(
                f"No file at {path}\n"
                f"Available configs for benchmark={benchmark}:\n  {avail_str}"
            )
        raise FileNotFoundError(
            f"No file at {path}\n"
            f"(no jsonl files found in {base} either)"
        )
    return path


def load_task(path: str, task_id: str = None, rep: int = 0, randomize: bool = False) -> dict:
    """Return a matching task record.

    - If task_id is given: return the matching record (with rep filter).
    - Else if randomize=True: pick a random non-error record.
    - Else: return the first non-error record.
    """
    candidates = []
    for line in open(path):
        try:
            r = json.loads(line)
        except Exception:
            continue
        if r.get("error"):
            continue
        if not r.get("call_records"):
            continue
        if task_id is not None and str(r.get("task_id")) != str(task_id):
            continue
        if not randomize and r.get("rep", 0) != rep:
            continue
        if randomize:
            candidates.append(r)
        else:
            return r
    if randomize and candidates:
        return random.choice(candidates)
    raise LookupError(f"No matching record in {path}")


# ----------------------------------------------------------------------
# Pretty printer (terminal + markdown)
# ----------------------------------------------------------------------

def _wrap(text: str, width: int = 100) -> list:
    """Simple word wrap for terminal output."""
    if not text:
        return []
    # Strip newlines and collapse whitespace for cleaner display
    text = " ".join(text.split())
    lines = []
    while len(text) > width:
        # Try to break at a space
        break_pt = text.rfind(" ", 0, width)
        if break_pt < width // 2:
            break_pt = width
        lines.append(text[:break_pt].rstrip())
        text = text[break_pt:].lstrip()
    if text:
        lines.append(text)
    return lines


def fmt_int(n):
    return f"{n:>6,}"


def fmt_energy(j):
    return f"{j:>8.0f}J"


def fmt_time(s):
    return f"{s:>6.1f}s"


def role_for(agent_id: str) -> str:
    if agent_id.startswith("orchestrator"):
        return "orchestrator"
    if agent_id.startswith("synth") or "synth" in agent_id:
        return "synthesizer"
    if agent_id.startswith("reviewer") or "review" in agent_id:
        return "reviewer"
    if "worker" in agent_id:
        m = re.search(r"_r(\d+)", agent_id)
        rnd = f"R{m.group(1)}" if m else ""
        return f"worker {rnd}"
    if "debater" in agent_id:
        m = re.search(r"_r(\d+)", agent_id)
        rnd = f"R{m.group(1)}" if m else ""
        return f"debater {rnd}"
    if agent_id.startswith("solver"):
        return "solver"
    return agent_id


def format_trace(record: dict, md: bool = False) -> str:
    out = []
    bench = record.get("benchmark", "?")
    topo = record.get("topology", "?")
    k = record.get("max_react_steps", "?")
    rounds = record.get("n_rounds_override")
    rounds_str = f"R={rounds}" if rounds else "R=2"
    task_id = record.get("task_id", "?")
    n_calls = len(record["call_records"])
    total_E = record.get("gpu_dynamic_energy_joules", 0) or 0
    total_wall = record.get("total_wall_seconds", 0) or 0
    total_P = record.get("total_prompt_tokens", 0) or 0
    total_C = record.get("total_completion_tokens", 0) or 0
    answer_full = record.get("answer") or ""
    answer_short = answer_full.replace("\n", " ").strip()
    correct = record.get("correct")
    acc_str = "✓" if correct else "✗" if correct is False else "?"
    subtasks = record.get("subtasks") or []

    # Try to load the task question and gold answer from benchmark sources
    qmeta = get_question_meta(bench, task_id)
    question_text = qmeta.get("question", "")
    gold_text = qmeta.get("gold", "")

    if md:
        out.append(f"### {bench} / {topo} k={k} {rounds_str} / task `{task_id}`")
        out.append("")
        out.append(f"- {n_calls} calls, total **{total_E:.0f}J**, **{total_wall:.1f}s** wall")
        out.append(f"- Σ prompt = {total_P:,} tok    Σ completion = {total_C:,} tok")
        out.append(f"- accuracy: **{acc_str}**")
        if question_text:
            out.append("")
            out.append("**Question:**")
            out.append("")
            out.append(f"> {question_text[:6000]}")
        if gold_text:
            out.append("")
            out.append(f"**Gold answer:** `{gold_text[:3000]}`")
        if subtasks:
            out.append("")
            out.append(f"**Orchestrator decomposition** ({len(subtasks)} subtasks):")
            out.append("")
            for i, st in enumerate(subtasks, 1):
                out.append(f"{i}. {str(st).strip()[:2000]}")
        if answer_full:
            out.append("")
            out.append(f"**Final answer:**")
            out.append("")
            out.append(f"> {answer_full[:8000]}")
        out.append("")
        out.append("**Per-call trace:**")
        out.append("")
        out.append("```")
    else:
        out.append("=" * 110)
        out.append(f"  {bench} / {topo} k={k} {rounds_str} / task {task_id}    [{acc_str}]")
        out.append(f"  {n_calls} calls   total {total_E:.0f}J   {total_wall:.1f}s wall   "
                   f"P_total={total_P:,}  C_total={total_C:,}")
        out.append("=" * 110)
        if question_text:
            out.append("QUESTION:")
            for line in _wrap(question_text[:6000], 104):
                out.append(f"  {line}")
            out.append("")
        if gold_text:
            out.append("GOLD ANSWER:")
            for line in _wrap(gold_text[:3000], 104):
                out.append(f"  {line}")
            out.append("")
        if subtasks:
            out.append(f"ORCHESTRATOR DECOMPOSITION ({len(subtasks)} subtasks):")
            for i, st in enumerate(subtasks, 1):
                wrapped = _wrap(str(st).strip()[:2000], 100)
                out.append(f"  {i}. {wrapped[0]}")
                for w in wrapped[1:]:
                    out.append(f"     {w}")
            out.append("")
        if answer_full:
            out.append("FINAL ANSWER:")
            for line in _wrap(answer_full[:8000], 104):
                out.append(f"  {line}")
            out.append("")
        out.append("PER-CALL TRACE:")

    # Header line
    header = (
        f"{'#':>3}  {'agent':<22} {'call_type':<18} "
        f"{'P':>7}  {'C':>7}  {'energy':>9}  {'wall':>7}  {'cum_wall':>8}"
    )
    out.append(header)
    out.append("-" * len(header))

    cum = 0.0
    last_role = None
    for i, c in enumerate(record["call_records"], 1):
        ct = c.get("call_type", "?")
        agent = c.get("agent_id", "?")
        role = role_for(agent)

        # Section break if role changed
        if role != last_role and last_role is not None:
            out.append("")
        last_role = role

        if ct == "tool_execution":
            tt = c.get("total_tokens", 0) or 0
            walls = c.get("wall_seconds", 0) or 0
            cum += walls
            out.append(
                f"{i:>3}  {agent:<22} {ct:<18} "
                f"{'-':>7}  {'-':>7}  {fmt_energy(0)}  {fmt_time(walls)}  {cum:>7.1f}s    "
                f"(tool result: {tt} tokens)"
            )
            tc = c.get("tool_call")
            if tc:
                out.append(f"      ↳ tool call: {tc.get('name')}({tc.get('arguments', '')[:1000]})")
                result_text = tc.get("result") or ""
                if result_text:
                    wrapped = _wrap(result_text, 96)
                    for line in wrapped[:40]:
                        out.append(f"         {line}")
                    if len(wrapped) > 40:
                        out.append(f"         ... ({len(wrapped) - 40} more lines)")
            continue

        P = c.get("prompt_tokens", 0) or 0
        C = c.get("completion_tokens", 0) or 0
        e = c.get("gpu_dynamic_energy_joules", 0) or 0
        walls = c.get("wall_seconds", 0) or 0
        cum += walls

        out.append(
            f"{i:>3}  {agent:<22} {ct:<18} "
            f"{fmt_int(P)}  {fmt_int(C)}  {fmt_energy(e)}  {fmt_time(walls)}  {cum:>7.1f}s"
        )

        # If transcript was saved, show prompt + response inline
        resp = c.get("response") or c.get("_transcript", {}).get("response_text")
        if isinstance(resp, dict):
            content = resp.get("content") or ""
            tcs = resp.get("tool_calls") or []
            if content:
                wrapped = _wrap(content, 96)
                for line in wrapped[:60]:
                    out.append(f"      ↳ {line}")
                if len(wrapped) > 60:
                    out.append(f"      ↳ ... ({len(wrapped) - 60} more lines)")
            for tc in tcs:
                out.append(f"      ↳ → tool call: {tc.get('name')}({tc.get('arguments', '')[:1000]})")
        elif isinstance(resp, str) and resp:
            wrapped = _wrap(resp, 96)
            for line in wrapped[:60]:
                out.append(f"      ↳ {line}")
            if len(wrapped) > 60:
                out.append(f"      ↳ ... ({len(wrapped) - 60} more lines)")

    if md:
        out.append("```")
        out.append("")
        # Aggregate by role for a quick summary
        out.append("**Per-role summary:**")
        out.append("")
        out.append("| Role | Calls | Σ P | Σ C | Σ Energy | Σ Wall |")
        out.append("|---|---|---|---|---|---|")
        agg = defaultdict(lambda: dict(n=0, P=0, C=0, E=0, W=0))
        for c in record["call_records"]:
            r = role_for(c.get("agent_id", "?"))
            ct = c.get("call_type", "?")
            if ct == "tool_execution":
                agg[r]["W"] += c.get("wall_seconds", 0) or 0
                continue
            agg[r]["n"] += 1
            agg[r]["P"] += c.get("prompt_tokens", 0) or 0
            agg[r]["C"] += c.get("completion_tokens", 0) or 0
            agg[r]["E"] += c.get("gpu_dynamic_energy_joules", 0) or 0
            agg[r]["W"] += c.get("wall_seconds", 0) or 0
        for r, d in agg.items():
            out.append(f"| {r} | {d['n']} | {d['P']:,} | {d['C']:,} | {d['E']:.0f}J | {d['W']:.1f}s |")
        out.append("")
    else:
        out.append("-" * len(header))
        # Per-role summary
        out.append("\nPer-role summary:")
        agg = defaultdict(lambda: dict(n=0, P=0, C=0, E=0, W=0))
        for c in record["call_records"]:
            r = role_for(c.get("agent_id", "?"))
            ct = c.get("call_type", "?")
            if ct == "tool_execution":
                agg[r]["W"] += c.get("wall_seconds", 0) or 0
                continue
            agg[r]["n"] += 1
            agg[r]["P"] += c.get("prompt_tokens", 0) or 0
            agg[r]["C"] += c.get("completion_tokens", 0) or 0
            agg[r]["E"] += c.get("gpu_dynamic_energy_joules", 0) or 0
            agg[r]["W"] += c.get("wall_seconds", 0) or 0
        out.append(f"  {'role':<18} {'n':>4}  {'Σ P':>10}  {'Σ C':>9}  {'Σ E':>10}  {'Σ wall':>9}")
        for r, d in agg.items():
            out.append(f"  {r:<18} {d['n']:>4}  {d['P']:>10,}  {d['C']:>9,}  "
                       f"{d['E']:>9.0f}J  {d['W']:>8.1f}s")
        out.append("")

    # Parallelism check note
    sum_wall = sum((c.get("wall_seconds", 0) or 0) for c in record["call_records"])
    if abs(sum_wall - total_wall) / max(total_wall, 1) < 0.05:
        note = "Σ per-call wall ≈ task wall → SERIAL execution (no overlap)"
    elif sum_wall > total_wall * 1.05:
        note = f"Σ per-call wall ({sum_wall:.1f}s) > task wall ({total_wall:.1f}s) → PARALLEL"
    else:
        note = f"Σ per-call wall ({sum_wall:.1f}s) < task wall ({total_wall:.1f}s) → idle gaps"
    if md:
        out.append(f"_Execution mode: {note}_")
        out.append("")
    else:
        out.append(f"Execution mode: {note}")
        out.append("")

    return "\n".join(out)


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Visualize MAS execution traces from a5000 result files.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
    # Specific task on Centralized
    python analysis/visualize_trace.py --benchmark qampari --topology centralized --k 5 \\
        --task-id 236__wikidata_intersection__dev

    # First non-error trace (default)
    python analysis/visualize_trace.py --benchmark qampari --topology decentralized --k 5

    # Random trace from the file
    python analysis/visualize_trace.py --benchmark qampari --topology decentralized --k 5 --random

    # Random trace from EVERY topology of a benchmark (one of each)
    python analysis/visualize_trace.py --benchmark qampari --random-each-topology

    # Random trace per topology, dumped to markdown
    python analysis/visualize_trace.py --benchmark qampari --random-each-topology --md traces.md
""",
    )
    ap.add_argument("--benchmark", default="qampari",
                    help="qampari | fanoutqa | browsecomp_plus | workbench | swebench")
    ap.add_argument("--topology", default="centralized",
                    help="sas | independent | centralized | decentralized | hybrid")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--task-id", default=None, help="specific task_id (overrides --random)")
    ap.add_argument("--rep", type=int, default=0)
    ap.add_argument("--random", action="store_true",
                    help="pick a random task from the matched file (instead of the first)")
    ap.add_argument("--all-topologies", action="store_true",
                    help="show traces for sas/independent/centralized/decentralized/hybrid (same task_id if given)")
    ap.add_argument("--random-each-topology", action="store_true",
                    help="show one RANDOM trace from each topology for the given benchmark")
    ap.add_argument("--seed", type=int, default=None,
                    help="random seed for reproducible random selection")
    ap.add_argument("--md", default=None, help="write markdown output to this file")
    args = ap.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    blocks = []

    if args.random_each_topology or args.all_topologies:
        avail = list_available_configs(args.benchmark)
        if not avail:
            print(f"No configs found for benchmark={args.benchmark}")
            return
        # One config per (topology, rounds), preferring user-specified k if it exists
        chosen = {}
        for topo, k, rounds in avail:
            key = (topo, rounds)
            if key not in chosen:
                chosen[key] = (topo, k, rounds)
            elif k == args.k:
                chosen[key] = (topo, k, rounds)
        # Stable canonical ordering
        order = ["sas", "independent", "centralized", "decentralized", "hybrid"]
        sorted_configs = sorted(
            chosen.values(),
            key=lambda c: (order.index(c[0]) if c[0] in order else 99, -c[2], c[1]),
        )
        for topo, k, rounds in sorted_configs:
            try:
                path = find_file(args.benchmark, topo, k, rounds)
                rec = load_task(
                    path,
                    task_id=args.task_id,
                    rep=args.rep,
                    randomize=args.random_each_topology or
                              (args.random and args.task_id is None),
                )
                blocks.append(format_trace(rec, md=bool(args.md)))
            except (FileNotFoundError, LookupError) as e:
                blocks.append(f"# {topo} k={k} R={rounds}: {e}\n")
    else:
        path = find_file(args.benchmark, args.topology, args.k, args.rounds)
        rec = load_task(path, args.task_id, args.rep, randomize=args.random and args.task_id is None)
        blocks.append(format_trace(rec, md=bool(args.md)))

    text = "\n\n".join(blocks)
    if args.md:
        with open(args.md, "w") as fh:
            fh.write("# MAS Execution Trace\n\n")
            fh.write(text)
        print(f"wrote {args.md}")
    else:
        print(text)


if __name__ == "__main__":
    main()
