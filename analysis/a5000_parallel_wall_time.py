"""Estimate parallel wall-clock time per task by re-aggregating per-call wall_seconds.

The dataset is recorded under serial execution: total_wall_seconds = sum over all
LLM calls. A production MAS system runs workers within the same round in parallel,
synchronizing only at round boundaries (orchestrator review, debate update, etc.).

For each task we compute:
  serial_wall    = sum over all calls (already in record)
  parallel_wall  = sum of phase wall times, where each phase's wall =
                     max over agents of (sum of that agent's calls in this phase)

Phase detection from call_records:
  - The call sequence is naturally grouped by (role, round_index).
  - We detect phase boundaries when the role-round changes.
  - Within a phase, multiple agents contribute calls in series (because the data is
    serialized), but in production they would run in parallel: max-over-agents.

Outputs:
    analysis/a5000_figs/parallel_wall_time.png
    analysis/a5000_figs/parallel_wall_time.csv
"""
from __future__ import annotations
import os
import json
import glob
import re
from collections import defaultdict, OrderedDict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

OUT_DIR = "analysis/a5000_figs"
os.makedirs(OUT_DIR, exist_ok=True)
RESULTS_GLOB = "mas-energy/results/a5000_*/Qwen_Qwen3.5-9B_*.jsonl"

BENCH_ORDER = ["qampari", "fanoutqa", "browsecomp_plus", "workbench", "swebench"]
BENCH_LABEL = {
    "qampari": "QAMPARI", "fanoutqa": "FanOutQA",
    "browsecomp_plus": "BrowseComp+", "workbench": "WorkBench",
    "swebench": "SWE-bench",
}
TOPO_ORDER = ["sas", "independent", "centralized_R1", "centralized",
              "decentralized", "hybrid_R1", "hybrid"]
TOPO_LABEL = {
    "sas": "SAS",
    "independent": "Independent",
    "centralized_R1": "Centralized R=1",
    "centralized": "Centralized R=2",
    "decentralized": "Decentralized R=2",
    "hybrid_R1": "Hybrid R=1",
    "hybrid": "Hybrid R=2",
}
TOPO_COLOR = {
    "sas":            "#1f77b4",
    "independent":    "#ff7f0e",
    "centralized_R1": "#98df8a",
    "centralized":    "#2ca02c",
    "decentralized":  "#d62728",
    "hybrid_R1":      "#c5b0d5",
    "hybrid":         "#9467bd",
}

CONFIG_RE = re.compile(r"Qwen_Qwen3\.5-9B_([a-z]+)_k(\d+)(?:_R(\d+))?\.jsonl")


# ----------------------------------------------------------------------
# Phase detection
# ----------------------------------------------------------------------

def parse_agent_role(agent_id: str) -> tuple:
    """Return (role, round_idx) where round_idx may be None for non-round agents."""
    if agent_id.startswith("orchestrator"):
        return ("orchestrator", None)
    if agent_id.startswith("synthesizer") or "synth" in agent_id:
        return ("synthesizer", None)
    if agent_id.startswith("sas") or agent_id == "sas_agent":
        return ("sas", None)
    if agent_id.startswith("independent_"):
        return ("independent", None)
    if agent_id.startswith("solver"):
        return ("solver", None)
    # Match worker/debater/peer with optional 'hybrid_' prefix.
    # Accepts:
    #   worker_0_r0,  worker_0_r1, ...
    #   debater_0_init,  debater_0_r0, ...
    #   hybrid_worker_0_r0, hybrid_peer_0_r0_p0, ...
    m = re.match(r"(?:hybrid_)?(worker|debater|peer)_(\d+)_(?:r(\d+)|(init))", agent_id)
    if m:
        role = m.group(1)
        if m.group(4) == "init":
            return (role, -1)
        return (role, int(m.group(3)))
    return (agent_id, None)


def detect_phases(call_records: list) -> list:
    """Group consecutive calls into phases based on (role, round_idx) of agent.

    Returns a list of phases, each as dict(role, round_idx, agents={agent_id: [calls]}).
    """
    phases = []
    cur = None
    for c in call_records:
        if c.get("call_type") == "tool_execution":
            # attribute the tool execution to the prior call's agent
            if cur is not None:
                agent = c.get("agent_id") or "?"
                cur["agents"].setdefault(agent, []).append(c)
            continue
        agent = c.get("agent_id") or "?"
        role, rnd = parse_agent_role(agent)
        # Phase key: orchestrator/synthesizer are sequential singletons,
        # workers/debaters group by round index, independent solvers form one phase.
        if role in ("orchestrator", "synthesizer"):
            # Disambiguate orchestrator phases by call_type (decompose, review_r0, review_r1, synthesis)
            ct = c.get("call_type", "?")
            phase_key = (role, ct)
        elif role in ("worker", "debater", "peer"):
            phase_key = (role, rnd)
        elif role == "independent":
            phase_key = ("independent", None)
        else:
            phase_key = (role, None)

        if cur is None or cur["key"] != phase_key:
            cur = dict(key=phase_key, role=role, round_idx=rnd, agents=OrderedDict())
            phases.append(cur)
        cur["agents"].setdefault(agent, []).append(c)
    return phases


def parallel_wall_time(call_records: list) -> tuple:
    """Return (serial_wall, parallel_wall, n_phases, n_parallel_phases)."""
    phases = detect_phases(call_records)
    serial = 0.0
    parallel = 0.0
    n_par = 0
    for ph in phases:
        agent_walls = []
        for agent, calls in ph["agents"].items():
            agent_walls.append(sum((c.get("wall_seconds", 0) or 0) for c in calls))
        if not agent_walls:
            continue
        serial += sum(agent_walls)
        # Parallel: if there are multiple agents AND this is a parallelizable role
        if len(ph["agents"]) > 1 and ph["role"] in ("worker", "debater", "peer", "independent"):
            parallel += max(agent_walls)
            n_par += 1
        else:
            parallel += sum(agent_walls)
    return serial, parallel, len(phases), n_par


# ----------------------------------------------------------------------
# Loader
# ----------------------------------------------------------------------

def load_records() -> pd.DataFrame:
    rows = []
    for path in sorted(glob.glob(RESULTS_GLOB)):
        if "/a5000_pilot/" in path:
            continue
        m = CONFIG_RE.match(os.path.basename(path))
        if not m:
            continue
        topo, k, r = m.group(1), int(m.group(2)), m.group(3)
        config_key = f"{topo}_R1" if r == "1" else topo
        for line in open(path):
            try:
                rec = json.loads(line)
            except Exception:
                continue
            if rec.get("error"):
                continue
            calls = rec.get("call_records") or []
            if not calls:
                continue
            e = rec.get("gpu_dynamic_energy_joules", 0) or 0
            P = rec.get("total_prompt_tokens", 0) or 0
            C = rec.get("total_completion_tokens", 0) or 0
            pred = -84 + 0.018 * P + 5.54 * C
            if pred > 0 and e / pred < 0.1:
                continue
            ser, par, n_ph, n_par = parallel_wall_time(calls)
            rows.append(dict(
                benchmark=rec.get("benchmark", "?"),
                topology=topo,
                config_key=config_key,
                k=k,
                task_id=str(rec.get("task_id", "?")),
                serial_wall=ser,
                parallel_wall=par,
                n_phases=n_ph,
                n_par_phases=n_par,
            ))
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------
# Aggregate + plot
# ----------------------------------------------------------------------

def aggregate(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["benchmark", "config_key", "k", "task_id"]).copy()
    df = df.groupby(["benchmark", "config_key", "k"]).head(50)
    return (
        df.groupby(["benchmark", "config_key", "k"], as_index=False)
          .agg(
              n=("task_id", "count"),
              serial_wall=("serial_wall", "mean"),
              parallel_wall=("parallel_wall", "mean"),
              speedup=("serial_wall", lambda s: float(s.mean())),
          )
    )


def plot(agg: pd.DataFrame, out: str) -> None:
    benches = [b for b in BENCH_ORDER if b in agg["benchmark"].unique()]
    n = len(benches)
    fig, axes = plt.subplots(1, n, figsize=(4.6 * n, 5.5), sharey=False)
    axes = np.atleast_1d(axes)

    for ax, bench in zip(axes, benches):
        sub = agg[agg["benchmark"] == bench].copy()
        if sub.empty:
            ax.set_visible(False)
            continue
        # For each (config, k), one bar pair (serial / parallel)
        sub["order"] = sub["config_key"].map(
            lambda c: TOPO_ORDER.index(c) if c in TOPO_ORDER else 99
        )
        sub = sub.sort_values(["order", "k"])
        labels = [f"{TOPO_LABEL.get(r['config_key'], r['config_key'])} k={int(r['k'])}"
                  for _, r in sub.iterrows()]
        x = np.arange(len(sub))
        w = 0.4
        ax.barh(x - w/2, sub["serial_wall"], w, color="#888888",
                edgecolor="black", linewidth=0.4, label="Serial (measured)")
        ax.barh(x + w/2, sub["parallel_wall"], w,
                color=[TOPO_COLOR.get(c, "gray") for c in sub["config_key"]],
                edgecolor="black", linewidth=0.4, label="Parallel (simulated)")
        for xi, (s, p) in enumerate(zip(sub["serial_wall"], sub["parallel_wall"])):
            speedup = s / max(p, 0.01)
            ax.text(max(s, p) * 1.04, xi, f"{speedup:.1f}×",
                    va="center", fontsize=7, color="#444")
        ax.set_yticks(x)
        ax.set_yticklabels(labels, fontsize=7)
        ax.set_xlabel("Wall-clock time per task (s)")
        ax.set_title(BENCH_LABEL.get(bench, bench), fontsize=11)
        ax.grid(True, axis="x", alpha=0.3)
        if ax is axes[0]:
            ax.legend(fontsize=8, loc="lower right")

    fig.suptitle(
        "Wall-clock time per task: serial (measured) vs parallel (simulated)\n"
        "Parallel = phase-by-phase reduction with max-over-agents at each round boundary\n"
        "Speedup ratios annotated to the right of each bar pair",
        y=1.03, fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"  saved {out}")


def main():
    print("Loading task records and computing parallel wall times...")
    df = load_records()
    print(f"  {len(df)} tasks processed")

    agg = aggregate(df)
    agg["speedup"] = agg["serial_wall"] / agg["parallel_wall"].clip(lower=0.01)
    agg.to_csv(os.path.join(OUT_DIR, "parallel_wall_time.csv"), index=False)
    print(f"  wrote parallel_wall_time.csv ({len(agg)} cells)")

    print("\n=== Median speedup per topology ===")
    by_topo = agg.groupby("config_key")["speedup"].agg(["mean", "median", "min", "max"])
    print(by_topo.sort_values("median", ascending=False).to_string(float_format=lambda x: f"{x:.2f}"))

    print("\n=== Per-benchmark median speedup ===")
    by_bench = agg.groupby("benchmark")["speedup"].agg(["mean", "median"])
    print(by_bench.to_string(float_format=lambda x: f"{x:.2f}"))

    plot(agg, os.path.join(OUT_DIR, "parallel_wall_time.png"))


if __name__ == "__main__":
    main()
