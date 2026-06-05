"""Quantify topology-dependent context growth from existing per-call records.

For each task, group call_records by agent_id (each agent is one trajectory),
and within each trajectory measure how prompt-token count P grows turn by turn,
plus the within-trajectory correlation between P and C (the "compound cost"
slope: longer context induces longer completions).

Reports:
  1. Per-(benchmark, topology) table:
       P_initial, P_final, multiplier, mean delta_P, mean trajectories per task,
       compound_cost_slope (within-trajectory C vs P, log-log).
  2. Scatter plot: C vs P faceted by topology, with per-topology fit lines.
  3. Stacked bar of P decomposition (initial vs growth from prior turns).

Run from repo root:
    python analysis/quantify_context_growth.py
"""
from __future__ import annotations
import json
import glob
import os
from collections import defaultdict

import numpy as np
import pandas as pd

RESULTS_GLOB = "mas-energy/results/a5000_*/Qwen_Qwen3.5-9B_*.jsonl"
OUT_DIR = "analysis/context_growth_out"
os.makedirs(OUT_DIR, exist_ok=True)


def load_call_level_df() -> pd.DataFrame:
    """Walk all per-task JSONLs and emit one row per LLM call (skip tool calls)."""
    rows = []
    for path in sorted(glob.glob(RESULTS_GLOB)):
        # Topology comes from filename: Qwen_Qwen3.5-9B_<topology>_k<k>[_R1].jsonl
        fname = os.path.basename(path)
        stem = fname.replace("Qwen_Qwen3.5-9B_", "").replace(".jsonl", "")
        # benchmark from parent dir
        benchmark = os.path.basename(os.path.dirname(path)).replace("a5000_", "").replace("_v4", "").replace("_pilot", "").replace("_v2", "")
        for line in open(path):
            r = json.loads(line)
            if r.get("error"):
                continue
            calls = r.get("call_records") or []
            if not calls:
                continue
            topology = r.get("topology", "?")
            task_id = r.get("task_id", "?")
            rep = r.get("rep", 0)
            # Walk calls in order; assign turn index per agent_id
            per_agent_turn = defaultdict(int)
            for c in calls:
                if c.get("call_type") == "tool_execution":
                    continue
                if "prompt_tokens" not in c or "completion_tokens" not in c:
                    continue
                agent = c.get("agent_id") or "default"
                turn = per_agent_turn[agent]
                per_agent_turn[agent] += 1
                rows.append(
                    dict(
                        benchmark=benchmark,
                        topology=topology,
                        config=stem,
                        task_id=task_id,
                        rep=rep,
                        agent_id=agent,
                        call_type=c.get("call_type", "?"),
                        turn_idx=turn,
                        P=int(c["prompt_tokens"]),
                        C=int(c["completion_tokens"]),
                        E=float(c.get("gpu_dynamic_energy_joules", 0.0)),
                    )
                )
    return pd.DataFrame(rows)


def topology_growth_table(df: pd.DataFrame) -> pd.DataFrame:
    """Per (benchmark, topology) summary of context growth and compound cost."""
    out = []
    for (bench, topo), g in df.groupby(["benchmark", "topology"]):
        # Per-(task, agent) trajectories
        traj_stats = []
        compound_slopes = []
        for (_, _, _), tr in g.groupby(["task_id", "rep", "agent_id"]):
            tr = tr.sort_values("turn_idx")
            if len(tr) < 1:
                continue
            P0 = int(tr["P"].iloc[0])
            Pf = int(tr["P"].iloc[-1])
            n_turns = len(tr)
            traj_stats.append((P0, Pf, n_turns))
            # Compound-cost slope: regress log(C+1) on log(P+1) within trajectory
            if len(tr) >= 3:
                lp = np.log1p(tr["P"].to_numpy(dtype=float))
                lc = np.log1p(tr["C"].to_numpy(dtype=float))
                if lp.std() > 1e-6:
                    slope = np.polyfit(lp, lc, 1)[0]
                    compound_slopes.append(slope)
        if not traj_stats:
            continue
        P0_arr = np.array([s[0] for s in traj_stats])
        Pf_arr = np.array([s[1] for s in traj_stats])
        nt_arr = np.array([s[2] for s in traj_stats])
        mult = np.where(P0_arr > 0, Pf_arr / np.maximum(P0_arr, 1), np.nan)
        out.append(
            dict(
                benchmark=bench,
                topology=topo,
                n_trajectories=len(traj_stats),
                mean_turns_per_traj=float(nt_arr.mean()),
                P_initial_mean=float(P0_arr.mean()),
                P_final_mean=float(Pf_arr.mean()),
                multiplier_mean=float(np.nanmean(mult)),
                multiplier_p50=float(np.nanmedian(mult)),
                compound_cost_slope_mean=float(np.mean(compound_slopes)) if compound_slopes else np.nan,
                compound_cost_slope_n=len(compound_slopes),
            )
        )
    return pd.DataFrame(out).sort_values(["benchmark", "topology"])


def cross_call_compound_slope(df: pd.DataFrame) -> pd.DataFrame:
    """Pooled C vs P slope per topology (all calls together, not within-trajectory).

    A positive slope is the unconditional compound-cost effect: across the
    population of calls, longer prompts come with longer completions. This is
    the headline statistic; the within-trajectory version above controls for
    task identity.
    """
    out = []
    for topo, g in df.groupby("topology"):
        if len(g) < 30:
            continue
        lp = np.log1p(g["P"].to_numpy(dtype=float))
        lc = np.log1p(g["C"].to_numpy(dtype=float))
        if lp.std() < 1e-6:
            continue
        slope, intercept = np.polyfit(lp, lc, 1)
        # Pearson r on log-log
        r = float(np.corrcoef(lp, lc)[0, 1])
        out.append(
            dict(
                topology=topo,
                n_calls=len(g),
                slope_loglog=float(slope),
                intercept_loglog=float(intercept),
                pearson_r_loglog=r,
                P_median=float(g["P"].median()),
                C_median=float(g["C"].median()),
            )
        )
    return pd.DataFrame(out).sort_values("slope_loglog", ascending=False)


def make_scatter(df: pd.DataFrame, path: str) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"matplotlib not available, skipping plot ({e})")
        return
    topos = sorted(df["topology"].dropna().unique())
    n = len(topos)
    cols = min(3, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4.5 * cols, 4 * rows), sharex=True, sharey=True)
    axes = np.atleast_1d(axes).ravel()
    for ax, topo in zip(axes, topos):
        g = df[df["topology"] == topo]
        if len(g) < 10:
            ax.set_visible(False)
            continue
        ax.scatter(g["P"], g["C"], s=4, alpha=0.25)
        # Fit line in log-log space
        lp = np.log1p(g["P"].to_numpy(dtype=float))
        lc = np.log1p(g["C"].to_numpy(dtype=float))
        if lp.std() > 1e-6:
            slope, intercept = np.polyfit(lp, lc, 1)
            xs = np.linspace(lp.min(), lp.max(), 50)
            ys = intercept + slope * xs
            ax.plot(np.expm1(xs), np.expm1(ys), color="red", lw=1.5,
                    label=f"slope(log-log)={slope:.2f}")
            ax.legend(fontsize=8, loc="upper left")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(topo)
        ax.set_xlabel("prompt tokens P")
        ax.set_ylabel("completion tokens C")
    for ax in axes[len(topos):]:
        ax.set_visible(False)
    fig.suptitle("Compound cost: completion tokens vs prompt tokens (per call, by topology)", y=1.02)
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    print(f"  saved {path}")


def main() -> None:
    print("Loading call-level dataframe...")
    df = load_call_level_df()
    print(f"  {len(df)} LLM calls across {df['task_id'].nunique()} unique task_ids, "
          f"{df['benchmark'].nunique()} benchmarks, {df['topology'].nunique()} topologies")

    df.to_parquet(os.path.join(OUT_DIR, "calls.parquet"), index=False)

    print("\n=== Per-(benchmark, topology) context growth ===")
    growth = topology_growth_table(df)
    print(growth.to_string(index=False, float_format=lambda x: f"{x:7.2f}"))
    growth.to_csv(os.path.join(OUT_DIR, "growth_table.csv"), index=False)

    print("\n=== Pooled compound-cost slope per topology (log-log) ===")
    slopes = cross_call_compound_slope(df)
    print(slopes.to_string(index=False, float_format=lambda x: f"{x:7.3f}"))
    slopes.to_csv(os.path.join(OUT_DIR, "compound_slopes.csv"), index=False)

    print("\nGenerating scatter plot...")
    make_scatter(df, os.path.join(OUT_DIR, "compound_cost_scatter.png"))

    print("\nDone. Outputs in", OUT_DIR)


if __name__ == "__main__":
    main()
