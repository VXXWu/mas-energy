"""Per-configuration energy amplification figure for the A5000 dataset.

Amplification = (energy_topology / energy_SAS) / (tokens_topology / tokens_SAS)

That ratio is the cleanest test of the "tokens != energy" claim:
  amp == 1 → tokens predict energy perfectly
  amp >  1 → energy is *more* expensive than tokens predict (decode-heavy)
  amp <  1 → energy is *cheaper* than tokens predict (prefill-heavy)

Each (topology, k) is a horizontal bar; one panel per benchmark.
SAS is anchored at k = max-of-the-config (matched).

Outputs:
    analysis/a5000_figs/amplification_per_config.png
    analysis/a5000_figs/amplification_per_config.csv
"""
from __future__ import annotations
import os
import json
import glob
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

RESULTS_GLOB = "mas-energy/results/a5000_*/Qwen_Qwen3.5-9B_*.jsonl"
OUT_DIR = "analysis/a5000_figs"
os.makedirs(OUT_DIR, exist_ok=True)

CONFIG_RE = re.compile(r"Qwen_Qwen3\.5-9B_([a-z]+)_k(\d+)(?:_R(\d+))?\.jsonl")

TOPO_COLOR = {
    "sas":            "#1f77b4",
    "independent":    "#ff7f0e",
    "centralized":    "#2ca02c",
    "centralized_R1": "#98df8a",
    "decentralized":  "#d62728",
    "hybrid":         "#9467bd",
    "hybrid_R1":      "#c5b0d5",
}
BENCH_ORDER = ["qampari", "fanoutqa", "browsecomp_plus", "workbench", "swebench"]
BENCH_LABEL = {
    "qampari": "QAMPARI", "fanoutqa": "FanOutQA",
    "browsecomp_plus": "BrowseComp+", "workbench": "WorkBench",
    "swebench": "SWE-bench",
}


def load() -> pd.DataFrame:
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
                d = json.loads(line)
            except Exception:
                continue
            if d.get("error"):
                continue
            e = d.get("gpu_dynamic_energy_joules", 0) or 0
            P = d.get("total_prompt_tokens", 0) or 0
            C = d.get("total_completion_tokens", 0) or 0
            pred = -84 + 0.018 * P + 5.54 * C
            if pred > 0 and e / pred < 0.1:
                continue
            rows.append(dict(
                benchmark=d.get("benchmark", "?"),
                config_key=config_key, k=k, task_id=d.get("task_id"),
                rep=d.get("rep", 0),
                energy_J=float(e),
                tokens=int(P) + int(C),
            ))
    return pd.DataFrame(rows)


def aggregate(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["benchmark", "config_key", "k", "task_id", "rep"]).copy()
    df = df.groupby(["benchmark", "config_key", "k"]).head(50)
    return (
        df.groupby(["benchmark", "config_key", "k"], as_index=False)
          .agg(n=("task_id", "count"),
               energy_J=("energy_J", "mean"),
               tokens=("tokens", "mean"))
    )


def amplification(agg: pd.DataFrame) -> pd.DataFrame:
    """For each (benchmark, k), compute amp = (E_topo/E_sas) / (T_topo/T_sas) per topology."""
    out = []
    for (bench, k), g in agg.groupby(["benchmark", "k"]):
        sas = g[g["config_key"] == "sas"]
        if sas.empty:
            continue
        sas_E = float(sas["energy_J"].iloc[0])
        sas_T = float(sas["tokens"].iloc[0])
        for _, r in g.iterrows():
            if r["config_key"] == "sas":
                continue
            if sas_T <= 0 or sas_E <= 0:
                continue
            energy_ratio = r["energy_J"] / sas_E
            token_ratio = r["tokens"] / sas_T
            amp = energy_ratio / token_ratio if token_ratio > 0 else np.nan
            out.append(dict(
                benchmark=bench, k=int(k), config_key=r["config_key"],
                energy_ratio=energy_ratio, token_ratio=token_ratio,
                amplification=amp,
            ))
    return pd.DataFrame(out)


def plot(amp: pd.DataFrame, out: str) -> None:
    benches = [b for b in BENCH_ORDER if b in amp["benchmark"].unique()]
    n = len(benches)
    fig, axes = plt.subplots(1, n, figsize=(4.4 * n, 6.5), squeeze=False, sharey=False)
    axes = axes[0]

    for ax, bench in zip(axes, benches):
        sub = amp[amp["benchmark"] == bench].copy()
        if sub.empty:
            ax.set_visible(False)
            continue
        # Sort: by topology (canonical order), then by k
        canonical = ["independent", "centralized_R1", "centralized",
                     "decentralized", "hybrid_R1", "hybrid"]
        sub["topo_order"] = sub["config_key"].map(
            lambda c: canonical.index(c) if c in canonical else 99)
        sub = sub.sort_values(["topo_order", "k"])
        labels = [f"{r['config_key']} k={int(r['k'])}" for _, r in sub.iterrows()]
        colors = [TOPO_COLOR.get(c, "gray") for c in sub["config_key"]]
        y = np.arange(len(sub))
        ax.barh(y, sub["amplification"], color=colors, edgecolor="black", linewidth=0.4)
        ax.axvline(1.0, color="black", linestyle=":", linewidth=1)
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=7)
        ax.set_xlabel("Amplification (energy / token cost ratio)", fontsize=8)
        ax.set_title(BENCH_LABEL.get(bench, bench), fontsize=11)
        ax.grid(True, axis="x", alpha=0.3)
        ax.tick_params(labelsize=8)
        # Add numeric values on the bars
        for yi, v in zip(y, sub["amplification"]):
            ax.text(v + 0.02, yi, f"{v:.2f}", va="center", fontsize=6.5)

    fig.suptitle(
        "A5000 Energy Amplification per Configuration\n"
        "(energy_ratio / token_ratio vs SAS at matched k)  —  >1 = energy costlier than tokens predict",
        y=1.01, fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"  saved {out}")


def main():
    df = load()
    print(f"Loaded {len(df)} records")
    agg = aggregate(df)
    amp = amplification(agg)
    amp.to_csv(os.path.join(OUT_DIR, "amplification_per_config.csv"), index=False)
    plot(amp, os.path.join(OUT_DIR, "amplification_per_config.png"))
    # Also print median amp per benchmark
    print("\nMedian amplification per benchmark:")
    print(amp.groupby("benchmark")["amplification"].median().round(3).to_string())


if __name__ == "__main__":
    main()
