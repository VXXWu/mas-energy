"""Fit per-call energy regression `E = a + b*P + c*C` on the A5000 dataset.

Operates on individual call_records (not task-aggregated rows), so we get
~tens of thousands of points and a tight per-call fit. Reports overall and
per-benchmark coefficients, R², and produces a predicted-vs-measured scatter.

Outputs:
    analysis/a5000_figs/energy_regression.png
    analysis/a5000_figs/energy_regression_table.csv
"""
from __future__ import annotations
import os
import json
import glob

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

OUT_DIR = "analysis/a5000_figs"
os.makedirs(OUT_DIR, exist_ok=True)

RESULTS_GLOB = "mas-energy/results/a5000_*/Qwen_Qwen3.5-9B_*.jsonl"


def load_call_level() -> pd.DataFrame:
    rows = []
    for path in sorted(glob.glob(RESULTS_GLOB)):
        if "/a5000_pilot/" in path:
            continue
        for line in open(path):
            try:
                rec = json.loads(line)
            except Exception:
                continue
            if rec.get("error"):
                continue
            calls = rec.get("call_records") or []
            for c in calls:
                if c.get("call_type") == "tool_execution":
                    continue
                if "prompt_tokens" not in c or "completion_tokens" not in c:
                    continue
                e = float(c.get("gpu_dynamic_energy_joules") or 0)
                P = int(c["prompt_tokens"])
                C = int(c["completion_tokens"])
                if e <= 0 and (P + C) > 0:
                    continue  # skip the zero-energy bug residue
                rows.append(dict(
                    benchmark=rec.get("benchmark", "?"),
                    topology=rec.get("topology", "?"),
                    P=P, C=C, E=e,
                    role=c.get("agent_id", "?"),
                    call_type=c.get("call_type", "?"),
                ))
    return pd.DataFrame(rows)


def fit(df: pd.DataFrame, label: str) -> dict:
    X = df[["P", "C"]].to_numpy(dtype=float)
    y = df["E"].to_numpy(dtype=float)
    model = LinearRegression().fit(X, y)
    pred = model.predict(X)
    return dict(
        label=label,
        n=len(df),
        a=float(model.intercept_),
        b=float(model.coef_[0]),
        c=float(model.coef_[1]),
        r2=float(r2_score(y, pred)),
        c_over_b=float(model.coef_[1] / model.coef_[0]) if model.coef_[0] != 0 else float("inf"),
    )


def plot(df: pd.DataFrame, fit_all: dict, out: str) -> None:
    a, b, c = fit_all["a"], fit_all["b"], fit_all["c"]
    pred = a + b * df["P"] + c * df["C"]
    df = df.copy()
    df["pred"] = pred

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: scatter measured vs predicted (log-log)
    ax = axes[0]
    ax.scatter(df["pred"], df["E"], s=3, alpha=0.18, color="#1f77b4")
    lo, hi = max(1, df["pred"].min()), df["pred"].max()
    ax.plot([lo, hi], [lo, hi], "k--", linewidth=1.2, label="y = x")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Predicted E (J) = a + b·P + c·C")
    ax.set_ylabel("Measured GPU dynamic energy (J)")
    ax.set_title(
        f"Per-call regression on A5000\n"
        f"E = {a:.2f} + {b:.4f}·P + {c:.3f}·C   |   R² = {fit_all['r2']:.4f}, n = {fit_all['n']}"
    )
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)

    # Right: residuals histogram
    ax = axes[1]
    resid = df["E"] - df["pred"]
    rel = resid / df["pred"].clip(lower=1)
    ax.hist(rel.clip(-2, 2), bins=80, color="#ff7f0e", edgecolor="black", linewidth=0.3)
    ax.axvline(0, color="black", linestyle="--", linewidth=1)
    ax.set_xlabel("Relative residual (E - Ê) / Ê")
    ax.set_ylabel("Calls")
    ax.set_title(f"Residual distribution\nmedian = {rel.median():.3f}, IQR = {rel.quantile(0.25):.2f}–{rel.quantile(0.75):.2f}")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"  saved {out}")


def main():
    print("Loading per-call records...")
    df = load_call_level()
    print(f"  {len(df)} call records, {df['benchmark'].nunique()} benchmarks, {df['topology'].nunique()} topologies")

    fits = []
    fits.append(fit(df, "ALL"))
    for b in sorted(df["benchmark"].unique()):
        fits.append(fit(df[df["benchmark"] == b], b))

    fdf = pd.DataFrame(fits)
    print()
    print(fdf.to_string(index=False, float_format=lambda x: f"{x:9.4f}"))
    fdf.to_csv(os.path.join(OUT_DIR, "energy_regression_table.csv"), index=False)
    plot(df, fits[0], os.path.join(OUT_DIR, "energy_regression.png"))


if __name__ == "__main__":
    main()
