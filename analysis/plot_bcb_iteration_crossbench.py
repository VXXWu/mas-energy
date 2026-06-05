"""
BCB cross-validation panel for the iteration-required-collapse claim.

Story: independent_share (single-shot, no debate) collapses on swebench
(-0.580 ΔAcc) but NOT on BigCodeBench (function-calling code). The
delta isolates "iterative write→test→fix cycles" from "code in general"
— repo-level multi-file edits drove the swebench collapse, not iteration
on any code task.

Inputs: paired Decent (baseline) vs IS jsonls, swebench + BCB.
Output: figures/bcb_iteration_crossbench.png
"""
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).parent.parent
RESULTS = ROOT / "mas-energy" / "results"

PAIRS = {
    "swebench": {
        "decent": RESULTS / "a5000_latent_transcripts" / "swebench" / "Qwen_Qwen3.5-9B_decentralized_k10.jsonl",
        "is":     RESULTS / "a5000_phase_b2_terse"   / "swebench" / "Qwen_Qwen3.5-9B_independent_share_k10.jsonl",
    },
    "bigcodebench": {
        "decent": RESULTS / "a5000_latent_transcripts" / "bigcodebench" / "Qwen_Qwen3.5-9B_decentralized_k10.jsonl",
        "is":     RESULTS / "a5000_phase_b2_terse"   / "bigcodebench" / "Qwen_Qwen3.5-9B_independent_share_k10.jsonl",
    },
}

def load_accs(path):
    """Returns dict {task_id: 1 or 0}."""
    out = {}
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            tid = r.get("task_id") or r.get("question_id") or r.get("instance_id")
            acc = r.get("correct", r.get("accuracy", r.get("acc")))
            if isinstance(acc, bool):
                acc = int(acc)
            if tid is not None and acc is not None:
                out[tid] = float(acc)
    return out

def paired_delta(d_decent, d_is):
    """Paired-set ΔAcc on shared task_ids; returns (n, mean_decent, mean_is, delta)."""
    shared = sorted(set(d_decent) & set(d_is))
    if not shared:
        return 0, np.nan, np.nan, np.nan
    a = np.array([d_decent[t] for t in shared])
    b = np.array([d_is[t]     for t in shared])
    return len(shared), a.mean(), b.mean(), (b - a).mean()

def boot_ci(d_decent, d_is, n_boot=10000, ci=0.95):
    shared = sorted(set(d_decent) & set(d_is))
    a = np.array([d_decent[t] for t in shared])
    b = np.array([d_is[t]     for t in shared])
    diffs = b - a
    n = len(diffs)
    rng = np.random.default_rng(0)
    boot = np.array([rng.choice(diffs, size=n, replace=True).mean() for _ in range(n_boot)])
    lo = np.quantile(boot, (1 - ci) / 2)
    hi = np.quantile(boot, 1 - (1 - ci) / 2)
    return lo, hi

rows = []
for bench, paths in PAIRS.items():
    d = load_accs(paths["decent"])
    i = load_accs(paths["is"])
    n, a_dec, a_is, delta = paired_delta(d, i)
    lo, hi = boot_ci(d, i)
    rows.append((bench, n, a_dec, a_is, delta, lo, hi))
    print(f"{bench:14s}  n={n:3d}  Decent={a_dec:.3f}  IS={a_is:.3f}  ΔAcc={delta:+.3f}  CI95=[{lo:+.3f},{hi:+.3f}]")

fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(10, 4.2), gridspec_kw={"width_ratios": [1.0, 1.0]})

# Panel 0: paired bars per benchmark
benches = [r[0].replace("bigcodebench", "BigCodeBench").replace("swebench", "SWE-bench") for r in rows]
decents = [r[2] for r in rows]
iss     = [r[3] for r in rows]
x = np.arange(len(benches))
w = 0.35
ax0.bar(x - w/2, decents, w, label="Decentralized (baseline)", color="#2E86AB")
ax0.bar(x + w/2, iss,     w, label="Independent + share (no debate)", color="#E63946")
ax0.set_xticks(x); ax0.set_xticklabels(benches, fontsize=10)
ax0.set_ylabel("Accuracy"); ax0.set_ylim(0, 1)
ax0.set_title("Decent vs Independent-share")
ax0.legend(loc="upper left", fontsize=8, frameon=False)
for xi, dec, isv in zip(x, decents, iss):
    ax0.text(xi - w/2, dec + 0.02, f"{dec:.2f}", ha="center", fontsize=9)
    ax0.text(xi + w/2, isv + 0.02, f"{isv:.2f}", ha="center", fontsize=9)

# Panel 1: ΔAcc with CI95 — the collapse delta
deltas = [r[4] for r in rows]
los    = [r[5] for r in rows]
his    = [r[6] for r in rows]
err_lo = [d - lo for d, lo in zip(deltas, los)]
err_hi = [hi - d for d, hi in zip(deltas, his)]
colors = ["#E63946" if d < -0.05 else "#777" for d in deltas]
ax1.barh(benches, deltas, color=colors, xerr=[err_lo, err_hi], capsize=4, alpha=0.85)
ax1.axvline(0,     color="k",   linestyle="-",  linewidth=0.8)
ax1.axvline(-0.05, color="grey", linestyle="--", linewidth=0.8, alpha=0.6, label="±0.05 null band")
ax1.axvline(+0.05, color="grey", linestyle="--", linewidth=0.8, alpha=0.6)
ax1.set_xlabel("ΔAcc (IS − Decent), paired"); ax1.set_title("Iteration-required collapse delta")
ax1.legend(loc="lower right", fontsize=8, frameon=False)
for i, (d, lo, hi) in enumerate(zip(deltas, los, his)):
    ax1.text(d + (0.02 if d >= 0 else -0.02), i,
             f"{d:+.2f} [{lo:+.2f},{hi:+.2f}]",
             va="center", ha="left" if d >= 0 else "right", fontsize=9)

fig.suptitle("BCB cross-validates: IS collapses on SWE-bench, not on BigCodeBench\n"
             "→ repo-level multi-file iteration, not code-task iteration in general",
             fontsize=11)
fig.tight_layout()
out = ROOT / "figures" / "bcb_iteration_crossbench.png"
fig.savefig(out, dpi=140, bbox_inches="tight")
print(f"\nsaved {out}")
