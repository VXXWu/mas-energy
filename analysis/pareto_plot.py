"""Plot Pareto curves from sweep results: accuracy vs cost (energy, wall time, tokens)."""

import json
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np

RESULTS_FILE = os.path.join(MAS_ENERGY_ROOT, "results/pareto/Qwen_Qwen3.5-9B.jsonl")
OUT_DIR = os.path.join(MAS_ENERGY_ROOT, "results/pareto")

TOPO_STYLE = {
    "sas":           {"color": "#1f77b4", "marker": "o", "label": "SAS (single agent)"},
    "independent":   {"color": "#ff7f0e", "marker": "s", "label": "Independent (3 agents)"},
    "centralized":   {"color": "#2ca02c", "marker": "^", "label": "Centralized (orchestrator)"},
    "decentralized": {"color": "#d62728", "marker": "D", "label": "Decentralized (debate)"},
}

STEP_BUDGETS = [2, 5, 10, 20]

# Load and group data
groups = defaultdict(list)
for line in open(RESULTS_FILE):
    d = json.loads(line)
    if "error" in d and "answer" not in d:
        continue
    key = (d["topology"], d.get("max_react_steps"))
    groups[key].append(d)

# Compute per-group aggregates
data = {}
for key, recs in groups.items():
    topo, steps = key
    n = len(recs)
    data[key] = {
        "acc": sum(1 for r in recs if r.get("correct")) / n,
        "acc_se": np.sqrt(sum(1 for r in recs if r.get("correct")) / n
                          * (1 - sum(1 for r in recs if r.get("correct")) / n) / n),
        "gpu_energy": sum(r.get("gpu_energy_joules", 0) for r in recs) / n,
        "wall": sum(r.get("total_wall_seconds", 0) for r in recs) / n,
        "tokens": sum(r.get("total_tokens", 0) for r in recs) / n,
        "n": n,
    }

# --- Plotting ---
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
metrics = [
    ("wall", "Avg Wall Time per Task (s)"),
    ("tokens", "Avg Tokens per Task"),
    ("gpu_energy", "Avg GPU Energy per Task (J)"),
]

for ax, (metric, xlabel) in zip(axes, metrics):
    for topo, style in TOPO_STYLE.items():
        xs, ys, se = [], [], []
        for s in STEP_BUDGETS:
            key = (topo, s)
            if key not in data:
                continue
            xs.append(data[key][metric])
            ys.append(data[key]["acc"] * 100)
            se.append(data[key]["acc_se"] * 100)

        ax.errorbar(xs, ys, yerr=se, **style, linewidth=1.5, markersize=7,
                     capsize=3, linestyle="-")

        # Annotate step budgets
        for i, s in enumerate(STEP_BUDGETS):
            if (topo, s) in data and i < len(xs):
                ax.annotate(f"{s}", (xs[i], ys[i]), textcoords="offset points",
                           xytext=(5, 5), fontsize=7, color=style["color"])

    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel("Accuracy (%)", fontsize=11)
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(True, alpha=0.3)

fig.suptitle("Pareto Curves: Accuracy vs Cost (Qwen3.5-9B, WorkBench, n=50)",
             fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/pareto_curves.png", dpi=150, bbox_inches="tight")
plt.savefig(f"{OUT_DIR}/pareto_curves.pdf", bbox_inches="tight")
print(f"Saved to {OUT_DIR}/pareto_curves.png")

# --- Print summary table ---
print("\nSummary Table:")
print(f"{'Topology':15s} {'Steps':>5s} {'N':>3s} {'Acc':>7s} {'Wall(s)':>8s} {'Tokens':>8s} {'GPU_E(J)':>9s}")
print("-" * 60)
for topo in TOPO_STYLE:
    for s in STEP_BUDGETS:
        key = (topo, s)
        if key not in data:
            print(f"{topo:15s} {s:>5d}   -- missing --")
            continue
        d = data[key]
        print(f"{topo:15s} {s:>5d} {d['n']:>3d} {d['acc']:>6.1%} {d['wall']:>8.1f} {d['tokens']:>8.0f} {d['gpu_energy']:>9.1f}")
