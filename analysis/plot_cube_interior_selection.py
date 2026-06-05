"""Plot where the cube-interior cells were sampled in (k, R, M) space, colored
by sampling source: standard 1D grid vs intermediate_grid vs LHS vs v3/v5
Pareto-dominant validation vs the Cent R=1 anchor cells.

Three 2D projections (k×R, k×M, R×M) per topology = 6 panels.
"""
import json, glob, os, re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Path resolution: scripts assume they live in mas-energy/analysis/ or
# mas-energy/code/. Override with PROJECT_ROOT or MAS_ENERGY_ROOT env vars.
_HERE = os.path.dirname(os.path.abspath(__file__))
MAS_ENERGY_ROOT = os.environ.get("MAS_ENERGY_ROOT", os.path.dirname(_HERE))
PROJECT_ROOT = os.environ.get("PROJECT_ROOT", os.path.dirname(MAS_ENERGY_ROOT))


BENCH_DIR = os.path.join(MAS_ENERGY_ROOT, "results/a5000_swebench")
CRE = re.compile(r"Qwen_Qwen3\.5-9B_([a-z]+)_k(\d+)(?:_R(\d+))?(?:_M(\d+))?\.jsonl$")

# Sampling provenance — manually-tagged cell sets per sbatch
LHS_CELLS = {  # from a5000_swebench_lhs.sbatch
    ("centralized",   2,  4, 3), ("centralized",   8, 1, 5), ("centralized",  22, 3, 2),
    ("centralized",   3,  5, 4), ("centralized",  13, 2, 2), ("centralized",  36, 3, 5),
    ("centralized",   5,  2, 4), ("centralized",   1, 5, 3),
    ("decentralized", 5,  4, 2), ("decentralized", 22, 5, 4), ("decentralized", 2, 3, 5),
    ("decentralized", 13, 4, 3), ("decentralized", 36, 1, 2), ("decentralized", 8, 2, 4),
    ("decentralized", 3,  3, 5), ("decentralized",  1, 2, 2),
}
INTERMED_CELLS = {  # from a5000_swebench_intermediate_grid.sbatch
    ("centralized",   5,  3, 4), ("centralized",  20, 3, 5), ("centralized",  30, 4, 2),
    ("centralized",   7,  4, 4),
    ("decentralized", 7,  3, 4), ("decentralized", 20, 4, 5), ("decentralized", 30, 3, 2),
    ("decentralized", 15, 5, 4),
}
V3_CELLS = {  # from a5000_validate_predictions_swebench_v3.sbatch
    ("decentralized", 10, 3, 2), ("centralized",   7, 5, 2), ("decentralized", 15, 2, 2),
    ("decentralized",  7, 5, 2), ("decentralized", 10, 4, 2), ("centralized",  30, 2, 2),
}
V5_CELLS = {  # from a5000_validate_predictions_swebench_v5.sbatch
    ("centralized",   50, 1, 2), ("decentralized",  3, 5, 4), ("centralized",  30, 1, 2),
    ("decentralized",  3, 5, 2), ("decentralized",  3, 5, 3), ("decentralized", 3, 4, 4),
    # v5 HistGBM-based (later batch)
    ("decentralized", 20, 4, 2), ("decentralized", 15, 4, 2), ("centralized",  20, 3, 2),
    ("decentralized", 20, 5, 2), ("centralized",  15, 4, 3), ("decentralized", 15, 3, 2),
}
R1_ANCHOR_CELLS = {  # from a5000_cent_R1_anchor.sbatch
    ("centralized", 15, 1, 2), ("centralized", 20, 1, 2), ("centralized", 22, 1, 2),
}


def load_cells():
    cells = []
    for f in sorted(glob.glob(f"{BENCH_DIR}/Qwen_Qwen3.5-9B_*.jsonl")):
        m = CRE.match(os.path.basename(f))
        if not m: continue
        topo, k, r_str, m_str = m.group(1), int(m.group(2)), m.group(3), m.group(4)
        if topo not in ("centralized", "decentralized"): continue
        R = int(r_str) if r_str else 2
        M = int(m_str) if m_str else 3
        n = 0; acc = 0
        for line in open(f):
            try: d = json.loads(line)
            except: continue
            if d.get("error"): continue
            n += 1; acc += (1 if d.get("correct") else 0)
        if n < 30: continue
        cells.append((topo, k, R, M, n, 100 * acc / n))
    return cells


def classify(cell):
    key = (cell[0], cell[1], cell[2], cell[3])
    if key in V5_CELLS:       return "v5 (HistGBM)"
    if key in V3_CELLS:       return "v3 (GP)"
    if key in INTERMED_CELLS: return "intermediate_grid"
    if key in LHS_CELLS:      return "LHS"
    if key in R1_ANCHOR_CELLS: return "R=1 anchor"
    return "1D scaling sweep"


SOURCE_STYLE = {
    "1D scaling sweep":    {"color": "#6699cc", "marker": "o", "size":  60, "z": 1},
    "intermediate_grid":   {"color": "#ff9933", "marker": "s", "size": 110, "z": 3},
    "LHS":                 {"color": "#cc66cc", "marker": "D", "size": 100, "z": 3},
    "v3 (GP)":             {"color": "#cc3333", "marker": "X", "size": 130, "z": 4},
    "v5 (HistGBM)":        {"color": "#33aa33", "marker": "^", "size": 130, "z": 5},
    "R=1 anchor":          {"color": "#ffcc33", "marker": "*", "size": 220, "z": 4},
}


def main():
    cells = load_cells()
    df = pd.DataFrame(cells, columns=["topo", "k", "R", "M", "n", "acc"])
    df["source"] = df.apply(lambda r: classify(r), axis=1)
    print(f"Loaded {len(df)} cells. Source distribution:")
    print(df["source"].value_counts())

    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    projections = [("k", "R", "k (max react steps)", "R (rounds)"),
                   ("k", "M", "k (max react steps)", "M (agent count)"),
                   ("R", "M", "R (rounds)",          "M (agent count)")]

    for row_idx, topo in enumerate(["centralized", "decentralized"]):
        sub = df[df["topo"] == topo]
        for col_idx, (xc, yc, xlabel, ylabel) in enumerate(projections):
            ax = axes[row_idx, col_idx]
            for source, s in SOURCE_STYLE.items():
                pts = sub[sub["source"] == source]
                if pts.empty: continue
                # Jitter slightly to avoid stacking when multiple cells overlap in 2D projection
                jx = pts[xc] * (1 + 0.04 * (np.arange(len(pts)) - len(pts)/2))
                jy = pts[yc] * (1 + 0.04 * (np.arange(len(pts)) - len(pts)/2))
                ax.scatter(jx, jy, c=s["color"], marker=s["marker"], s=s["size"],
                           edgecolors="black", linewidths=0.8, alpha=0.85,
                           zorder=s["z"], label=source if (row_idx == 0 and col_idx == 0) else None)
            ax.set_xscale("log"); ax.set_yscale("log")
            ax.set_xlabel(xlabel, fontsize=11)
            ax.set_ylabel(ylabel, fontsize=11)
            ax.grid(True, alpha=0.3)
            if col_idx == 0:
                ax.set_ylabel(f"{topo}\n{ylabel}", fontsize=12, fontweight="bold")
            if row_idx == 0:
                ax.set_title(f"{xc} × {yc}", fontsize=13)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.06),
               ncol=6, fontsize=11, frameon=False)
    fig.suptitle("SWE-bench cube interior sampling — where cells were placed in (k, R, M) space",
                 fontsize=15, fontweight="bold", y=0.99)
    fig.tight_layout(rect=[0, 0.08, 1, 0.97])
    out = os.path.join(PROJECT_ROOT, "figures/cube_interior_selection.png")
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"saved {out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
