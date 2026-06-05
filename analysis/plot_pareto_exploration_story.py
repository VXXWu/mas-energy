"""SWE-bench Pareto frontier exploration story figure.

Two-panel:
  (a) All cells in (energy, accuracy) space, color-coded by sampling source.
      Pareto frontier highlighted. Shows where each exploration strategy
      placed cells and which ones made the frontier.
  (b) "Stacking" view — for cells on the Pareto frontier, show how (k, R, M)
      varies across the energy budget. Demonstrates that Pareto cells use
      moderate values of all 3 axes (not pure extension of any one).

Highlights:
  - 6 of 15 Pareto cells (40%) are cube-interior (v3/v5/LHS), not 1D sweeps
  - 7 of 9 MAS Pareto cells use M=2 (low-M dominates Pareto)
  - The 100% champion (Decent R=10) is the only pure-axis Pareto extreme
"""
import json, glob, os, re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

# Path resolution: scripts assume they live in mas-energy/analysis/ or
# mas-energy/code/. Override with PROJECT_ROOT or MAS_ENERGY_ROOT env vars.
_HERE = os.path.dirname(os.path.abspath(__file__))
MAS_ENERGY_ROOT = os.environ.get("MAS_ENERGY_ROOT", os.path.dirname(_HERE))
PROJECT_ROOT = os.environ.get("PROJECT_ROOT", os.path.dirname(MAS_ENERGY_ROOT))


BENCH_DIR = os.path.join(MAS_ENERGY_ROOT, "results/a5000_swebench")
CRE = re.compile(r"Qwen_Qwen3\.5-9B_([a-z]+)_k(\d+)(?:_R(\d+))?(?:_M(\d+))?\.jsonl$")

LHS = {("centralized",2,4,3),("centralized",8,1,5),("centralized",22,3,2),("centralized",3,5,4),
       ("centralized",13,2,2),("centralized",36,3,5),("centralized",5,2,4),("centralized",1,5,3),
       ("decentralized",5,4,2),("decentralized",22,5,4),("decentralized",2,3,5),("decentralized",13,4,3),
       ("decentralized",36,1,2),("decentralized",8,2,4),("decentralized",3,3,5),("decentralized",1,2,2)}
INTERMED = {("centralized",5,3,4),("centralized",20,3,5),("centralized",30,4,2),("centralized",7,4,4),
            ("decentralized",7,3,4),("decentralized",20,4,5),("decentralized",30,3,2),("decentralized",15,5,4)}
V3 = {("decentralized",10,3,2),("centralized",7,5,2),("decentralized",15,2,2),
      ("decentralized",7,5,2),("decentralized",10,4,2),("centralized",30,2,2)}
V5 = {("decentralized",20,4,2),("decentralized",15,4,2),("centralized",20,3,2),
      ("decentralized",20,5,2),("centralized",15,4,3),("decentralized",15,3,2)}
R1 = {("centralized",15,1,2),("centralized",20,1,2),("centralized",22,1,2)}


def classify(t, k, R, M):
    key = (t, k, R, M)
    if key in V5: return "v5 (HistGBM)"
    if key in V3: return "v3 (GP)"
    if key in INTERMED: return "intermediate_grid"
    if key in LHS: return "LHS space-filling"
    if key in R1: return "Cent R=1 anchor"
    return "1D scaling sweep"


SOURCE_STYLE = {
    "1D scaling sweep":    {"color": "#7799cc", "marker": "o", "size":  55, "z": 1},
    "intermediate_grid":   {"color": "#ff9933", "marker": "s", "size":  90, "z": 3},
    "LHS space-filling":   {"color": "#cc66cc", "marker": "D", "size":  85, "z": 3},
    "v3 (GP)":             {"color": "#cc3333", "marker": "X", "size": 130, "z": 4},
    "v5 (HistGBM)":        {"color": "#33aa44", "marker": "^", "size": 130, "z": 5},
    "Cent R=1 anchor":     {"color": "#ffcc33", "marker": "*", "size": 200, "z": 4},
}


def main():
    rows = []
    for f in sorted(glob.glob(f"{BENCH_DIR}/Qwen_Qwen3.5-9B_*.jsonl")):
        m = CRE.match(os.path.basename(f))
        if not m: continue
        topo, k, r_str, m_str = m.group(1), int(m.group(2)), m.group(3), m.group(4)
        if topo == "hybrid": continue
        R = int(r_str) if r_str else (5 if topo == "centralized" else 2)
        M = int(m_str) if m_str else 3
        n=0; acc=0; energy=0
        for line in open(f):
            try: d=json.loads(line)
            except: continue
            if d.get("error"): continue
            n+=1; acc+=(1 if d.get("correct") else 0); energy+=d.get("gpu_dynamic_energy_joules",0) or 0
        if n < 30: continue
        rows.append((topo, k, R, M, n, 100*acc/n, energy/n/1000))

    df = pd.DataFrame(rows, columns=["topo","k","R","M","n","acc","energy_kJ"])
    df["source"] = df.apply(lambda r: classify(r["topo"], r["k"], r["R"], r["M"])
                            if r["topo"] in ("centralized","decentralized") else "SAS / Independent",
                            axis=1)

    # Pareto frontier (min energy, max acc) across ALL cells
    rows_sorted = df.sort_values("energy_kJ").reset_index(drop=True)
    pareto_idx = []
    max_acc = -1
    for i, r in rows_sorted.iterrows():
        if r["acc"] > max_acc:
            pareto_idx.append(i)
            max_acc = r["acc"]
    pareto = rows_sorted.loc[pareto_idx].reset_index(drop=True)

    # ============ FIGURE ============
    fig = plt.figure(figsize=(18, 8))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.6, 1])

    # ── Panel (a): exploration scatter ──
    ax = fig.add_subplot(gs[0, 0])
    # SAS / Independent first (background)
    sai = df[df["topo"].isin(["sas", "independent"])]
    ax.scatter(sai["energy_kJ"], sai["acc"], c="#999999", marker="x", s=40,
               alpha=0.7, label="SAS / Independent", zorder=0)
    for source, s in SOURCE_STYLE.items():
        sub = df[df["source"] == source]
        if sub.empty: continue
        ax.scatter(sub["energy_kJ"], sub["acc"], c=s["color"], marker=s["marker"],
                   s=s["size"], edgecolors="black", linewidths=0.6, alpha=0.85,
                   zorder=s["z"], label=f"{source} ({len(sub)})")
    # Pareto frontier line
    ax.plot(pareto["energy_kJ"], pareto["acc"], color="black", lw=1.8, ls="-",
            zorder=10, label="Pareto frontier", alpha=0.7)
    # Annotate Pareto cells with their source-marker shape (large)
    for _, r in pareto.iterrows():
        if r["topo"] in ("sas", "independent"): continue
        s = SOURCE_STYLE[r["source"]]
        ax.scatter(r["energy_kJ"], r["acc"], c=s["color"], marker=s["marker"],
                   s=s["size"]*1.4, edgecolors="black", linewidths=1.5,
                   zorder=20)
    # Annotate headline cells
    headlines = [
        ("decentralized", 15, 4, 2, "Decent k=15 R=4 M=2\n98% @ 177 kJ\nv5 HistGBM"),
        ("decentralized", 10, 10, 3, "Decent k=10 R=10 M=3\n100% @ 354 kJ\nstandard R-sweep"),
        ("centralized", 20, 3, 2, "Cent k=20 R=3 M=2\n94% @ 136 kJ\nv5 HistGBM"),
    ]
    for topo, k, R, M, label in headlines:
        sub = df[(df["topo"]==topo) & (df["k"]==k) & (df["R"]==R) & (df["M"]==M)]
        if sub.empty: continue
        r = sub.iloc[0]
        ax.annotate(label, (r["energy_kJ"], r["acc"]),
                    xytext=(35, -30), textcoords="offset points",
                    fontsize=9, ha="left",
                    arrowprops=dict(arrowstyle="->", color="black", lw=1.0),
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", lw=0.5, alpha=0.9))
    ax.set_xscale("log")
    ax.set_xlabel("Energy per task (kJ, log)", fontsize=12)
    ax.set_ylabel("Accuracy (%)", fontsize=12)
    ax.set_title("(a) SWE-bench exploration: Pareto frontier extended by cube-interior cells (40%)", fontsize=13)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9, loc="lower right")

    # ── Panel (b): stacking story for Pareto cells ──
    ax2 = fig.add_subplot(gs[0, 1])
    # For each Pareto MAS cell, show (k, R, M) as small bars / a strip
    mas_pareto = pareto[pareto["topo"].isin(["centralized","decentralized"])].sort_values("energy_kJ").reset_index(drop=True)
    n_cells = len(mas_pareto)
    bar_width = 0.25
    x_pos = np.arange(n_cells)
    # Normalize axes for visual
    k_norm = mas_pareto["k"] / 50.0  # k=50 = 1.0
    r_norm = mas_pareto["R"] / 10.0  # R=10 = 1.0
    m_norm = mas_pareto["M"] / 5.0   # M=5 = 1.0
    ax2.bar(x_pos - bar_width, k_norm, bar_width, label="k (/50)", color="#3366cc", edgecolor="black", linewidth=0.5)
    ax2.bar(x_pos,             r_norm, bar_width, label="R (/10)", color="#cc6633", edgecolor="black", linewidth=0.5)
    ax2.bar(x_pos + bar_width, m_norm, bar_width, label="M (/5)",  color="#33aa44", edgecolor="black", linewidth=0.5)
    # Label each cell with its accuracy
    for i, r in mas_pareto.iterrows():
        topo_short = "C" if r["topo"] == "centralized" else "D"
        ax2.text(i, 1.08, f"{int(r['acc'])}%", ha="center", fontsize=9, fontweight="bold")
        ax2.text(i, -0.08, f"{topo_short}\n{int(r['energy_kJ'])} kJ", ha="center", va="top",
                 fontsize=8, color="#444")
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels([f"k={int(r['k'])}\nR={int(r['R'])}\nM={int(r['M'])}"
                         for _, r in mas_pareto.iterrows()], fontsize=8)
    ax2.set_ylabel("Normalized axis value", fontsize=11)
    ax2.set_ylim(-0.18, 1.18)
    ax2.set_title("(b) Pareto cells stack moderate (k, R, M) — M=2 dominates",
                  fontsize=12)
    ax2.legend(fontsize=10, loc="upper left")
    ax2.grid(True, alpha=0.3, axis="y")
    ax2.axhline(0, color="black", lw=0.5)

    fig.suptitle("SWE-bench Pareto frontier: cube-interior exploration + stacking pattern",
                 fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    out = os.path.join(PROJECT_ROOT, "figures/pareto_exploration_story.png")
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"saved {out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
