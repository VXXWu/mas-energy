"""Single-panel figure showing all SWE-bench cube-interior cells tested,
with Pareto-dominant ones highlighted and labeled.

Layout:
- Background (gray): standard 1D scaling sweeps (k/R/M-sweeps from main grid)
- Foreground (colored, by source): intermediate_grid / LHS / v3 GP / v5 HistGBM / R=1 anchors
- Pareto frontier drawn as black line
- Pareto-dominant cube-interior cells: large markers + label with (k, R, M, acc, E)
"""
import json, glob, os, re, sys
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(__file__))
from _accuracy_helper import record_accuracy

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
    if key in V5: return "v5 HistGBM"
    if key in V3: return "v3 GP"
    if key in INTERMED: return "intermediate_grid"
    if key in LHS: return "LHS"
    if key in R1: return "Cent R=1 anchor"
    return "1D scaling sweep"


SOURCE_STYLE = {
    "1D scaling sweep":    {"color": "#aaaaaa", "marker": "o", "size":  35, "z": 1, "alpha": 0.5},
    "intermediate_grid":   {"color": "#ff9933", "marker": "s", "size":  95, "z": 3, "alpha": 0.95},
    "LHS":                 {"color": "#cc66cc", "marker": "D", "size":  85, "z": 3, "alpha": 0.95},
    "v3 GP":               {"color": "#cc3333", "marker": "X", "size": 130, "z": 4, "alpha": 0.95},
    "v5 HistGBM":          {"color": "#33aa44", "marker": "^", "size": 130, "z": 5, "alpha": 0.95},
    "Cent R=1 anchor":     {"color": "#ffcc33", "marker": "*", "size": 200, "z": 4, "alpha": 0.95},
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
            a, ok = record_accuracy(d)
            if not ok: continue
            n+=1; acc+=a; energy+=d.get("gpu_dynamic_energy_joules",0) or 0
        if n < 20: continue
        rows.append({"topo": topo, "k": k, "R": R, "M": M, "n": n,
                     "acc": 100*acc/n, "energy_kJ": energy/n/1000})

    # MAS only — drop SAS/Independent for clarity
    rows = [r for r in rows if r["topo"] in ("centralized","decentralized")]

    # Pareto frontier
    rows_sorted = sorted(rows, key=lambda r: r["energy_kJ"])
    pareto = []
    max_acc = -1
    for r in rows_sorted:
        if r["acc"] > max_acc:
            pareto.append(r); max_acc = r["acc"]

    pareto_keys = {(r["topo"], r["k"], r["R"], r["M"]) for r in pareto}

    # Build figure
    fig, ax = plt.subplots(figsize=(15, 9))

    # Plot all cells by source
    plotted_legends = set()
    counts = {}
    for r in rows:
        src = classify(r["topo"], r["k"], r["R"], r["M"])
        counts[src] = counts.get(src, 0) + 1

    for src in ["1D scaling sweep", "intermediate_grid", "LHS",
                "Cent R=1 anchor", "v3 GP", "v5 HistGBM"]:
        s = SOURCE_STYLE[src]
        sub = [r for r in rows if classify(r["topo"], r["k"], r["R"], r["M"]) == src]
        if not sub: continue
        xs = [r["energy_kJ"] for r in sub]
        ys = [r["acc"] for r in sub]
        ax.scatter(xs, ys, c=s["color"], marker=s["marker"], s=s["size"],
                   edgecolors="black", linewidths=0.6, alpha=s["alpha"],
                   zorder=s["z"], label=f"{src} ({counts[src]})")

    # Pareto frontier line
    ax.plot([r["energy_kJ"] for r in pareto], [r["acc"] for r in pareto],
            color="black", lw=1.8, ls="-", zorder=10, label=f"Pareto frontier ({len(pareto)} cells)")

    # Highlight + label Pareto cells from cube interior (the discovery story)
    for r in pareto:
        src = classify(r["topo"], r["k"], r["R"], r["M"])
        if src == "1D scaling sweep":
            # Still mark on Pareto, but with smaller emphasis
            ax.scatter([r["energy_kJ"]], [r["acc"]], c="#444",
                       marker="o", s=110, edgecolors="black", linewidths=1.2,
                       zorder=20, facecolors="white")
        else:
            s = SOURCE_STYLE[src]
            ax.scatter([r["energy_kJ"]], [r["acc"]], c=s["color"],
                       marker=s["marker"], s=s["size"] * 1.7,
                       edgecolors="black", linewidths=1.8, zorder=20)
            # Label: topology shorthand + (k, R, M) + acc% + energy kJ
            topo_short = "C" if r["topo"] == "centralized" else "D"
            label = f"{topo_short}: k={r['k']} R={r['R']} M={r['M']}\n{r['acc']:.0f}% @ {r['energy_kJ']:.0f} kJ"
            ax.annotate(label, (r["energy_kJ"], r["acc"]),
                        xytext=(15, -30), textcoords="offset points", fontsize=9,
                        ha="left",
                        arrowprops=dict(arrowstyle="->", color="black", lw=0.7),
                        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black",
                                  lw=0.6, alpha=0.92))

    # Also label the standard-1D-sweep Pareto cells (less prominent)
    for r in pareto:
        src = classify(r["topo"], r["k"], r["R"], r["M"])
        if src != "1D scaling sweep": continue
        topo_short = "C" if r["topo"] == "centralized" else "D"
        label = f"{topo_short}: k={r['k']} R={r['R']} M={r['M']}\n{r['acc']:.0f}% @ {r['energy_kJ']:.0f} kJ"
        ax.annotate(label, (r["energy_kJ"], r["acc"]),
                    xytext=(15, 20), textcoords="offset points", fontsize=8.5,
                    ha="left", color="#333",
                    arrowprops=dict(arrowstyle="->", color="#666", lw=0.6),
                    bbox=dict(boxstyle="round,pad=0.25", fc="#f0f0f0", ec="#888",
                              lw=0.5, alpha=0.9))

    ax.set_xscale("log")
    ax.set_xlabel("Energy per task (kJ, log)", fontsize=13)
    ax.set_ylabel("Accuracy (%)", fontsize=13)
    n_total = len(rows)
    n_cube = sum(1 for r in rows
                 if classify(r["topo"], r["k"], r["R"], r["M"]) != "1D scaling sweep")
    n_pareto_cube = sum(1 for r in pareto
                        if classify(r["topo"], r["k"], r["R"], r["M"]) != "1D scaling sweep")
    ax.set_title(f"SWE-bench cube-interior exploration: {n_cube} cells tested across 5 sampling methods · "
                 f"{n_pareto_cube} of {len(pareto)} Pareto cells from cube interior",
                 fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10, loc="lower right", framealpha=0.95)
    fig.tight_layout()
    out = os.path.join(PROJECT_ROOT, "figures/swebench_intermediate_cells_pareto.png")
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"saved {out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
