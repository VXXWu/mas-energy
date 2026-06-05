"""Canonical SWE-bench Pareto frontier figure — paper-ready single panel.

All MAS cells colored by topology, Pareto envelope drawn, frontier cells
labeled with (acc, energy). Clean styling, no source attribution clutter.
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

TOPO_STYLE = {
    "sas":            {"color": "#1f77b4", "marker": "o", "size": 60, "label": "SAS"},
    "independent":    {"color": "#ff7f0e", "marker": "s", "size": 60, "label": "Independent (M=3)"},
    "centralized":    {"color": "#2ca02c", "marker": "^", "size": 80, "label": "Centralized"},
    "decentralized":  {"color": "#d62728", "marker": "D", "size": 70, "label": "Decentralized"},
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
        if n < 20: continue  # was 30; lowered to surface high-k SAS cells
        rows.append({"topo": topo, "k": k, "R": R, "M": M, "n": n,
                     "acc": 100*acc/n, "energy_kJ": energy/n/1000})

    # Pareto frontier
    rows_sorted = sorted(rows, key=lambda r: r["energy_kJ"])
    pareto = []
    max_acc = -1
    for r in rows_sorted:
        if r["acc"] > max_acc:
            pareto.append(r); max_acc = r["acc"]

    # Build figure
    fig, ax = plt.subplots(figsize=(12, 8))

    for topo, s in TOPO_STYLE.items():
        sub = [r for r in rows if r["topo"] == topo]
        if not sub: continue
        xs = [r["energy_kJ"] for r in sub]
        ys = [r["acc"] for r in sub]
        ax.scatter(xs, ys, c=s["color"], marker=s["marker"], s=s["size"],
                   edgecolors="black", linewidths=0.5, alpha=0.55,
                   label=s["label"], zorder=2)

    # Pareto frontier line + highlighted points
    pe = [r["energy_kJ"] for r in pareto]
    pa = [r["acc"] for r in pareto]
    ax.plot(pe, pa, color="black", lw=2.2, ls="-", zorder=10,
            label=f"Pareto frontier ({len(pareto)} cells)")
    for r in pareto:
        s = TOPO_STYLE[r["topo"]]
        ax.scatter([r["energy_kJ"]], [r["acc"]],
                   c=s["color"], marker=s["marker"], s=s["size"] * 2.0,
                   edgecolors="black", linewidths=1.6, zorder=20)

    # Label Pareto MAS cells
    for r in pareto:
        if r["topo"] in ("sas", "independent"): continue
        topo_short = "C" if r["topo"] == "centralized" else "D"
        label = f"{topo_short}: k={r['k']} R={r['R']} M={r['M']}\n{r['acc']:.0f}% @ {r['energy_kJ']:.0f} kJ"
        ax.annotate(label, (r["energy_kJ"], r["acc"]),
                    xytext=(15, -25), textcoords="offset points", fontsize=9,
                    ha="left",
                    arrowprops=dict(arrowstyle="->", color="black", lw=0.7),
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black",
                              lw=0.5, alpha=0.92))

    ax.set_xscale("log")
    ax.set_xlabel("Energy per task (kJ, log)", fontsize=13)
    ax.set_ylabel("Accuracy (%)", fontsize=13)
    ax.set_title(f"SWE-bench Pareto frontier · Qwen3.5-9B · {len(rows)} cells",
                 fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11, loc="lower right", framealpha=0.95)
    fig.tight_layout()
    out = os.path.join(PROJECT_ROOT, "figures/swebench_pareto_canonical.png")
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"saved {out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
