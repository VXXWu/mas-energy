"""SWE-bench γ exponent visualization — per-axis acc-vs-energy efficiency.

Shows the finding: per-axis acc ~ E^γ with γ varying 10× across axes:
  - k-axis: γ ≈ 1.14-1.24 (near-linear, most efficient)
  - R-axis: γ ≈ 0.45-0.69 (diminishing)
  - M-axis: γ ≈ 0.13-0.30 (heavy diminishing, least efficient)

Figure layout:
  Row 1: Cent (left), Decent (right) — log-log scatter of acc vs energy with
         each axis (k, R, M) as a separate trajectory + fitted γ line annotated.
  Row 2: bar chart of γ values per topology × axis (clean comparison).
"""
import json, glob, os, re, sys
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _accuracy_helper import record_accuracy

# Path resolution: scripts assume they live in mas-energy/analysis/ or
# mas-energy/code/. Override with PROJECT_ROOT or MAS_ENERGY_ROOT env vars.
_HERE = os.path.dirname(os.path.abspath(__file__))
MAS_ENERGY_ROOT = os.environ.get("MAS_ENERGY_ROOT", os.path.dirname(_HERE))
PROJECT_ROOT = os.environ.get("PROJECT_ROOT", os.path.dirname(MAS_ENERGY_ROOT))


BENCH_DIR = os.path.join(MAS_ENERGY_ROOT, "results/a5000_swebench")
CRE = re.compile(r"Qwen_Qwen3\.5-9B_([a-z]+)_k(\d+)(?:_R(\d+))?(?:_M(\d+))?\.jsonl$")

AXIS_STYLE = {
    "k": {"color": "#3366cc", "marker": "o", "label": "k-axis (steps)"},
    "R": {"color": "#cc6633", "marker": "s", "label": "R-axis (rounds)"},
    "M": {"color": "#33aa44", "marker": "^", "label": "M-axis (agents)"},
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
        n=0; acc=0; energy=0
        for line in open(f):
            try: d=json.loads(line)
            except: continue
            if d.get("error"): continue
            a, ok = record_accuracy(d)
            if not ok: continue
            n+=1; acc+=a; energy+=d.get("gpu_dynamic_energy_joules",0) or 0
        if n < 30: continue
        cells.append((topo, k, R, M, n, 100*acc/n, energy/n/1000))
    return cells


def main():
    cells = load_cells()
    # Standard 1D sweep filters
    sweeps = {
        "k": {"filter": lambda c: c[2]==2 and c[3]==3, "idx": 1},
        "R": {"filter": lambda c: c[1]==10 and c[3]==3, "idx": 2},
        "M": {"filter": lambda c: c[1]==10 and c[2]==2, "idx": 3},
    }

    # Compute per (topology, axis) trajectory + γ exponent
    trajs = {}  # (topo, axis) -> (energies, accs, gamma, R²)
    for topo in ["centralized", "decentralized"]:
        for axis, spec in sweeps.items():
            sub = [c for c in cells if c[0]==topo and spec["filter"](c)]
            if len(sub) < 3: continue
            energies = np.array([c[6] for c in sub])
            accs = np.array([c[5] for c in sub])
            axis_vals = np.array([c[spec["idx"]] for c in sub])
            order = np.argsort(axis_vals)
            energies = energies[order]; accs = accs[order]; axis_vals = axis_vals[order]
            # Fit log(acc) vs log(E) on cells with positive accuracy
            mask = (accs > 1) & (energies > 0)
            if mask.sum() >= 3:
                sl, ic, r, _, _ = stats.linregress(np.log(energies[mask]), np.log(accs[mask]))
                trajs[(topo, axis)] = (energies, accs, axis_vals, sl, r**2, ic)

    # ── Build figure ──
    fig = plt.figure(figsize=(15, 10))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.4, 1])

    # Row 1: log-log scatter per topology
    for col, topo in enumerate(["centralized", "decentralized"]):
        ax = fig.add_subplot(gs[0, col])
        for axis in ["k", "R", "M"]:
            if (topo, axis) not in trajs: continue
            energies, accs, axis_vals, gamma, r2, ic = trajs[(topo, axis)]
            s = AXIS_STYLE[axis]
            ax.plot(energies, accs, color=s["color"], marker=s["marker"],
                    markersize=8, lw=2, label=f"{s['label']}  γ={gamma:+.2f} (R²={r2:.2f})",
                    alpha=0.9)
            # Fitted power-law line
            mask = (accs > 1) & (energies > 0)
            if mask.sum() >= 3:
                e_fit = np.geomspace(energies[mask].min(), energies[mask].max(), 50)
                a_fit = np.exp(ic) * e_fit**gamma
                ax.plot(e_fit, a_fit, color=s["color"], lw=1.2, ls="--", alpha=0.55)
            # Endpoint label
            ax.annotate(f"{axis}={int(axis_vals[-1])}", (energies[-1], accs[-1]),
                        xytext=(6, 0), textcoords="offset points",
                        fontsize=9, color=s["color"], va="center", fontweight="bold")
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel("Energy per task (kJ, log)", fontsize=11)
        ax.set_ylabel("Accuracy (%, log)", fontsize=11)
        ax.set_title(f"{topo.capitalize()}", fontsize=14, fontweight="bold")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=10, loc="lower right")

    # Row 2: γ bar chart
    ax_bar = fig.add_subplot(gs[1, :])
    axes_order = ["k", "R", "M"]
    topos = ["centralized", "decentralized"]
    n_topos = len(topos)
    width = 0.36
    x = np.arange(len(axes_order))
    bars = []
    for i, topo in enumerate(topos):
        gammas = [trajs.get((topo, a), (None,None,None,np.nan))[3] for a in axes_order]
        offset = (i - (n_topos-1)/2) * width
        b = ax_bar.bar(x + offset, gammas, width,
                       label=topo.capitalize(),
                       color="#2ca02c" if topo == "centralized" else "#d62728",
                       edgecolor="black", linewidth=0.8)
        bars.append((b, gammas))
    # Labels above each bar
    for bgroup, gammas in bars:
        for rect, g in zip(bgroup, gammas):
            if g is None or np.isnan(g): continue
            ax_bar.text(rect.get_x() + rect.get_width()/2, g + 0.03,
                        f"{g:+.2f}", ha="center", va="bottom",
                        fontsize=11, fontweight="bold")
    # Regime annotations on the right
    ax_bar.axhspan(0.9, 1.35, color="#3366cc", alpha=0.08)
    ax_bar.axhspan(0.4, 0.75, color="#cc6633", alpha=0.08)
    ax_bar.axhspan(0.08, 0.35, color="#33aa44", alpha=0.08)
    ax_bar.text(2.7, 1.18, "near-linear\nmost efficient\n(k-axis)", fontsize=9,
                ha="left", va="center", color="#3366cc")
    ax_bar.text(2.7, 0.575, "diminishing\nreturns\n(R-axis)", fontsize=9,
                ha="left", va="center", color="#cc6633")
    ax_bar.text(2.7, 0.215, "heavy diminishing\nleast efficient\n(M-axis)", fontsize=9,
                ha="left", va="center", color="#33aa44")
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels([AXIS_STYLE[a]["label"] for a in axes_order], fontsize=12)
    ax_bar.set_ylabel("γ exponent (acc ~ E^γ)", fontsize=12)
    ax_bar.set_title("Per-axis energy efficiency: γ varies ~10× across axes",
                     fontsize=13, fontweight="bold")
    ax_bar.set_ylim(0, 1.45)
    ax_bar.set_xlim(-0.5, 3.5)
    ax_bar.legend(fontsize=11, loc="upper left")
    ax_bar.grid(True, alpha=0.3, axis="y")
    ax_bar.axhline(1.0, color="black", lw=0.5, ls=":", alpha=0.5)

    fig.suptitle("SWE-bench per-axis energy efficiency — γ exponent of accuracy vs energy power law",
                 fontsize=15, fontweight="bold", y=1.00)
    fig.tight_layout()
    out = os.path.join(PROJECT_ROOT, "figures/swebench_gamma_per_axis.png")
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"saved {out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
