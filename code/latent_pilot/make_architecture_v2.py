"""Architecture comparison v2: Diffusion HYBRID (Round 0 = text decode + re-encode)
vs Pure-latent diffusion (Round 0 = latent thoughts direct). Cleaner layout —
no inset boxes, no overlapping text.

Key visual difference:
  Hybrid (left):     Sender → decode text → re-encode → hidden states → bridge → soft prompts
  Pure-latent (right): Sender → latent thoughts → bridge → soft prompts
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from pathlib import Path
import os
_HERE = Path(__file__).resolve().parent
MAS_ENERGY_ROOT = Path(os.environ.get("MAS_ENERGY_ROOT", _HERE.parent.parent))
PROJECT_ROOT = Path(os.environ.get("PROJECT_ROOT", MAS_ENERGY_ROOT.parent))

OUT = (PROJECT_ROOT / "latent_diffusion"))
OUT.mkdir(exist_ok=True)


def box(ax, x, y, w, h, label, color="#cfe2f3", fontsize=10, lw=1.5):
    """Draw a rounded-rectangle box with label."""
    b = FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.02",
        facecolor=color, edgecolor="black", linewidth=lw,
    )
    ax.add_patch(b)
    ax.text(x + w / 2, y + h / 2, label, ha="center", va="center",
            fontsize=fontsize, fontweight="bold")


def arrow_vert(ax, x, y0, y1, color="black", lw=1.5):
    """Vertical arrow (no labels — keeps the layout clean)."""
    arr = FancyArrowPatch(
        (x, y0), (x, y1), arrowstyle="-|>",
        mutation_scale=15, color=color, lw=lw,
        connectionstyle="arc3,rad=0",
    )
    ax.add_patch(arr)


def arrow_diag(ax, x0, y0, x1, y1, color="black", lw=1.0):
    """Thin diagonal arrow."""
    arr = FancyArrowPatch(
        (x0, y0), (x1, y1), arrowstyle="->",
        mutation_scale=12, color=color, lw=lw,
        connectionstyle="arc3,rad=0",
    )
    ax.add_patch(arr)


def draw_panel(ax, variant):
    """variant ∈ {'hybrid', 'pure'} — controls the Round 0 sender path."""
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 14)
    ax.axis("off")

    AGENT_X = [1.5, 5.0, 8.5]
    AGENT_W, AGENT_H = 1.6, 0.9
    SENDER_COLOR = "#f4cccc" if variant == "pure" else "#fce5cd"
    RECEIVER_COLOR = "#f4cccc" if variant == "pure" else "#fce5cd"
    INTER_TEXT_COLOR = "#e8f0fa"
    BRIDGE_COLOR = "#fff5e6"

    # ─── Title ───
    if variant == "hybrid":
        title = "(a) Diffusion HYBRID\nRound 0 senders decode text → re-encode → bridge"
    else:
        title = "(b) Pure-latent diffusion\nRound 0 senders go directly to bridge"
    ax.text(5, 13.5, title, ha="center", va="top",
            fontsize=12, fontweight="bold")

    # ─── Row 1: Round 0 senders ───
    for x in AGENT_X:
        box(ax, x - AGENT_W/2, 11.6, AGENT_W, AGENT_H,
            "Agent\nRound 0", color=SENDER_COLOR, fontsize=9)

    if variant == "hybrid":
        # Row 1.5: each agent decodes text, then re-encodes
        for x in AGENT_X:
            arrow_vert(ax, x, 11.55, 10.7, lw=1.2)
        for x in AGENT_X:
            box(ax, x - 0.85, 10.0, 1.7, 0.65,
                "decoded text\n(~500 tok)", color=INTER_TEXT_COLOR, fontsize=7, lw=1.0)
        # Down to "re-encode" step
        for x in AGENT_X:
            arrow_vert(ax, x, 9.95, 9.1, lw=1.2)
        for x in AGENT_X:
            box(ax, x - 0.85, 8.5, 1.7, 0.55,
                "re-encode\nbackbone forward", color="#ffe9d5", fontsize=7, lw=1.0)
        # Down to hidden states label
        for x in AGENT_X:
            arrow_vert(ax, x, 8.45, 7.7, lw=1.2)
        # Hidden states row
        for x in AGENT_X:
            ax.text(x, 7.5, "hidden states", ha="center", fontsize=7.5,
                    style="italic", color="#a05a00")
        ROUTE_TO_BRIDGE_Y = 7.3
    else:
        # Pure: direct latent thoughts to bridge (no decode)
        for x in AGENT_X:
            arrow_vert(ax, x, 11.55, 10.5, lw=1.5)
        for x in AGENT_X:
            box(ax, x - 0.85, 9.85, 1.7, 0.6,
                "latent thoughts\n(no text decode)",
                color="#ffe0e0", fontsize=7.2, lw=1.0)
        for x in AGENT_X:
            arrow_vert(ax, x, 9.8, 7.7, lw=1.5)
        for x in AGENT_X:
            ax.text(x, 7.5, "hidden states", ha="center", fontsize=7.5,
                    style="italic", color="#a05a00")
        ROUTE_TO_BRIDGE_Y = 7.3

    # ─── Bridge box (centered, wide) ───
    arrow_vert_to_bridge = lambda x: arrow_vert(ax, x, ROUTE_TO_BRIDGE_Y, 6.5, lw=1.5, color="#a05a00")
    for x in AGENT_X:
        arrow_vert_to_bridge(x)

    bridge_y, bridge_h = 5.2, 1.3
    box_x, box_w = 0.6, 8.8
    ax.add_patch(FancyBboxPatch(
        (box_x, bridge_y), box_w, bridge_h, boxstyle="round,pad=0.04",
        facecolor=BRIDGE_COLOR, edgecolor="#a05a00", linewidth=2,
    ))
    ax.text(5, bridge_y + bridge_h*0.7, "Diffusion Bridge",
            ha="center", va="center", fontsize=12, fontweight="bold",
            color="#a05a00")
    ax.text(5, bridge_y + bridge_h*0.3,
            "K diffusion sampling steps  |  W_e warm-start  |  AdaLN conditioning",
            ha="center", va="center", fontsize=8, style="italic")

    # ─── Soft prompts emerging from bridge ───
    for x in AGENT_X:
        arrow_vert(ax, x, bridge_y - 0.05, 4.3, lw=1.5, color="#a05a00")
    for x in AGENT_X:
        ax.text(x, 3.85, "k=16 soft\nprompts", ha="center", fontsize=7.5,
                style="italic", color="#a05a00")

    # ─── Row 3: Round 1 receivers ───
    for x in AGENT_X:
        arrow_vert(ax, x, 3.6, 3.05, lw=1.5)
    for x in AGENT_X:
        box(ax, x - AGENT_W/2, 2.1, AGENT_W, AGENT_H,
            "Agent\nRound 1", color=RECEIVER_COLOR, fontsize=9)

    # ─── Down to synthesizer ───
    SYNTH_CX = 5.0
    for x in AGENT_X:
        arrow_diag(ax, x, 2.05, SYNTH_CX, 1.0, lw=1.0, color="#666666")

    box(ax, SYNTH_CX - 1.0, 0.2, 2.0, 0.75,
        "Synthesizer", color="#dddddd", fontsize=10)


# ──────────────────────────────────────────────────────────────────────
# Build the figure
# ──────────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(1, 2, figsize=(15, 9))
draw_panel(axes[0], "hybrid")
draw_panel(axes[1], "pure")

fig.suptitle("Diffusion bridge inter-agent communication:\n"
             "Hybrid (Round 0 = decoded text + re-encode) vs Pure-latent (Round 0 = direct latent)",
             fontsize=13, fontweight="bold", y=0.99)

fig.tight_layout()
fig.savefig(OUT / "fig5_architecture.png", dpi=150, bbox_inches="tight")
fig.savefig(OUT / "fig5_architecture.pdf", bbox_inches="tight")
plt.close(fig)
print(f"Saved: {OUT / 'fig5_architecture.png'}")
