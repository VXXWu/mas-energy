"""Bridge internals diagram (fig6) — clean redesign.

Shows the per-agent inference path: how the diffusion bridge takes sender
hidden states and produces receiver-injectable soft prompts.

Layout: horizontal flow with three macro stages (Sender → Bridge → Receiver),
bridge internals shown as a vertical stack of clearly-labeled operations.
All labels positioned to NOT overlap with arrows or other visual elements.
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

OUT = PROJECT_ROOT / "latent_diffusion"


def box(ax, x, y, w, h, label, color="#cfe2f3", fontsize=10, lw=1.5,
        text_color="black"):
    b = FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.02",
        facecolor=color, edgecolor="black", linewidth=lw,
    )
    ax.add_patch(b)
    ax.text(x + w / 2, y + h / 2, label, ha="center", va="center",
            fontsize=fontsize, fontweight="bold", color=text_color)


def arrow(ax, x0, y0, x1, y1, color="black", lw=1.8, style="-|>"):
    arr = FancyArrowPatch(
        (x0, y0), (x1, y1), arrowstyle=style,
        mutation_scale=18, color=color, lw=lw,
        connectionstyle="arc3,rad=0",
    )
    ax.add_patch(arr)


fig, ax = plt.subplots(figsize=(15, 7))
ax.set_xlim(0, 16)
ax.set_ylim(0, 9)
ax.axis("off")

# ──────────────────────────────────────────────────────────────────────
# Stage 1: Sender (left)
# ──────────────────────────────────────────────────────────────────────
SENDER_X, SENDER_Y = 0.5, 4.5
SENDER_W, SENDER_H = 2.5, 1.6
ax.text(SENDER_X + SENDER_W/2, SENDER_Y + SENDER_H + 0.4, "SENDER AGENT",
        ha="center", fontsize=11.5, fontweight="bold")
box(ax, SENDER_X, SENDER_Y, SENDER_W, SENDER_H,
    "Qwen3.5-9B\nbackbone\n[frozen]",
    color="#f4cccc", fontsize=9.5)
ax.text(SENDER_X + SENDER_W/2, SENDER_Y - 0.35,
        "Latent CoT or\ntext re-encode",
        ha="center", va="top", fontsize=8.5, style="italic", color="#666")

# Arrow: hidden states out (sender → bridge)
SENDER_END_X = SENDER_X + SENDER_W
BRIDGE_START_X = 4.5
arrow(ax, SENDER_END_X + 0.05, SENDER_Y + SENDER_H/2,
      BRIDGE_START_X - 0.05, SENDER_Y + SENDER_H/2,
      color="#a05a00", lw=2.2)
# Label BELOW the arrow (not on it)
arrow_mid_x = (SENDER_END_X + BRIDGE_START_X) / 2
ax.text(arrow_mid_x, SENDER_Y + SENDER_H/2 - 0.5,
        "hidden\nstates\nL=3, m=32",
        ha="center", va="top", fontsize=8.5, color="#a05a00",
        fontweight="bold")

# ──────────────────────────────────────────────────────────────────────
# Stage 2: Diffusion Bridge (center) — vertical stack of internals
# ──────────────────────────────────────────────────────────────────────
BRIDGE_X, BRIDGE_Y = BRIDGE_START_X, 1.5
BRIDGE_W, BRIDGE_H = 6.5, 6.5

# Outer bridge container
ax.add_patch(FancyBboxPatch(
    (BRIDGE_X, BRIDGE_Y), BRIDGE_W, BRIDGE_H,
    boxstyle="round,pad=0.05",
    facecolor="#fff5e6", edgecolor="#a05a00", linewidth=2.5,
))
ax.text(BRIDGE_X + BRIDGE_W/2, BRIDGE_Y + BRIDGE_H - 0.4,
        "DIFFUSION BRIDGE",
        ha="center", va="center", fontsize=13, fontweight="bold",
        color="#a05a00")

# Three internal components stacked vertically
INTERNAL_W = 5.0
INTERNAL_X = BRIDGE_X + (BRIDGE_W - INTERNAL_W) / 2

# 1. W_e warm-start (initialization)
box(ax, INTERNAL_X, 5.7, INTERNAL_W, 0.7,
    "(1) W_e warm-start  —  K=0 gives closed-form alignment",
    color="#ffe9c5", fontsize=9.5, lw=1.2)

# 2. AdaLN conditioning
box(ax, INTERNAL_X, 4.6, INTERNAL_W, 0.7,
    "(2) AdaLN conditioning on multi-layer source",
    color="#ffe9c5", fontsize=9.5, lw=1.2)

# 3. DDPM sampling
box(ax, INTERNAL_X, 3.5, INTERNAL_W, 0.7,
    "(3) DDPM sampling  —  K iterative refinements",
    color="#ffe9c5", fontsize=9.5, lw=1.2)

# Down-arrows between internals
arrow(ax, INTERNAL_X + INTERNAL_W/2, 5.65, INTERNAL_X + INTERNAL_W/2, 5.35,
      color="#a05a00", lw=1.2)
arrow(ax, INTERNAL_X + INTERNAL_W/2, 4.55, INTERNAL_X + INTERNAL_W/2, 4.25,
      color="#a05a00", lw=1.2)

# Output indication at bottom of bridge
ax.text(BRIDGE_X + BRIDGE_W/2, 2.85,
        "Output: k=16 soft prompts (d=4096 each)",
        ha="center", va="center", fontsize=10, fontweight="bold",
        color="#a05a00",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#fff",
                  edgecolor="#a05a00", linewidth=1.0))

# Trained on note (below output)
ax.text(BRIDGE_X + BRIDGE_W/2, 2.0,
        "Trained on 6,523 inter-agent pairs from main-study runs",
        ha="center", va="center", fontsize=8.5, style="italic", color="#666")

# ──────────────────────────────────────────────────────────────────────
# Stage 3: Receiver (right)
# ──────────────────────────────────────────────────────────────────────
RECEIVER_X = BRIDGE_X + BRIDGE_W + 1.5
RECEIVER_Y = SENDER_Y
RECEIVER_W, RECEIVER_H = SENDER_W, SENDER_H

# Arrow: soft prompts → receiver
BRIDGE_END_X = BRIDGE_X + BRIDGE_W
arrow(ax, BRIDGE_END_X + 0.05, SENDER_Y + SENDER_H/2,
      RECEIVER_X - 0.05, SENDER_Y + SENDER_H/2,
      color="#a05a00", lw=2.2)
arrow_mid_x_r = (BRIDGE_END_X + RECEIVER_X) / 2
ax.text(arrow_mid_x_r, SENDER_Y + SENDER_H/2 - 0.5,
        "injected at\nembedding layer",
        ha="center", va="top", fontsize=8.5, color="#a05a00",
        fontweight="bold")

ax.text(RECEIVER_X + RECEIVER_W/2, RECEIVER_Y + RECEIVER_H + 0.4, "RECEIVER AGENT",
        ha="center", fontsize=11.5, fontweight="bold")
box(ax, RECEIVER_X, RECEIVER_Y, RECEIVER_W, RECEIVER_H,
    "Qwen3.5-9B\nbackbone\n[frozen]",
    color="#f4cccc", fontsize=9.5)
ax.text(RECEIVER_X + RECEIVER_W/2, RECEIVER_Y - 0.35,
        "Prefill + decode\nfinal answer",
        ha="center", va="top", fontsize=8.5, style="italic", color="#666")

# Title
ax.set_title("Diffusion Bridge: per-agent inference path",
             fontsize=14, fontweight="bold", pad=15)

fig.tight_layout()
fig.savefig(OUT / "fig6_bridge_internals.png", dpi=150, bbox_inches="tight")
fig.savefig(OUT / "fig6_bridge_internals.pdf", bbox_inches="tight")
plt.close(fig)
print(f"Saved: {OUT / 'fig6_bridge_internals.png'}")
