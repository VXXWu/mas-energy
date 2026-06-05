"""Architecture diagram: how diffusion bridge communication works in a
3-agent decentralized debate setup. Two side-by-side panels:
  Left: Text MAS (decoded peer messages flow as context)
  Right: Diffusion bridge (hidden states → bridge → soft prompts)

And a "Pure-latent diffusion" inset showing what skipping Round 0 decode means.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from pathlib import Path

OUT = Path("latent_diffusion")
OUT.mkdir(exist_ok=True)


def agent_box(ax, x, y, w, h, label, color="#cfe2f3", fontsize=11):
    """Draw a rounded-rectangle agent box with label."""
    box = FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.02",
        facecolor=color, edgecolor="black", linewidth=1.5,
    )
    ax.add_patch(box)
    ax.text(x + w / 2, y + h / 2, label, ha="center", va="center",
            fontsize=fontsize, fontweight="bold")


def arrow(ax, x0, y0, x1, y1, color="black", lw=1.5, style="-|>",
          label=None, label_pos=0.5, label_color=None, label_above=True):
    """Draw an annotated arrow."""
    arr = FancyArrowPatch(
        (x0, y0), (x1, y1), arrowstyle=style,
        mutation_scale=18, color=color, lw=lw,
        connectionstyle="arc3,rad=0",
    )
    ax.add_patch(arr)
    if label:
        lx = x0 + (x1 - x0) * label_pos
        ly = y0 + (y1 - y0) * label_pos
        offset = 0.05 if label_above else -0.05
        ax.text(lx, ly + offset, label, ha="center", va="center",
                fontsize=8.5, color=label_color or color,
                fontweight="bold")


# ──────────────────────────────────────────────────────────────────────
# Main figure
# ──────────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(1, 2, figsize=(15, 7))

# ─── LEFT PANEL: Text MAS (baseline) ─────────────────────────────────
ax = axes[0]
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.axis("off")
ax.set_title("(a) Text MAS — baseline\n"
             "Inter-agent comm = full decoded text",
             fontsize=13, fontweight="bold", pad=12)

# Three senders at top
for i, agent_x in enumerate([1, 4.5, 8]):
    agent_box(ax, agent_x - 0.7, 7.5, 1.4, 1.0,
              f"Agent {i}\n(Round 0)",
              color="#cfe2f3")
    # Down-arrow with decoded text label
    arrow(ax, agent_x, 7.4, agent_x, 6.2,
          label="decode\n~500 tokens", label_color="#3a6ea5",
          label_pos=0.45)

# Three text-blob boxes (each agent's decoded response)
for i, agent_x in enumerate([1, 4.5, 8]):
    ax.add_patch(patches.Rectangle(
        (agent_x - 0.75, 5.2), 1.5, 1.0,
        facecolor="#e8f0fa", edgecolor="#3a6ea5",
        linewidth=1.5,
    ))
    ax.text(agent_x, 5.7, f"Text response {i}", ha="center", va="center",
            fontsize=9, style="italic")

# Triple-arrows from text blobs to each receiver (each receiver sees peer texts)
for receiver_x in [1, 4.5, 8]:
    for sender_x in [1, 4.5, 8]:
        if sender_x == receiver_x: continue
        arrow(ax, sender_x, 5.15, receiver_x, 3.8,
              lw=0.9, color="#3a6ea5", style="->")

# Three receivers (Round 1)
for i, agent_x in enumerate([1, 4.5, 8]):
    agent_box(ax, agent_x - 0.7, 2.8, 1.4, 1.0,
              f"Agent {i}\n(Round 1)",
              color="#cfe2f3")
    # Down-arrow to synthesizer
    arrow(ax, agent_x, 2.7, 4.5, 1.5,
          lw=0.8, color="#666666", style="->")

# Synthesizer at bottom
agent_box(ax, 3.7, 0.2, 1.6, 1.0, "Synthesizer", color="#dddddd")

ax.text(5, -0.5, "Per-task energy: ~16 kJ",
        ha="center", fontsize=10, fontweight="bold", color="#3a6ea5")


# ─── RIGHT PANEL: Diffusion bridge ──────────────────────────────────
ax = axes[1]
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.axis("off")
ax.set_title("(b) Diffusion bridge — hidden states → soft prompts\n"
             "Inter-agent comm = bridge-translated latents",
             fontsize=13, fontweight="bold", pad=12)

# Three senders at top
for i, agent_x in enumerate([1, 4.5, 8]):
    agent_box(ax, agent_x - 0.7, 7.5, 1.4, 1.0,
              f"Agent {i}\n(Round 0)",
              color="#f4cccc")
    # Arrow showing hidden-state extraction
    arrow(ax, agent_x, 7.4, agent_x, 6.2,
          label="hidden states\n(latent thoughts)",
          label_color="#a04444", label_pos=0.45)

# Bridge mechanism box (central)
ax.add_patch(FancyBboxPatch(
    (0.5, 4.4), 9, 1.4, boxstyle="round,pad=0.05",
    facecolor="#fce5cd", edgecolor="#a05a00",
    linewidth=2,
))
ax.text(5, 5.45, "Diffusion Bridge", ha="center", va="center",
        fontsize=14, fontweight="bold", color="#a05a00")
ax.text(5, 4.85,
        "Conditional DDPM, K=5-20 sampling steps, AdaLN-zero | "
        "W_e warm-start | 12% KL reduction vs closed-form",
        ha="center", va="center", fontsize=8.5, style="italic")

# Down-arrows to receivers
for i, agent_x in enumerate([1, 4.5, 8]):
    arrow(ax, agent_x, 4.35, agent_x, 3.8,
          label="k=16 soft prompts" if i == 1 else None,
          label_color="#a05a00", lw=1.5, label_pos=0.0, label_above=False)

# Three receivers (Round 1)
for i, agent_x in enumerate([1, 4.5, 8]):
    agent_box(ax, agent_x - 0.7, 2.8, 1.4, 1.0,
              f"Agent {i}\n(Round 1)",
              color="#f4cccc")
    # Down-arrow to synthesizer
    arrow(ax, agent_x, 2.7, 4.5, 1.5,
          lw=0.8, color="#666666", style="->")

# Synthesizer at bottom
agent_box(ax, 3.7, 0.2, 1.6, 1.0, "Synthesizer", color="#dddddd")

ax.text(5, -0.5, "Per-task energy: ~40 kJ (pure-latent variant)",
        ha="center", fontsize=10, fontweight="bold", color="#a04444")


# ──────────────────────────────────────────────────────────────────────
# Highlight the "pure-latent" variant — skip Round 0 decode
# ──────────────────────────────────────────────────────────────────────

# Add a small inset on the right panel explaining the pure-latent variant
inset_x, inset_y = 5.5, 8.6
ax.text(inset_x + 1.6, inset_y + 0.35,
        "PURE-LATENT VARIANT:\nRound 0 senders do NOT\ndecode text — they only\nproduce latent thoughts.\nSaves 47% vs decoded\ndiffusion hybrid.",
        ha="left", va="center", fontsize=8, color="#a04444",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#fff5f5",
                  edgecolor="#a04444", linewidth=1.2))

fig.suptitle("Diffusion bridge inter-agent communication: architecture overview",
             fontsize=14, fontweight="bold", y=1.02)

fig.tight_layout()
fig.savefig(OUT / "fig5_architecture.png", dpi=150, bbox_inches="tight")
fig.savefig(OUT / "fig5_architecture.pdf", bbox_inches="tight")
plt.close(fig)
print(f"Saved: {OUT / 'fig5_architecture.png'}")


# ──────────────────────────────────────────────────────────────────────
# Bonus: detailed bridge-internals diagram
# ──────────────────────────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(13, 5.5))
ax.set_xlim(0, 14)
ax.set_ylim(0, 7)
ax.axis("off")

# Sender side
ax.text(1.5, 6.3, "Sender Agent",
        ha="center", fontsize=12, fontweight="bold")
agent_box(ax, 0.5, 4.5, 2, 1.5, "Backbone\n(Qwen3.5-9B)\n[frozen]",
          color="#f4cccc", fontsize=10)
ax.text(1.5, 4.2, "Latent CoT or\ntext re-encode",
        ha="center", va="top", fontsize=8.5, style="italic")

# Arrow: hidden states out
arrow(ax, 2.6, 5.3, 4.0, 5.3,
      label="L=3 layers ×\nm=32 positions\nhidden states",
      label_pos=0.5, label_above=True, color="#a05a00")

# Bridge box (central)
ax.add_patch(FancyBboxPatch(
    (4.1, 2.5), 5.0, 4.0, boxstyle="round,pad=0.05",
    facecolor="#fff5e6", edgecolor="#a05a00", linewidth=2,
))
ax.text(6.6, 6.0, "Conditional Diffusion Bridge",
        ha="center", fontsize=12, fontweight="bold", color="#a05a00")

# Bridge internals
ax.text(4.4, 5.3, "Source\nconditioning", ha="left", fontsize=8.5)
ax.add_patch(patches.Rectangle((5.6, 5.0), 1.0, 0.6,
              facecolor="#ffe0b3", edgecolor="black"))
ax.text(6.1, 5.3, "AdaLN", ha="center", va="center", fontsize=8)

ax.text(4.4, 4.4, "DDPM sampling\n(K=5 inference)", ha="left", fontsize=8.5)
ax.add_patch(patches.Rectangle((5.6, 4.1), 1.0, 0.6,
              facecolor="#ffe0b3", edgecolor="black"))
ax.text(6.1, 4.4, "× K steps", ha="center", va="center", fontsize=8)

ax.text(4.4, 3.4, "W_e warm-start\n(K=0: closed-form)", ha="left", fontsize=8.5)
ax.add_patch(patches.Rectangle((5.6, 3.1), 1.0, 0.6,
              facecolor="#ffe0b3", edgecolor="black"))
ax.text(6.1, 3.4, "Init", ha="center", va="center", fontsize=8)

ax.text(7.0, 4.4, "→ k=16\nsoft prompts\n(d=4096 each)",
        ha="left", va="center", fontsize=9.5, fontweight="bold")

# Arrow: soft prompts out
arrow(ax, 9.2, 4.5, 10.7, 4.5,
      label="injected at\nembedding layer",
      label_pos=0.5, label_above=True, color="#a05a00")

# Receiver side
ax.text(12.0, 6.3, "Receiver Agent",
        ha="center", fontsize=12, fontweight="bold")
agent_box(ax, 11.0, 3.7, 2, 1.6, "Backbone\n(Qwen3.5-9B)\n[frozen]",
          color="#f4cccc", fontsize=10)
ax.text(12.0, 3.4, "Prefill + decode\nfinal answer",
        ha="center", va="top", fontsize=8.5, style="italic")

# KL evidence
ax.text(7.0, 1.5,
        "Trained on 6,523 inter-agent pairs from main study Qwen3.5-9B decentralized runs\n"
        "Held-out KL: W_e baseline = 2.68 → K=5 diffusion = 2.35 nats/token (−12% reduction)",
        ha="center", fontsize=9, style="italic",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#f0f0f0", edgecolor="gray"))

ax.set_title("Diffusion bridge: per-agent inference path",
             fontsize=13, fontweight="bold", pad=12)
fig.tight_layout()
fig.savefig(OUT / "fig6_bridge_internals.png", dpi=150, bbox_inches="tight")
fig.savefig(OUT / "fig6_bridge_internals.pdf", bbox_inches="tight")
plt.close(fig)
print(f"Saved: {OUT / 'fig6_bridge_internals.png'}")
