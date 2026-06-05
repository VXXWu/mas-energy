"""Text-output visualization: side-by-side panels showing what each of the 4
communication channels actually produces on the same FanOutQA tasks.

Highlights the "agents fail to respond → synthesizer falls back" pattern that
makes pure-latent diffusion interesting (cheaper than diffusion hybrid, often
matches text MAS on F1 despite agents producing nothing).
"""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path

PILOT = Path("mas-energy/results/diffusion_pilot_fanoutqa_Qwen_Qwen3_5-9B_specmatched")
OUT = Path("latent_diffusion")
OUT.mkdir(exist_ok=True)

# Load the broken_prompt data (has all 4 conditions on overlapping task IDs)
records = {}
for line in open(PILOT / "end_to_end_results.broken_prompt.jsonl"):
    try: r = json.loads(line)
    except: continue
    if r.get("error") or r.get("condition") is None: continue
    records[(r["task_id"], r["condition"])] = r

# Find 11 tasks where all 4 conditions are present
tids_per_cond = {}
for (tid, cond) in records:
    tids_per_cond.setdefault(cond, set()).add(tid)
common = set.intersection(*tids_per_cond.values())
common = sorted(common)

# Show two contrasting tasks: one where text MAS hallucinates wrong, one where
# both text MAS and pure-latent get F1=1.00
TASKS_TO_SHOW = []
for tid in common:
    text_f1 = records[(tid, "text")]["f1"]
    pure_f1 = records[(tid, "latent_diffusion_pure")]["f1"]
    if text_f1 == 0.0 and pure_f1 == 1.0:
        TASKS_TO_SHOW.append(tid)
        break
for tid in common:
    text_f1 = records[(tid, "text")]["f1"]
    pure_f1 = records[(tid, "latent_diffusion_pure")]["f1"]
    if text_f1 == 1.0 and pure_f1 == 1.0 and tid not in TASKS_TO_SHOW:
        TASKS_TO_SHOW.append(tid)
        break

print(f"Showing tasks: {TASKS_TO_SHOW}")

CONDITIONS = ["text", "latent_we", "latent_diffusion", "latent_diffusion_pure"]
LABELS = {
    "text": "Text MAS",
    "latent_we": "W_e hybrid",
    "latent_diffusion": "Diffusion hybrid",
    "latent_diffusion_pure": "Pure-latent diffusion",
}
COLORS = {
    "text": "#cfe2f3",
    "latent_we": "#d9ead3",
    "latent_diffusion": "#fce5cd",
    "latent_diffusion_pure": "#f4cccc",
}

# Truncate answers for readability
MAX_CHARS = 350

fig, axes = plt.subplots(len(TASKS_TO_SHOW), 4, figsize=(20, 4.5 * len(TASKS_TO_SHOW)))
if len(TASKS_TO_SHOW) == 1:
    axes = axes.reshape(1, -1)

for ti, tid in enumerate(TASKS_TO_SHOW):
    q = records[(tid, "text")].get("question", "")
    for ci, cond in enumerate(CONDITIONS):
        ax = axes[ti, ci]
        r = records[(tid, cond)]
        ans = (r.get("synthesized_answer") or "")[:MAX_CHARS]
        if len(r.get("synthesized_answer") or "") > MAX_CHARS:
            ans += "..."
        f1 = r["f1"]
        en = r["energy"]["gpu_dynamic_energy_joules"] / 1000
        wall = r["energy"]["wall_seconds"]

        # Top row of task: show question once spanning the row
        # Actually just put it in title of first column
        if ci == 0:
            ax.set_title(f"Q: {q[:100]}",
                         fontsize=10, loc="left", color="black",
                         pad=8, fontweight="bold")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")

        # Background fill
        ax.add_patch(patches.Rectangle((0, 0), 1, 1, transform=ax.transAxes,
                                        facecolor=COLORS[cond], alpha=0.5,
                                        edgecolor="black", linewidth=1))

        # Condition label + metrics at top
        header = f"{LABELS[cond]}\nF1={f1:.2f}  |  {en:.1f} kJ  |  {wall:.0f} s"
        ax.text(0.02, 0.96, header, transform=ax.transAxes,
                fontsize=11, fontweight="bold", verticalalignment="top",
                color="black")

        # Answer text below
        ax.text(0.02, 0.78, ans, transform=ax.transAxes,
                fontsize=8.5, verticalalignment="top",
                wrap=True, family="monospace",
                color="#1a1a1a")

fig.suptitle("Per-task text outputs across 4 inter-agent communication channels\n"
             "FanOutQA, Qwen3.5-9B, R=2 debate. Top row: hard task where text MAS hallucinates. "
             "Bottom row: easy task where all conditions succeed.",
             fontsize=12, fontweight="bold", y=1.0)
fig.tight_layout()
fig.savefig(OUT / "fig4_text_output_comparison.png", dpi=150, bbox_inches="tight")
fig.savefig(OUT / "fig4_text_output_comparison.pdf", bbox_inches="tight")
plt.close(fig)
print(f"Saved: {OUT / 'fig4_text_output_comparison.png'}")
