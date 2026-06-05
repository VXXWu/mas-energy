"""Honest comparison: text MAS / latent_we / diffusion variants on FanOutQA,
with explicit accounting for the synthesizer-fallback pattern where the
synthesizer reports 'agents failed to generate' and then answers from
parametric knowledge.

For the diffusion variants, F1 from fallback cases reflects ONLY the
synthesizer's parametric-knowledge answer — the bridge mechanism contributed
zero substantive content. We report this transparently:
  - Reported F1   = FanOutQA scorer's loose accuracy on the synthesizer output
  - Bridge F1     = F1 with fallback records set to 0 (only count cases where
                    the bridge mechanism actually delivered usable content)
"""
import json
import re
import statistics as s
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PILOT = (MAS_ENERGY_ROOT / "results/diffusion_pilot_fanoutqa_Qwen_Qwen3_5-9B_specmatched")
OUT = (PROJECT_ROOT / "latent_diffusion")
OUT.mkdir(exist_ok=True)

FALLBACK_PATTERNS = [
    r"agents.*failed to generate",
    r"No response.*without any tool calls",
    r"all three agents.*returned .No response.",
    r"no.*data.*was generated",
    r"no.*information.*generated",
    r"no synthesized data",
    r"no.*substantive.*answer",
    r"failed to generate a substantive",
]
patterns_re = [re.compile(p, re.IGNORECASE) for p in FALLBACK_PATTERNS]


def is_synth_fallback(answer: str) -> bool:
    return any(p.search(answer or "") for p in patterns_re)


# Load + group
by_cond = defaultdict(list)
records = {}
for line in open(PILOT / "end_to_end_results.broken_prompt.jsonl"):
    try: r = json.loads(line)
    except: continue
    if r.get("error") or r.get("condition") is None: continue
    cond = r["condition"]
    tid = r["task_id"]
    r["_is_fallback"] = is_synth_fallback(r.get("synthesized_answer") or "")
    by_cond[cond].append(r)
    records[(tid, cond)] = r

# Aggregate stats per condition
def stats_for(cond):
    recs = by_cond.get(cond, [])
    n = len(recs)
    if n == 0: return None
    reported_f1 = [r["f1"] for r in recs]
    bridge_f1 = [r["f1"] if not r["_is_fallback"] else 0.0 for r in recs]
    energy = [r["energy"]["gpu_dynamic_energy_joules"] / 1000 for r in recs]
    n_fallback = sum(1 for r in recs if r["_is_fallback"])
    return {
        "n": n,
        "n_fallback": n_fallback,
        "fallback_rate": n_fallback / n,
        "reported_f1": s.mean(reported_f1),
        "bridge_f1": s.mean(bridge_f1),
        "energy_kJ": s.mean(energy),
    }


CONDITIONS = [
    ("text", "Text MAS"),
    ("latent_we", "W_e hybrid"),
    ("latent_diffusion", "Diffusion hybrid"),
    ("latent_diffusion_pure", "Pure-latent diffusion"),
]

# ──────────────────────────────────────────────────────────────────────
# Markdown table
# ──────────────────────────────────────────────────────────────────────

md = ["# Honest comparison: bridge contribution vs synthesizer fallback\n\n",
      "**Two F1 numbers per condition:**\n\n",
      "- **Reported F1**: FanOutQA loose-accuracy scorer on the synthesized answer "
      "(what the scoring pipeline returns).\n",
      "- **Bridge F1**: same scorer, but with all synthesizer-fallback cases set "
      "to 0. A fallback case is one where the synthesizer explicitly states "
      "'agents failed to generate substantive answer' and then answers from "
      "parametric knowledge. In those cases, the bridge mechanism delivered "
      "ZERO substantive content to the synthesizer — the F1 reflects only "
      "the model's training-data knowledge, not the bridge.\n\n",
      "## Per-condition summary\n\n",
      "| Condition | n | Reported F1 | Bridge F1 (fallback → 0) | Synth-fallback rate | Mean energy (kJ) |\n",
      "|---|---|---|---|---|---|\n"]

for cond_key, cond_label in CONDITIONS:
    st = stats_for(cond_key)
    if st is None: continue
    md.append(f"| **{cond_label}** | {st['n']} | {st['reported_f1']:.3f} | "
              f"{st['bridge_f1']:.3f} | {100*st['fallback_rate']:.0f}% "
              f"({st['n_fallback']}/{st['n']}) | {st['energy_kJ']:.1f} |\n")

md.append("\n")
md.append("## Per-task breakdown across all 4 conditions\n\n")
common = sorted(set.intersection(*[set(r['task_id'] for r in by_cond[c])
                                     for c, _ in CONDITIONS if by_cond.get(c)]))
md.append(f"On the n={len(common)} tasks where all 4 conditions produced valid records:\n\n")

md.append("| Task | Text F1 | W_e F1 | Diff. hybrid F1 (fallback?) | Pure-latent F1 (fallback?) |\n")
md.append("|---|---|---|---|---|\n")
for tid in common:
    t = records[(tid, "text")]
    w = records[(tid, "latent_we")]
    h = records[(tid, "latent_diffusion")]
    p = records[(tid, "latent_diffusion_pure")]
    h_mark = "↻ fallback" if h["_is_fallback"] else "ok"
    p_mark = "↻ fallback" if p["_is_fallback"] else "ok"
    md.append(f"| `{tid[:8]}` | {t['f1']:.2f} | {w['f1']:.2f} | "
              f"{h['f1']:.2f} ({h_mark}) | {p['f1']:.2f} ({p_mark}) |\n")

md.append("\n")
md.append("## Headline\n\n")
hyb = stats_for("latent_diffusion")
pure = stats_for("latent_diffusion_pure")
txt = stats_for("text")
md.append(f"- **Synthesizer fallback dominates the diffusion variants**: "
          f"{hyb['fallback_rate']*100:.0f}% of diffusion-hybrid runs and "
          f"{pure['fallback_rate']*100:.0f}% of pure-latent runs end with the "
          f"synthesizer reporting that agents produced nothing substantive.\n")
md.append(f"- **In those cases the reported F1 (≈0.64) is the synthesizer's "
          f"parametric-knowledge answer quality, not the bridge mechanism's "
          f"contribution.**\n")
md.append(f"- **Bridge F1 (fallback set to 0) tells the honest story**:\n")
md.append(f"    - Text MAS:                {txt['bridge_f1']:.3f} (= reported, no fallback in this condition)\n")
md.append(f"    - Diffusion hybrid:        {hyb['bridge_f1']:.3f}\n")
md.append(f"    - Pure-latent diffusion:   {pure['bridge_f1']:.3f}\n")
md.append(f"- The diffusion mechanism's downstream task contribution on FanOutQA "
          f"is currently NOT measurable above the synthesizer's parametric noise. "
          f"The bridge KL evidence (12% reduction vs W_e baseline) shows the mechanism "
          f"works at the representation level — translating that into "
          f"downstream task F1 is the open implementation gap.\n")
md.append(f"- **Energy savings (hybrid → pure-latent) remain real**: "
          f"{100*(hyb['energy_kJ']-pure['energy_kJ'])/hyb['energy_kJ']:.0f}% reduction. "
          f"Skipping Round 0 text decode saves energy regardless of whether the "
          f"bridge content is reaching the synthesizer.\n")

(OUT / "hybrid_vs_pure_table.md").write_text("".join(md))
print(f"Wrote: {OUT / 'hybrid_vs_pure_table.md'}")


# ──────────────────────────────────────────────────────────────────────
# Companion figure: reported vs bridge F1
# ──────────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(1, 2, figsize=(11, 5))

ax = axes[0]
labels = [lbl for (_, lbl) in CONDITIONS]
reported = [stats_for(c)["reported_f1"] for (c, _) in CONDITIONS]
bridge_f1 = [stats_for(c)["bridge_f1"] for (c, _) in CONDITIONS]

x = np.arange(len(labels))
w = 0.38
b1 = ax.bar(x - w/2, reported, w, color="#9ec5f5",
            edgecolor="black", linewidth=1, alpha=0.85, label="Reported F1")
b2 = ax.bar(x + w/2, bridge_f1, w, color="#e87a7a",
            edgecolor="black", linewidth=1, alpha=0.85,
            label="Bridge F1 (fallback → 0)")

for i, (rep, bri) in enumerate(zip(reported, bridge_f1)):
    ax.text(i - w/2, rep + 0.02, f"{rep:.2f}", ha="center", fontsize=9, fontweight="bold")
    ax.text(i + w/2, bri + 0.02, f"{bri:.2f}", ha="center", fontsize=9, fontweight="bold")

ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=9, rotation=10, ha="right")
ax.set_ylabel("F1 (loose accuracy)", fontsize=11)
ax.set_title("Reported F1 includes synthesizer fallback;\n"
             "Bridge F1 zeros out fallback cases (honest mechanism F1)",
             fontsize=10)
ax.set_ylim(0, 0.75)
ax.legend(loc="upper right", fontsize=9)
ax.grid(True, axis="y", alpha=0.3)

# Right panel: synthesizer fallback rate
ax = axes[1]
rates = [stats_for(c)["fallback_rate"] * 100 for (c, _) in CONDITIONS]
bars = ax.bar(x, rates, color=["#cfe2f3", "#d9ead3", "#fce5cd", "#f4cccc"],
              edgecolor="black", linewidth=1, alpha=0.85)
for i, r in enumerate(rates):
    if r > 0:
        ax.text(i, r + 2, f"{r:.0f}%", ha="center", fontsize=10, fontweight="bold")
    else:
        ax.text(i, 2, "0%", ha="center", fontsize=10)
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=9, rotation=10, ha="right")
ax.set_ylabel("Synthesizer-fallback rate (%)", fontsize=11)
ax.set_title("How often the synthesizer reports\n"
             "'agents produced nothing → I'll guess from training data'",
             fontsize=10)
ax.set_ylim(0, 100)
ax.grid(True, axis="y", alpha=0.3)

fig.suptitle("Bridge mechanism contribution vs synthesizer-fallback",
             fontsize=13, fontweight="bold")
fig.tight_layout()
fig.savefig(OUT / "fig_honest_f1_comparison.png", dpi=150, bbox_inches="tight")
fig.savefig(OUT / "fig_honest_f1_comparison.pdf", bbox_inches="tight")
plt.close(fig)
print(f"Saved: {OUT / 'fig_honest_f1_comparison.png'}")


# ──────────────────────────────────────────────────────────────────────
# Update text_outputs_comparison.md to mark fallback cases
# ──────────────────────────────────────────────────────────────────────

# Pick 2 contrasting tasks as before
hybrid_tids = {t for (t, c) in records if c == "latent_diffusion"}
pure_tids = {t for (t, c) in records if c == "latent_diffusion_pure"}
common_tids = sorted(set.intersection(*[
    set(r["task_id"] for r in by_cond[c]) for c, _ in CONDITIONS]))

TASKS_TO_SHOW = []
for tid in common_tids:
    if records[(tid, "text")]["f1"] == 0.0 and records[(tid, "latent_diffusion_pure")]["f1"] == 1.0:
        TASKS_TO_SHOW.append(tid)
        break
for tid in common_tids:
    if records[(tid, "text")]["f1"] == 1.0 and records[(tid, "latent_diffusion_pure")]["f1"] == 1.0 and tid not in TASKS_TO_SHOW:
        TASKS_TO_SHOW.append(tid)
        break

out_lines = ["# Per-Task Text Outputs Across Four Inter-Agent Communication Channels\n\n",
             "**IMPORTANT**: In the diffusion variants, the synthesizer often explicitly "
             "reports that agents failed to produce substantive content and then answers "
             "from parametric knowledge. F1 in those cases reflects ONLY the model's "
             "training-data answer quality, not bridge content. Look for the "
             "**↻ SYNTH FALLBACK** marker on each panel.\n\n",
             "## Per-condition summary\n\n",
             "| Condition | n | Reported F1 | Bridge F1 (fallback → 0) | Synth-fallback rate |\n",
             "|---|---|---|---|---|\n"]
for cond_key, cond_label in CONDITIONS:
    st = stats_for(cond_key)
    out_lines.append(f"| **{cond_label}** | {st['n']} | {st['reported_f1']:.3f} | "
                     f"{st['bridge_f1']:.3f} | {100*st['fallback_rate']:.0f}% |\n")
out_lines.append("\n---\n\n")

for ti, tid in enumerate(TASKS_TO_SHOW):
    q = records[(tid, "text")].get("question", "")
    out_lines.append(f"## Task {ti+1}: `{tid}`\n\n**Q**: {q}\n\n")
    for cond_key, cond_label in CONDITIONS:
        r = records[(tid, cond_key)]
        ans = r.get("synthesized_answer") or ""
        f1 = r["f1"]
        en = r["energy"]["gpu_dynamic_energy_joules"] / 1000
        wall = r["energy"]["wall_seconds"]
        fallback_mark = "  ↻ **SYNTH FALLBACK**" if r["_is_fallback"] else ""
        bridge_f1 = 0.0 if r["_is_fallback"] else f1
        out_lines.append(f"### {cond_label}{fallback_mark}\n")
        out_lines.append(f"**Reported F1**: {f1:.2f}  |  **Bridge F1**: {bridge_f1:.2f}  "
                         f"|  **Energy**: {en:.1f} kJ  |  **Wall**: {wall:.0f} s\n\n")
        out_lines.append("```\n" + ans + "\n```\n\n")
    out_lines.append("---\n\n")

(OUT / "text_outputs_comparison.md").write_text("".join(out_lines))
print(f"Wrote: {OUT / 'text_outputs_comparison.md'}")
