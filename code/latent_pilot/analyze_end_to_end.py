"""Aggregate Stage 8 end-to-end pilot results into a text summary + figures.

Reads `end_to_end_results.jsonl` (output of `eval_hybrid_diffusion.py`) and
emits:

  - stdout: per-condition table (n, mean/median F1, mean energy_J, mean
    energy-per-correct, mean wall) and paired-bootstrap deltas between the
    three conditions on F1 and energy.
  - <out-dir>/summary.json:  same numbers in machine-readable form.
  - <out-dir>/fig_pareto.png:    mean energy vs mean F1 per condition, with
                                 bootstrap 95% CI bars.
  - <out-dir>/fig_energy_box.png: per-task energy distribution by condition.
  - <out-dir>/fig_f1_box.png:     per-task F1 distribution by condition.

Usage:
  python -m latent_pilot.analyze_end_to_end \
      --results .../end_to_end_results.jsonl \
      --out-dir .../analysis
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

CONDITIONS = ["sas", "main_study_decent", "text", "latent_we",
              "latent_diffusion", "latent_diffusion_pure"]
COND_LABEL = {
    "sas": "SAS (1 agent)",
    "main_study_decent": "Main-study Decent",
    "text": "Text MAS (pilot)",
    "latent_we": "W_e hybrid",
    "latent_diffusion": "Diffusion hybrid",
    "latent_diffusion_pure": "Pure-latent diffusion",
}

# Default path to main-study SAS jsonl for FanOutQA on Qwen3.5-9B.
# k=5 is picked when available (k-matched to the pilot's MAS conditions which
# use max_react_steps=5). Falls back to higher-k files otherwise — but the
# unfair-comparison guard in auto_conclusion will flag the mismatch.
def _default_sas_path():
    import glob
    candidates = sorted(glob.glob(
        "mas-energy/results/fanoutqa_v*/Qwen_Qwen3.5-9B_sas_k*.jsonl"
    ))
    # Prefer k=5 (k-matched to pilot MAS), else any available.
    for c in candidates:
        if "_sas_k5.jsonl" in c:
            return Path(c)
    return Path(candidates[0]) if candidates else Path(
        "mas-energy/results/fanoutqa_v4/Qwen_Qwen3.5-9B_sas_k5.jsonl"
    )


DEFAULT_SAS_PATH = _default_sas_path()


def _default_main_decent_path():
    """Best available main-study Decentralized k=5 R=2 result. This is the
    real MAS-performance ceiling the latent variants should be measured
    against, since the pilot's text MAS reimplementation undershoots by
    ~0.07 F1 on overlapping tasks. Prefer k=5 (k-matched to pilot)."""
    import glob
    candidates = sorted(glob.glob(
        "mas-energy/results/fanoutqa_v*/Qwen_Qwen3.5-9B_decentralized_k*.jsonl"
    ))
    # Prefer k=5 explicitly so the comparison is k-matched.
    for c in candidates:
        if "_decentralized_k5." in c:
            return Path(c)
    return Path(candidates[0]) if candidates else None


DEFAULT_MAIN_DECENT_PATH = _default_main_decent_path()


def load_main_decent_records(path, task_id_filter=None,
                              required_k=5, required_R=2):
    """Pull main-study Decentralized records at matching k and R as a separate
    condition. Same schema mapping as load_sas_records."""
    out = []
    if path is None or not path.exists():
        return out
    with open(path) as f:
        for line in f:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("max_react_steps") != required_k:
                continue
            R = r.get("rounds_used") or r.get("n_rounds_override")
            if R != required_R:
                continue
            tid = r.get("task_id")
            if not tid:
                continue
            if task_id_filter is not None and tid not in task_id_filter:
                continue
            la = r.get("loose_accuracy", 0) or 0
            out.append({
                "condition": "main_study_decent",
                "n_rounds": required_R,
                "task_id": tid,
                "f1": la,
                "loose_accuracy": la,
                "evaluation": {},
                "energy": {
                    "gpu_dynamic_energy_joules": r.get("gpu_dynamic_energy_joules", 0),
                    "wall_seconds": r.get("total_wall_seconds", 0),
                },
                "_k_budget": required_k,
                "_source_label": f"main_study_decent:{path.name}",
                "error": None,
            })
    return out


def load_sas_records(sas_path: Path, task_id_filter=None, source_label="external_k20"):
    """Load main-study SAS jsonl and convert to the Stage 8 record schema.

    Schema mapping (SAS file → Stage 8):
      task_id                       → task_id
      loose_accuracy                → f1 (Stage 8's `f1` field happens to be
                                         FanOutQA loose accuracy, so map 1:1)
      loose_accuracy                → loose_accuracy
      gpu_dynamic_energy_joules     → energy.gpu_dynamic_energy_joules
      total_wall_seconds            → energy.wall_seconds
      max_react_steps               → _k_budget  (tagged for unfair-comparison detection)

    SAS has no debate rounds, so n_rounds=1 by definition.

    The `source_label` is stored on each record so the analysis can detect
    when SAS data is k-budget-mismatched against the MAS conditions (e.g.,
    k=20 SAS vs k=5 MAS — the comparison is unfair and the dominance headline
    must be suppressed).
    """
    out = []
    if not sas_path.exists():
        return out
    with open(sas_path) as f:
        for line in f:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            tid = r.get("task_id")
            if not tid:
                continue
            if task_id_filter is not None and tid not in task_id_filter:
                continue
            la = r.get("loose_accuracy", 0) or 0
            out.append({
                "condition": "sas",
                "n_rounds": 1,
                "task_id": tid,
                "f1": la,
                "loose_accuracy": la,
                "evaluation": {},
                "energy": {
                    "gpu_dynamic_energy_joules": r.get("gpu_dynamic_energy_joules", 0),
                    "wall_seconds": r.get("total_wall_seconds", 0),
                },
                "_k_budget": r.get("max_react_steps"),
                "_source_label": source_label,
                "error": None,
            })
    return out


def load_sas_k_sensitivity():
    """Discover SAS results at multiple k budgets across the project. Returns
    a list of (k, F1_mean, energy_mean, n_tasks, source_path) tuples sorted by k.

    Sources scanned:
      - mas-energy/results/diffusion_pilot_fanoutqa_*/mechanism_diagnosis.json
        (k=10 SAS from Stage 1 of the pilot, labelled `single`)
      - mas-energy/results/fanoutqa_v*/Qwen_Qwen3.5-9B_sas_k*.jsonl
        (main-study SAS at various k)
    """
    import glob
    out = []

    # Mechanism Stage 1 (k=10)
    mech_paths = glob.glob(
        "mas-energy/results/diffusion_pilot_fanoutqa_*/mechanism_diagnosis.json"
    )
    for mp in mech_paths:
        try:
            d = json.load(open(mp))
            single = d.get("means", {}).get("single")
            if single is None:
                continue
            out.append((
                10,  # mechanism stage 1 uses MECH_K=10
                float(single.get("f1", 0)),
                float(single.get("energy_j", 0)),
                int(d.get("n_records", 0)),
                mp,
            ))
        except Exception:
            continue

    # Main-study SAS files at various k
    main_paths = sorted(glob.glob(
        "mas-energy/results/fanoutqa_v*/Qwen_Qwen3.5-9B_sas_k*.jsonl"
    ))
    for mp in main_paths:
        try:
            f1s, ens, k_seen = [], [], None
            for line in open(mp):
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                la = r.get("loose_accuracy", 0) or 0
                en = r.get("gpu_dynamic_energy_joules", 0)
                k_seen = r.get("max_react_steps", k_seen)
                f1s.append(la)
                ens.append(en)
            if not f1s:
                continue
            # Infer k from filename if not in records
            if k_seen is None:
                import re
                m = re.search(r"_sas_k(\d+)", mp)
                k_seen = int(m.group(1)) if m else None
            if k_seen is None:
                continue
            out.append((
                k_seen,
                sum(f1s) / len(f1s),
                sum(ens) / len(ens),
                len(f1s),
                mp,
            ))
        except Exception:
            continue

    return sorted(out, key=lambda x: x[0])


def load_records(path: Path):
    """Yield per-task records; skip the first line if it's a session_meta dict
    without a `condition` field, and drop errored records.

    Older records may be missing `n_rounds` — these are treated as R=1 since
    that was the hardcoded value before the R=N refactor."""
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "condition" not in rec:
                continue  # session_meta header
            if rec.get("error"):
                continue
            rec.setdefault("n_rounds", 1)
            out.append(rec)
    return out


def bootstrap_ci(values, n_boot=2000, alpha=0.05, rng=None):
    if len(values) == 0:
        return (float("nan"), float("nan"), float("nan"))
    rng = rng or np.random.default_rng(0)
    arr = np.asarray(values, dtype=float)
    idx = rng.integers(0, len(arr), size=(n_boot, len(arr)))
    means = arr[idx].mean(axis=1)
    return float(arr.mean()), float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


def paired_bootstrap_delta(a_by_task, b_by_task, n_boot=2000, alpha=0.05, rng=None):
    """Paired delta b - a across tasks that have both conditions scored."""
    common = sorted(set(a_by_task) & set(b_by_task))
    if not common:
        return None
    a = np.array([a_by_task[t] for t in common], dtype=float)
    b = np.array([b_by_task[t] for t in common], dtype=float)
    d = b - a
    rng = rng or np.random.default_rng(0)
    idx = rng.integers(0, len(d), size=(n_boot, len(d)))
    boot_means = d[idx].mean(axis=1)
    return {
        "n_paired": len(common),
        "mean_delta": float(d.mean()),
        "ci_lo": float(np.quantile(boot_means, alpha / 2)),
        "ci_hi": float(np.quantile(boot_means, 1 - alpha / 2)),
    }


def per_condition(records, cond, n_rounds=None):
    """If n_rounds is None, include all rounds (legacy behavior). Otherwise
    filter to records with that specific R value."""
    rs = [r for r in records if r["condition"] == cond
          and (n_rounds is None or r.get("n_rounds", 1) == n_rounds)]
    f1 = np.array([r.get("f1", 0.0) or 0.0 for r in rs], dtype=float)
    loose = np.array([r.get("loose_accuracy", 0.0) or 0.0 for r in rs], dtype=float)
    energy = np.array(
        [r["energy"].get("gpu_dynamic_energy_joules", float("nan")) for r in rs],
        dtype=float,
    )
    wall = np.array(
        [r["energy"].get("wall_seconds", float("nan")) for r in rs],
        dtype=float,
    )
    # energy-per-correct: avoid /0 by skipping when n_correct == 0
    n_correct = int((f1 > 0).sum())
    epc = float(np.nansum(energy) / n_correct) if n_correct > 0 else float("nan")
    label_suffix = f" (R={n_rounds})" if n_rounds is not None else ""
    return {
        "condition": cond,
        "n_rounds": n_rounds,
        "label": COND_LABEL[cond] + label_suffix,
        "n_tasks": len(rs),
        "mean_f1": bootstrap_ci(f1),
        "mean_loose": bootstrap_ci(loose),
        "mean_energy_J": bootstrap_ci(energy[~np.isnan(energy)].tolist()),
        "mean_wall_s": bootstrap_ci(wall[~np.isnan(wall)].tolist()),
        "energy_per_correct_J": epc,
        "n_correct": n_correct,
        "_f1_by_task": {r["task_id"]: r.get("f1", 0.0) or 0.0 for r in rs if r.get("task_id") is not None},
        "_energy_by_task": {r["task_id"]: r["energy"].get("gpu_dynamic_energy_joules", float("nan")) for r in rs if r.get("task_id") is not None},
        "_raw_f1": f1.tolist(),
        "_raw_energy": energy.tolist(),
    }


def fmt_ci(triple, digits=3):
    m, lo, hi = triple
    if np.isnan(m):
        return "   nan"
    return f"{m:.{digits}f} [{lo:.{digits}f}, {hi:.{digits}f}]"


def interpret_delta(metric, d, lower_is_better=False):
    """Map a paired delta (mean + CI) to a one-line natural-language verdict.
    `metric` is the human-readable name; `d` is {mean_delta, ci_lo, ci_hi, ...}.
    Returns: (verdict_str, sign) where sign ∈ {-1, 0, 1} for worse / no-diff / better."""
    if d is None:
        return (f"{metric}: insufficient paired data", 0)
    lo, hi, m = d["ci_lo"], d["ci_hi"], d["mean_delta"]
    crosses_zero = (lo < 0 < hi)
    if crosses_zero:
        return (
            f"{metric}: indistinguishable — Δ={m:+.3f} [{lo:+.3f}, {hi:+.3f}] crosses 0",
            0,
        )
    # CI is wholly on one side.
    if lower_is_better:
        # "lower" means b - a < 0 is the improvement
        if hi < 0:
            return (f"{metric}: B is lower — Δ={m:+.3f} [{lo:+.3f}, {hi:+.3f}]", 1)
        else:
            return (f"{metric}: B is higher — Δ={m:+.3f} [{lo:+.3f}, {hi:+.3f}]", -1)
    else:
        if lo > 0:
            return (f"{metric}: B is higher — Δ={m:+.3f} [{lo:+.3f}, {hi:+.3f}]", 1)
        else:
            return (f"{metric}: B is lower — Δ={m:+.3f} [{lo:+.3f}, {hi:+.3f}]", -1)


def auto_conclusion(deltas):
    """Map all paired deltas to a structured, scenario-aware multi-line verdict.
    Returns list of lines to print."""
    lines = ["", "Auto-interpretation (CI-based; bullets fire only when supported):", "=" * 130]

    def fmt(a, b):
        return f"{COND_LABEL[b]} vs {COND_LABEL[a]}"

    for key, d in deltas.items():
        if not isinstance(key, tuple) or len(key) != 2 or d is None:
            continue
        a, b = key
        f1v, _ = interpret_delta("F1", d["f1"], lower_is_better=False)
        env, _ = interpret_delta("Energy", d["energy"], lower_is_better=True)
        lines.append(f"  {fmt(a, b)}:")
        lines.append(f"    {f1v}")
        lines.append(f"    {env}")

    # SAS floor block — fires only when the SAS comparison is k-matched
    # against the MAS conditions. The pilot's MAS uses k=5 (max_react_steps);
    # if the SAS records come from main-study k=20, the comparison is unfair
    # because FanOutQA F1 is highly k-sensitive (see SAS k-sensitivity panel).
    sas_pairs = [("sas", "text"), ("sas", "latent_diffusion"),
                 ("sas", "latent_diffusion_pure")]
    if any(p in deltas and deltas[p] is not None for p in sas_pairs):
        sas_k_budget = deltas.get("_sas_k_budget")  # set in main() before calling auto_conclusion
        mas_k_budget = deltas.get("_mas_k_budget", 5)
        k_matched = (sas_k_budget is None) or (sas_k_budget == mas_k_budget)
        lines.append("")
        lines.append("SAS floor comparison (does MAS justify its energy over single agent?):")
        if not k_matched:
            lines.append(f"  ⚠️  UNFAIR COMPARISON: SAS uses k={sas_k_budget} ReAct steps, MAS uses k={mas_k_budget}.")
            lines.append(f"      FanOutQA F1 is highly k-sensitive — see 'SAS k-sensitivity' panel below.")
            lines.append(f"      Dominance verdicts SUPPRESSED. Submit in-pilot k={mas_k_budget} SAS for")
            lines.append(f"      a clean comparison: E2E_CONDITIONS=\"sas\" sbatch run_pilot_4b.sbatch")
        for p in sas_pairs:
            if p not in deltas or deltas[p] is None:
                continue
            f1_v, f1_s = interpret_delta("F1", deltas[p]["f1"])
            en_v, en_s = interpret_delta("Energy", deltas[p]["energy"], lower_is_better=True)
            a, b = p
            lines.append(f"  {COND_LABEL[b]} vs {COND_LABEL[a]}:")
            lines.append(f"    {f1_v}")
            lines.append(f"    {en_v}")
            # Headline per pair — only fire when k-matched.
            if not k_matched:
                continue
            if f1_s == 1 and en_s == -1:
                lines.append(f"    → {COND_LABEL[b]} buys higher F1 at higher energy vs SAS — pay-to-play tradeoff.")
            elif f1_s == 1 and en_s == 0:
                lines.append(f"    → {COND_LABEL[b]} improves F1 over SAS at indistinguishable energy — free accuracy.")
            elif f1_s == 0 and en_s == -1:
                lines.append(f"    → {COND_LABEL[b]} is DOMINATED by SAS — same F1, more energy.")
            elif f1_s == -1:
                lines.append(f"    → {COND_LABEL[b]} is strictly dominated by SAS — lower F1 AND higher energy.")

    # Pure-latent variant fires its own headline if present (this tests the
    # design-as-intended energy claim — does latent comm actually save energy?)
    pure_pairs = [("text", "latent_diffusion_pure"),
                  ("latent_diffusion", "latent_diffusion_pure")]
    if all(p in deltas and deltas[p] is not None for p in pure_pairs):
        pure_vs_text_f1 = interpret_delta("", deltas[("text", "latent_diffusion_pure")]["f1"])[1]
        pure_vs_text_en = interpret_delta("", deltas[("text", "latent_diffusion_pure")]["energy"], lower_is_better=True)[1]
        pure_vs_diff_en = interpret_delta("", deltas[("latent_diffusion", "latent_diffusion_pure")]["energy"], lower_is_better=True)[1]
        lines.append("")
        lines.append("Pure-latent variant headline (tests the design-as-intended energy claim):")
        if pure_vs_diff_en == 1:
            lines.append("  Pure-latent diffusion uses LOWER energy than the text-Round-0 diffusion variant — "
                         "skipping Round 0 decode helps as predicted.")
        elif pure_vs_diff_en == 0:
            lines.append("  Pure-latent diffusion uses indistinguishable energy from the text-Round-0 variant — "
                         "Round 0 decode wasn't the dominant cost.")
        else:
            lines.append("  Pure-latent diffusion uses HIGHER energy than text-Round-0 variant — unexpected; "
                         "investigate latent CoT step cost vs decoded ReAct.")
        if pure_vs_text_en == 1 and pure_vs_text_f1 in (0, 1):
            lines.append("  Pure-latent diffusion is Pareto-competitive with text MAS — "
                         "latent communication's energy claim is validated.")
        elif pure_vs_text_en == 1 and pure_vs_text_f1 == -1:
            lines.append("  Pure-latent diffusion saves energy at the cost of F1 — accuracy-cost tradeoff.")
        elif pure_vs_text_en == -1:
            lines.append("  Pure-latent diffusion is STILL more expensive than text MAS — even the corrected "
                         "design doesn't beat text on energy at this corpus/model.")

    # Headline conclusion — fires only if all the relevant pairs are present.
    needed = [("text", "latent_we"), ("text", "latent_diffusion"), ("latent_we", "latent_diffusion")]
    if all(p in deltas and deltas[p] is not None for p in needed):
        we_f1_sign = interpret_delta("", deltas[("text", "latent_we")]["f1"])[1]
        we_en_sign = interpret_delta("", deltas[("text", "latent_we")]["energy"], lower_is_better=True)[1]
        df_f1_sign = interpret_delta("", deltas[("text", "latent_diffusion")]["f1"])[1]
        df_en_sign = interpret_delta("", deltas[("text", "latent_diffusion")]["energy"], lower_is_better=True)[1]
        df_vs_we_f1_sign = interpret_delta("", deltas[("latent_we", "latent_diffusion")]["f1"])[1]

        lines.append("")
        lines.append("Headline:")
        # Diffusion vs text: accuracy & energy
        if df_f1_sign == 0 and df_en_sign == 1:
            lines.append("  Diffusion hybrid matches text F1 at LOWER energy — energy-efficient substitution succeeds.")
        elif df_f1_sign == 1 and df_en_sign == 1:
            lines.append("  Diffusion hybrid IMPROVES F1 AND lowers energy — strict Pareto improvement over text MAS.")
        elif df_f1_sign == -1 and df_en_sign == 1:
            lines.append("  Diffusion hybrid trades F1 for energy savings — accuracy-cost frontier shift.")
        elif df_f1_sign == 0 and df_en_sign == 0:
            lines.append("  Diffusion hybrid matches text MAS on BOTH F1 and energy — no measurable downstream effect.")
        elif df_f1_sign == -1 and df_en_sign != 1:
            lines.append("  Diffusion hybrid is dominated by text MAS — F1 lower without energy gain.")
        else:
            lines.append("  Diffusion hybrid result is mixed; see paired deltas above.")

        # Diffusion vs W_e: does diffusion's KL gain transfer to F1?
        if df_vs_we_f1_sign == 1:
            lines.append("  Diffusion BEATS W_e closed-form on F1 — KL advantage transfers downstream.")
        elif df_vs_we_f1_sign == 0:
            lines.append("  Diffusion ≈ W_e closed-form on F1 — bridge's KL gain does not translate to accuracy "
                         "(consistent with channel-muting story).")
        else:
            lines.append("  Diffusion UNDERPERFORMS W_e closed-form on F1 — learned bridge harms what alignment alone gets right.")

        # W_e vs text: does latent substitution work at all?
        if we_f1_sign == 0:
            lines.append("  W_e hybrid matches text MAS on F1 — latent channel substitution preserves accuracy.")
        elif we_f1_sign == 1:
            lines.append("  W_e hybrid IMPROVES on text MAS F1 — latent channel adds signal.")
        else:
            lines.append("  W_e hybrid degrades text MAS F1 — closed-form alignment is lossy in this setting.")

    return lines


def print_summary(stats, deltas, out_lines):
    def w(line=""):
        out_lines.append(line)
        print(line)
    w()
    w(f"{'condition':>20}  {'n':>5}  {'F1 (mean [95% CI])':>28}  {'energy J':>26}  {'wall s':>22}  {'J/correct':>10}")
    w("=" * 130)
    for s in stats:
        w(
            f"{s['label']:>20}  {s['n_tasks']:>5}  "
            f"{fmt_ci(s['mean_f1']):>28}  "
            f"{fmt_ci(s['mean_energy_J'], digits=1):>26}  "
            f"{fmt_ci(s['mean_wall_s'], digits=1):>22}  "
            f"{s['energy_per_correct_J']:>10.1f}"
        )
    w()
    w("Paired deltas (B - A) with bootstrap 95% CI on per-task differences:")
    w("=" * 130)
    for key, d in deltas.items():
        # Skip sentinel keys (used to pass metadata to auto_conclusion).
        if not isinstance(key, tuple) or len(key) != 2 or d is None:
            continue
        a, b = key
        f1d = d["f1"]
        ed = d["energy"]
        w(
            f"  {COND_LABEL[b]:>17} vs {COND_LABEL[a]:>17}  "
            f"n={f1d['n_paired']:>3}  "
            f"ΔF1={f1d['mean_delta']:+.3f} [{f1d['ci_lo']:+.3f}, {f1d['ci_hi']:+.3f}]  "
            f"ΔEnergy={ed['mean_delta']:+.1f}J [{ed['ci_lo']:+.1f}, {ed['ci_hi']:+.1f}]"
        )
    for line in auto_conclusion(deltas):
        w(line)
    w()


def plot_pareto(stats, out_path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 4.5))
    colors = {"text": "tab:blue", "latent_we": "tab:orange", "latent_diffusion": "tab:green"}
    for s in stats:
        ef, lo_f, hi_f = s["mean_f1"]
        ee, lo_e, hi_e = s["mean_energy_J"]
        ax.errorbar(
            ee, ef,
            xerr=[[ee - lo_e], [hi_e - ee]],
            yerr=[[ef - lo_f], [hi_f - ef]],
            fmt="o", markersize=10, capsize=4,
            color=colors.get(s["condition"], "gray"),
            label=f"{s['label']} (n={s['n_tasks']})",
        )
        ax.annotate(s["label"], (ee, ef), xytext=(8, 8), textcoords="offset points", fontsize=9)
    ax.set_xlabel("Mean GPU dynamic energy per task (J)")
    ax.set_ylabel("Mean F1")
    ax.set_title("Pareto: accuracy vs energy by communication channel")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_box(stats, key_raw, ylabel, title, out_path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 4.5))
    data = [np.array(s[key_raw])[~np.isnan(np.array(s[key_raw]))] for s in stats]
    labels = [s["label"] for s in stats]
    try:
        bp = ax.boxplot(data, tick_labels=labels, showmeans=True, meanline=True, patch_artist=True)
    except TypeError:
        bp = ax.boxplot(data, labels=labels, showmeans=True, meanline=True, patch_artist=True)
    palette = ["#aec7e8", "#ffbb78", "#98df8a"]
    for patch, color in zip(bp["boxes"], palette[: len(bp["boxes"])]):
        patch.set_facecolor(color)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _analyze_single_R(R_records, out_dir, rounds_label=""):
    """Run the per-condition table + paired deltas + figures for a single R slice."""
    stats = [per_condition(R_records, c) for c in CONDITIONS]
    stats = [s for s in stats if s["n_tasks"] > 0]
    if not stats:
        return
    cond_to_stats = {s["condition"]: s for s in stats}
    deltas = {}
    pairs = [
        ("sas", "main_study_decent"),
        ("sas", "text"),
        ("sas", "latent_we"),
        ("sas", "latent_diffusion"),
        ("sas", "latent_diffusion_pure"),
        ("main_study_decent", "text"),
        ("main_study_decent", "latent_we"),
        ("main_study_decent", "latent_diffusion"),
        ("main_study_decent", "latent_diffusion_pure"),
        ("text", "latent_we"),
        ("text", "latent_diffusion"),
        ("text", "latent_diffusion_pure"),
        ("latent_we", "latent_diffusion"),
        ("latent_we", "latent_diffusion_pure"),
        ("latent_diffusion", "latent_diffusion_pure"),
    ]
    rng = np.random.default_rng(0)
    for a, b in pairs:
        if a not in cond_to_stats or b not in cond_to_stats:
            continue
        sa, sb = cond_to_stats[a], cond_to_stats[b]
        f1_d = paired_bootstrap_delta(sa["_f1_by_task"], sb["_f1_by_task"], rng=rng)
        en_d = paired_bootstrap_delta(sa["_energy_by_task"], sb["_energy_by_task"], rng=rng)
        deltas[(a, b)] = {"f1": f1_d, "energy": en_d} if f1_d and en_d else None
    lines = [f"=== {rounds_label} ==="] if rounds_label else []
    print_summary(stats, deltas, lines)
    plot_pareto(stats, out_dir / "fig_pareto.png")
    plot_box(stats, "_raw_energy", "GPU dynamic energy per task (J)",
             f"Energy distribution {rounds_label}".rstrip(), out_dir / "fig_energy_box.png")
    plot_box(stats, "_raw_f1", "F1 per task",
             f"F1 distribution {rounds_label}".rstrip(), out_dir / "fig_f1_box.png")
    with open(out_dir / "summary.txt", "w") as f:
        f.write("\n".join(lines) + "\n")


def _emit_sas_k_sensitivity(out_dir, mas_stats=None):
    """Scan the project for SAS results at multiple k budgets and emit:
      - sas_k_sensitivity.txt: a table of k → F1, energy across all sources
      - fig_sas_k_sensitivity.png: F1 vs k scatter, with MAS conditions overlaid
        as horizontal reference lines if their F1 values are provided
    """
    rows = load_sas_k_sensitivity()
    if not rows:
        return None
    lines = ["", "SAS k-sensitivity on FanOutQA (Qwen3.5-9B):",
             "=" * 130,
             f"{'k':>4}  {'F1 (loose)':>12}  {'energy J':>12}  {'n':>5}  source"]
    for k, f1, en, n, src in rows:
        lines.append(f"{k:>4}  {f1:>12.3f}  {en:>12.0f}  {n:>5}  {src}")
    text = "\n".join(lines)
    print(text)
    with open(out_dir / "sas_k_sensitivity.txt", "w") as f:
        f.write(text + "\n")

    # Figure: F1 vs k
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        ks = [r[0] for r in rows]
        f1s = [r[1] for r in rows]
        ens = [r[2] for r in rows]
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
        ax1.plot(ks, f1s, marker="o", linewidth=2, color="tab:blue")
        ax1.set_xlabel("SAS k (max ReAct steps)")
        ax1.set_ylabel("F1 (loose accuracy)")
        ax1.set_title("SAS budget sensitivity: F1 vs k")
        ax1.grid(True, alpha=0.3)
        for k, f1 in zip(ks, f1s):
            ax1.annotate(f"k={k}\nF1={f1:.2f}", (k, f1), xytext=(5, 5),
                         textcoords="offset points", fontsize=8)
        if mas_stats:
            for s in mas_stats:
                ax1.axhline(s["mean_f1"][0], linestyle="--", alpha=0.5,
                            label=f"{s['label']} (MAS k=5)")
            ax1.legend(loc="lower right", fontsize=7)

        ax2.plot(ks, ens, marker="s", linewidth=2, color="tab:orange")
        ax2.set_xlabel("SAS k (max ReAct steps)")
        ax2.set_ylabel("Energy per task (J)")
        ax2.set_title("SAS budget sensitivity: Energy vs k")
        ax2.grid(True, alpha=0.3)

        fig.tight_layout()
        fig.savefig(out_dir / "fig_sas_k_sensitivity.png", dpi=150)
        plt.close(fig)
    except Exception as e:
        print(f"  (k-sensitivity figure skipped: {e})")

    return rows


def _print_R_effect(records, out_dir):
    """Within-condition R=1 vs R=2 paired delta. Shows whether more debate
    rounds change F1/energy for each condition."""
    R_vals = sorted({r.get("n_rounds", 1) for r in records})
    if len(R_vals) < 2:
        return
    lines = ["", "R-effect (R=2 vs R=1) paired delta per condition:", "=" * 130]
    rng = np.random.default_rng(0)
    for c in CONDITIONS:
        r1 = per_condition([r for r in records if r.get("n_rounds", 1) == 1], c)
        r2 = per_condition([r for r in records if r.get("n_rounds", 1) == 2], c)
        if r1["n_tasks"] == 0 or r2["n_tasks"] == 0:
            continue
        f1_d = paired_bootstrap_delta(r1["_f1_by_task"], r2["_f1_by_task"], rng=rng)
        en_d = paired_bootstrap_delta(r1["_energy_by_task"], r2["_energy_by_task"], rng=rng)
        if f1_d is None or en_d is None:
            continue
        lines.append(
            f"  {COND_LABEL[c]:>22}  n={f1_d['n_paired']:>3}  "
            f"ΔF1={f1_d['mean_delta']:+.3f} [{f1_d['ci_lo']:+.3f}, {f1_d['ci_hi']:+.3f}]  "
            f"ΔEnergy={en_d['mean_delta']:+.1f}J [{en_d['ci_lo']:+.1f}, {en_d['ci_hi']:+.1f}]"
        )
    text = "\n".join(lines) + "\n"
    print(text)
    with open(out_dir / "R_effect.txt", "w") as f:
        f.write(text)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True, type=Path,
                    help="Path to end_to_end_results.jsonl")
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="Directory for summary.json and figures. "
                         "Defaults to <results>.parent / 'analysis'.")
    ap.add_argument("--sas-records", type=Path, default=DEFAULT_SAS_PATH,
                    help="Optional path to main-study SAS jsonl. When present, "
                         "SAS is added as a 5th condition, filtered to the "
                         "intersection of task IDs with --results. Set to a "
                         "nonexistent path to skip.")
    ap.add_argument("--no-sas", action="store_true",
                    help="Skip SAS even if the default path exists.")
    args = ap.parse_args()

    out_dir = args.out_dir or args.results.parent / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    records = load_records(args.results)
    if not records:
        raise SystemExit(f"No valid records in {args.results}")

    # SAS k-budget detection: in-pilot SAS records (condition="sas" already in
    # the JSONL) carry the MAS k-budget (max_react_steps=5) so are k-matched.
    # External SAS from the main-study file is k=20 — record its budget so the
    # auto-interpretation can flag the unfair comparison.
    in_pilot_sas_present = any(
        r.get("condition") == "sas" for r in records
    )
    sas_k_budget = None
    if not args.no_sas and args.sas_records is not None and not in_pilot_sas_present:
        stage8_task_ids = {r["task_id"] for r in records if r.get("task_id")}
        sas_recs = load_sas_records(args.sas_records,
                                    task_id_filter=stage8_task_ids,
                                    source_label=f"external:{args.sas_records.name}")
        if sas_recs:
            records.extend(sas_recs)
            sas_k_budget = sas_recs[0].get("_k_budget")
            print(f"Added {len(sas_recs)} SAS records from {args.sas_records} "
                  f"(intersection with Stage 8 task IDs; k={sas_k_budget or 'unknown'})")
        elif args.sas_records.exists():
            print(f"  (no overlap between Stage 8 task IDs and {args.sas_records})")
    elif in_pilot_sas_present:
        # In-pilot SAS records are k-matched against MAS by construction (the
        # pilot's --max-react-steps is the same for SAS and MAS conditions).
        sas_k_budget = 5
        print(f"Using in-pilot SAS records (k-matched, k={sas_k_budget})")

    # Pull main-study Decentralized as the real MAS-performance baseline. The
    # pilot's text MAS reimplementation is ~0.07 F1 below main-study Decent
    # on overlapping tasks; this lets the latent variants be measured against
    # the actual MAS ceiling.
    if DEFAULT_MAIN_DECENT_PATH is not None:
        stage8_task_ids = {r["task_id"] for r in records if r.get("task_id")}
        md_recs = load_main_decent_records(
            DEFAULT_MAIN_DECENT_PATH,
            task_id_filter=stage8_task_ids,
            required_k=5, required_R=2,
        )
        if md_recs:
            records.extend(md_recs)
            print(f"Added {len(md_recs)} main-study Decentralized k=5 R=2 records "
                  f"(intersection with Stage 8 task IDs) as 'Main-study Decent' condition")

    rounds_present = sorted({r.get("n_rounds", 1) for r in records})

    # If multiple R values present, run the analysis once per R into a
    # subdirectory R{R}/, then continue producing the legacy single-R output
    # for whichever R has the most records (so the top-level summary.txt /
    # figs still exist for whatever analysis tooling expects them).
    if len(rounds_present) > 1:
        per_R_n = {R: sum(1 for r in records if r.get("n_rounds", 1) == R) for R in rounds_present}
        primary_R = max(per_R_n, key=per_R_n.get)
        print(f"Detected R values: {rounds_present}; running analysis per R.")
        for R in rounds_present:
            R_records = [r for r in records if r.get("n_rounds", 1) == R]
            R_dir = out_dir / f"R{R}"
            R_dir.mkdir(parents=True, exist_ok=True)
            _analyze_single_R(R_records, R_dir, rounds_label=f"R={R}")
        # Add a cross-R R-effect comparison.
        _print_R_effect(records, out_dir)
        # Continue using primary_R for legacy top-level output.
        records = [r for r in records if r.get("n_rounds", 1) == primary_R]

    stats = [per_condition(records, c) for c in CONDITIONS]
    stats = [s for s in stats if s["n_tasks"] > 0]

    cond_to_stats = {s["condition"]: s for s in stats}
    deltas = {}
    pairs = [
        ("sas", "main_study_decent"),
        ("sas", "text"),
        ("sas", "latent_we"),
        ("sas", "latent_diffusion"),
        ("sas", "latent_diffusion_pure"),
        ("main_study_decent", "text"),
        ("main_study_decent", "latent_we"),
        ("main_study_decent", "latent_diffusion"),
        ("main_study_decent", "latent_diffusion_pure"),
        ("text", "latent_we"),
        ("text", "latent_diffusion"),
        ("text", "latent_diffusion_pure"),
        ("latent_we", "latent_diffusion"),
        ("latent_we", "latent_diffusion_pure"),
        ("latent_diffusion", "latent_diffusion_pure"),
    ]
    rng = np.random.default_rng(0)
    for a, b in pairs:
        if a not in cond_to_stats or b not in cond_to_stats:
            continue
        sa, sb = cond_to_stats[a], cond_to_stats[b]
        f1_d = paired_bootstrap_delta(sa["_f1_by_task"], sb["_f1_by_task"], rng=rng)
        en_d = paired_bootstrap_delta(sa["_energy_by_task"], sb["_energy_by_task"], rng=rng)
        deltas[(a, b)] = {"f1": f1_d, "energy": en_d} if f1_d and en_d else None

    # Pass SAS k-budget through to auto_conclusion so it can flag unfair comparisons.
    # Use sentinel keys (string keys, not tuples) so they don't collide with delta pairs.
    deltas["_sas_k_budget"] = sas_k_budget
    deltas["_mas_k_budget"] = 5  # pilot Stage 8 uses --max-react-steps 5

    summary_lines = []
    print_summary(stats, deltas, summary_lines)

    # Plots
    plot_pareto(stats, out_dir / "fig_pareto.png")
    plot_box(stats, "_raw_energy", "GPU dynamic energy per task (J)",
             "Energy distribution by condition", out_dir / "fig_energy_box.png")
    plot_box(stats, "_raw_f1", "F1 per task",
             "F1 distribution by condition", out_dir / "fig_f1_box.png")

    # SAS k-sensitivity panel + figure (always emitted; provides budget context
    # so a reader can immediately see whether SAS dominance is from topology or k).
    _emit_sas_k_sensitivity(out_dir, mas_stats=[s for s in stats if s["condition"] != "sas"])

    # Machine-readable summary
    summary_obj = {
        "n_records": len(records),
        "conditions": [
            {k: v for k, v in s.items() if not k.startswith("_")}
            for s in stats
        ],
        "deltas": {
            f"{key[0]}__to__{key[1]}": d
            for key, d in deltas.items()
            if isinstance(key, tuple) and len(key) == 2 and d is not None
        },
        "sas_k_budget": sas_k_budget,
        "mas_k_budget": 5,
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary_obj, f, indent=2, default=str)
    with open(out_dir / "summary.txt", "w") as f:
        f.write("\n".join(summary_lines) + "\n")

    print(f"\nWrote: {out_dir/'summary.json'}")
    print(f"       {out_dir/'summary.txt'}")
    print(f"       {out_dir/'fig_pareto.png'}")
    print(f"       {out_dir/'fig_energy_box.png'}")
    print(f"       {out_dir/'fig_f1_box.png'}")


if __name__ == "__main__":
    main()
