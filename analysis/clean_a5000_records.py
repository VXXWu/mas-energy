"""Filter a5000 result JSONLs in place, removing records where the recorded
energy is implausible relative to what the regression `E = a + bP + cC`
predicts from token counts.

Threshold: keep records with `measured / predicted >= 0.1`.
The empirical distribution is cleanly bimodal (broken at ~0.005, real at ~1.0)
so the cut is unambiguous.

Each modified file gets a `.preclean.bak` next to it.

Run:
    python analysis/clean_a5000_records.py            # actually rewrite
    python analysis/clean_a5000_records.py --dry-run  # just print summary
"""
from __future__ import annotations
import json
import glob
import os
import sys
import shutil

# Energy regression coefficients (project-wide canonical)
A_INTERCEPT = -84.0
B_PROMPT = 0.018
C_COMP = 5.54
RATIO_FLOOR = 0.1

PATH_GLOB = "mas-energy/results/a5000_*/Qwen_Qwen3.5-9B_*.jsonl"


def classify(rec: dict) -> str:
    """Return one of: 'error', 'bad_energy', 'good'."""
    if rec.get("error"):
        return "error"
    e = rec.get("gpu_dynamic_energy_joules")
    if e is None:
        return "bad_energy"
    P = rec.get("total_prompt_tokens", 0) or 0
    C = rec.get("total_completion_tokens", 0) or 0
    pred = A_INTERCEPT + B_PROMPT * P + C_COMP * C
    if pred <= 0:
        # tiny task, fall back to a 50J floor
        return "good" if e >= 50 else "bad_energy"
    if e / pred < RATIO_FLOOR:
        return "bad_energy"
    return "good"


def main() -> None:
    dry = "--dry-run" in sys.argv
    files = sorted(glob.glob(PATH_GLOB))
    print(f"{'file':<70} {'tot':>5} {'err':>5} {'badE':>5} {'good':>5}")
    grand = dict(tot=0, err=0, bad=0, good=0, files_changed=0)
    for path in files:
        records = []
        for line in open(path):
            try:
                records.append(json.loads(line))
            except Exception:
                continue
        n_err = n_bad = n_good = 0
        keep = []
        for r in records:
            cat = classify(r)
            if cat == "error":
                n_err += 1
            elif cat == "bad_energy":
                n_bad += 1
            else:
                n_good += 1
                keep.append(r)
        grand["tot"] += len(records)
        grand["err"] += n_err
        grand["bad"] += n_bad
        grand["good"] += n_good
        short = path.replace("mas-energy/results/", "").replace("Qwen_Qwen3.5-9B_", "")
        flag = ""
        if (n_err + n_bad) > 0:
            flag = " *"
        print(f"{short:<70} {len(records):>5} {n_err:>5} {n_bad:>5} {n_good:>5}{flag}")
        if (n_err + n_bad) > 0 and not dry:
            bak = path + ".preclean.bak"
            if not os.path.exists(bak):
                shutil.copy2(path, bak)
            with open(path, "w") as fh:
                for r in keep:
                    fh.write(json.dumps(r) + "\n")
            grand["files_changed"] += 1

    print()
    print(f"TOTAL: tot={grand['tot']}  err={grand['err']}  bad_energy={grand['bad']}  good={grand['good']}")
    if dry:
        print("(dry-run — no files were modified)")
    else:
        print(f"Files rewritten: {grand['files_changed']}  (backups at *.preclean.bak)")


if __name__ == "__main__":
    main()
