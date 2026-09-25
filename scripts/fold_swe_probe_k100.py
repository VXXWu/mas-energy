#!/usr/bin/env python3
"""Fold the SWE-bench K=100 breadth-ladder fill runs into final_data/kxn_grid.

Adds the (topology, k=100, N) cells missing from kxn_9b_per_record.csv / kxn_9b.csv:
  independent   N in {5, 10}          from a5000_swebench_probeBkstar
  decentralized N in {4, 5, 10}       from a5000_swebench_probeBkstar
  centralized   N in {2, 4, 5, 10}    from a5000_swebench_probe_centstd
                                       (N=4 completed 2026-09-20/21 after two OOM
                                        rounds; see project_log 2026-09-20/21)

real_correct is resolved from the cell-keyed harness report families
(logs/run_evaluation + mas-energy/logs/run_evaluation, _M-named dirs), which were
audited and de-staled on 2026-09-21 (stale no-patch verdicts for the re-run
tasks removed, fresh n4interim_0920 verdicts installed). Unharnessed -> 0.

Idempotent + self-repairing: cells already present in the CSVs are skipped for
append, but per-cell rows with empty mean_total_tokens/source_file are repaired
in place (left blank by the 2026-09-21 first run of this script).
Run AFTER extract_kxn_per_record.py / extract_kxn_grid.py, which do not scan the
SWE probe dirs (their real_correct lives in report families, not the dedupe
mapping; durable integration into swe_realcorrect's index is TODO).
Authorized fold: user, 2026-09-21 (3-topology D-panel adoption).
"""
import csv, glob, json, os, statistics as st, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
from extract_kxn_per_record import utilization

RESULTS = ROOT / "mas-energy/results"
LOGS = [ROOT / "logs/run_evaluation", ROOT / "mas-energy/logs/run_evaluation"]
PER_REC = ROOT / "final_data/kxn_grid/kxn_9b_per_record.csv"
PER_CELL = ROOT / "final_data/kxn_grid/kxn_9b.csv"

NEED = {  # topo -> (results dir, N-file pattern, Ns, R, report-family pattern (_M-named))
    "independent":   ("a5000_swebench_probeBkstar",  "independent_k100_N{n}",      [5, 10],       "", "independent_k100_M{n}"),
    "decentralized": ("a5000_swebench_probeBkstar",  "decentralized_k100_R2_N{n}", [4, 5, 10],    "2", "decentralized_k100_R2_M{n}"),
    "centralized":   ("a5000_swebench_probe_centstd","centralized_k100_R2_N{n}",   [2, 4, 5, 10], "2", "centralized_k100_R2_M{n}"),
}

def family_resolved(fam):
    res = {}
    for lb in LOGS:
        for rep in glob.glob(f"{lb}/{fam}/*/*/report.json") + glob.glob(f"{lb}/{fam}/*/report.json"):
            try: d = json.load(open(rep))
            except Exception: continue
            for k, v in d.items():
                if isinstance(v, dict) and "resolved" in v:
                    res[k] = res.get(k, False) or bool(v["resolved"])
    return res

def read_cell(topo, d, pat, n, R, fampat, swetasks):
    """-> (per-record dicts, per-cell dict) or (None, None) if the file is absent."""
    f = RESULTS / d / f"Qwen_Qwen3.5-9B_{pat.format(n=n)}.jsonl"
    if not f.exists():
        print(f"MISSING FILE: {f}"); return None, None
    res = family_resolved(f"Qwen_Qwen3.5-9B_{fampat.format(n=n)}")
    seen, cell, toks = set(), [], []
    for line in open(f):
        try: r = json.loads(line)
        except Exception: continue
        if r.get("error"): continue
        tid = r.get("task_id")
        if tid in seen or tid not in swetasks: continue
        seen.add(tid)
        toks.append(r.get("total_tokens") or 0)
        cell.append(dict(benchmark="SWE_bench", topology=topo, k=100,
            R=R, N=n, task_id=tid,
            score=1.0 if res.get(tid) else 0.0,
            gpu_dynamic_kJ=(r.get("gpu_dynamic_energy_joules") or 0) / 1000.0,
            utilization=utilization(r, 100), answered=1,
            source_file=f"{d}/{f.name}",
            decent_convention=""))
    acc = 100 * st.mean(c["score"] for c in cell) if cell else 0
    kj = st.mean(c["gpu_dynamic_kJ"] for c in cell) if cell else 0
    tok = st.mean(toks) if toks else 0
    meta = dict(benchmark="SWE_bench", topology=topo, k=100, R=R,
        N=n, n=len(cell), acc_pct=round(acc, 1),
        mean_dyn_kJ=round(kj, 3), mean_total_tokens=round(tok, 1),
        acc_field="real_correct", source_file=f"{d}/{f.name}")
    return cell, meta

def main():
    rows = list(csv.DictReader(open(PER_REC)))
    fields = list(rows[0].keys())
    have = {(r["benchmark"], r["topology"], int(r["k"]), int(r["N"])) for r in rows}
    swetasks = {r["task_id"] for r in rows if r["benchmark"] == "SWE_bench"}

    new_rows, metas = [], {}
    for topo, (d, pat, Ns, R, fampat) in NEED.items():
        for n in Ns:
            cell, meta = read_cell(topo, d, pat, n, R, fampat, swetasks)
            if cell is None: continue
            metas[("SWE_bench", topo, "100", str(n))] = meta
            if ("SWE_bench", topo, 100, n) in have:
                print(f"present: {topo} N={n} (repair-only)")
            else:
                new_rows += cell
                print(f"fold: {topo} N={n} n={len(cell)} acc={meta['acc_pct']}%")

    # per-record: append new rows
    with open(PER_REC, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader(); w.writerows(rows)
        for r in new_rows: w.writerow(r)

    # per-cell: repair blank mean_total_tokens/source_file on existing rows, append new
    cell_rows = list(csv.DictReader(open(PER_CELL)))
    cf = list(cell_rows[0].keys())
    cell_fixed, present = 0, set()
    for r in cell_rows:
        key = (r["benchmark"], r["topology"], r["k"], r["N"])
        present.add(key)
        m = metas.get(key)
        if m and (not r["mean_total_tokens"] or not r["source_file"]):
            r["mean_total_tokens"] = m["mean_total_tokens"]
            r["source_file"] = m["source_file"]; cell_fixed += 1
    appended = [m for k, m in metas.items() if k not in present]
    with open(PER_CELL, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cf, extrasaction="ignore")
        w.writeheader(); w.writerows(cell_rows)
        for m in appended: w.writerow(m)
    print(f"appended {len(new_rows)} per-record rows, "
          f"{len(appended)} per-cell rows ({cell_fixed} field repairs)")

if __name__ == "__main__":
    main()
