# final_data/ — single home for all analysis-ready data

**This directory is the one place all figures and tables read from.** Raw per-task
JSONLs stay in `mas-energy/results/` (the source of truth); `final_data/` is the
consolidated CSV layer built from them. Rebuild everything with:

```bash
python3 analysis/build_final_data.py
```

## Layout

| path | granularity | figures/tables | built by |
|------|-------------|----------------|----------|
| `raw/qwen9b/{bench}/{k,n,r}_sweep/*.jsonl` | **raw per-task run records (9B)** | source of everything below | the experiment runs |
| `qwen9b_percell/{bench}_{k,m,r}_sweep.csv` | per-cell (9B, 4 benchmarks) | **Figs 1-7** (k/N/R scaling + gamma) | `extract_final_metrics.py` (reads `raw/qwen9b/`) |
| `qwen9b_percell/SWE_bench_per_record.csv` | per-task (9B SWE) | supporting | `extract_swebench_harness_records.py` |
| `cross_model/per_record_master.csv` | **per-task, all 3 models** | **Figs 8-10** (cross-model) + frontier | `build_per_record_master.py` |
| `cross_model/{k,m,r}sweep_3model.csv`, `cells_*.csv` | per-cell (3 models) | cross-model per-cell | `extract_3model_metrics.py` |
| `kxn_grid/kxn_9b.csv` | per-cell (9B joint k×N) | coverage / audit | `extract_kxn_grid.py` |
| `kxn_grid/kxn_9b_per_record.csv` | **per-task (9B joint k×N)** | **probe / §6 config-policy** | `extract_kxn_per_record.py` |
| `MANIFEST.json` | — | machine-readable inventory | `build_final_data.py` |

**Raw runs live here too (2026-08-23).** The 9B run JSONLs moved from
`mas-energy/results/final_qwen9b_canonical/{bench}/` into `final_data/raw/qwen9b/{bench}/`, so
the whole chain (raw runs -> per-axis consolidated metrics -> figures) sits in one directory.
`extract_final_metrics.py` reads the new location and the 12 per-cell CSVs rebuild byte-identically
from it. The old paths remain as **symlinks** into `raw/qwen9b/`, so every other script that still
references `final_qwen9b_canonical/{bench}/` keeps resolving; do not delete them. One code change
was required: `swe_realcorrect.cell_name_from_path` dir-qualifies harness keys by the path, so it
now maps `final_data/raw/qwen9b/...` back onto the recorded `final_qwen9b_canonical/...` key.
Without that mapping every SWE harness lookup misses and `real_correct` silently empties.
Note `_backup_pre_n100_*` were left in place; they are backups, not live data.

**Primary file for most uses:** `cross_model/per_record_master.csv` — one row per
task, all 3 models × benchmarks, with per-phase energy
(`inference/coordination/tool_exec/gpu_total/total_system_kJ`), harness accuracy
(`real_correct`, `harness_category`), `loose_accuracy`, tokens, call counts, wall,
`cell_rank` (filter `cell_rank < 50` for the paper's per-cell n).

## Figure → file mapping (authoritative)

- **Fig 1/2** (k accuracy/energy) → `qwen9b_percell/{bench}_k_sweep.csv`
- **Fig 3/4** (N accuracy/energy) → `qwen9b_percell/{bench}_n_sweep.csv`
- **Fig 5/6** (R accuracy/energy) → `qwen9b_percell/{bench}_r_sweep.csv`
- **Fig 7** (gamma) → same `qwen9b_percell/*` per-cell CSVs
- **Fig 8/9** (cross-model grid) → `cross_model/per_record_master.csv`
- **Fig 10** (cross-model gamma), SAS-vs-MAS frontier → `cross_model/per_record_master.csv`
- **§6 probe / config-policy, k×N** → `kxn_grid/kxn_9b_per_record.csv` (per-task; the §6 protocol splits each cell 25 probe / 25 eval over 200 draws, so per-cell aggregates cannot serve it). `kxn_9b.csv` is the per-cell view of the same rows and is used for coverage.
- Figs 11-13 (energy regression, batch coefficients) draw on the batch-probe data
  (`mas-energy/results/a5000_batch_coeff_sweep_bf16`) + regression outputs; those
  are not part of the scaling-grid CSVs and are listed here for completeness only.

## Design decisions (log)

**2026-07-28 — consolidation.** All final CSVs were previously in two scattered
homes: `analysis/csv_3model/` (3-model) and
`mas-energy/results/final_qwen9b_canonical/csv/` (9B per-cell), and the k×N grid had
**no** consolidated CSV at all (read raw from `*_canonical_bf16` by
`config_policy_sweep.py`). ~58 scripts referenced the two old paths.

Consolidation chosen (low-risk, zero break to the 58 consumers):
1. The two CSV homes were **moved** into `final_data/` (`cross_model/`,
   `qwen9b_percell/`) and the old paths left as **symlinks** back
   (`analysis/csv_3model -> ../final_data/cross_model`,
   `.../final_qwen9b_canonical/csv -> ../../../final_data/qwen9b_percell`). Every
   existing script therefore reads/writes `final_data/` transparently.
2. Core paper-figure producers/consumers were **repointed explicitly** to
   `final_data/` paths (not via symlink): `build_per_record_master.py`,
   `extract_3model_metrics.py`, `extract_final_metrics.py`, `plot_kRM_from_csv.py`,
   `plot_cross_model_grid.py`, `cross_model_sas_vs_mas_frontier.py`.
3. The **k×N gap** was closed: `extract_kxn_grid.py` emits `kxn_grid/kxn_9b.csv`
   from the `*_canonical_bf16` dirs (the joint (k,N) cells don't fit the
   k/m/r_sweep buckets of `consolidate_final_qwen9b_data.py`, so they reached no
   CSV before). New k×N fills from the probe agent land in those same
   `*_canonical_bf16` dirs and appear here on the next `build_final_data.py`.
4. `build_final_data.py` orchestrates the whole rebuild and writes `MANIFEST.json`.

**Why symlinks + explicit repoint (not repoint-all-58):** repointing 58 scripts
by hand is high-risk and many are exploratory/superseded. Symlinks give a single
physical home immediately; the core figures additionally have an explicit
`final_data/` dependency so the paper pipeline never relies on the symlink.

**Current cell completion (n_clean per cell): see `COVERAGE.md`** (regenerate with `python3 analysis/coverage_grid.py`).

**Which cell is which serving era, and where each lives: see `ERA_MAP.md`** (empirical
date + context-fingerprint audit; the data spans Apr/May/Jun/Jul eras, not a binary).

**Era caveat carried in the data, not hidden by consolidation:** `per_record_master`
still prefers the era-consistent sources and overrides (highk_rerun, strongmodel
SAS k100/200) via its `SOURCES` list; `kxn_9b.csv` and `kxn_9b_per_record.csv` are the `_canonical_bf16` era
(the probe's intended source). Consolidation does not re-mix eras.

**2026-07-28 — §6 became a dumb reader.** `config_policy_sweep.py` previously
re-read raw JSONLs and re-implemented dedup, the 50-task cap, SWE `real_correct`
resolution and the utilization derivation; each drifted from the consolidated
CSVs at some point (worst: 37 of 102 SWE cells, after an experiment repointed it
at `final_qwen9b_canonical`, which put families A/C on one era and B/D on
another). It now loads `kxn_9b_per_record.csv` only, and reproduces `kxn_9b.csv`
on all 262 comparable cells. Both k×N CSVs are emitted by `build_final_data.py`
from the same source and dedup rules, so they cannot disagree.
