# Serving-era map: what data is which era, and where it lives

**"Era" = the serving configuration a cell was run under** (SGLANG_CONTEXT_LENGTH,
sglang version, GPU, date), which affects **energy** (long-context prefill
efficiency) but not accuracy. Established empirically from record timestamps +
per-call max-context fingerprints (2026-07-28 audit).

**There is no clean binary "old vs new."** The data spans four calendar eras. What
matters is that **each sweep axis is internally ~consistent, but the axes differ
from each other**, and cross-model is inherently cross-era. Details below.

## Era timeline (9B, the multi-era model)

| era | data | serving | physical source (raw) | → final_data CSV |
|-----|------|---------|-----------------------|------------------|
| **Apr 2026** (base) | 9B k-sweep k=1..50, all 4 topos; SAS/Indep deep-k k=100/200 | A5000, runner cap 49,152 (truncation engages k≥30) | `a5000_swebench` → `final_qwen9b_canonical/SWE_bench/k_sweep/` | `qwen9b_percell/SWE_bench_k_sweep.csv` + master |
| **May 2026** (N) | 9B Cent/Decent k=10 N-variants (N=2,4,5,10); decent k10_R2 baseline | A5000, 49,152 | `a5000_swebench` (n=100 runs) → `.../n_sweep/` | `qwen9b_percell/SWE_bench_n_sweep.csv` + master |
| **Jun 2026** (R + deep + rerun) | 9B R-sweep (kbudget30); Cent/Decent deep-k k=100/200; Indep k=20 N-sweep; canonical_bf16 re-runs | A5000, 49,152 | `a5000_swebench` + `a5000_swebench_canonical_bf16` → `.../r_sweep/`, `.../k_sweep/` (deep) | `qwen9b_percell/SWE_bench_r_sweep.csv` + master |
| **Jul 2026** (cross-model) | **MoE** + **Gemma**, all SWE + BrowseComp | A6000, MoE 49,152 / Gemma ~30,312 KV-cap | `a5000_swebench_{qwen35b_a3b_fp8,gemma4_31b_dense}_canonical` | master + `cross_model/*` |
| **Jul 28+** (fills, pending) | 9B decent k=50 re-run; MoE/Gemma SAS k=100/200; 9B N=1 anchors | current env | `a5000_swebench_highk_rerun`, `..._strongmodel_sas_highk`, `..._9b_n1_anchor` (**empty until jobs land**) | master (via SOURCES override) |

Prose benchmarks (FanOutQA/WorkBench/BrowseComp+) 9B: k-sweep = **Apr 2026**,
m/r-sweep = **Jun 2026**, all runner cap 49,152. Same axis-era structure as SWE.

## VERIFIED audit (2026-07-29): era + context-cap per model/benchmark/sweep

From record timestamps + per-call max-prompt-token fingerprints:

| model / benchmark / sweep | dominant era | ctx cap | spread |
|---|---|---|---|
| 9B SWE k / N / R | Apr / May / Jun | 49k / 27k / 48k | all mixed 04-07 |
| 9B FanOutQA k / N / R | Apr / Jul / Jun | ~49k | mixed |
| 9B WorkBench k / N / R | Apr / Jul / Jul | 36k / 21k / 48k | mixed |
| 9B BrowseComp k / N / R | Apr / Jul / Jun | ~49k | mixed |
| **MoE** SWE + BrowseComp | **Jul (clean)** | ~40-45k | 07 |
| **Gemma** SWE + BrowseComp | **Jul (clean)** | **~30k (KV cap)** | 07 |

Takeaways: (1) every 9B sweep is era-MIXED (month-dominant but cells spread Apr-Jul);
(2) MoE and Gemma are each a single clean July era; (3) the 3 models differ on ALL
THREE axes at once — era, GPU (9B A5000 vs MoE/Gemma A6000), and context cap (9B/MoE
~49k vs Gemma ~30k).

**Gemma ~30k cap is a MEMORY constraint, not a chosen setting:** Gemma launched with
`--context-length 65536 --mem-fraction-static 0.85` on the A6000 (48GB), but Gemma-4-31B
is a DENSE 31B — after w4a16 weights (~15.5GB) + the mf-0.85 reservation, the leftover
VRAM fits a KV-cache pool of only ~30,312 tokens. That KV pool (not `--context-length`)
is the binding cap; longer sequences are rejected/truncated (58/50 rejection events
logged). Architecture × hardware: dense-31B has large per-token KV → ~30k; MoE 35B-A3B
is sparse (~3B active KV) → ~49k on the same A6000; 9B is small → ~49k on the A5000.
The mf=0.85 was the anti-OOM ceiling for the dense 31B, NOT a tuning oversight:
`mem-fraction-static` reserves weights+KV, leaving the rest for activation/prefill
headroom, which a dense 31B (all params active, long-context prefill) needs a lot of.
MoE ran fine at 0.90 (tiny ~3B activations); Gemma capped at 0.85 and its BrowseComp
even fell to 0.80. Raising mf would OOM and only reach ~35k, not 49k — the dense-31B
footprint on 48GB is the fundamental limit.

**Fig 9 (cross-model energy) confound — verified:** the "9B > Gemma energy" and the
"Gemma energy plateaus/ticks-down at k≈15" are BOTH the Gemma ~30k KV-pool cap, not
model efficiency. Gemma's max context pins at ~30k from k≈15 across every topology, so
it truncates trajectories early and does ~half the tokens/calls at high k (k=50 SWE SAS:
9B 608k prompt-tok / Gemma 307k). At matched work below the cap (k≲10-15), Gemma ≈ or
ABOVE 9B per token (it is a bigger model; only the 4-bit weights help). So do NOT read
Fig 9 high-k energy as "Gemma more efficient" — restrict cross-model energy claims to
k below the cap, or lead with Fig 8 (accuracy) and caveat Fig 9's context/GPU asymmetry.

## Where every relevant piece lives (physical → final_data)

- **9B scaling (Figs 1-7):** raw in `mas-energy/results/final_qwen9b_canonical/{bench}/{k,m,r}_sweep/` (itself consolidated from `a5000_swebench` + `a5000_*_canonical_bf16` + older dirs by `consolidate_final_qwen9b_data.py`) → `final_data/qwen9b_percell/*_sweep.csv`.
- **Cross-model (Figs 8-10, frontier):** 9B from `final_qwen9b_canonical`; MoE from `a5000_swebench_qwen35b_a3b_fp8_canonical` + BrowseComp dir; Gemma from `a5000_swebench_gemma4_31b_dense_canonical` + BrowseComp dir → `final_data/cross_model/per_record_master.csv`.
- **k×N grid (probe/§6):** `a5000_*_canonical_bf16` (Jun/Jul) → `final_data/kxn_grid/kxn_9b.csv`.
- **Pending fills** (override the era-outlier cells once landed): `a5000_swebench_highk_rerun` (decent k50), `a5000_swebench_strongmodel_sas_highk` (MoE/Gemma SAS k100/200) — the master's `SOURCES` list prefers these over the older cells by sorted-path precedence.

## Comparability rules (what the eras allow)

1. **Within one axis of one model → comparable.** The k-axis is one era (Apr), the R-axis another (Jun), etc.; each is internally consistent, so shapes and within-axis energy ratios are sound.
2. **Across axes of the same model → energy is era-confounded.** 9B k-axis (Apr) vs N-axis (May) vs R-axis (Jun) differ in serving; compare accuracy freely, but absolute energy across axes carries a serving caveat.
3. **Cross-model (Figs 8/9) → inherently cross-era AND cross-GPU.** 9B (Apr-Jun, A5000) vs MoE/Gemma (Jul, A6000). Unavoidable — MoE/Gemma need the A6000. Accuracy is clean; energy differences fold in a serving+hardware component. The SAS-vs-MAS domination (10-22×) is far larger than era drift (~15-50%), so it survives.
4. **Deep-k boundary (SWE):** SAS/Indep k=100/200 are Apr era; **Cent/Decent k=200 is OLD-only** and reads ~2.6× its k=100 — that is era drift, not depth. Do not compare Cent/Decent k=100↔200 as a scaling point. (See `reference_swe_deep_k_eras`.)

## Decentralized R convention (the one that changes what R MEANS)

`MAS_DECENT_R_INCLUDES_INIT` flips whether a debater's initial ReAct loop counts
toward R. Verified per cell by counting distinct debate-round phases (init/r0/r1…)
vs labeled R:

| convention | R=2 = | decentralized data | era |
|-----------|-------|--------------------|-----|
| **EARLY** (R excl init) | **3 loops** (init + 2 debate) | ALL 9B default-R decent: k-sweep, N-sweep, deep-k k100/200, all 4 benchmarks | Apr-Jun |
| **CURRENT** (R incl init; paper convention) | **2 loops** (init + 1 debate) | 9B explicit R-sweep (`_R{R}_N3` kbudget30) + ALL MoE + ALL Gemma decent | Jun-Jul |

Consequences:
1. **Cross-model decent (Figs 8/9) is confounded**: 9B runs 3 loops at "R2", MoE/Gemma
   run 2 → 9B decent energy inflated ~1.5× (and likely accuracy). SAS-vs-MAS
   domination survives (9B decent k30 would be ~195 kJ not 292; MoE-SAS still ~15×).
2. **9B R-sweep (CURRENT) ≠ 9B k/N-sweep decent (EARLY)**: the R-sweep R=2 point
   (2 loops) is a different protocol than base decent k15 (3 loops); the R-axis is
   internally consistent but does not join the k-axis at low R.
Non-decentralized topologies have no init/debate split → unaffected.
To de-confound cross-model decent, re-run 9B decent under CURRENT convention (decCurR).

### CAVEAT: decCurR vs EARLY overlay is DOUBLE-confounded on prose (verified 2026-07-29)

Do NOT read a within-9B EARLY-vs-CURRENT decent overlay as "the R-convention effect."
It conflates TWO changes: (1) the convention (−1 debate loop) AND (2) an era-shift in
how many ReAct steps agents take before self-terminating. The second is large and noisy
on the easy/parallelizable prose benchmarks.

Evidence: FanOutQA decent **init-round** react-calls (convention-independent work) vary
by ERA: Apr(master)=32.0, Jun=14.3, Jul(decCurR)=18.8, Jul(kxn-grid)=26.3. decCurR (18.8)
sits INSIDE the current-era spread (Jun/kxn) — it is CORRECT, not broken; EARLY (32.0) is
the high outlier. On SWE-bench, init react is steady (~91-122, agents use full budget), so
this ambiguity is prose-only.

Consequences:
- decCurR decent energy ≈ Independent on prose is REAL current-era behavior (agents
  early-terminate, so 2-loop decent barely exceeds 1-loop independent), not a bug.
- EARLY decent is NOT a clean "3k budget": its compute is front-loaded (init≫r0≫r1) and
  totals ≈ Centralized R2 in react-calls/tokens/energy. Calling it 3k overstates it.
- SAFE use: decCurR IS era- and convention-consistent with MoE/Gemma decent (both CURRENT
  conv, current serving), so the CROSS-MODEL comparison is clean. The confound bites ONLY
  the within-9B EARLY-vs-CURRENT prose overlay — do not use it for causal inference.

## Depth-column precedence in the k x N grid (set 2026-07-30)

`kxn_9b_per_record.csv` draws from two trees. Which one wins is now decided by
COLUMN, not by list order:

| column | source | why |
|---|---|---|
| **depth column**: SAS (N=1) and MAS N=3 | `final_qwen9b_canonical/{bench}/{k,m}_sweep` wherever it has the cell, else `*_canonical_bf16` | this is the same column Figs 1-2 plot, so §6 must deploy the cells the paper's own k-sweep shows |
| **k x N fill**: N not in {1,3} | `*_canonical_bf16` only | these cells exist nowhere else and no figure uses them |

Before this rule `*_canonical_bf16` won everything it had, which put 8 prose and
31 SWE depth cells on a different serving era than Figs 1-2 inside an otherwise
era-consistent curve. At matched (topo, k, N=3, R=2) the two trees disagree by
**0.34x to 1.38x in energy** (canonical_bf16 mostly cheaper; the Apr k-sweep ran
at a lower-throughput serving config) and by up to 14 pp in accuracy on the SWE
deep-k rungs. §6's deployed rung is 41-59% of family A's total energy, so the
mix distorted the shape of every policy curve.

After the rule the depth column agrees with `qwen9b_percell/*_k_sweep.csv` to
0.00 pp / 1.00x on all four benchmarks (WorkBench shows 1.24 pp, which is the
47-vs-50 task intersection, not a source mismatch). Verify with the check in
`analysis/gen_policy_tables.py`'s companion diff, or re-run
`python3 analysis/extract_kxn_per_record.py`.

Effect on §6: A vs C energy went 0.51/0.40/0.56/0.53x -> 0.35/0.42/0.50/0.63x
(A still cheaper in all 12 cells); SWE-bench A(tau) Independent moved 43.2% ->
57.1% because its deep-k SAS rungs are now the k-sweep tree's. Conclusions are
unchanged in direction.

## Known era anomalies + status

- **9B decent SWE k=50 = early-Apr (Apr-08)** while its k-sweep neighbors are late-Apr → non-physical energy dip (271 vs k30's 292 kJ). Fix staged: `a5000_atlas24_decent_k50_rerun.sbatch` → `highk_rerun` overrides it once landed.
- **13 SWE cells silently resolve to OLD (n=100) over NEW (n=50)** in `config_policy_sweep.load_cell` (most-records rule) at Cent/Decent k=10 N-ladder. Deliberately unresolved judgement call. (See `reference_load_cell_prefers_old_era`.)
- **Gemma effective context ~30,312** (KV-pool cap, not --context-length) vs 9B/MoE ~49,600 → an architecture asymmetry in cross-model long-context cells, censored not natural. (See `reference_ctx_serving_eras`.)

Companion: `README.md` (file layout), memories `reference_ctx_serving_eras`,
`reference_swe_deep_k_eras`, `reference_load_cell_prefers_old_era`.
