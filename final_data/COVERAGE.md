# Cell completion grid (living)

Data as of `per_record_master.csv` mtime: **2026-08-26 10:46**. Regenerate after each pull: `python3 analysis/coverage_grid.py`.

Legend: plain = 50 (full) · `N!` = 40-49 (plots) · `N**` = <40 (**gated, will NOT plot**).
9B decent rows are EARLY convention until the decent_currentR fill lands.

## QWEN9B / SWE_bench
```
  sas   k: k1:50  k2:50  k3:50  k5:50  k7:50  k10:50  k15:50  k20:50  k30:50  k50:50  k100:50  k200:50
  indep k: k1:50  k2:50  k3:50  k5:50  k7:50  k10:50  k15:50  k20:50  k30:50  k50:50  k100:50  k200:50
  indep N: N1:50  N2:50  N3:50  N4:50  N5:50  N10:50
  centr k: k1:50  k2:50  k3:50  k5:50  k7:50  k10:50  k15:50  k20:50  k30:50  k50:48!  k100:50  k200:50
  centr N: N1:50  N2:50  N3:50  N4:50  N5:50  N10:50
  centr R: R1:50  R2:50  R3:50  R5:50  R6:50  R10:50  R15:50  R30:50
  decen k: k1:50  k2:50  k3:50  k5:50  k7:50  k10:50  k15:50  k20:50  k30:50  k50:50  k100:50  k200:50
  decen N: N1:50  N2:50  N3:50  N4:50  N5:50  N10:50
  decen R: R1:50  R2:50  R3:50  R5:50  R6:50  R10:50  R15:50  R30:50
```

## QWEN9B / FanOutQA
```
  sas   k: k1:50  k2:50  k3:50  k5:50  k7:50  k10:50  k15:50  k20:50  k30:50  k50:50
  indep k: k1:50  k2:50  k3:50  k5:50  k7:50  k10:50  k15:50  k20:50  k30:50  k50:50
  indep N: N2:50  N3:50  N4:50  N5:50  N10:50
  centr k: k1:50  k2:50  k3:50  k5:50  k7:50  k10:50  k15:50  k20:50  k30:50  k50:50
  centr N: N1:50  N2:50  N3:50  N4:50  N5:50  N10:50
  centr R: R1:50  R2:50  R3:50  R5:50  R6:50  R10:50  R15:50  R30:50
  decen k: k1:50  k2:50  k3:50  k5:50  k7:50  k10:50  k15:50  k20:50  k30:50  k50:50
  decen N: N1:50  N2:50  N3:50  N4:50  N5:50  N10:50
  decen R: R1:50  R2:50  R3:50  R5:50  R6:50  R10:50  R15:50  R30:50
```

## QWEN9B / WorkBench
```
  sas   k: k1:50  k2:50  k3:50  k5:50  k7:50  k10:50  k15:50  k20:50  k30:50  k50:50
  indep k: k1:50  k2:50  k3:50  k5:50  k7:50  k10:50  k15:50  k20:50  k30:50  k50:50
  indep N: N2:50  N3:50  N4:50  N5:50
  centr k: k1:50  k2:50  k3:50  k5:50  k7:50  k10:50  k15:50  k20:50  k30:50  k50:50
  centr N: N1:50  N2:50  N3:50  N4:50  N5:50
  centr R: R1:50  R2:50  R3:50  R5:50  R6:50  R10:50  R15:50  R30:50
  decen k: k1:50  k2:50  k3:50  k5:50  k7:50  k10:50  k15:50  k20:50  k30:50  k50:50
  decen N: N1:50  N2:50  N3:50  N4:50  N5:50
  decen R: R1:50  R2:50  R3:50  R5:50  R6:50  R10:50  R15:50  R30:50
```

## QWEN9B / BrowseCompplus
```
  sas   k: k1:50  k2:50  k3:50  k5:50  k7:50  k10:50  k15:50  k20:50  k30:50  k50:50
  indep k: k1:50  k2:50  k3:50  k5:50  k7:50  k10:50  k15:50  k20:50  k30:50  k50:50
  indep N: N2:50  N3:50  N4:50  N5:50  N10:50
  centr k: k1:50  k2:50  k3:50  k5:50  k7:50  k10:50  k15:50  k20:50  k30:50  k50:50
  centr N: N1:50  N2:50  N3:50  N4:50  N5:50  N10:50
  centr R: R1:50  R2:50  R3:50  R5:50  R6:50  R10:50  R15:50  R30:50
  decen k: k1:50  k2:50  k3:50  k5:50  k7:50  k10:50  k15:50  k20:50  k30:50  k50:50
  decen N: N1:50  N2:50  N3:50  N4:50  N5:50  N10:50
  decen R: R1:50  R2:50  R3:50  R5:50  R6:50  R10:50  R15:50  R30:50
```

## MOE / SWE_bench
```
  sas   k: k1:50  k2:50  k3:50  k5:50  k7:50  k10:50  k15:50  k20:50  k30:50  k50:50
  indep k: k1:50  k2:50  k3:50  k5:50  k7:50  k10:50  k15:50  k20:50  k30:50  k50:50
  indep N: N1:50  N2:50  N3:50  N4:50  N5:50  N10:50
  centr k: k1:50  k2:50  k3:50  k5:50  k7:50  k10:50  k15:50  k20:50  k30:50  k50:50
  centr N: N1:50  N2:50  N3:50  N4:50  N5:50  N10:50
  centr R: R1:50  R2:50  R3:50  R5:50  R6:50  R10:50  R15:50  R30:50
  decen k: k1:50  k2:50  k3:50  k5:50  k7:50  k10:50  k15:50  k20:50  k30:50  k50:50
  decen N: N1:50  N2:50  N3:50  N4:50  N5:50  N10:50
  decen R: R1:50  R2:50  R3:50  R5:50  R6:50  R10:50  R15:50  R30:50
```

## MOE / BrowseCompplus
```
  centr k: k15:50
  centr R: R1:50  R2:50  R3:50  R5:50  R6:50  R10:50  R15:50  R30:50
  decen k: k15:50
  decen R: R1:50  R2:50  R3:50  R5:50  R6:50  R10:50  R15:50  R30:50
```

## GEMMA / SWE_bench
```
  sas   k: k1:50  k2:50  k3:50  k5:50  k7:50  k10:50  k15:50  k20:50  k30:50  k50:50
  indep k: k1:50  k2:50  k3:50  k5:50  k7:50  k10:50  k15:50  k20:50  k30:50  k50:50
  indep N: N1:50  N2:50  N3:50  N4:50  N5:50  N10:50
  centr k: k1:50  k2:50  k3:50  k5:50  k7:50  k10:50  k15:50  k20:50  k30:50  k50:50
  centr N: N1:50  N2:50  N3:50  N4:50  N5:50  N10:50
  centr R: R1:50  R2:50  R3:50  R5:50  R6:50  R10:50  R15:50  R30:49!
  decen k: k1:50  k2:50  k3:50  k5:50  k7:50  k10:50  k15:18**  k20:50  k30:50  k50:50
  decen N: N1:50  N2:50  N3:50  N4:50  N5:50  N10:50
  decen R: R1:50  R2:18**  R3:50  R5:50  R6:50  R10:50  R15:50  R30:50
```

## GEMMA / BrowseCompplus
```
  centr k: k15:50
  centr R: R1:50  R2:50  R3:50  R5:50  R6:50  R10:50  R15:50  R30:50
  decen k: k15:49!
  decen R: R1:50  R2:49!  R3:50  R5:50  R6:50  R10:50  R15:50  R30:33**
```

## Gated cells (n_clean < 40, will not plot)
- gemma BrowseCompplus decentralized k1 R30 N3: **33**/50
- gemma SWE_bench decentralized k15 R2 N3: **18**/50

## k×N grid (`kxn_grid/kxn_9b.csv`): 512 cells, 76 SWE joint-(k,N) cells at N∈{2,4,5,10}
