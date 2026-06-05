"""v6-extended: per-benchmark HistGBM candidate picks for cross-bench cube sampling.

Strategy: rather than strict Pareto-domination filter (which returns 0 on the
saturated prose benchmarks), score candidates by their predicted position
relative to the observed frontier. Pick cells that:
  - Predicted accuracy ≥ 80% of observed max accuracy on that benchmark
  - Predicted energy is in a reasonable mid-range
  - Are not already in the observed cells

For prose benchmarks (FanOutQA/WorkBench/BrowseComp+), the predicted cells
likely confirm saturation rather than extend frontiers — but running them gives
direct empirical evidence vs current claim resting on negative model output.

For SWE-bench, picks include the 3 ceiling-extrapolation cells (M=2→M=3
variants of v5 successes).

Outputs: top 4 cells per benchmark, ready for a single array sbatch.
"""
import json, glob, os, re, itertools
import numpy as np
import pandas as pd
import warnings; warnings.filterwarnings("ignore")
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression

# Path resolution: scripts assume they live in mas-energy/analysis/ or
# mas-energy/code/. Override with PROJECT_ROOT or MAS_ENERGY_ROOT env vars.
_HERE = os.path.dirname(os.path.abspath(__file__))
MAS_ENERGY_ROOT = os.environ.get("MAS_ENERGY_ROOT", os.path.dirname(_HERE))
PROJECT_ROOT = os.environ.get("PROJECT_ROOT", os.path.dirname(MAS_ENERGY_ROOT))


CRE = re.compile(r"Qwen_Qwen3\.5-9B_([a-z]+)_k(\d+)(?:_R(\d+))?(?:_M(\d+))?\.jsonl$")
BENCH_DIRS = {
    "FanOutQA":   "a5000_fanoutqa_v4",
    "WorkBench":  "a5000_workbench_v2",
    "BrowseComp+":"a5000_browsecomp_pilot",
    "SWE-bench":  "a5000_swebench",
}
BENCH_KEY = {"FanOutQA": "fanoutqa", "WorkBench": "workbench",
             "BrowseComp+": "browsecomp_plus", "SWE-bench": "swebench"}


def load_bench(bench_dir):
    base = fos.path.join(MAS_ENERGY_ROOT, "results/{bench_dir}")
    cells = []
    for f in sorted(glob.glob(f"{base}/Qwen_Qwen3.5-9B_*.jsonl")):
        m = CRE.match(os.path.basename(f))
        if not m: continue
        topo, k, r_str, m_str = m.group(1), int(m.group(2)), m.group(3), m.group(4)
        if topo not in ("centralized", "decentralized"): continue
        R = int(r_str) if r_str else 2
        M = int(m_str) if m_str else 3
        n=0; acc=0; energy=0; P=0; C=0
        for line in open(f):
            try: d=json.loads(line)
            except: continue
            if d.get("error"): continue
            n+=1
            a = (float(d["loose_accuracy"]) if d.get("loose_accuracy") is not None
                 else (1.0 if d.get("correct") else 0.0))
            acc+=a; energy+=d.get("gpu_dynamic_energy_joules",0) or 0
            P+=d.get("total_prompt_tokens",0) or 0; C+=d.get("total_completion_tokens",0) or 0
        if n < 30: continue
        cells.append({"topo": topo, "k": k, "R": R, "M": M,
                      "acc": 100*acc/n, "energy_kJ": energy/n/1000,
                      "P": P/n, "C": C/n, "n": n})
    return pd.DataFrame(cells)


def per_bench_picks(df, n_picks=4):
    """Return top n_picks unique cells predicted to be near Pareto frontier."""
    X = np.column_stack([np.log(df["k"]), np.log(df["R"]), np.log(df["M"]),
                         (df["topo"]=="centralized").astype(int)])
    acc_m = HistGradientBoostingRegressor(max_iter=300, max_depth=5, learning_rate=0.05, random_state=0).fit(X, df["acc"])
    p_m = HistGradientBoostingRegressor(max_iter=300, max_depth=5, learning_rate=0.05, random_state=0).fit(X, df["P"])
    c_m = HistGradientBoostingRegressor(max_iter=300, max_depth=5, learning_rate=0.05, random_state=0).fit(X, df["C"])
    PC = np.column_stack([df["P"], df["C"]])
    e_reg = LinearRegression().fit(PC, df["energy_kJ"])

    max_acc = df["acc"].max()
    min_e = df["energy_kJ"].min()
    max_e = df["energy_kJ"].max()
    acc_threshold = 0.80 * max_acc  # 80% of observed max — finds frontier-relevant cells

    obs = set((r["topo"], r["k"], r["R"], r["M"]) for _, r in df.iterrows())

    # Build observed MAS-only Pareto frontier
    mas_df = df[df["topo"].isin(["centralized", "decentralized"])].sort_values("energy_kJ")
    pareto = []; max_a = -1
    for _, r in mas_df.iterrows():
        if r["acc"] > max_a:
            pareto.append((r["energy_kJ"], r["acc"])); max_a = r["acc"]

    # Constrain candidate space to plausibly-meaningful configs:
    #  - k >= 7: avoid low-k cliff (consistently underperforms on all benchmarks)
    #  - R >= 2: skip R=1 (essentially Independent + extra step; usually worse than R=2-5)
    #  - M in [2, 5]: M=5 is the cap before context-overflow issues on prose
    KS = [7, 10, 13, 15, 20, 22, 25, 30]
    RS = [2, 3, 4, 5]
    MS = [2, 3, 4, 5]
    cands = []
    for topo, k, R, M in itertools.product(["centralized","decentralized"], KS, RS, MS):
        if (topo, k, R, M) in obs: continue
        x = np.array([[np.log(k), np.log(R), np.log(M), 1 if topo=="centralized" else 0]])
        pred_p = max(0, p_m.predict(x)[0]); pred_c = max(0, c_m.predict(x)[0])
        pred_e = max(min_e, float(e_reg.predict(np.array([[pred_p, pred_c]]))[0]))
        pred_acc = acc_m.predict(x)[0]
        # Filter: must clear acc_threshold and be within observed energy range
        if pred_acc < acc_threshold: continue
        if pred_e > max_e: continue  # cell must be in observed energy range (no extrapolation past max)
        # Score: distance to Pareto frontier (positive if dominates, negative if dominated)
        # Use min-distance-to-Pareto as score; cells near or above frontier score high
        domination_score = 0
        for (oe, oa) in pareto:
            if pred_e <= oe and pred_acc >= oa:
                domination_score += 1
        # Distance above frontier accuracy at this energy budget (interpolated)
        # Score by (pred_acc - frontier_acc_at_pred_e) + small (1 / pred_e) tiebreak
        cands.append({"topo": topo, "k": k, "R": R, "M": M,
                      "pred_acc": pred_acc, "pred_e": pred_e,
                      "dom": domination_score})

    if not cands:
        return pd.DataFrame()
    cand_df = pd.DataFrame(cands)
    # Sort: prefer high acc, low energy
    cand_df["score"] = cand_df["pred_acc"] - 0.02 * cand_df["pred_e"]
    cand_df = cand_df.sort_values("score", ascending=False)
    cand_df = cand_df.drop_duplicates(subset=["pred_acc", "pred_e"])
    return cand_df.head(n_picks)


def main():
    print("=" * 100)
    print("v6-extended: per-benchmark HistGBM candidates (top 4 per bench by acc-energy score)")
    print("=" * 100)

    all_picks = []
    for bench, d in BENCH_DIRS.items():
        df = load_bench(d)
        if df.empty: continue
        max_acc = df["acc"].max()
        threshold = 0.80 * max_acc
        picks = per_bench_picks(df, n_picks=4)
        print(f"\n=== {bench} ({len(df)} observed cells, max_acc={max_acc:.0f}%, threshold={threshold:.0f}%) ===")
        if picks.empty:
            print("  No candidates after filter — frontier already saturated")
            continue
        print(f"  {'cell':<32} {'pred acc':>9} {'pred E':>9} {'dom':>4}")
        for _, r in picks.iterrows():
            cell = f"{r['topo']} k={int(r['k'])} R={int(r['R'])} M={int(r['M'])}"
            print(f"  {cell:<32} {r['pred_acc']:>8.1f}% {r['pred_e']:>7.0f}kJ {int(r['dom']):>3d}")
            all_picks.append({"bench": bench, "bench_key": BENCH_KEY[bench],
                              "topo": r["topo"], "k": int(r["k"]),
                              "R": int(r["R"]), "M": int(r["M"]),
                              "pred_acc": r["pred_acc"], "pred_e": r["pred_e"]})

    print("\n" + "=" * 100)
    print(f"Total cells to validate: {len(all_picks)} (4 per benchmark × 4 benchmarks)")
    print("=" * 100)
    print("\nFor sbatch case statement (copy-paste ready):")
    for i, p in enumerate(all_picks):
        topo, k, R, M = p["topo"], p["k"], p["R"], p["M"]
        bench_key = p["bench_key"]
        out_dir = "a5000_swebench" if bench_key == "swebench" else \
                  "a5000_fanoutqa_v4" if bench_key == "fanoutqa" else \
                  "a5000_workbench_v2" if bench_key == "workbench" else \
                  "a5000_browsecomp_pilot"
        print(f"    {i:>2d}) BENCH={bench_key:<16s} TOPO={topo:<14s} K={k:>3d}; R={R}; M={M};  OUTPUT_DIR=/atlas2/u/$USER/mas_project/mas-energy/results/{out_dir} ;;")


if __name__ == "__main__":
    main()
