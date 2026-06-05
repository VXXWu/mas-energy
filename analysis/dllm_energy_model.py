"""
Theoretical energy model: dLLM vs AR for MAS topologies.

Uses empirical per-call token counts from 80K+ inference calls
to compute what would happen if decode cost structure changed
from sequential (AR) to parallel (dLLM).
"""
import json

# Empirical per-call token profiles (MEDIAN values from CSV)
# Format: (prompt_tokens, completion_tokens, n_calls)
CALL_PROFILES = {
    "react_step":       {"P": 5168, "C": 75,  "C_P": 0.015},
    "decompose":        {"P": 305,  "C": 117, "C_P": 0.384},
    "review":           {"P": 1037, "C": 845, "C_P": 0.815},
    "synthesis":        {"P": 1016, "C": 247, "C_P": 0.243},
    "debate_synthesis":  {"P": 1690, "C": 354, "C_P": 0.209},
    "wrapup":           {"P": 3852, "C": 121, "C_P": 0.031},
}

# AR energy model coefficients (R² = 0.976 on 80K calls)
B_PROMPT = 0.018   # J per prompt token
B_COMPL = 5.54     # J per completion token
INTERCEPT = -84    # J per call (fixed overhead)

# Topology call structures (from Kim et al. framework, M=3, R=2)
# Each entry: list of (call_type, count)
# Using empirical median token profiles, not theoretical
TOPOLOGIES = {
    "SAS": [
        # 1 agent, multiple react steps. Median ~5 steps at k=5.
        ("react_step", 5),
        ("wrapup", 1),
    ],
    "Independent": [
        # 3 parallel agents (each ~5 react steps) + 1 synthesis
        ("react_step", 15),  # 3 agents × 5 steps
        ("synthesis", 1),
        ("wrapup", 1),
    ],
    "Centralized": [
        # 1 decompose + 2 rounds × (3 workers × ~3 react steps + 1 review) + 1 synthesis
        ("decompose", 1),
        ("react_step", 18),  # 2 rounds × 3 workers × 3 steps
        ("review", 2),       # 2 rounds
        ("synthesis", 1),
        ("wrapup", 1),
    ],
    "Decentralized": [
        # 3 initial agents (each ~5 react steps) + 2 debate rounds × 3 agents
        ("react_step", 15),  # 3 agents × 5 steps
        ("debate_synthesis", 6),  # 2 rounds × 3 agents
        ("wrapup", 1),
    ],
}


def ar_energy(P, C):
    """AR energy per call (empirical model)."""
    return max(0, INTERCEPT + B_PROMPT * P + B_COMPL * C)


def ar_energy_decomposed(P, C):
    """Return (prompt_energy, completion_energy) for AR."""
    return B_PROMPT * P, B_COMPL * C


def dllm_energy_physics(P, C, S=20, alpha=1.0):
    """
    Physics-based dLLM energy model.

    Each of S denoising steps runs a full forward pass over (P+C) positions.
    No KV-cache reuse. Each step processes all positions in parallel (like prefill).

    Cost per step ≈ alpha × B_PROMPT × (P + C)
    (same throughput as prefill, since all positions processed in parallel)

    Total cost = S × alpha × B_PROMPT × (P + C)
    """
    return S * alpha * B_PROMPT * (P + C)


def dllm_energy_uniform(P, C, r=0.5):
    """
    Uniform decode reduction model.
    dLLM decode costs r× the AR decode cost. Prefill unchanged.
    """
    return max(0, INTERCEPT + B_PROMPT * P + r * B_COMPL * C)


def crossover_completion(P, S=20):
    """
    Completion length where dLLM breaks even with AR.
    Below this: dLLM is more expensive. Above this: dLLM is cheaper.
    """
    numerator = B_PROMPT * P * (S - 1)
    denominator = B_COMPL - B_PROMPT * S
    if denominator <= 0:
        return float('inf')  # dLLM never cheaper
    return numerator / denominator


def compute_topology_energy(topology_name, energy_fn):
    """Compute total energy for a topology using given energy function."""
    total = 0
    breakdown = {}
    for call_type, count in TOPOLOGIES[topology_name]:
        prof = CALL_PROFILES[call_type]
        e = energy_fn(prof["P"], prof["C"]) * count
        total += e
        breakdown[call_type] = {"energy": e, "count": count, "per_call": e / count}
    return total, breakdown


def main():
    print("=" * 80)
    print("dLLM vs AR Energy Model for MAS Topologies")
    print("Using empirical token counts from 80K+ inference calls")
    print("=" * 80)

    # --- Section 1: AR Baseline ---
    print("\n### 1. AR BASELINE (empirical model: E = -84 + 0.018P + 5.54C)")
    print()

    ar_results = {}
    for topo in TOPOLOGIES:
        total, breakdown = compute_topology_energy(topo, ar_energy)
        ar_results[topo] = {"total": total, "breakdown": breakdown}

        # Compute completion fraction
        total_prompt_e = 0
        total_compl_e = 0
        for call_type, count in TOPOLOGIES[topo]:
            prof = CALL_PROFILES[call_type]
            pe, ce = ar_energy_decomposed(prof["P"], prof["C"])
            total_prompt_e += pe * count
            total_compl_e += ce * count

        compl_frac = total_compl_e / (total_prompt_e + total_compl_e) if (total_prompt_e + total_compl_e) > 0 else 0

        print(f"  {topo:15s}: {total:8.0f} J | {total/ar_results['SAS']['total']:5.1f}x SAS | "
              f"completion fraction: {compl_frac:.1%}")
        for call_type, count in TOPOLOGIES[topo]:
            bd = breakdown[call_type]
            print(f"    {call_type:20s}: {count:2d} calls × {bd['per_call']:7.0f} J = {bd['energy']:8.0f} J")

    sas_ar = ar_results["SAS"]["total"]

    # --- Section 2: Per-call crossover analysis ---
    print("\n### 2. CROSSOVER ANALYSIS: When is dLLM cheaper than AR?")
    print()

    for S in [10, 20, 50]:
        print(f"  S = {S} denoising steps:")
        for call_type, prof in CALL_PROFILES.items():
            P, C = prof["P"], prof["C"]
            e_ar = ar_energy(P, C)
            e_dllm = dllm_energy_physics(P, C, S=S)
            ratio = e_dllm / e_ar if e_ar > 0 else float('inf')
            c_cross = crossover_completion(P, S=S)
            status = "CHEAPER" if ratio < 1.0 else "MORE EXPENSIVE"
            print(f"    {call_type:20s}: P={P:5d}, C={C:4d} | "
                  f"AR={e_ar:7.0f}J, dLLM={e_dllm:7.0f}J | "
                  f"ratio={ratio:.2f}x | crossover_C={c_cross:.0f} | {status}")
        print()

    # --- Section 3: Physics-based dLLM (S=20) ---
    print("\n### 3. dLLM PHYSICS MODEL (S=20 denoising steps)")
    print()

    dllm_results_physics = {}
    for topo in TOPOLOGIES:
        total, breakdown = compute_topology_energy(
            topo, lambda P, C: dllm_energy_physics(P, C, S=20))
        dllm_results_physics[topo] = {"total": total, "breakdown": breakdown}
        ar_total = ar_results[topo]["total"]
        print(f"  {topo:15s}: {total:8.0f} J | {total/dllm_results_physics['SAS']['total']:5.1f}x SAS | "
              f"vs AR: {total/ar_total:.2f}x | AR was {ar_total:.0f} J")

    # --- Section 4: Disproportionality analysis ---
    print("\n### 4. DISPROPORTIONALITY: Does dLLM benefit Centralized more than SAS?")
    print()

    for S in [10, 20, 50]:
        print(f"  S = {S}:")
        ratios_to_sas_ar = {}
        ratios_to_sas_dllm = {}

        for topo in TOPOLOGIES:
            ar_total = ar_results[topo]["total"]
            dllm_total, _ = compute_topology_energy(
                topo, lambda P, C, s=S: dllm_energy_physics(P, C, S=s))

            sas_ar_total = ar_results["SAS"]["total"]
            sas_dllm_total, _ = compute_topology_energy(
                "SAS", lambda P, C, s=S: dllm_energy_physics(P, C, S=s))

            ratio_ar = ar_total / sas_ar_total
            ratio_dllm = dllm_total / sas_dllm_total

            compression = ratio_dllm / ratio_ar  # <1 means disproportionate benefit

            print(f"    {topo:15s}: AR ratio={ratio_ar:5.1f}x SAS → "
                  f"dLLM ratio={ratio_dllm:5.1f}x SAS | "
                  f"compression={compression:.2f}x {'← DISPROPORTIONATE BENEFIT' if compression < 0.8 else ''}")
        print()

    # --- Section 5: Uniform decode reduction ---
    print("\n### 5. UNIFORM DECODE REDUCTION (E = -84 + 0.018P + r×5.54C)")
    print()

    for r in [0.1, 0.2, 0.5]:
        print(f"  r = {r} (decode cost = {r:.0%} of AR):")
        for topo in TOPOLOGIES:
            total, _ = compute_topology_energy(
                topo, lambda P, C, r_=r: dllm_energy_uniform(P, C, r=r_))
            ar_total = ar_results[topo]["total"]
            sas_total_r, _ = compute_topology_energy(
                "SAS", lambda P, C, r_=r: dllm_energy_uniform(P, C, r=r_))
            ratio_to_sas = total / sas_total_r if sas_total_r > 0 else float('inf')
            ar_ratio = ar_results[topo]["total"] / ar_results["SAS"]["total"]
            compression = ratio_to_sas / ar_ratio if ar_ratio > 0 else float('inf')
            print(f"    {topo:15s}: {total:8.0f} J ({total/ar_total:.2f}x AR) | "
                  f"{ratio_to_sas:5.1f}x SAS (was {ar_ratio:.1f}x) | "
                  f"compression={compression:.2f}x")
        print()

    # --- Section 6: Energy decomposition by call category ---
    print("\n### 6. ENERGY DECOMPOSITION: Coordination vs Work (AR baseline)")
    print()

    COORD_CALLS = {"decompose", "review", "synthesis", "debate_synthesis"}
    WORK_CALLS = {"react_step", "wrapup"}

    for topo in TOPOLOGIES:
        coord_e = 0
        work_e = 0
        for call_type, count in TOPOLOGIES[topo]:
            prof = CALL_PROFILES[call_type]
            e = ar_energy(prof["P"], prof["C"]) * count
            if call_type in COORD_CALLS:
                coord_e += e
            else:
                work_e += e
        total = coord_e + work_e
        print(f"  {topo:15s}: coord={coord_e:8.0f}J ({coord_e/total:.0%}) | "
              f"work={work_e:8.0f}J ({work_e/total:.0%}) | total={total:8.0f}J")

    # --- Section 7: The definitive answer ---
    print("\n" + "=" * 80)
    print("### 7. DEFINITIVE ANSWER: Does dLLM disproportionately benefit Centralized?")
    print("=" * 80)
    print()

    # Compute the "disproportionality factor" for each topology
    # = (dLLM energy savings %) for topology X / (dLLM energy savings %) for SAS
    # > 1 means MORE benefit than SAS (disproportionate benefit)
    # < 1 means LESS benefit than SAS

    for S in [10, 20, 50]:
        print(f"  S = {S} denoising steps:")
        sas_ar = ar_results["SAS"]["total"]
        sas_dllm, _ = compute_topology_energy(
            "SAS", lambda P, C, s=S: dllm_energy_physics(P, C, S=s))
        sas_change = (sas_dllm - sas_ar) / sas_ar

        for topo in TOPOLOGIES:
            ar_total = ar_results[topo]["total"]
            dllm_total, _ = compute_topology_energy(
                topo, lambda P, C, s=S: dllm_energy_physics(P, C, S=s))
            topo_change = (dllm_total - ar_total) / ar_total

            # Relative benefit vs SAS
            # If SAS gets 50% worse and Centralized gets 10% worse,
            # Centralized has disproportionate benefit
            relative_benefit = sas_change - topo_change  # positive = topo benefits more

            # Also: ratio compression
            ar_ratio = ar_total / sas_ar
            dllm_ratio = dllm_total / sas_dllm
            compression = dllm_ratio / ar_ratio

            print(f"    {topo:15s}: AR→dLLM change: {topo_change:+.0%} | "
                  f"SAS change: {sas_change:+.0%} | "
                  f"relative benefit: {relative_benefit:+.0%} | "
                  f"ratio compression: {compression:.2f}x")

        print()

    # --- Section 8: Sensitivity to S ---
    print("\n### 8. SENSITIVITY: How many steps S needed for dLLM to help each topology?")
    print()

    for topo in TOPOLOGIES:
        ar_total = ar_results[topo]["total"]
        # Find S where dLLM total = AR total (break-even)
        best_s = None
        for s_test in range(1, 200):
            dllm_total, _ = compute_topology_energy(
                topo, lambda P, C, s=s_test: dllm_energy_physics(P, C, S=s))
            if dllm_total <= ar_total:
                best_s = s_test
                break

        if best_s is not None:
            dllm_at_breakeven, _ = compute_topology_energy(
                topo, lambda P, C, s=best_s: dllm_energy_physics(P, C, S=s))
            print(f"  {topo:15s}: break-even at S ≤ {best_s:3d} steps | "
                  f"AR={ar_total:.0f}J, dLLM(S={best_s})={dllm_at_breakeven:.0f}J")
        else:
            print(f"  {topo:15s}: dLLM never cheaper (even at S=1)")

    # Recompute properly (the lambda closure was wrong above)
    print("\n  [Recomputed with explicit S values:]")
    for topo in TOPOLOGIES:
        ar_total = ar_results[topo]["total"]
        for s_val in [1, 2, 5, 10, 20, 50]:
            dllm_total = 0
            for call_type, count in TOPOLOGIES[topo]:
                prof = CALL_PROFILES[call_type]
                dllm_total += dllm_energy_physics(prof["P"], prof["C"], S=s_val) * count
            ratio = dllm_total / ar_total
            marker = " ← BREAK-EVEN" if abs(ratio - 1.0) < 0.15 else ""
            marker = " ← dLLM WINS" if ratio < 0.85 else marker
            print(f"    {topo:15s} S={s_val:2d}: dLLM={dllm_total:8.0f}J | "
                  f"{ratio:.2f}x AR{marker}")

    # --- Save summary for the writeup ---
    print("\n\n" + "=" * 80)
    print("SUMMARY TABLE FOR PAPER")
    print("=" * 80)

    header = f"{'Topology':15s} | {'AR (J)':>10s} | {'AR/SAS':>8s}"
    for S in [10, 20]:
        header += f" | {'dLLM S='+str(S)+' (J)':>15s} | {'dLLM/SAS':>8s} | {'Compress':>8s}"
    print(header)
    print("-" * len(header))

    for topo in TOPOLOGIES:
        ar_total = ar_results[topo]["total"]
        ar_ratio = ar_total / ar_results["SAS"]["total"]
        row = f"{topo:15s} | {ar_total:10.0f} | {ar_ratio:8.1f}x"

        for S in [10, 20]:
            dllm_total = sum(
                dllm_energy_physics(CALL_PROFILES[ct]["P"], CALL_PROFILES[ct]["C"], S=S) * n
                for ct, n in TOPOLOGIES[topo]
            )
            sas_dllm = sum(
                dllm_energy_physics(CALL_PROFILES[ct]["P"], CALL_PROFILES[ct]["C"], S=S) * n
                for ct, n in TOPOLOGIES["SAS"]
            )
            dllm_ratio = dllm_total / sas_dllm
            compression = dllm_ratio / ar_ratio
            row += f" | {dllm_total:15.0f} | {dllm_ratio:8.1f}x | {compression:8.2f}x"

        print(row)


if __name__ == "__main__":
    main()
