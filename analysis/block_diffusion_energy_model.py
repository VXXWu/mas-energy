"""
Block diffusion energy model for MAS topologies.

Block diffusion (BD3-LMs, SDAR, LLaDA2.0) generates text in blocks:
- AR conditioning BETWEEN blocks (with KV-cache)
- Diffusion denoising WITHIN each block (S steps over B tokens)

This is fundamentally different from pure diffusion:
- Pure dLLM: S steps over ALL (P+C) tokens every step (no KV-cache)
- Block diffusion: S steps over just B tokens per block, with KV-cache for prior context

Energy model for block diffusion:
- Prefill: same as AR (one forward pass over P tokens) = 0.018 * P
- Decode: ceil(C/B) blocks, each block costs:
  - S denoising steps, each processing B tokens + attending to KV-cache of prior context
  - Per-step cost ≈ cost_of_prefilling_B_tokens + attention_over_cached_context
  - ≈ 0.018 * B + attention_cost(context_length)

  But attention cost is small relative to FFN cost for small B.
  Simplify: per-block cost ≈ S * 0.018 * B (FFN-dominated for small B)
  Total decode: ceil(C/B) * S * 0.018 * B = S * 0.018 * C (when C is divisible by B)

  Wait -- this simplifies to the same as pure dLLM decode but WITHOUT
  reprocessing the prompt at every step!

  Pure dLLM: E_decode = S * 0.018 * (P + C)  [reprocesses P every step]
  Block diff: E_decode = S * 0.018 * C         [KV-cache handles P]

  Plus the AR conditioning between blocks adds a small per-block
  forward pass for the AR "glue" step. Model this as:
  E_ar_glue = ceil(C/B) * 5.54  [one AR decode step per block boundary]

  Total block diffusion energy:
  E_block = 0.018 * P + S * 0.018 * C + ceil(C/B) * 5.54

  But actually, within each block, the S denoising steps need to attend
  to the full prior context (P + previously generated blocks) via
  cross-attention or concatenated context. With KV-cache, this is
  just the attention component, not the full FFN recomputation.

  For transformer models, FFN typically dominates compute (~2/3 of FLOPs).
  Attention scales with sequence length but is cached.

  More precise model:
  - Each denoising step within block k processes B new positions
  - FFN cost: proportional to B (fixed per step)
  - Attention cost: B queries attending to (P + k*B) cached keys
  - For large models, FFN dominates, so per-step ≈ 0.018 * B
  - For long contexts, attention contribution grows but is still < FFN

  Let's model with an attention overhead factor:
  E_block_k = S * (0.018 * B + attn_cost * B * (P + k*B))

  This is getting complex. Let me just compute with the simplified models
  and show the range.
"""

import math

# Empirical per-call token profiles (MEDIAN values from 80K+ calls)
CALL_PROFILES = {
    "react_step":       {"P": 5168, "C": 75},
    "decompose":        {"P": 305,  "C": 117},
    "review":           {"P": 1037, "C": 845},
    "synthesis":        {"P": 1016, "C": 247},
    "debate_synthesis": {"P": 1690, "C": 354},
    "wrapup":           {"P": 3852, "C": 121},
}

B_PROMPT = 0.018   # J per prompt token (prefill)
B_COMPL = 5.54     # J per completion token (AR decode)

TOPOLOGIES = {
    "SAS": [
        ("react_step", 5),
        ("wrapup", 1),
    ],
    "Independent": [
        ("react_step", 15),
        ("synthesis", 1),
        ("wrapup", 1),
    ],
    "Centralized": [
        ("decompose", 1),
        ("react_step", 18),
        ("review", 2),
        ("synthesis", 1),
        ("wrapup", 1),
    ],
    "Decentralized": [
        ("react_step", 15),
        ("debate_synthesis", 6),
        ("wrapup", 1),
    ],
}


def ar_energy(P, C):
    """Standard AR energy."""
    return max(0, -84 + B_PROMPT * P + B_COMPL * C)


def pure_dllm_energy(P, C, S=20):
    """Pure dLLM: S steps over full (P+C) sequence, no KV-cache."""
    return S * B_PROMPT * (P + C)


def block_diffusion_energy(P, C, S=10, B=64):
    """
    Block diffusion energy model.

    - Prefill: one forward pass over P tokens (same as AR)
    - Decode: ceil(C/B) blocks, each with S denoising steps over B tokens
    - AR glue: one AR-style forward pass per block boundary
    - KV-cache: prior context cached, not reprocessed in FFN

    Simplified (FFN-dominated, attention cached):
    E = 0.018*P + ceil(C/B) * S * 0.018 * B + ceil(C/B) * ar_step_cost

    The ar_step_cost per block boundary is approximately one AR decode step
    (generating the conditioning token for the next block).
    """
    n_blocks = max(1, math.ceil(C / B))

    # Prefill prompt (same as AR)
    e_prefill = B_PROMPT * P

    # Diffusion within blocks: S steps per block, each processing B tokens
    # Cost per step ≈ prefilling B tokens (FFN-dominated)
    e_diffusion = n_blocks * S * B_PROMPT * B

    # AR conditioning between blocks: ~1 AR decode step per boundary
    # This is the "glue" that provides left-to-right coherence
    e_ar_glue = n_blocks * B_COMPL  # one decode-equivalent step per block

    return e_prefill + e_diffusion + e_ar_glue


def block_diffusion_with_attention(P, C, S=10, B=64, attn_fraction=0.3):
    """
    Block diffusion with attention overhead.

    Each denoising step within a block also does cross-attention to cached
    prior context. This adds cost proportional to B * context_length,
    but it's only the attention component (not FFN recomputation).

    attn_fraction: what fraction of per-token compute is attention vs FFN.
    Typically 0.2-0.4 for transformer models.
    """
    n_blocks = max(1, math.ceil(C / B))

    e_prefill = B_PROMPT * P

    e_total_blocks = 0
    for k in range(n_blocks):
        context_len = P + k * B  # cached context from prior blocks
        actual_B = min(B, C - k * B)  # last block may be shorter

        # FFN cost: processing actual_B new tokens
        e_ffn = S * B_PROMPT * actual_B

        # Attention cost: actual_B queries attending to context_len cached keys
        # Relative to FFN, attention is attn_fraction of total per-token cost
        # But attention scales with context_len / B_typical
        # For a single AR step on 1 token with context L, attention cost ∝ L
        # For B tokens in a block with context L, attention cost ∝ B * L
        # Normalize: attention per AR step ≈ attn_fraction * B_COMPL
        # For block of B tokens with context L:
        #   attn ≈ attn_fraction * B_PROMPT * actual_B * (context_len / P_ref)
        # where P_ref is some reference prompt length
        #
        # Actually, simpler: the attention component of one AR decode step is
        # attn_fraction * B_COMPL ≈ 0.3 * 5.54 = 1.66 J
        # For a block of B tokens, it's B * 1.66 J per step
        # But with KV-cache, it scales linearly with context, not quadratically
        # The key point: this is MUCH less than FFN

        # Simple model: attention adds attn_fraction * (context_len/1000) * B_PROMPT * actual_B per step
        # This captures that attention cost grows with context length
        e_attn = S * attn_fraction * B_PROMPT * actual_B * (context_len / 1000)

        e_total_blocks += e_ffn + e_attn

    e_ar_glue = n_blocks * B_COMPL

    return e_prefill + e_total_blocks + e_ar_glue


def topology_energy(topo_name, energy_fn):
    """Compute total topology energy."""
    total = 0
    for call_type, count in TOPOLOGIES[topo_name]:
        prof = CALL_PROFILES[call_type]
        total += energy_fn(prof["P"], prof["C"]) * count
    return total


def main():
    print("=" * 90)
    print("BLOCK DIFFUSION vs PURE dLLM vs AR: Energy Model for MAS Topologies")
    print("=" * 90)

    # --- Section 1: Per-call comparison ---
    print("\n### 1. PER-CALL ENERGY COMPARISON")
    print()

    for S in [10, 20]:
        for B in [32, 64, 128]:
            print(f"  S={S} steps, B={B} block size:")
            print(f"  {'Call':20s} | {'P':>5s} {'C':>5s} | {'AR':>8s} | {'Pure dLLM':>10s} | {'Block Diff':>10s} | {'Block+Attn':>10s} | {'BD/AR':>6s}")
            print(f"  {'-'*20}-+-{'-'*11}-+-{'-'*8}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}-+-{'-'*6}")

            for call_type in ["react_step", "decompose", "review", "synthesis", "debate_synthesis", "wrapup"]:
                prof = CALL_PROFILES[call_type]
                P, C = prof["P"], prof["C"]

                e_ar = ar_energy(P, C)
                e_pure = pure_dllm_energy(P, C, S=S)
                e_block = block_diffusion_energy(P, C, S=S, B=B)
                e_block_attn = block_diffusion_with_attention(P, C, S=S, B=B)

                ratio_bd_ar = e_block / e_ar if e_ar > 0 else float('inf')

                print(f"  {call_type:20s} | {P:5d} {C:5d} | {e_ar:8.0f} | {e_pure:10.0f} | {e_block:10.0f} | {e_block_attn:10.0f} | {ratio_bd_ar:6.2f}x")
            print()

    # --- Section 2: Per-topology comparison ---
    print("\n### 2. PER-TOPOLOGY ENERGY: AR vs Pure dLLM vs Block Diffusion")
    print()

    # Reference: AR
    ar_results = {t: topology_energy(t, ar_energy) for t in TOPOLOGIES}
    ar_sas = ar_results["SAS"]

    for S in [10, 20]:
        for B in [64]:
            print(f"  S={S}, B={B}:")
            print(f"  {'Topology':15s} | {'AR (J)':>10s} {'AR/SAS':>8s} | {'Pure dLLM':>10s} {'Pd/SAS':>8s} | {'Block Diff':>10s} {'BD/SAS':>8s} | {'BD+Attn':>10s} {'BA/SAS':>8s}")
            print(f"  {'-'*15}-+-{'-'*19}-+-{'-'*19}-+-{'-'*19}-+-{'-'*19}")

            pure_sas = topology_energy("SAS", lambda P, C: pure_dllm_energy(P, C, S=S))
            block_sas = topology_energy("SAS", lambda P, C: block_diffusion_energy(P, C, S=S, B=B))
            blocka_sas = topology_energy("SAS", lambda P, C: block_diffusion_with_attention(P, C, S=S, B=B))

            for topo in TOPOLOGIES:
                e_ar = ar_results[topo]
                e_pure = topology_energy(topo, lambda P, C: pure_dllm_energy(P, C, S=S))
                e_block = topology_energy(topo, lambda P, C: block_diffusion_energy(P, C, S=S, B=B))
                e_blocka = topology_energy(topo, lambda P, C: block_diffusion_with_attention(P, C, S=S, B=B))

                print(f"  {topo:15s} | {e_ar:10.0f} {e_ar/ar_sas:7.1f}x | {e_pure:10.0f} {e_pure/pure_sas:7.1f}x | {e_block:10.0f} {e_block/block_sas:7.1f}x | {e_blocka:10.0f} {e_blocka/blocka_sas:7.1f}x")
            print()

    # --- Section 3: Compression factors ---
    print("\n### 3. COMPRESSION FACTORS (ratio_dLLM/SAS / ratio_AR/SAS)")
    print("    <1.0 = disproportionate benefit vs SAS")
    print()

    print(f"  {'Topology':15s} | {'AR/SAS':>8s} | ", end="")
    configs = []
    for S in [10, 20]:
        for B in [32, 64, 128]:
            label = f"BD S={S},B={B}"
            configs.append((S, B, label))
            print(f"{label:>14s} | ", end="")
    # Also pure dLLM
    print(f"{'Pure S=10':>14s} | {'Pure S=20':>14s}")

    print(f"  {'-'*15}-+-{'-'*8}-+-" + "-+-".join(['-'*14] * len(configs)) + f"-+-{'-'*14}-+-{'-'*14}")

    for topo in TOPOLOGIES:
        ar_ratio = ar_results[topo] / ar_sas
        row = f"  {topo:15s} | {ar_ratio:7.1f}x | "

        for S, B, label in configs:
            block_topo = topology_energy(topo, lambda P, C, s=S, b=B: block_diffusion_energy(P, C, S=s, B=b))
            block_sas = topology_energy("SAS", lambda P, C, s=S, b=B: block_diffusion_energy(P, C, S=s, B=b))
            bd_ratio = block_topo / block_sas
            compression = bd_ratio / ar_ratio
            row += f"{compression:13.2f}x | "

        # Pure dLLM
        for S in [10, 20]:
            pure_topo = topology_energy(topo, lambda P, C, s=S: pure_dllm_energy(P, C, S=s))
            pure_sas = topology_energy("SAS", lambda P, C, s=S: pure_dllm_energy(P, C, S=s))
            pure_ratio = pure_topo / pure_sas
            compression = pure_ratio / ar_ratio
            row += f"{compression:13.2f}x | "

        print(row)

    print()

    # --- Section 4: Why block diffusion differs from pure dLLM ---
    print("\n### 4. KEY INSIGHT: WHY BLOCK DIFFUSION BEHAVES DIFFERENTLY")
    print()
    print("  Pure dLLM reprocesses the ENTIRE prompt at every denoising step (no KV-cache).")
    print("  This penalizes high-P calls (react_step: P=5168) disproportionately.")
    print()
    print("  Block diffusion caches the prompt via KV-cache (like AR).")
    print("  Each denoising step only processes B tokens (the current block).")
    print("  So the prompt cost is paid ONCE (prefill), not S times.")
    print()
    print("  This means block diffusion:")
    print("  - Does NOT penalize high-P, low-C calls (react_step)")
    print("  - Still helps high-C calls (S * 0.018 * B per block < 5.54 per token)")
    print("  - The benefit is more UNIFORM across call types")
    print()

    # Show the decomposition
    print("  Energy decomposition for react_step (P=5168, C=75, S=10, B=64):")
    P, C, S, B = 5168, 75, 10, 64
    n_blocks = max(1, math.ceil(C / B))
    e_prefill = B_PROMPT * P
    e_diff = n_blocks * S * B_PROMPT * B
    e_glue = n_blocks * B_COMPL
    e_total = e_prefill + e_diff + e_glue
    e_ar = ar_energy(P, C)
    print(f"    Prefill:    {e_prefill:8.1f} J (same as AR)")
    print(f"    Diffusion:  {e_diff:8.1f} J ({n_blocks} blocks × {S} steps × {B_PROMPT}×{B} = {n_blocks*S*B_PROMPT*B:.1f})")
    print(f"    AR glue:    {e_glue:8.1f} J ({n_blocks} blocks × {B_COMPL})")
    print(f"    Total:      {e_total:8.1f} J")
    print(f"    AR total:   {e_ar:8.1f} J")
    print(f"    Ratio:      {e_total/e_ar:.2f}x")
    print()

    print("  Energy decomposition for review (P=1037, C=845, S=10, B=64):")
    P, C, S, B = 1037, 845, 10, 64
    n_blocks = max(1, math.ceil(C / B))
    e_prefill = B_PROMPT * P
    e_diff = n_blocks * S * B_PROMPT * B
    e_glue = n_blocks * B_COMPL
    e_total = e_prefill + e_diff + e_glue
    e_ar = ar_energy(P, C)
    print(f"    Prefill:    {e_prefill:8.1f} J")
    print(f"    Diffusion:  {e_diff:8.1f} J ({n_blocks} blocks × {S} steps × {B_PROMPT}×{B})")
    print(f"    AR glue:    {e_glue:8.1f} J ({n_blocks} blocks × {B_COMPL})")
    print(f"    Total:      {e_total:8.1f} J")
    print(f"    AR total:   {e_ar:8.1f} J")
    print(f"    Ratio:      {e_total/e_ar:.2f}x")
    print()

    # --- Section 5: Break-even analysis ---
    print("\n### 5. BREAK-EVEN: What (S, B) makes block diffusion match AR?")
    print()

    for call_type in CALL_PROFILES:
        prof = CALL_PROFILES[call_type]
        P, C = prof["P"], prof["C"]
        e_ar = ar_energy(P, C)

        print(f"  {call_type} (P={P}, C={C}, AR={e_ar:.0f}J):")
        for B in [32, 64, 128]:
            # Find max S where block_diffusion <= AR
            # E_block = 0.018*P + ceil(C/B)*S*0.018*B + ceil(C/B)*5.54
            # Set = AR = -84 + 0.018*P + 5.54*C
            # ceil(C/B)*S*0.018*B + ceil(C/B)*5.54 = -84 + 5.54*C
            # ceil(C/B)*S*0.018*B = -84 + 5.54*C - ceil(C/B)*5.54
            n_blocks = max(1, math.ceil(C / B))
            rhs = -84 + B_COMPL * C - n_blocks * B_COMPL
            if rhs <= 0:
                print(f"    B={B:3d}: block diff always more expensive (AR glue alone exceeds AR decode)")
                continue
            s_max = rhs / (n_blocks * B_PROMPT * B)
            print(f"    B={B:3d}: n_blocks={n_blocks:2d}, break-even S ≤ {s_max:.1f}")
        print()

    # --- Section 6: The answer ---
    print("\n" + "=" * 90)
    print("### 6. ANSWER: Does block diffusion maintain disproportionate MAS benefit?")
    print("=" * 90)
    print()

    # Compare compression factors: pure dLLM vs block diffusion
    print("  Compression factors (how much the topology ratio to SAS shrinks):")
    print(f"  {'Topology':15s} | {'AR/SAS':>8s} | {'Pure dLLM S=10':>15s} | {'Block S=10,B=64':>16s} | {'Block S=10,B=128':>17s}")
    print(f"  {'-'*15}-+-{'-'*8}-+-{'-'*15}-+-{'-'*16}-+-{'-'*17}")

    for topo in TOPOLOGIES:
        ar_ratio = ar_results[topo] / ar_sas

        pure_topo = topology_energy(topo, lambda P, C: pure_dllm_energy(P, C, S=10))
        pure_sas_e = topology_energy("SAS", lambda P, C: pure_dllm_energy(P, C, S=10))
        pure_compress = (pure_topo / pure_sas_e) / ar_ratio

        for B in [64, 128]:
            block_topo = topology_energy(topo, lambda P, C, b=B: block_diffusion_energy(P, C, S=10, B=b))
            block_sas_e = topology_energy("SAS", lambda P, C, b=B: block_diffusion_energy(P, C, S=10, B=b))
            if B == 64:
                b64_compress = (block_topo / block_sas_e) / ar_ratio
            else:
                b128_compress = (block_topo / block_sas_e) / ar_ratio

        print(f"  {topo:15s} | {ar_ratio:7.1f}x | {pure_compress:14.2f}x | {b64_compress:15.2f}x | {b128_compress:16.2f}x")

    print()
    print("  Interpretation:")
    print("  - Pure dLLM: strong compression (0.47-0.49x) because prompt reprocessing")
    print("    penalizes SAS's high-P react calls disproportionately.")
    print("  - Block diffusion: compression depends on whether the topology's")
    print("    coordination calls generate completions >> B (block size).")
    print()

    # Detailed explanation
    print("  Why block diffusion is different:")
    print("  Pure dLLM's disproportionate benefit came from TWO sources:")
    print("    1. Penalizing SAS react calls (P=5168, C=75) by reprocessing P at every step")
    print("    2. Helping coordination calls (high C) via parallel decode")
    print()
    print("  Block diffusion REMOVES source #1 (KV-cache handles P).")
    print("  It KEEPS source #2 (parallel decode within blocks).")
    print("  But source #2 alone is weaker: the benefit is more uniform across")
    print("  call types, so the DISPROPORTIONATE advantage shrinks.")


if __name__ == "__main__":
    main()
