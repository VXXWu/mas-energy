"""
Calculate maximum context length for each model on A5000 (24GB)
and A6000 (48GB) for comparison.

KV cache per token = 2 (K+V) × n_layers × n_kv_heads × head_dim × 2 (FP16 bytes)

SGLang memory layout:
  total_available = gpu_vram × mem_fraction_static
  kv_cache_budget = total_available - model_weights
  max_context = kv_cache_budget / kv_bytes_per_token
"""
import math

MODELS = {
    "Qwen2.5-7B-Instruct": {
        "layers": 28,
        "kv_heads": 4,
        "head_dim": 128,
        "weights_gb_bf16": 15.0,
        "weights_gb_int4": 4.5,  # ~4-5GB INT4
        "precision": "BF16",
        "tool_calling": "Strong (BFCL tested)",
        "sglang_parser": "qwen25",
    },
    "Qwen3-8B": {
        "layers": 32,
        "kv_heads": 32,  # Full MHA, no GQA
        "head_dim": 128,
        "weights_gb_bf16": 16.0,
        "weights_gb_int4": 5.0,
        "precision": "BF16",
        "tool_calling": "Native, qwen3_coder",
        "sglang_parser": "qwen3_coder",
    },
    "Llama-3.1-8B-Instruct": {
        "layers": 32,
        "kv_heads": 8,  # GQA 4:1
        "head_dim": 128,
        "weights_gb_bf16": 16.0,
        "weights_gb_int4": 5.0,
        "precision": "BF16",
        "tool_calling": "Decent",
        "sglang_parser": "llama3",
    },
    "Mistral-7B-Instruct-v0.3": {
        "layers": 32,
        "kv_heads": 8,  # GQA
        "head_dim": 128,
        "weights_gb_bf16": 14.0,
        "weights_gb_int4": 4.0,
        "precision": "BF16",
        "tool_calling": "Native function calling",
        "sglang_parser": "mistral",
    },
    "Phi-3.5-mini-instruct (3.8B)": {
        "layers": 32,
        "kv_heads": 8,  # GQA
        "head_dim": 96,
        "weights_gb_bf16": 7.6,
        "weights_gb_int4": 2.5,
        "precision": "BF16",
        "tool_calling": "Decent",
        "sglang_parser": "phi3",
    },
    "Qwen3.5-9B (current, reference)": {
        "layers": 40,  # estimated for hybrid model
        "kv_heads": 4,  # GQA for attention layers
        "head_dim": 128,
        # Hybrid model: 18GB weights + 10.5GB GDN state = 28.5GB BF16
        "weights_gb_bf16": 28.5,  # includes GDN state
        "weights_gb_int4": None,  # hybrid arch, complex quantization
        "precision": "BF16",
        "tool_calling": "Native, qwen3_coder",
        "sglang_parser": "qwen3_coder",
        "note": "Hybrid Mamba/GDN - 75% of layers use GDN (no KV cache needed). Only ~25% attention layers need KV. Effective KV layers ~10.",
        "effective_kv_layers": 10,  # only attention layers need KV cache
    },
}

GPUS = {
    "A5000 (24GB)": {"vram_gb": 24},
    "A6000 Ada (48GB)": {"vram_gb": 48},
}

TARGET_CONTEXT = 131072


def kv_bytes_per_token(layers, kv_heads, head_dim):
    """KV cache bytes per token for all layers."""
    return 2 * layers * kv_heads * head_dim * 2  # 2 for K+V, 2 for FP16 bytes


def max_context(vram_gb, weights_gb, kv_per_token, mem_fraction=0.85):
    """Maximum context length given VRAM budget."""
    available_gb = vram_gb * mem_fraction
    kv_budget_gb = available_gb - weights_gb
    if kv_budget_gb <= 0:
        return 0
    kv_budget_bytes = kv_budget_gb * 1024**3
    return int(kv_budget_bytes / kv_per_token)


def mem_fraction_for_target(vram_gb, weights_gb, kv_per_token, target_ctx):
    """What mem_fraction_static is needed to reach target context length?"""
    kv_needed_bytes = target_ctx * kv_per_token
    kv_needed_gb = kv_needed_bytes / 1024**3
    total_needed_gb = weights_gb + kv_needed_gb
    fraction = total_needed_gb / vram_gb
    return fraction


def main():
    print("=" * 100)
    print("MODEL CONTEXT LENGTH CALCULATOR: A5000 (24GB) vs A6000 (48GB)")
    print("=" * 100)

    # Section 1: KV cache sizes
    print("\n### 1. KV CACHE SIZE PER TOKEN")
    print(f"{'Model':40s} | {'Layers':>6s} | {'KV Heads':>8s} | {'Head Dim':>8s} | {'KV/token':>10s} | {'KV/token':>10s}")
    print(f"{'':40s} | {'':>6s} | {'':>8s} | {'':>8s} | {'(bytes)':>10s} | {'(KB)':>10s}")
    print("-" * 100)

    for name, cfg in MODELS.items():
        eff_layers = cfg.get("effective_kv_layers", cfg["layers"])
        kv = kv_bytes_per_token(eff_layers, cfg["kv_heads"], cfg["head_dim"])
        print(f"{name:40s} | {eff_layers:6d} | {cfg['kv_heads']:8d} | {cfg['head_dim']:8d} | {kv:10,d} | {kv/1024:10.1f}")

    # Section 2: Max context per GPU
    print("\n### 2. MAX CONTEXT LENGTH BY GPU AND MEM_FRACTION")
    print()

    for gpu_name, gpu in GPUS.items():
        print(f"  --- {gpu_name} ---")
        print(f"  {'Model':40s} | {'Weights':>8s} | {'frac=0.80':>10s} | {'frac=0.85':>10s} | {'frac=0.90':>10s} | {'frac=0.95':>10s} | {'Need for 131K':>13s}")
        print(f"  {'-'*40}-+-{'-'*8}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}-+-{'-'*13}")

        for name, cfg in MODELS.items():
            weights = cfg["weights_gb_bf16"]
            if weights is None:
                continue

            eff_layers = cfg.get("effective_kv_layers", cfg["layers"])
            kv = kv_bytes_per_token(eff_layers, cfg["kv_heads"], cfg["head_dim"])

            contexts = []
            for frac in [0.80, 0.85, 0.90, 0.95]:
                ctx = max_context(gpu["vram_gb"], weights, kv, frac)
                contexts.append(ctx)

            # What fraction needed for 131072?
            frac_needed = mem_fraction_for_target(gpu["vram_gb"], weights, kv, TARGET_CONTEXT)

            def fmt_ctx(c):
                if c <= 0:
                    return "NO FIT"
                elif c >= 1_000_000:
                    return f"{c/1000:.0f}K"
                elif c >= 10_000:
                    return f"{c/1000:.0f}K"
                else:
                    return f"{c:,d}"

            frac_str = f"{frac_needed:.2f}" if frac_needed <= 1.0 else "IMPOSSIBLE"

            print(f"  {name:40s} | {weights:7.1f}G | {fmt_ctx(contexts[0]):>10s} | {fmt_ctx(contexts[1]):>10s} | {fmt_ctx(contexts[2]):>10s} | {fmt_ctx(contexts[3]):>10s} | {frac_str:>13s}")

        print()

    # Section 3: Recommendation
    print("\n### 3. MODELS THAT ACHIEVE 131K CONTEXT ON A5000")
    print()

    for name, cfg in MODELS.items():
        weights = cfg["weights_gb_bf16"]
        if weights is None:
            continue

        eff_layers = cfg.get("effective_kv_layers", cfg["layers"])
        kv = kv_bytes_per_token(eff_layers, cfg["kv_heads"], cfg["head_dim"])

        frac_needed = mem_fraction_for_target(24, weights, kv, TARGET_CONTEXT)

        if frac_needed <= 0.95:
            ctx_at_90 = max_context(24, weights, kv, 0.90)
            print(f"  YES: {name}")
            print(f"       Weights: {weights}GB | KV: {kv/1024:.1f} KB/tok | "
                  f"Needs mem_fraction >= {frac_needed:.2f} | "
                  f"Context at 0.90: {ctx_at_90/1000:.0f}K")
        else:
            ctx_at_90 = max_context(24, weights, kv, 0.90)
            max_ctx = max_context(24, weights, kv, 0.95)
            print(f"  NO:  {name}")
            print(f"       Weights: {weights}GB | KV: {kv/1024:.1f} KB/tok | "
                  f"Would need mem_fraction={frac_needed:.2f} (>0.95) | "
                  f"Max context at 0.95: {max_ctx/1000:.0f}K")


if __name__ == "__main__":
    main()
