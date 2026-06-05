# Latent Communication Pilot

Pilot experiments for the latent-communication MAS project. Goal: decide in
~2 days whether Option 1 (training-free KV-cache injection) is viable on
Qwen3.5-9B's hybrid Gated DeltaNet architecture.

See `<PROJECT_ROOT>/latent_communication_mas_project_brief.md`
for the full project context.

## Core risk

Qwen3.5-9B uses hybrid attention: only ~8 of 32 layers are standard softmax
attention with a KV cache. The other ~24 are DeltaNet (linear-attention /
state-space-style) layers that maintain a recurrent state, not per-token KV.

KV-cache injection can only transmit information through the attention
layers. If DeltaNet carries most of the signal flow, Option 1 has a low
ceiling regardless of compression scheme. This pilot measures whether that
is the case.

## Tests (sequenced)

### Step 0. Architecture inspection (prerequisite)

`inspect_model.py` loads Qwen3.5-9B and dumps:
- Module hierarchy (layer class per index)
- Which layers are softmax-attention vs. DeltaNet
- Cache data structures used by each
- Shape of per-layer K/V tensors

Output: `results/latent_pilot/model_structure.json`. Informs all downstream
tests' hook targets. Runs in ~5 minutes.

### Test 1. Layer-attribution ablation (the gate)

`test1_layer_ablation.py` tests: if we ablate softmax-attention layer
outputs, how much does the model's next-token prediction change? Same for
DeltaNet layers.

Procedure: 200 prompts sampled from existing QAMPARI + WorkBench debate
traces. For each, compare three forward passes:
- **Baseline**: unmodified
- **Attn-ablated**: zero the output of every softmax-attention sublayer
- **DeltaNet-ablated**: zero the output of every DeltaNet sublayer

Metric: top-1 token agreement with baseline, averaged over the last 50
positions of each prompt.

**Decision rule:**
- Attn-ablation → >50% disagreement: attention layers carry substantial
  signal → Option 1 viable → proceed to Test 2
- Attn-ablation → <20% disagreement: DeltaNet dominates → Option 1 has low
  ceiling → fall back to Option 2 (Q-Former compressor + soft embedding)
- Between: Option 1 works but expect lossy transmission; prioritize Test 5

### Test 2. KV-injection functional equivalence

`test2_injection.py` tests: given Agent A's output, does Agent B with
injected KV behave like Agent B reading A's text?

Procedure: 100 debate turns constructed from existing Decentralized R=2
traces. For each sampled turn:
- `B_text`: run Agent B forward with prefix = [system, A's text, B's hint]
- `B_latent`: run Agent B forward with prefix = [system, B's hint],
  injecting the last K KV entries from A's attention layers at the gap

Metric: first-token top-1 agreement; ROUGE-L over first 64 generated tokens.

**Decision thresholds** (at K=64, generous to isolate architecture from
compression):
- Agreement >60%: injection transmits useful signal; proceed to Tests 3-5
- 20-60%: marginal; consider learned projection on top of KV
- <20%: injection broken at this architecture; abandon Option 1

### Test 3. RoPE sensitivity (extends Test 2)

Repeat Test 2 with three RoPE handling strategies:
- (a) Inject K with A's original rotary positions intact
- (b) Strip A's RoPE, re-rotate to B's target positions  
- (c) SGLang-equivalent default prefix-cache treatment

Determines whether manual position correction is required.

### Test 4. Compression knee (extends Test 2)

Repeat Test 2 at K ∈ {8, 16, 32, 64, 128}. Plot first-token agreement vs K.
Identifies the compression Pareto knee.

### Test 5. End-to-end task accuracy

Only if Tests 1-4 pass. Run 50 QAMPARI + 50 WorkBench tasks through the full
Decentralized M=3, R=2 pipeline with K=64 latent messages. Compare accuracy
and energy to existing text baseline.

Kill criterion: >15% accuracy drop → Option 1 does not work even at generous
K → fall back to Option 2.

## What NOT to do in this pilot

- Do not use SGLang for Tests 1-4. Needs raw HF transformers for hook
  access.
- Do not implement learned compressor yet. Option 1 must be ruled out first.
- Do not run on benchmarks beyond QAMPARI + WorkBench. 100 tasks × 2
  benchmarks is sufficient statistical power for the gate decision.
- Do not skip the inspection step. Hook targets depend on exact module
  class names in Qwen3.5-9B.

## Execution order on cluster

```bash
# Step 0 (~5 min): architecture dump
sbatch mas-energy/scripts/latent_pilot_inspect.sbatch

# Step 1 (~1 hr): layer ablation — the gate
sbatch mas-energy/scripts/latent_pilot_test1.sbatch

# Step 2 (~2 hr): only if Test 1 passes
sbatch mas-energy/scripts/latent_pilot_test2.sbatch
```

All outputs land in `/atlas2/u/$USER/mas_project/mas-energy/results/latent_pilot/`.
