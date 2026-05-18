"""Test 1: Layer-attribution ablation.

Question: how much of Qwen3.5-9B's next-token prediction depends on softmax
attention layers vs. DeltaNet layers?

Procedure: for N prompts, compare baseline next-token predictions to
predictions when a specific layer kind is fully ablated (output zeroed
inside every matching sublayer). The residual stream still flows through
MLPs and the other layer type, so an ablation that preserves prediction
quality means the ablated pathway was redundant.

Decision rule (see README):
  attn disagreement rate > 0.50  → Option 1 viable, proceed
  attn disagreement rate < 0.20  → DeltaNet dominates, abandon Option 1
  in between                     → proceed but expect lossy

Outputs:
  results/latent_pilot/test1_ablation.jsonl   per-prompt records
  results/latent_pilot/test1_summary.json     aggregate decision
"""
from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
from latent_pilot.model_probe import (  # noqa: E402
    capture_sequence_logits,
    load_model,
    register_output_zero_hooks,
)
from latent_pilot.utils import (  # noqa: E402
    append_jsonl,
    results_root,
    save_json,
    top1_agreement,
)


def _load_qampari_questions(n: int, seed: int = 42) -> list[str]:
    """Load QAMPARI questions from the HF dataset (cached on cluster).
    Each question is a list-answer QA prompt with a system instruction,
    giving ~100-300 tokens after tokenization — enough for ablation."""
    try:
        from datasets import load_dataset
        ds = load_dataset("valanth/qampari", split="test")
        rng = random.Random(seed)
        indices = list(range(len(ds)))
        rng.shuffle(indices)
        prompts = []
        for idx in indices[:n]:
            row = ds[idx]
            q = row.get("question", "")
            prompt = (
                "You are a helpful research assistant. Answer the following question "
                "by listing all relevant entities. Provide as many correct answers as "
                "possible, each on its own line.\n\n"
                f"Question: {q}\n\nAnswer:"
            )
            prompts.append(prompt)
        return prompts
    except Exception as e:
        print(f"Could not load QAMPARI dataset: {e}")
        return []


def _synthetic_debate_prompts(n: int, seed: int = 42) -> list[str]:
    """Generate synthetic multi-agent debate prompts that are long enough
    to produce meaningful ablation measurements (~100-500 tokens each).
    These simulate the structure of inter-agent debate messages without
    needing saved transcripts."""
    rng = random.Random(seed)
    topics = [
        "the benefits and risks of multi-agent AI systems for scientific research",
        "whether large language models can genuinely reason or merely pattern match",
        "the energy efficiency tradeoffs of sparse mixture-of-experts vs dense models",
        "how to evaluate factual accuracy in open-ended question answering systems",
        "the role of inter-agent communication overhead in multi-agent system scaling",
        "whether retrieval-augmented generation reduces or increases hallucination",
        "the tradeoffs between model size, quantization level, and task performance",
        "how attention mechanisms contribute to in-context learning capabilities",
        "the environmental impact of scaling language model training and inference",
        "whether chain-of-thought prompting improves reasoning or just lengthens outputs",
    ]
    templates = [
        (
            "You are Agent {agent_id} in a multi-agent debate system. The topic is: {topic}.\n\n"
            "Agent 1 previously argued:\n"
            "The key consideration here is that current approaches tend to overlook the "
            "fundamental tradeoff between computational cost and marginal accuracy gains. "
            "When we examine the empirical evidence from recent benchmarks, we find that "
            "the relationship between resource expenditure and performance improvement is "
            "highly non-linear, with diminishing returns setting in much earlier than most "
            "practitioners assume. Furthermore, the environmental implications of scaling "
            "these systems without careful measurement are significant and underappreciated.\n\n"
            "Agent 2 previously argued:\n"
            "While Agent 1 raises valid concerns about efficiency, the counterargument is "
            "that capability improvements unlock qualitatively new use cases that cannot be "
            "achieved at lower scales. The history of deep learning shows that scale has "
            "consistently been the primary driver of emergent capabilities, and premature "
            "optimization of resource usage risks missing transformative applications. The "
            "real question is not whether to scale, but how to scale responsibly while "
            "measuring the true costs including energy, carbon, and opportunity cost.\n\n"
            "Now provide your synthesis of both arguments. Identify where they agree, where "
            "they disagree, and what the strongest combined position would be.\n\nSynthesis:"
        ),
        (
            "System: You are a research assistant helping to evaluate claims about {topic}.\n\n"
            "The following evidence has been gathered from multiple sources:\n\n"
            "Source A (supporting): Recent empirical studies demonstrate significant "
            "improvements when applying the proposed methodology across diverse evaluation "
            "settings. The measured effect sizes range from moderate to large, with "
            "statistical significance achieved in the majority of tested conditions. "
            "Ablation studies confirm that each component of the system contributes "
            "meaningfully to the overall performance, ruling out the possibility that "
            "gains are driven by a single factor.\n\n"
            "Source B (opposing): However, a critical reanalysis of the same data reveals "
            "several methodological concerns. First, the evaluation benchmarks used may not "
            "be representative of real-world deployment conditions. Second, the comparison "
            "baselines were not optimally tuned, potentially inflating the apparent "
            "improvement. Third, the computational cost of the proposed approach is "
            "substantially higher than reported when accounting for all preprocessing "
            "and infrastructure overhead.\n\n"
            "Source C (contextual): A broader meta-analysis of the field suggests that "
            "contradictory findings are common when different research groups evaluate "
            "similar approaches, often due to subtle differences in evaluation protocol, "
            "data preprocessing, or hyperparameter selection rather than fundamental "
            "disagreements about the underlying methodology.\n\n"
            "Based on all three sources, provide a balanced assessment of the current "
            "evidence and identify the most critical open questions.\n\nAssessment:"
        ),
    ]
    prompts = []
    for i in range(n):
        topic = rng.choice(topics)
        template = rng.choice(templates)
        prompt = template.format(agent_id=rng.randint(1, 5), topic=topic)
        prompts.append(prompt)
    return prompts


def build_prompts(n: int, seed: int = 42) -> list[str]:
    """Gather n prompts for ablation testing. Tries QAMPARI dataset first
    (real questions), falls back to synthetic debate-style prompts that
    are long enough for meaningful measurement (~100-500 tokens each)."""
    out = _load_qampari_questions(n, seed=seed)
    n_qampari = len(out)
    if n_qampari < n:
        synthetic = _synthetic_debate_prompts(n - n_qampari, seed=seed)
        out.extend(synthetic)
    print(f"Prompt sources: {n_qampari} QAMPARI, {len(out) - n_qampari} synthetic")
    return out[:n]


def run_pass(
    probed, input_ids: torch.Tensor, attention_mask: torch.Tensor, ablate: str
) -> torch.Tensor:
    """Return logits [batch, seq, vocab] under the named ablation."""
    probed.clear_hooks()
    if ablate == "softmax_attn":
        register_output_zero_hooks(probed, "softmax_attn")
    elif ablate == "deltanet":
        register_output_zero_hooks(probed, "deltanet")
    elif ablate != "none":
        raise ValueError(f"Unknown ablation: {ablate}")
    logits = capture_sequence_logits(probed, input_ids, attention_mask)
    probed.clear_hooks()
    return logits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-prompts", type=int, default=200)
    ap.add_argument("--last-k-positions", type=int, default=50,
                    help="measure agreement only over the last K positions of each prompt")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    out_dir = results_root()
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = out_dir / "test1_ablation.jsonl"
    if jsonl_path.exists():
        jsonl_path.unlink()

    probed = load_model()
    prompts = build_prompts(args.n_prompts, seed=args.seed)
    print(f"Loaded {len(prompts)} prompts")

    agg = {
        "attn_agreement_sum": 0.0,
        "deltanet_agreement_sum": 0.0,
        "count": 0,
    }

    t0 = time.time()
    for i, prompt in enumerate(prompts):
        enc = probed.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=4096,
        )
        ids, mask = enc.input_ids, enc.attention_mask
        seq_len = ids.shape[1]
        if seq_len < 4:
            continue  # need at least a few positions

        try:
            baseline = run_pass(probed, ids, mask, "none")
            attn_abl = run_pass(probed, ids, mask, "softmax_attn")
            dn_abl = run_pass(probed, ids, mask, "deltanet")
        except Exception as e:
            print(f"[{i}] forward failure: {e}")
            continue

        # Measure over last K positions (adaptive: use what's available)
        kpos = min(args.last_k_positions, seq_len - 1)
        attn_agree = top1_agreement(baseline[:, -kpos:, :], attn_abl[:, -kpos:, :])
        dn_agree = top1_agreement(baseline[:, -kpos:, :], dn_abl[:, -kpos:, :])

        rec = {
            "i": i,
            "seq_len": seq_len,
            "measured_positions": kpos,
            "attn_agreement": attn_agree,
            "deltanet_agreement": dn_agree,
        }
        append_jsonl(rec, jsonl_path)

        agg["attn_agreement_sum"] += attn_agree
        agg["deltanet_agreement_sum"] += dn_agree
        agg["count"] += 1

        if (i + 1) % 10 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            print(f"[{i+1:4d}/{len(prompts)}] "
                  f"attn_agree={agg['attn_agreement_sum']/agg['count']:.3f} "
                  f"dn_agree={agg['deltanet_agreement_sum']/agg['count']:.3f} "
                  f"({rate:.1f} prompts/s)")

    if agg["count"] == 0:
        print("ERROR: no prompts produced results")
        return

    attn_mean = agg["attn_agreement_sum"] / agg["count"]
    dn_mean = agg["deltanet_agreement_sum"] / agg["count"]
    # Disagreement = 1 - agreement. Higher = that pathway mattered more.
    attn_disagreement = 1 - attn_mean
    dn_disagreement = 1 - dn_mean

    if attn_disagreement > 0.50:
        verdict = "VIABLE: softmax attention carries substantial signal; proceed to Test 2"
    elif attn_disagreement < 0.20:
        verdict = "KILL: DeltaNet dominates; abandon Option 1 (training-free KV injection)"
    else:
        verdict = "MARGINAL: proceed but expect lossy transmission; prioritize Test 5"

    summary = {
        "n_prompts_measured": agg["count"],
        "attn_agreement_mean": attn_mean,
        "deltanet_agreement_mean": dn_mean,
        "attn_disagreement": attn_disagreement,
        "deltanet_disagreement": dn_disagreement,
        "verdict": verdict,
        "elapsed_sec": time.time() - t0,
    }
    save_json(summary, out_dir / "test1_summary.json")
    print("\n" + "=" * 60)
    print("TEST 1 RESULT")
    for k, v in summary.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
