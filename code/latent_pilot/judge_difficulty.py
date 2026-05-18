"""LLM-as-judge: rate each question's reasoning complexity 1-5 using
Qwen3-8B (same model family as the main study).

Runs one short forward pass per question, far cheaper than running MAS.
Output is a single feature ('llm_difficulty_rating') appended to the
enriched CD jsonl for downstream use by train_cd_classifier.py.

Rating scale:
  1 = direct factual lookup (single entity, single fact)
  2 = list retrieval (entities matching a property)
  3 = composition / intersection of known facts
  4 = multi-hop reasoning across entities
  5 = analytical reasoning / causal inference

Output format: adds r["text_features"]["llm_difficulty_rating"] (float 1-5)
and r["text_features"]["llm_difficulty_logits"] (list[float] length 5) for
downstream use.

Usage (cluster, with Qwen3-8B):
    python judge_difficulty.py \
        --in mas-energy/results/latent_pilot/collaboration_dividend_final.jsonl \
        --out mas-energy/results/latent_pilot/cd_with_llm_judge.jsonl \
        --model Qwen/Qwen3-8B

Usage (local test with small model):
    python judge_difficulty.py \
        --in ... --out ... \
        --model Qwen/Qwen3-0.6B-Base  # or any small CausalLM

Runtime: ~500 questions × ~1 short forward pass each. On A5000 bf16:
~2-3 minutes. On CPU with a small model: ~30-60 minutes.
"""
import argparse
import json
import os
import sys
from pathlib import Path


JUDGE_PROMPT = (
    "You are rating the reasoning complexity of a question on a 1-5 scale:\n"
    "1 = direct factual lookup (single entity, single fact)\n"
    "2 = list retrieval (find multiple entities matching a property)\n"
    "3 = composition or intersection of known facts\n"
    "4 = multi-hop reasoning across entities or documents\n"
    "5 = analytical reasoning, causal inference, or synthesis\n\n"
    "Respond with ONLY a single digit (1, 2, 3, 4, or 5). No explanation.\n\n"
    "Question: {question}\n\n"
    "Rating:"
)


def build_prompt(tokenizer, question):
    """Format the judge prompt via chat template (supports Qwen-style models).
    Falls back to raw text for models without chat templates."""
    raw = JUDGE_PROMPT.format(question=question)
    try:
        messages = [{"role": "user", "content": raw}]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=False,
        )
        return text
    except Exception:
        return raw


def rate_question(model, tokenizer, question, digit_token_ids, device="cuda"):
    """Return (rating, logits) where rating is expected 1-5 and logits is
    the raw log-prob over digits 1-5 at the first generation position."""
    import torch
    text = build_prompt(tokenizer, question)
    enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=2048)
    ids = enc.input_ids.to(device)
    with torch.no_grad():
        out = model(ids)
    # Logits at the LAST position (what the model would emit next)
    last_logits = out.logits[0, -1, :].float()
    # Restrict to digit tokens 1..5
    digit_logits = [last_logits[tid].item() for tid in digit_token_ids]
    # Expected rating = argmax over digits
    import math
    # Softmax over the 5 digit tokens only
    m = max(digit_logits)
    exps = [math.exp(x - m) for x in digit_logits]
    Z = sum(exps)
    probs = [e / Z for e in exps]
    expected = sum((i + 1) * p for i, p in enumerate(probs))
    return expected, digit_logits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path", type=str, required=True)
    ap.add_argument("--out", dest="out_path", type=str, required=True)
    ap.add_argument("--model", type=str, default="Qwen/Qwen3-8B")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--max-tasks", type=int, default=None,
                    help="Cap on questions to judge (for quick tests).")
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Loading {args.model} on {args.device}...")
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16 if "cuda" in args.device else torch.float32,
        trust_remote_code=True,
    ).to(args.device)
    model.eval()

    # Token IDs for digits 1-5 (Qwen-style BPE; typically one token each).
    digit_token_ids = []
    for d in ("1", "2", "3", "4", "5"):
        ids = tok.encode(d, add_special_tokens=False)
        if len(ids) != 1:
            print(f"  WARNING: digit '{d}' tokenizes to {ids}; using first token.")
        digit_token_ids.append(ids[0])
    print(f"  Digit token ids: {dict(zip('12345', digit_token_ids))}")

    rows = [json.loads(l) for l in open(args.in_path)]
    n_questions = sum(1 for r in rows if r.get("question_text"))
    print(f"Questions to judge: {n_questions}/{len(rows)} "
          f"(max={args.max_tasks or 'all'})")

    judged = 0
    import time
    t0 = time.time()
    for i, r in enumerate(rows):
        q = r.get("question_text")
        if not q:
            continue
        if args.max_tasks and judged >= args.max_tasks:
            break
        rating, logits = rate_question(model, tok, q, digit_token_ids, device=args.device)
        tf = r.get("text_features") or {}
        tf["llm_difficulty_rating"] = rating
        tf["llm_difficulty_logits"] = logits
        r["text_features"] = tf
        judged += 1
        if (judged % 20) == 0:
            elapsed = time.time() - t0
            rate = judged / max(elapsed, 1e-3)
            print(f"  [{judged}/{n_questions}] rating mean so far: "
                  f"{sum(r['text_features'].get('llm_difficulty_rating', 0) for r in rows[:i+1] if r.get('text_features')) / max(judged, 1):.2f}  "
                  f"({rate:.1f} q/s, {elapsed:.0f}s elapsed)")

    Path(args.out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"\nJudged {judged} questions.")
    print(f"Output: {args.out_path}")


if __name__ == "__main__":
    main()
