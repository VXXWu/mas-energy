"""Concision SFT: fine-tune a LoRA adapter to output concise entity lists.

Takes (verbose, concise) pairs from concision_data_gen.py and trains a
LoRA on (context → concise_target) with standard cross-entropy. No
mechanism change, just biasing output toward brevity.

Usage:
    python concision_train.py --data concise_pairs.jsonl --out-dir concise_lora
"""
import argparse
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, required=True)
    ap.add_argument("--model", type=str, default="Qwen/Qwen3-8B")
    ap.add_argument("--out-dir", type=str, required=True)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lora-rank", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--max-seq-len", type=int, default=4096)
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, get_scheduler
    from peft import LoraConfig, get_peft_model, TaskType
    from torch.utils.data import DataLoader

    print(f"Loading {args.model} in bf16 + LoRA rank {args.lora_rank}...")
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, trust_remote_code=True,
    ).to("cuda")
    lora = LoraConfig(
        r=args.lora_rank, lora_alpha=16, lora_dropout=0.05,
        bias="none", task_type=TaskType.CAUSAL_LM,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    rows = [json.loads(l) for l in open(args.data)]
    print(f"Loaded {len(rows)} (verbose, concise) pairs")

    def build_example(row):
        msgs = row["messages_pre_answer"] + [
            {"role": "assistant", "content": row["concise_target"]}
        ]
        text = tok.apply_chat_template(msgs, tokenize=False, enable_thinking=False)
        enc = tok(text, return_tensors="pt", truncation=True, max_length=args.max_seq_len)
        input_ids = enc.input_ids[0]
        labels = input_ids.clone()

        # Mask everything before the final assistant turn so we only learn
        # the concise_target tokens (not the prompt).
        prefix_msgs = row["messages_pre_answer"]
        prefix_text = tok.apply_chat_template(prefix_msgs, tokenize=False,
                                               add_generation_prompt=True,
                                               enable_thinking=False)
        prefix_ids = tok(prefix_text, add_special_tokens=False).input_ids
        labels[:len(prefix_ids)] = -100
        return {"input_ids": input_ids, "labels": labels}

    dataset = [build_example(r) for r in rows]
    print(f"  {len(dataset)} training examples")

    def collate(batch):
        max_len = max(x["input_ids"].size(0) for x in batch)
        ids = torch.full((len(batch), max_len), tok.pad_token_id, dtype=torch.long)
        labs = torch.full((len(batch), max_len), -100, dtype=torch.long)
        mask = torch.zeros((len(batch), max_len), dtype=torch.long)
        for i, x in enumerate(batch):
            n = x["input_ids"].size(0)
            ids[i, :n] = x["input_ids"]
            labs[i, :n] = x["labels"]
            mask[i, :n] = 1
        return {"input_ids": ids, "labels": labs, "attention_mask": mask}

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    total = max((len(loader) // args.grad_accum) * args.epochs, 10)
    sched = get_scheduler("cosine", optimizer=opt, num_warmup_steps=10, num_training_steps=total)

    model.train()
    for epoch in range(args.epochs):
        for bi, batch in enumerate(loader):
            batch = {k: v.to("cuda") for k, v in batch.items()}
            out = model(**batch)
            loss = out.loss / args.grad_accum
            loss.backward()
            if (bi + 1) % args.grad_accum == 0:
                opt.step(); sched.step(); opt.zero_grad()
            if bi % 20 == 0:
                print(f"epoch={epoch} batch={bi} loss={out.loss.item():.4f}")

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.out_dir)
    tok.save_pretrained(args.out_dir)
    print(f"Saved concision LoRA to {args.out_dir}")


if __name__ == "__main__":
    main()
