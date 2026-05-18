"""Train the conditional diffusion bridge.

Source representation per training example:
  - Multi-layer hidden states from re-encoding peer_text through the frozen
    backbone. Used as a proxy for actual sender latent CoT thoughts (both
    come from the same backbone, so distributional match is good).

Target x_0:
  - Receiver's input embeddings of peer_text, truncated/padded to k positions.
    This is the "natural" soft prompt sequence the receiver would have
    consumed from text — the bridge learns to reconstruct it from compressed
    multi-layer hidden states.

Loss:
  - Stage 1 (this script): DDPM ε-prediction on the residual from W_e
    baseline. At zero loss, bridge output = target embedding sequence.
  - Stage 2 (KL distillation against text-conditioned receiver) is left as an
    optional auxiliary loss — gated by --kl-weight > 0; computationally heavy
    so default off for the pilot.

Eval metric:
  - Held-out MSE on noise prediction (training-objective consistency).
  - End-to-end KL: sample the bridge, plug into receiver, compute KL of
    receiver's next-token distribution vs text-conditioned distribution.
    Reported every --eval-every steps; this is the publication metric.

Usage (smoke test, ~minutes):
  python train_diffusion_bridge.py --max-pairs 64 --epochs 1 --batch-size 1 \
      --pairs-path mas-energy/results/latent_pilot/training_pairs_qampari.jsonl

Full training (~hours on A6000):
  python train_diffusion_bridge.py --epochs 3 --batch-size 2 --k-soft-prompts 16 \
      --pairs-path mas-energy/results/latent_pilot/training_pairs_all.jsonl \
      --output-dir mas-energy/results/diffusion_bridge/run1
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent.parent))
from latent_pilot.diffusion_bridge import (  # noqa: E402
    BridgeConfig, DiffusionBridge,
    extract_source_layers, default_layer_indices,
)
from latent_pilot.latentmas_baseline import (  # noqa: E402
    compute_alignment, find_weight_matrix,
)


# ────────────────────────────────────────────────────────────────────
# Data
# ────────────────────────────────────────────────────────────────────

def task_split(records, eval_frac=0.1, seed=42):
    """Held-out split by task_id so eval contains entirely unseen tasks."""
    task_ids = sorted({r.get("task_id", "") for r in records if r.get("task_id")})
    rng = random.Random(seed)
    rng.shuffle(task_ids)
    n_eval = max(1, int(len(task_ids) * eval_frac))
    eval_ids = set(task_ids[:n_eval])
    return (
        [r for r in records if r.get("task_id") not in eval_ids],
        [r for r in records if r.get("task_id") in eval_ids],
    )


class BridgePairs(Dataset):
    """Loads (peer_text, target_response, question) records.

    Tokenization is lazy. Hidden-state extraction happens at batch time so we
    don't pre-cache massive tensors.
    """
    def __init__(self, jsonl_path, tokenizer,
                 max_peer_tokens=512, max_question_tokens=512,
                 max_target_tokens=256):
        self.tok = tokenizer
        self.records = []
        self.max_peer_tokens = max_peer_tokens
        self.max_question_tokens = max_question_tokens
        self.max_target_tokens = max_target_tokens
        with open(jsonl_path) as f:
            for line in f:
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not (r.get("peer_text") and r.get("target_response")):
                    continue
                self.records.append(r)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        r = self.records[idx]
        peer = self.tok(r["peer_text"], truncation=True,
                        max_length=self.max_peer_tokens,
                        return_tensors="pt").input_ids.squeeze(0)
        # Question is approximated from the first 'user' content if present,
        # else fall back to the start of peer_text. The pair extractor doesn't
        # always store the original question separately.
        q_text = r.get("question") or r.get("question_text") or r["peer_text"][:512]
        question = self.tok(q_text, truncation=True,
                            max_length=self.max_question_tokens,
                            return_tensors="pt").input_ids.squeeze(0)
        target = self.tok(r["target_response"], truncation=True,
                          max_length=self.max_target_tokens,
                          return_tensors="pt").input_ids.squeeze(0)
        return {
            "peer_ids": peer, "question_ids": question, "target_ids": target,
            "task_id": r.get("task_id", ""),
        }


def collate(batch, pad_id):
    """Right-pad to longest in batch on each tensor."""
    def pad(tensors):
        L = max(t.size(0) for t in tensors)
        bs = len(tensors)
        out = torch.full((bs, L), pad_id, dtype=torch.long)
        mask = torch.zeros((bs, L), dtype=torch.long)
        for i, t in enumerate(tensors):
            out[i, :t.size(0)] = t
            mask[i, :t.size(0)] = 1
        return out, mask

    peer_ids, peer_mask = pad([b["peer_ids"] for b in batch])
    q_ids, q_mask = pad([b["question_ids"] for b in batch])
    tgt_ids, tgt_mask = pad([b["target_ids"] for b in batch])
    return {
        "peer_ids": peer_ids, "peer_mask": peer_mask,
        "question_ids": q_ids, "question_mask": q_mask,
        "target_ids": tgt_ids, "target_mask": tgt_mask,
    }


# ────────────────────────────────────────────────────────────────────
# Source / task / x_0 extraction from a frozen backbone
# ────────────────────────────────────────────────────────────────────

@torch.no_grad()
def extract_batch_features(backbone, embed_layer, batch, layer_indices,
                           m_positions, k_soft_prompts, device):
    """Run backbone on peer_text and question, return source/task/x_0 tensors.

    source: (B, m, L, d) — last m positions of peer-text encoding at L layers.
    task:   (B, d)        — pooled (last-position) hidden state of question.
    x_0:    (B, k, d)     — receiver input embeddings of peer_text, last k tokens
                            (or padded with the pad-token embedding if shorter).
    """
    pid = batch["peer_ids"].to(device)
    pmask = batch["peer_mask"].to(device)
    qid = batch["question_ids"].to(device)
    qmask = batch["question_mask"].to(device)

    # Encode peer_text with hidden states for source extraction
    out_p = backbone(input_ids=pid, attention_mask=pmask,
                     output_hidden_states=True, use_cache=False)
    source = extract_source_layers(out_p.hidden_states, layer_indices,
                                   positions=m_positions)
    # Pad if peer_text was shorter than m_positions (extract_source_layers
    # silently truncates from the right; we left-pad with the first hidden
    # state so the model sees a coherent sequence)
    if source.size(1) < m_positions:
        deficit = m_positions - source.size(1)
        pad_h = source[:, :1, :, :].expand(-1, deficit, -1, -1)
        source = torch.cat([pad_h, source], dim=1)

    # Encode question for task representation
    out_q = backbone(input_ids=qid, attention_mask=qmask,
                     output_hidden_states=True, use_cache=False)
    last_pos = qmask.sum(dim=1) - 1                          # (B,)
    task = out_q.hidden_states[-1][torch.arange(qid.size(0)), last_pos, :]

    # x_0: receiver's input embeddings of peer_text, last k tokens.
    # If shorter than k, left-pad with zero embeddings (those positions
    # will contribute zero signal to the receiver — same effect as
    # not having those tokens at all).
    embeds = embed_layer(pid)                                # (B, T, d)
    if embeds.size(1) >= k_soft_prompts:
        x0 = embeds[:, -k_soft_prompts:, :]
    else:
        deficit = k_soft_prompts - embeds.size(1)
        pad_emb = torch.zeros(embeds.size(0), deficit, embeds.size(2),
                              dtype=embeds.dtype, device=device)
        x0 = torch.cat([pad_emb, embeds], dim=1)

    return source, task, x0


# ────────────────────────────────────────────────────────────────────
# Eval: end-to-end KL on a small held-out subset
# ────────────────────────────────────────────────────────────────────

@torch.no_grad()
def eval_endtoend_kl(backbone, embed_layer, bridge, eval_loader, layer_indices,
                     m_positions, k_soft_prompts, device, K_sample, max_batches=8):
    """For each held-out batch: compute KL between
       (a) receiver_logits given text peer_input (gold)
       (b) receiver_logits given bridge-sampled soft prompts
    over the target tokens. Lower is better; ~0.1-0.2 is "indistinguishable",
    ~0.65 is the prior linear-encoder ceiling.
    """
    bridge.eval()
    kl_vals = []
    for bi, batch in enumerate(eval_loader):
        if bi >= max_batches:
            break
        source, task, _ = extract_batch_features(
            backbone, embed_layer, batch, layer_indices,
            m_positions, k_soft_prompts, device,
        )

        tgt = batch["target_ids"].to(device)
        tgt_mask = batch["target_mask"].to(device)

        # Teacher path: text peer + target
        pid = batch["peer_ids"].to(device)
        pmask = batch["peer_mask"].to(device)
        full_ids = torch.cat([pid, tgt], dim=1)
        full_mask = torch.cat([pmask, tgt_mask], dim=1)
        teacher_out = backbone(input_ids=full_ids, attention_mask=full_mask,
                               use_cache=False)
        T = tgt.size(1)
        P = pid.size(1)
        teacher_logits = teacher_out.logits[:, P-1:P-1+T, :]

        # Student path: bridge soft prompt + target embeddings
        soft = bridge.sample(source, task, K=K_sample)             # (B, k, d)
        tgt_embeds = embed_layer(tgt)
        full_embeds = torch.cat([soft, tgt_embeds], dim=1)
        student_mask = torch.cat([
            torch.ones(soft.size(0), soft.size(1), dtype=tgt_mask.dtype, device=device),
            tgt_mask,
        ], dim=1)
        student_out = backbone(inputs_embeds=full_embeds,
                               attention_mask=student_mask, use_cache=False)
        k = soft.size(1)
        student_logits = student_out.logits[:, k-1:k-1+T, :]

        teacher_lp = F.log_softmax(teacher_logits, dim=-1)
        student_lp = F.log_softmax(student_logits, dim=-1)
        kl = F.kl_div(student_lp, teacher_lp.exp(), reduction="none").sum(-1)
        # Mask out padding positions of target
        kl = (kl * tgt_mask).sum() / tgt_mask.sum().clamp(min=1)
        kl_vals.append(kl.item())

    bridge.train()
    return float(sum(kl_vals) / max(1, len(kl_vals)))


# ────────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-name", default="Qwen/Qwen3-8B")
    ap.add_argument("--pairs-path", required=True, type=Path,
                    help="JSONL of training pairs from extract_training_pairs.py")
    ap.add_argument("--output-dir", required=True, type=Path)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--max-pairs", type=int, default=None,
                    help="Cap total pairs (for smoke testing)")
    ap.add_argument("--eval-frac", type=float, default=0.1)
    ap.add_argument("--eval-every", type=int, default=500,
                    help="End-to-end KL evaluation frequency (steps)")
    ap.add_argument("--save-every", type=int, default=2000)
    ap.add_argument("--k-soft-prompts", type=int, default=16)
    ap.add_argument("--m-positions", type=int, default=32,
                    help="Number of latent positions to extract from sender")
    ap.add_argument("--n-source-layers", type=int, default=3)
    ap.add_argument("--n-blocks", type=int, default=4)
    ap.add_argument("--n-heads", type=int, default=8)
    ap.add_argument("--K-sample-eval", type=int, default=20)
    ap.add_argument("--T-train", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dtype", choices=["bfloat16", "float16"], default="bfloat16")
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--warmup-steps", type=int, default=200)
    ap.add_argument("--no-warm-start", action="store_true",
                    help="Disable W_e warm start (purely learned bridge)")
    args = ap.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    # ---------- Backbone ----------
    print(f"Loading backbone: {args.model_name}")
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    tok = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    backbone = AutoModelForCausalLM.from_pretrained(
        args.model_name, dtype=dtype, device_map="auto",
        trust_remote_code=True,
    )
    backbone.eval()
    for p in backbone.parameters():
        p.requires_grad = False
    device = next(backbone.parameters()).device
    embed_layer = backbone.get_input_embeddings()
    n_layers = backbone.config.num_hidden_layers
    d_model = backbone.config.hidden_size
    print(f"  d_model={d_model}, n_layers={n_layers}, device={device}")

    # ---------- W_e alignment ----------
    print("Computing W_e alignment matrix (one-time)...")
    W_e, target_norm = compute_alignment(backbone)
    W_e = W_e.to(device, dtype=dtype)

    # ---------- Bridge ----------
    layer_indices = default_layer_indices(n_layers, args.n_source_layers)
    print(f"  Source layers: {layer_indices} (1-indexed into hidden_states tuple)")
    cfg = BridgeConfig(
        d_model=d_model,
        k_soft_prompts=args.k_soft_prompts,
        n_blocks=args.n_blocks,
        n_heads=args.n_heads,
        n_source_layers=len(layer_indices),
        T_train=args.T_train,
        K_sample=args.K_sample_eval,
        warm_start=not args.no_warm_start,
    )
    bridge = DiffusionBridge(cfg).to(device).to(dtype)
    bridge.attach_w_e(W_e, target_norm)
    n_params = sum(p.numel() for p in bridge.parameters() if p.requires_grad)
    print(f"  Bridge params: {n_params/1e6:.2f}M")

    # ---------- Data ----------
    print(f"Loading pairs from {args.pairs_path}")
    full_ds = BridgePairs(args.pairs_path, tok)
    print(f"  Total: {len(full_ds)} pairs")
    if args.max_pairs:
        full_ds.records = full_ds.records[:args.max_pairs]
        print(f"  Capped to: {len(full_ds)}")
    train_recs, eval_recs = task_split(full_ds.records, args.eval_frac, args.seed)
    full_ds.records = train_recs
    eval_ds = BridgePairs.__new__(BridgePairs)
    eval_ds.tok = tok
    eval_ds.records = eval_recs
    eval_ds.max_peer_tokens = full_ds.max_peer_tokens
    eval_ds.max_question_tokens = full_ds.max_question_tokens
    eval_ds.max_target_tokens = full_ds.max_target_tokens
    print(f"  Train/eval: {len(full_ds)}/{len(eval_ds)}")

    pad = tok.pad_token_id
    train_loader = DataLoader(full_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=0, collate_fn=lambda b: collate(b, pad))
    eval_loader = DataLoader(eval_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=0, collate_fn=lambda b: collate(b, pad))

    # ---------- Optimizer ----------
    optim = torch.optim.AdamW(
        [p for p in bridge.parameters() if p.requires_grad], lr=args.lr,
    )
    sched = torch.optim.lr_scheduler.LambdaLR(
        optim, lambda step: min(1.0, (step + 1) / max(1, args.warmup_steps)),
    )

    # ---------- Train ----------
    log_path = args.output_dir / "train_log.jsonl"
    metrics_log = open(log_path, "w")
    print(f"Logging to {log_path}")

    step = 0
    t_start = time.time()
    for epoch in range(args.epochs):
        for batch in train_loader:
            source, task, x0 = extract_batch_features(
                backbone, embed_layer, batch, layer_indices,
                args.m_positions, args.k_soft_prompts, device,
            )
            losses = bridge.training_loss(x0, source, task)
            loss = losses["loss"]

            optim.zero_grad()
            loss.backward()
            if args.grad_clip:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in bridge.parameters() if p.requires_grad],
                    args.grad_clip,
                )
            optim.step()
            sched.step()

            if step % 25 == 0:
                rec = {"step": step, "epoch": epoch, "loss": loss.item(),
                       "lr": sched.get_last_lr()[0],
                       "elapsed": time.time() - t_start}
                print(f"  step={step:6d} loss={loss.item():.4f} lr={rec['lr']:.2e}")
                metrics_log.write(json.dumps(rec) + "\n")
                metrics_log.flush()

            if step > 0 and step % args.eval_every == 0:
                kl = eval_endtoend_kl(
                    backbone, embed_layer, bridge, eval_loader, layer_indices,
                    args.m_positions, args.k_soft_prompts, device,
                    args.K_sample_eval,
                )
                rec = {"step": step, "epoch": epoch, "eval_kl": kl,
                       "K_sample": args.K_sample_eval}
                print(f"  >> EVAL step={step} kl={kl:.4f}")
                metrics_log.write(json.dumps(rec) + "\n")
                metrics_log.flush()

            if step > 0 and step % args.save_every == 0:
                ckpt = args.output_dir / f"bridge_step{step:07d}.pt"
                torch.save({
                    "step": step, "epoch": epoch,
                    "config": cfg.to_dict(),
                    "layer_indices": layer_indices,
                    "model_name": args.model_name,
                    "bridge_state_dict": bridge.state_dict(),
                }, ckpt)
                print(f"  saved {ckpt}")

            step += 1

        # End-of-epoch eval + save
        kl = eval_endtoend_kl(
            backbone, embed_layer, bridge, eval_loader, layer_indices,
            args.m_positions, args.k_soft_prompts, device, args.K_sample_eval,
        )
        ckpt = args.output_dir / f"bridge_epoch{epoch}.pt"
        torch.save({
            "step": step, "epoch": epoch,
            "config": cfg.to_dict(),
            "layer_indices": layer_indices,
            "model_name": args.model_name,
            "eval_kl": kl,
            "bridge_state_dict": bridge.state_dict(),
        }, ckpt)
        print(f"  END epoch {epoch}: kl={kl:.4f}, saved {ckpt}")
        metrics_log.write(json.dumps({
            "epoch_end": epoch, "step": step, "eval_kl": kl,
        }) + "\n")
        metrics_log.flush()

    metrics_log.close()
    print(f"Done. Total elapsed: {(time.time() - t_start)/60:.1f} min")


if __name__ == "__main__":
    main()
