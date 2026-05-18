"""Phase B: train a latent encoder + adapter that compresses inter-agent
peer text into k continuous vectors, via KL distillation against the
text-conditioned receiver distribution.

Setup:
  - Frozen backbone: Qwen3.5-9B (same as the agents themselves).
  - Trainable encoder: k learned query vectors that cross-attend over the
    peer_text's final hidden states. Output z ∈ R^{k × d}.
  - Trainable adapter: identity by default (z used directly as input
    embeddings in B's prefill); optionally a small MLP.
  - Loss: KL(p_B(text) || p_B(latent)) computed over each token of the
    target response. Backbone is frozen; only encoder + adapter params receive
    gradient updates.

Skeleton scope:
  This file implements the dataset, model architecture, and training loop.
  It is runnable end-to-end on a small subset; for full-scale training it
  expects ~17GB VRAM (single A5000) and a few hours.

Usage (smoke-test on tiny subset):
  python train_latent_encoder.py --max-pairs 16 --epochs 1 --k 4 --batch-size 1

Full training:
  python train_latent_encoder.py --epochs 3 --k 8 --batch-size 2 \\
    --output-dir /atlas2/u/$USER/mas_project/mas-energy/results/latent_encoder

Required pip packages: torch, transformers, accelerate.
"""
import argparse
import json
import math
import os
import random
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM


def task_split(records, eval_frac, seed=42):
    """Split records into train/eval by unique task_id so the eval set
    contains entirely held-out tasks (not just held-out responses for tasks
    the model has already seen). Deterministic via fixed seed."""
    task_ids = sorted({r.get('task_id', '') for r in records if r.get('task_id')})
    rng = random.Random(seed)
    rng.shuffle(task_ids)
    n_eval = max(1, int(len(task_ids) * eval_frac))
    eval_set = set(task_ids[:n_eval])
    train_recs = [r for r in records if r.get('task_id') not in eval_set]
    eval_recs  = [r for r in records if r.get('task_id') in eval_set]
    return train_recs, eval_recs


# ────────────────────────────────────────────────────────────────────
# Dataset
# ────────────────────────────────────────────────────────────────────

class PeerTextPairs(Dataset):
    """Loads (peer_text, target_response) pairs from extract_training_pairs.py
    output. Tokenizes lazily."""

    def __init__(self, jsonl_path, tokenizer, max_peer_tokens=2048,
                 max_target_tokens=512):
        self.path = jsonl_path
        self.tokenizer = tokenizer
        self.max_peer_tokens = max_peer_tokens
        self.max_target_tokens = max_target_tokens
        self.records = []
        with open(jsonl_path) as f:
            for line in f:
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if r.get('peer_text') and r.get('target_response'):
                    self.records.append(r)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        r = self.records[idx]
        peer_ids = self.tokenizer(
            r['peer_text'], truncation=True,
            max_length=self.max_peer_tokens, return_tensors='pt',
        ).input_ids.squeeze(0)
        target_ids = self.tokenizer(
            r['target_response'], truncation=True,
            max_length=self.max_target_tokens, return_tensors='pt',
        ).input_ids.squeeze(0)
        return {
            'peer_ids': peer_ids,
            'target_ids': target_ids,
            'task_id': r.get('task_id', ''),
            'benchmark': r.get('benchmark', ''),
        }


def collate(batch, pad_id):
    """Pad to longest in batch on each side (peer / target separately)."""
    peer_lens = [b['peer_ids'].size(0) for b in batch]
    tgt_lens = [b['target_ids'].size(0) for b in batch]
    P = max(peer_lens); T = max(tgt_lens)
    bs = len(batch)
    peer_pad = torch.full((bs, P), pad_id, dtype=torch.long)
    peer_mask = torch.zeros((bs, P), dtype=torch.long)
    tgt_pad = torch.full((bs, T), pad_id, dtype=torch.long)
    tgt_mask = torch.zeros((bs, T), dtype=torch.long)
    for i, b in enumerate(batch):
        pl, tl = b['peer_ids'].size(0), b['target_ids'].size(0)
        peer_pad[i, :pl] = b['peer_ids']
        peer_mask[i, :pl] = 1
        tgt_pad[i, :tl] = b['target_ids']
        tgt_mask[i, :tl] = 1
    return {
        'peer_ids': peer_pad, 'peer_mask': peer_mask,
        'target_ids': tgt_pad, 'target_mask': tgt_mask,
    }


# ────────────────────────────────────────────────────────────────────
# Encoder + Adapter
# ────────────────────────────────────────────────────────────────────

class LatentEncoder(nn.Module):
    """k learned query vectors cross-attend over the peer-text final hidden
    states (from the frozen backbone) to produce z ∈ R^{k × d}.

    Trainable: query bank, cross-attention proj, optional MLP adapter.
    Frozen:    everything else (the backbone is held externally).
    """
    def __init__(self, d_model, k=8, n_heads=8, mlp_adapter=False):
        super().__init__()
        self.k = k
        self.d_model = d_model
        # k learned query tokens, initialized small to encourage drift from
        # zero rather than random scrambling
        self.query_bank = nn.Parameter(torch.randn(k, d_model) * 0.02)
        # Single multi-head cross-attention layer
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=n_heads, batch_first=True,
        )
        self.norm = nn.LayerNorm(d_model)
        if mlp_adapter:
            self.adapter = nn.Sequential(
                nn.Linear(d_model, d_model * 2),
                nn.GELU(),
                nn.Linear(d_model * 2, d_model),
            )
        else:
            self.adapter = nn.Identity()

    def forward(self, peer_hidden, peer_mask):
        """peer_hidden: (B, P, d) — frozen backbone's final hidden states for peer_text.
           peer_mask:   (B, P)    — 1 for real tokens, 0 for pad.
           returns z:   (B, k, d) — soft prompt vectors."""
        B = peer_hidden.size(0)
        q = self.query_bank.unsqueeze(0).expand(B, -1, -1)  # (B, k, d)
        # MultiheadAttention's key_padding_mask: True where pad
        kpm = (peer_mask == 0)
        z, _ = self.cross_attn(q, peer_hidden, peer_hidden, key_padding_mask=kpm)
        z = self.norm(z + q)  # residual
        z = self.adapter(z)
        return z


# ────────────────────────────────────────────────────────────────────
# Training loop
# ────────────────────────────────────────────────────────────────────

def get_peer_hiddens(backbone, peer_ids, peer_mask):
    """Run backbone on peer_text, return final hidden states. No grad."""
    with torch.no_grad():
        out = backbone(input_ids=peer_ids, attention_mask=peer_mask,
                       output_hidden_states=True, use_cache=False)
        return out.hidden_states[-1]  # (B, P, d)


def text_path_logits(backbone, peer_ids, peer_mask, target_ids, target_mask):
    """Teacher: receiver's distribution over target tokens given full peer_text."""
    B = peer_ids.size(0)
    full_ids = torch.cat([peer_ids, target_ids], dim=1)
    full_mask = torch.cat([peer_mask, target_mask], dim=1)
    with torch.no_grad():
        out = backbone(input_ids=full_ids, attention_mask=full_mask, use_cache=False)
    P = peer_ids.size(1)
    T = target_ids.size(1)
    # logits at each position predict the *next* token; for target token at
    # position P+t, the predicting position is P+t-1
    return out.logits[:, P-1:P-1+T, :]  # (B, T, V)


def latent_path_logits(backbone, embed_layer, latent_z, target_ids, target_mask):
    """Student: receiver's distribution over target tokens given latent z
    injected as input embeddings in place of the peer_text."""
    target_embeds = embed_layer(target_ids)  # (B, T, d)
    full_embeds = torch.cat([latent_z, target_embeds], dim=1)
    B, T = target_ids.size()
    k = latent_z.size(1)
    full_mask = torch.cat([
        torch.ones(B, k, dtype=target_mask.dtype, device=target_mask.device),
        target_mask,
    ], dim=1)
    out = backbone(inputs_embeds=full_embeds, attention_mask=full_mask, use_cache=False)
    return out.logits[:, k-1:k-1+T, :]  # (B, T, V)


def kl_loss(student_logits, teacher_log_probs, target_mask):
    """Token-level KL(teacher || student), masked to non-pad positions.

    teacher_log_probs is already log-softmaxed and detached, so the caller can
    free teacher_logits before running the student forward — keeps two
    full-vocab tensors from being alive concurrently. F.kl_div is fused and
    avoids materializing p_t * (log_p_t - log_p_s) as separate intermediates.
    """
    log_p_s = F.log_softmax(student_logits, dim=-1)
    kl_per_tok = F.kl_div(log_p_s, teacher_log_probs,
                          reduction='none', log_target=True).sum(dim=-1)
    mask = target_mask.float()
    return (kl_per_tok * mask).sum() / mask.sum().clamp(min=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--training-pairs',
                    default='mas-energy/results/latent_pilot/training_pairs.jsonl')
    ap.add_argument('--backbone', default='Qwen/Qwen3.5-9B')
    ap.add_argument('--output-dir',
                    default='mas-energy/results/latent_pilot/encoder_ckpt')
    ap.add_argument('--k', type=int, default=8, help='Number of latent vectors')
    ap.add_argument('--mlp-adapter', action='store_true',
                    help='Add 2-layer MLP after cross-attention pool')
    ap.add_argument('--n-heads', type=int, default=8)
    ap.add_argument('--epochs', type=int, default=1)
    ap.add_argument('--batch-size', type=int, default=1)
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--max-peer-tokens', type=int, default=2048,
                    help='Truncate peer_text to this many tokens. Drop to 1024 on A5000 to fit memory.')
    ap.add_argument('--max-target-tokens', type=int, default=512,
                    help='Truncate target_response. Drop to 256 on A5000 to fit memory.')
    ap.add_argument('--no-gradient-checkpointing', action='store_true',
                    help='Disable backbone gradient checkpointing. Faster (~25%) but uses '
                         'more memory. Safe to use on A6000 (48G); required to enable on A5000 (24G).')
    ap.add_argument('--eval-frac', type=float, default=0.1,
                    help='Fraction of unique task_ids held out as eval set. '
                         '0.0 disables validation entirely.')
    ap.add_argument('--eval-max-batches', type=int, default=200,
                    help='Cap the number of eval batches per epoch end (keeps eval cheap).')
    ap.add_argument('--max-pairs', type=int, default=None,
                    help='Cap dataset size for smoke tests')
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--dtype', default='bfloat16', choices=['float16', 'bfloat16', 'float32'])
    ap.add_argument('--log-every', type=int, default=10)
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading tokenizer + backbone ({args.backbone})...")
    tokenizer = AutoTokenizer.from_pretrained(args.backbone, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    dtype = {'float16': torch.float16, 'bfloat16': torch.bfloat16,
             'float32': torch.float32}[args.dtype]
    backbone = AutoModelForCausalLM.from_pretrained(
        args.backbone, torch_dtype=dtype, trust_remote_code=True,
    )
    backbone.eval()
    for p in backbone.parameters():
        p.requires_grad_(False)
    # Recompute backbone activations during backward pass instead of storing
    # them — required to fit Qwen3.5-9B + a forward pass with grad-tracking on
    # a single 24GB A5000. Skip on A6000 (48G) for ~25% faster training.
    if not args.no_gradient_checkpointing and hasattr(backbone, 'gradient_checkpointing_enable'):
        backbone.gradient_checkpointing_enable()
        print('Gradient checkpointing: ON (slower, lower memory).')
    else:
        print('Gradient checkpointing: OFF (faster, higher memory).')
    backbone = backbone.to(args.device)

    d_model = backbone.config.hidden_size
    embed_layer = backbone.get_input_embeddings()

    print(f"Loading training pairs from {args.training_pairs}...")
    ds = PeerTextPairs(args.training_pairs, tokenizer,
                       max_peer_tokens=args.max_peer_tokens,
                       max_target_tokens=args.max_target_tokens)
    if args.max_pairs:
        ds.records = ds.records[:args.max_pairs]
    print(f"  {len(ds)} pairs loaded (k={args.k}, d_model={d_model})")

    # Split by unique task_id so the eval set is entirely held-out tasks.
    if args.eval_frac > 0:
        train_recs, eval_recs = task_split(ds.records, args.eval_frac, seed=42)
        train_ds = PeerTextPairs.__new__(PeerTextPairs)
        train_ds.path = ds.path; train_ds.tokenizer = ds.tokenizer
        train_ds.max_peer_tokens = ds.max_peer_tokens
        train_ds.max_target_tokens = ds.max_target_tokens
        train_ds.records = train_recs
        eval_ds = PeerTextPairs.__new__(PeerTextPairs)
        eval_ds.path = ds.path; eval_ds.tokenizer = ds.tokenizer
        eval_ds.max_peer_tokens = ds.max_peer_tokens
        eval_ds.max_target_tokens = ds.max_target_tokens
        eval_ds.records = eval_recs
        n_train_tasks = len({r.get('task_id') for r in train_recs})
        n_eval_tasks = len({r.get('task_id') for r in eval_recs})
        print(f"  Split: {len(train_recs)} train pairs ({n_train_tasks} tasks), "
              f"{len(eval_recs)} eval pairs ({n_eval_tasks} held-out tasks)")
    else:
        train_ds = ds
        eval_ds = None

    loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        collate_fn=lambda b: collate(b, tokenizer.pad_token_id),
        num_workers=0,
    )
    eval_loader = None
    if eval_ds is not None and len(eval_ds) > 0:
        eval_loader = DataLoader(
            eval_ds, batch_size=args.batch_size, shuffle=False,
            collate_fn=lambda b: collate(b, tokenizer.pad_token_id),
            num_workers=0,
        )

    encoder = LatentEncoder(d_model=d_model, k=args.k, n_heads=args.n_heads,
                            mlp_adapter=args.mlp_adapter).to(args.device, dtype=dtype)
    n_trainable = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
    print(f"Encoder trainable params: {n_trainable:,}")

    optim = torch.optim.AdamW(encoder.parameters(), lr=args.lr)

    print("Starting training...")
    step = 0
    for epoch in range(args.epochs):
        for batch in loader:
            peer_ids = batch['peer_ids'].to(args.device)
            peer_mask = batch['peer_mask'].to(args.device)
            target_ids = batch['target_ids'].to(args.device)
            target_mask = batch['target_mask'].to(args.device)

            # Teacher: full-text path (no_grad inside text_path_logits already).
            # Convert to log-probs immediately so we can free the full
            # (B, T, V) teacher_logits tensor before doing the student forward,
            # which is the memory-heavy part (gradients flow through backbone).
            teacher_logits = text_path_logits(backbone, peer_ids, peer_mask,
                                              target_ids, target_mask)
            teacher_log_probs = F.log_softmax(teacher_logits.float(), dim=-1).detach()
            del teacher_logits
            torch.cuda.empty_cache()

            # Student: latent path
            peer_h = get_peer_hiddens(backbone, peer_ids, peer_mask)
            z = encoder(peer_h, peer_mask)
            student_logits = latent_path_logits(backbone, embed_layer, z,
                                                target_ids, target_mask)

            loss = kl_loss(student_logits, teacher_log_probs, target_mask)
            del teacher_log_probs, student_logits
            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(encoder.parameters(), 1.0)
            optim.step()

            if step % args.log_every == 0:
                print(f"  epoch {epoch} step {step}: kl={loss.item():.4f}")
            step += 1

        # ── End-of-epoch validation ──
        eval_kl = None
        if eval_loader is not None:
            encoder.eval()
            kl_sum = 0.0
            n_seen = 0
            with torch.no_grad():
                for ei, ebatch in enumerate(eval_loader):
                    if ei >= args.eval_max_batches:
                        break
                    e_peer_ids = ebatch['peer_ids'].to(args.device)
                    e_peer_mask = ebatch['peer_mask'].to(args.device)
                    e_tgt_ids = ebatch['target_ids'].to(args.device)
                    e_tgt_mask = ebatch['target_mask'].to(args.device)
                    e_teacher = text_path_logits(backbone, e_peer_ids, e_peer_mask,
                                                  e_tgt_ids, e_tgt_mask)
                    e_teacher_lp = F.log_softmax(e_teacher.float(), dim=-1).detach()
                    del e_teacher
                    e_peer_h = get_peer_hiddens(backbone, e_peer_ids, e_peer_mask)
                    e_z = encoder(e_peer_h, e_peer_mask)
                    e_student = latent_path_logits(backbone, embed_layer, e_z,
                                                    e_tgt_ids, e_tgt_mask)
                    e_kl = kl_loss(e_student, e_teacher_lp, e_tgt_mask)
                    bs = e_peer_ids.size(0)
                    kl_sum += float(e_kl) * bs
                    n_seen += bs
            eval_kl = kl_sum / max(1, n_seen)
            print(f"  epoch {epoch} EVAL: held-out kl={eval_kl:.4f}  (n_batches={ei+1}, n_pairs={n_seen})")
            encoder.train()

        # Save per-epoch checkpoint (include eval_kl for tracking)
        ckpt = out_dir / f"encoder_epoch{epoch}.pt"
        torch.save({
            'encoder_state_dict': encoder.state_dict(),
            'config': {
                'k': args.k, 'd_model': d_model, 'n_heads': args.n_heads,
                'mlp_adapter': args.mlp_adapter, 'backbone': args.backbone,
            },
            'epoch': epoch, 'step': step, 'eval_kl': eval_kl,
        }, ckpt)
        print(f"Saved {ckpt}")

    print("Training complete.")


if __name__ == "__main__":
    main()
