"""Conditional diffusion bridge for inter-agent latent communication.

Sits between two agents in a hybrid latent-text MAS pipeline. The sender
runs m steps of latent CoT (last-layer hidden states, optionally with
multi-layer features extracted at the same positions). The bridge maps
those latent thoughts to k soft prompt vectors that the receiver consumes
at its input embedding layer.

Replaces LatentMAS's closed-form W_e linear projection with a conditional
DDPM. Warm-started from W_e so K=0 sampling steps approximately recovers
the LatentMAS baseline; K>0 refines on top.

Architecture:
    Source: (B, m, n_layers, d) — m latent thoughts at n_layers selected layers
    Task condition: (B, d) — receiver's last-layer pooled state of the question
    Output: (B, k, d) — soft prompt vectors in receiver's input-embedding space

Training objective: ε-prediction (DDPM) with cross-attention conditioning on
source + task. KL distillation against the text-conditioned receiver
distribution is applied separately in train_diffusion_bridge.py.

The W_e warm-start is implemented as an additive skip path: at K=0 the model
returns h_last @ W_e (LatentMAS), and learned residual refinements compose on
top during training. This makes the worst case "approximately what LatentMAS
does" rather than random output.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


# ────────────────────────────────────────────────────────────────────
# Noise schedule (cosine, Nichol & Dhariwal 2021)
# ────────────────────────────────────────────────────────────────────

class CosineNoiseSchedule:
    """Cosine β_t schedule. Provides forward q(x_t | x_0) and reverse-step
    coefficients for DDIM-style deterministic sampling.

    All tensors live on CPU; index with .to(device) when needed in a forward
    pass. T (training timesteps) defaults to 1000; sampling can use any K ≤ T
    via uniform spacing along [0, T-1]."""

    def __init__(self, T: int = 1000, s: float = 0.008):
        self.T = T
        steps = torch.arange(T + 1, dtype=torch.float64) / T
        f = torch.cos((steps + s) / (1 + s) * math.pi / 2) ** 2
        alpha_bar = (f / f[0]).clamp(min=1e-8, max=1.0)
        self.alpha_bar = alpha_bar[1:].float()              # (T,)
        self.alpha_bar_prev = alpha_bar[:-1].float()        # (T,)
        self.beta = (1.0 - self.alpha_bar / self.alpha_bar_prev).clamp(0, 0.999)
        self.alpha = 1.0 - self.beta

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        ab = self.alpha_bar.to(x0.device)[t].view(-1, *([1] * (x0.dim() - 1)))
        return ab.sqrt() * x0 + (1 - ab).sqrt() * noise

    def sampling_timesteps(self, K: int) -> torch.Tensor:
        """Uniform spacing of K steps along [0, T-1] for DDIM-like sampling.
        Returned in descending order (T-1 → 0) for the reverse loop."""
        if K <= 0:
            return torch.empty(0, dtype=torch.long)
        idx = torch.linspace(0, self.T - 1, K).round().long()
        return idx.flip(0)


# ────────────────────────────────────────────────────────────────────
# Sinusoidal timestep embedding
# ────────────────────────────────────────────────────────────────────

class TimestepEmbed(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model
        self.proj = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.SiLU(),
            nn.Linear(d_model * 4, d_model),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # t: (B,) integer timestep indices
        half = self.d_model // 2
        freqs = torch.exp(
            -math.log(10000.0) *
            torch.arange(half, device=t.device, dtype=torch.float32) / half
        )
        args = t.float()[:, None] * freqs[None]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if emb.size(-1) < self.d_model:
            emb = F.pad(emb, (0, self.d_model - emb.size(-1)))
        return self.proj(emb)


# ────────────────────────────────────────────────────────────────────
# Conditional transformer block (denoiser core)
# ────────────────────────────────────────────────────────────────────

class CondTransformerBlock(nn.Module):
    """Pre-norm self-attention + cross-attention to source/task condition + MLP.
    AdaLN-zero modulation from timestep + task vector (DiT-style)."""

    def __init__(self, d: int, n_heads: int, d_ff: int = None):
        super().__init__()
        d_ff = d_ff or 4 * d
        self.norm1 = nn.LayerNorm(d, elementwise_affine=False)
        self.self_attn = nn.MultiheadAttention(d, n_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(d, elementwise_affine=False)
        self.cross_attn = nn.MultiheadAttention(d, n_heads, batch_first=True)
        self.norm3 = nn.LayerNorm(d, elementwise_affine=False)
        self.mlp = nn.Sequential(
            nn.Linear(d, d_ff), nn.GELU(), nn.Linear(d_ff, d),
        )
        # AdaLN-zero: 9 modulation channels (3 each for self/cross/mlp)
        self.adaLN = nn.Sequential(
            nn.SiLU(), nn.Linear(d, 9 * d, bias=True),
        )
        nn.init.zeros_(self.adaLN[-1].weight)
        nn.init.zeros_(self.adaLN[-1].bias)

    def _mod(self, x, scale, shift):
        return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

    def forward(self, x: torch.Tensor, cond_seq: torch.Tensor,
                cond_vec: torch.Tensor, cond_mask: torch.Tensor = None) -> torch.Tensor:
        # x: (B, k, d) noisy soft prompts
        # cond_seq: (B, S, d) source + task tokens for cross-attention
        # cond_vec: (B, d) timestep+task summary for AdaLN modulation
        params = self.adaLN(cond_vec).chunk(9, dim=-1)
        sa_s, sa_sh, sa_g, ca_s, ca_sh, ca_g, mlp_s, mlp_sh, mlp_g = params

        h = self._mod(self.norm1(x), sa_s, sa_sh)
        h, _ = self.self_attn(h, h, h, need_weights=False)
        x = x + sa_g.unsqueeze(1) * h

        h = self._mod(self.norm2(x), ca_s, ca_sh)
        h, _ = self.cross_attn(h, cond_seq, cond_seq,
                               key_padding_mask=cond_mask, need_weights=False)
        x = x + ca_g.unsqueeze(1) * h

        h = self._mod(self.norm3(x), mlp_s, mlp_sh)
        h = self.mlp(h)
        x = x + mlp_g.unsqueeze(1) * h
        return x


# ────────────────────────────────────────────────────────────────────
# DiffusionBridge: the full module
# ────────────────────────────────────────────────────────────────────

@dataclass
class BridgeConfig:
    d_model: int                  # backbone hidden dim
    k_soft_prompts: int = 16      # output sequence length
    n_blocks: int = 4             # number of transformer blocks
    n_heads: int = 8
    n_source_layers: int = 3      # how many backbone layers we condition on
    T_train: int = 1000           # training noise steps
    K_sample: int = 20            # default sampling steps
    target_norm_match: bool = True  # renormalize output to match input embed L2 norm
    warm_start: bool = True       # add W_e skip path so K=0 ≈ LatentMAS

    def to_dict(self):
        return {k: getattr(self, k) for k in self.__dataclass_fields__}


class DiffusionBridge(nn.Module):
    """Conditional DDPM. Source = sender's m latent thoughts at n_source_layers
    layers. Task condition = receiver's pooled question state. Output = k soft
    prompt vectors in receiver's input-embedding space.

    W_e warm-start: register W_e as a non-trainable buffer; final output =
    skip(h_last @ W_e) + denoised_residual. At init the residual MLP head is
    zero, so output strictly equals the W_e baseline.
    """

    def __init__(self, cfg: BridgeConfig):
        super().__init__()
        self.cfg = cfg
        d = cfg.d_model

        # Project source layer slices into the bridge's working space.
        # We tag each layer with a learned layer embedding so cross-attention
        # can disambiguate which layer a token came from.
        self.source_layer_embed = nn.Parameter(
            torch.randn(cfg.n_source_layers, d) * 0.02
        )
        self.task_token = nn.Parameter(torch.randn(d) * 0.02)
        # Position embedding on the m latent positions (max 256 latent steps)
        self.source_pos_embed = nn.Parameter(torch.randn(256, d) * 0.02)
        # Position embedding on the k output soft prompts
        self.output_pos_embed = nn.Parameter(torch.randn(cfg.k_soft_prompts, d) * 0.02)

        # Learned start-of-sequence x_T (queries for cross-attention from noise)
        self.x_init = nn.Parameter(torch.randn(cfg.k_soft_prompts, d) * 0.02)

        self.t_embed = TimestepEmbed(d)
        self.task_proj = nn.Linear(d, d)

        self.blocks = nn.ModuleList([
            CondTransformerBlock(d, cfg.n_heads) for _ in range(cfg.n_blocks)
        ])

        # Final residual head (zero-initialized for the W_e warm start)
        self.head = nn.Sequential(
            nn.LayerNorm(d),
            nn.Linear(d, d),
        )
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

        self.schedule = CosineNoiseSchedule(T=cfg.T_train)

        # Buffers populated by `attach_w_e()`. Non-trainable.
        self.register_buffer("W_e", torch.zeros(d, d), persistent=False)
        self.register_buffer("target_norm", torch.tensor(0.0), persistent=False)
        self._w_e_attached = False

    # ----------------------------------------------------------------
    # External setup
    # ----------------------------------------------------------------

    def attach_w_e(self, W_e: torch.Tensor, target_norm: float):
        """Inject the closed-form alignment matrix as the warm-start skip path.
        Call once after `compute_alignment(model)` in latentmas_baseline.

        W_e: (d, d) ridge-regression alignment.
        target_norm: scalar mean L2 norm of input embeddings.
        """
        assert W_e.shape == (self.cfg.d_model, self.cfg.d_model), \
            f"W_e shape {W_e.shape} does not match d_model {self.cfg.d_model}"
        self.W_e = W_e.to(self.W_e.device, dtype=self.W_e.dtype).detach()
        self.target_norm = torch.tensor(float(target_norm),
                                        device=self.target_norm.device,
                                        dtype=self.target_norm.dtype)
        self._w_e_attached = True

    # ----------------------------------------------------------------
    # Forward pieces
    # ----------------------------------------------------------------

    def _build_condition(self, source: torch.Tensor, task: torch.Tensor):
        """Pack source layers + task token into one cross-attention sequence.

        source: (B, m, L, d) — m latent thoughts at L source layers.
        task:   (B, d)        — task representation.

        Returns: cond_seq (B, m·L+1, d), cond_mask (B, m·L+1) bool padding mask.
        """
        B, m, L, d = source.shape
        assert L == self.cfg.n_source_layers, \
            f"source has {L} layers, expected {self.cfg.n_source_layers}"
        assert m <= self.source_pos_embed.size(0), \
            f"source length {m} exceeds max {self.source_pos_embed.size(0)}"

        # Add layer + position embeddings
        layer_emb = self.source_layer_embed.view(1, 1, L, d)        # (1,1,L,d)
        pos_emb = self.source_pos_embed[:m].view(1, m, 1, d)         # (1,m,1,d)
        src = source + layer_emb + pos_emb
        src = src.reshape(B, m * L, d)

        task_token = (task + self.task_token).unsqueeze(1)           # (B,1,d)
        cond_seq = torch.cat([task_token, src], dim=1)               # (B, m·L+1, d)
        cond_mask = torch.zeros(B, cond_seq.size(1), dtype=torch.bool,
                                device=cond_seq.device)
        return cond_seq, cond_mask

    def denoise(self, x_t: torch.Tensor, t: torch.Tensor,
                source: torch.Tensor, task: torch.Tensor) -> torch.Tensor:
        """Predict ε from noisy x_t. Output shape matches x_t (B, k, d)."""
        cond_seq, cond_mask = self._build_condition(source, task)
        cond_vec = self.t_embed(t) + self.task_proj(task)            # (B, d)

        # Add output position embedding to noisy queries
        x = x_t + self.output_pos_embed.unsqueeze(0)
        for blk in self.blocks:
            x = blk(x, cond_seq, cond_vec, cond_mask=cond_mask)
        return self.head(x)

    # ----------------------------------------------------------------
    # Warm-start: W_e baseline path
    # ----------------------------------------------------------------

    def w_e_baseline(self, source: torch.Tensor) -> torch.Tensor:
        """Apply LatentMAS's closed-form alignment to the LAST source layer's
        last k positions. Returns (B, k, d) in input-embedding space.

        If source has fewer than k positions, repeats the last position to fill.
        """
        if not self._w_e_attached:
            raise RuntimeError("W_e not attached. Call attach_w_e() first.")

        B, m, L, d = source.shape
        h = source[:, :, -1, :]  # (B, m, d) — last layer of source
        if m < self.cfg.k_soft_prompts:
            # Repeat the last latent position to fill k
            pad = h[:, -1:, :].expand(B, self.cfg.k_soft_prompts - m, d)
            h = torch.cat([h, pad], dim=1)
        elif m > self.cfg.k_soft_prompts:
            # Take last k positions (most recent thoughts)
            h = h[:, -self.cfg.k_soft_prompts:, :]

        # h @ W_e, then renormalize to target_norm (matches LatentMAS apply_alignment)
        h_f = h.float()
        e = h_f @ self.W_e.float()
        if self.cfg.target_norm_match:
            n = e.norm(dim=-1, keepdim=True).clamp(min=1e-8)
            e = e * (self.target_norm.float() / n)
        return e.to(source.dtype)

    # ----------------------------------------------------------------
    # Training: noise prediction loss
    # ----------------------------------------------------------------

    def training_loss(self, x0: torch.Tensor, source: torch.Tensor,
                      task: torch.Tensor) -> dict:
        """Standard DDPM ε-prediction loss on the *residual* from W_e baseline.

        Target is the residual x0 - w_e_baseline(source); the model learns to
        produce that residual via diffusion. At zero loss, output = x0.
        """
        B = x0.size(0)
        t = torch.randint(0, self.cfg.T_train, (B,), device=x0.device)
        baseline = self.w_e_baseline(source) if self.cfg.warm_start else 0.0
        residual_target = x0 - baseline

        noise = torch.randn_like(residual_target)
        x_t = self.schedule.q_sample(residual_target, t, noise)

        noise_pred = self.denoise(x_t, t, source, task)
        loss = F.mse_loss(noise_pred, noise)
        return {"loss": loss, "noise_pred": noise_pred, "noise_target": noise}

    # ----------------------------------------------------------------
    # Inference: K-step DDIM sampling
    # ----------------------------------------------------------------

    @torch.no_grad()
    def sample(self, source: torch.Tensor, task: torch.Tensor,
               K: int = None, generator: torch.Generator = None) -> torch.Tensor:
        """Sample k soft prompt vectors. K=0 returns the W_e baseline directly.

        DDIM deterministic sampling. With K=20 (default) this typically
        produces stable results; set K higher for quality, lower for speed.
        """
        if K is None:
            K = self.cfg.K_sample
        B = source.size(0)
        d = self.cfg.d_model
        device = source.device
        baseline = self.w_e_baseline(source) if self.cfg.warm_start else \
            torch.zeros(B, self.cfg.k_soft_prompts, d, device=device, dtype=source.dtype)

        if K == 0:
            return baseline

        # Initialize residual from noise
        if generator is None:
            x = torch.randn(B, self.cfg.k_soft_prompts, d, device=device, dtype=source.dtype)
        else:
            x = torch.randn(B, self.cfg.k_soft_prompts, d, device=device,
                            dtype=source.dtype, generator=generator)

        ts = self.schedule.sampling_timesteps(K).to(device)
        ab = self.schedule.alpha_bar.to(device)

        for i in range(len(ts)):
            t = ts[i]
            t_b = t.expand(B)
            eps = self.denoise(x, t_b, source, task).to(x.dtype)
            ab_t = ab[t]
            x0_pred = (x - (1 - ab_t).sqrt() * eps) / ab_t.sqrt().clamp(min=1e-8)
            if i + 1 < len(ts):
                t_next = ts[i + 1]
                ab_next = ab[t_next]
                x = ab_next.sqrt() * x0_pred + (1 - ab_next).sqrt() * eps
            else:
                x = x0_pred

        residual = x
        # baseline is already target_norm-normalized inside w_e_baseline().
        # x0 (input embeddings) has its own per-position norms, so
        # residual carries the norm-correction. Do NOT re-renormalize the
        # sum — that would distort the trained residual.
        return baseline + residual


# ────────────────────────────────────────────────────────────────────
# Convenience: extract multi-layer source from a backbone forward pass
# ────────────────────────────────────────────────────────────────────

def extract_source_layers(hidden_states: tuple, layer_indices: list,
                          positions: int = None) -> torch.Tensor:
    """Slice a HF transformers hidden_states tuple into the bridge's source
    tensor.

    hidden_states: tuple of length (n_layers + 1), each (B, T, d). Element 0
        is the embedding output; elements 1..L are the L transformer layers.
    layer_indices: list of layer indices to use (1-indexed into the tuple).
    positions: optional. If set, take the LAST `positions` token positions
        from each layer. If None, take all positions.

    Returns: (B, m, L, d) where m = positions (or T) and L = len(layer_indices).
    """
    slices = []
    for li in layer_indices:
        h = hidden_states[li]                              # (B, T, d)
        if positions is not None:
            h = h[:, -positions:, :]
        slices.append(h.unsqueeze(2))                      # (B, T, 1, d)
    out = torch.cat(slices, dim=2)                         # (B, T, L, d)
    return out


def default_layer_indices(n_layers: int, n_keep: int = 3) -> list:
    """Pick `n_keep` layer indices spread across the stack (early-mid-late).
    Indices are 1-based for the HF hidden_states tuple."""
    if n_keep == 1:
        return [n_layers]
    pts = torch.linspace(n_layers // 2, n_layers, n_keep).round().long().tolist()
    return list(dict.fromkeys(pts))  # dedupe while preserving order
