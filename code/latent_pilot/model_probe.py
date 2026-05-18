"""Load Qwen3.5-9B via HuggingFace and expose per-layer hooks for attention
vs. linear-attention (DeltaNet/Mamba-style) sublayers.

Qwen3.5-9B is a vision-language model with architecture
`Qwen3_5ForConditionalGeneration`. Its text decoder stack is found via
`_locate_decoder_layers` (actual path: `model.model.layers`). Each layer has
EITHER `.self_attn` (Qwen3_5Attention, softmax, 25% of layers at indices
3/7/11/15/19/23/27/31) OR `.linear_attn` (Qwen3_5GatedDeltaNet, 75% of
layers). Ground truth is `config.text_config.layer_types`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


MODEL_ID = "Qwen/Qwen3.5-9B"


@dataclass
class LayerInfo:
    index: int
    kind: str           # "softmax_attn" | "deltanet" | "other"
    attr_name: str      # "self_attn" | "linear_attn" — actual submodule attribute
    class_name: str     # for reporting
    cfg_layer_type: str # raw value from config.text_config.layer_types


@dataclass
class ProbedModel:
    model: torch.nn.Module
    tokenizer: object
    config: object
    decoder_layers: torch.nn.Module       # the ModuleList of decoder blocks
    decoder_path: str                      # dotted path to decoder_layers (for reporting)
    layers: list[LayerInfo] = field(default_factory=list)
    _hook_handles: list = field(default_factory=list)

    @property
    def device(self) -> str:
        return next(self.model.parameters()).device

    def get_sublayer(self, index: int) -> torch.nn.Module:
        """Resolve the attention or linear-attention submodule for a given layer index."""
        li = self.layers[index]
        return getattr(self.decoder_layers[li.index], li.attr_name)

    def attn_layer_indices(self) -> list[int]:
        return [li.index for li in self.layers if li.kind == "softmax_attn"]

    def deltanet_layer_indices(self) -> list[int]:
        return [li.index for li in self.layers if li.kind == "deltanet"]

    def clear_hooks(self) -> None:
        for h in self._hook_handles:
            h.remove()
        self._hook_handles.clear()


def _locate_decoder_layers(model: torch.nn.Module) -> tuple[torch.nn.Module, str]:
    """Walk common HF attribute paths to find the decoder-layer ModuleList."""
    candidates = [
        ("model", "language_model", "layers"),
        ("language_model", "layers"),
        ("model", "layers"),              # legacy single-modal
        ("transformer", "h"),              # GPT-2 style
    ]
    for path in candidates:
        obj = model
        ok = True
        for attr in path:
            obj = getattr(obj, attr, None)
            if obj is None:
                ok = False
                break
        if ok and hasattr(obj, "__len__") and len(obj) > 0:
            return obj, "model." + ".".join(path)
    raise RuntimeError(
        "Could not locate decoder layer stack; tried: "
        + ", ".join("model." + ".".join(p) for p in candidates)
    )


def _classify_from_cfg(cfg_type: str) -> str:
    t = (cfg_type or "").lower()
    if t in ("full_attention", "attention", "softmax_attention"):
        return "softmax_attn"
    if t in ("linear_attention", "deltanet", "gated_deltanet", "mamba", "ssm"):
        return "deltanet"
    return "other"


def _pick_sublayer_attr(block: torch.nn.Module, kind: str) -> str:
    """Find which attribute name holds the attention (or linear-attn) submodule."""
    if kind == "softmax_attn":
        for cand in ("self_attn", "attn", "attention"):
            if hasattr(block, cand):
                return cand
    elif kind == "deltanet":
        for cand in ("linear_attn", "deltanet", "mixer", "ssm"):
            if hasattr(block, cand):
                return cand
    # Last resort: whichever of the two families is present
    for cand in ("self_attn", "linear_attn", "attn", "mixer"):
        if hasattr(block, cand):
            return cand
    raise RuntimeError(f"Block {block.__class__.__name__} has no recognized attention submodule")


def load_model(model_id: str = MODEL_ID, dtype: str = "bfloat16") -> ProbedModel:
    torch_dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16}[dtype]

    print(f"Loading config for {model_id}...")
    config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
    text_cfg = getattr(config, "text_config", config)
    layer_types = getattr(text_cfg, "layer_types", None)
    if layer_types is not None:
        print(f"config.text_config.layer_types: {len(layer_types)} entries "
              f"({layer_types.count('full_attention')} full_attention, "
              f"{layer_types.count('linear_attention')} linear_attention)")
    else:
        print("WARNING: config has no layer_types; will classify from module class names")

    print(f"Loading tokenizer...")
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

    # Prefer AutoModelForCausalLM (handles VLMs with causal-LM head); fall back to
    # generic AutoModel if the architecture class isn't registered.
    try:
        from transformers import AutoModel  # noqa: F401  (imported for fallback)
        print(f"Loading model weights (dtype={dtype})...")
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            dtype=torch_dtype,
            device_map="auto",
            trust_remote_code=True,
        )
    except (ValueError, KeyError) as e:
        print(f"AutoModelForCausalLM failed ({e}); falling back to AutoModel")
        from transformers import AutoModel
        model = AutoModel.from_pretrained(
            model_id,
            dtype=torch_dtype,
            device_map="auto",
            trust_remote_code=True,
        )
    model.eval()

    decoder_layers, decoder_path = _locate_decoder_layers(model)
    print(f"Decoder stack at {decoder_path}, len={len(decoder_layers)}")

    layers: list[LayerInfo] = []
    for i, block in enumerate(decoder_layers):
        cfg_type = layer_types[i] if (layer_types is not None and i < len(layer_types)) else ""
        kind = _classify_from_cfg(cfg_type)

        # If config didn't tell us, infer from which submodule is present
        if kind == "other":
            if hasattr(block, "self_attn"):
                kind = "softmax_attn"
            elif hasattr(block, "linear_attn") or hasattr(block, "mixer"):
                kind = "deltanet"

        attr_name = _pick_sublayer_attr(block, kind)
        sub = getattr(block, attr_name)
        layers.append(LayerInfo(
            index=i,
            kind=kind,
            attr_name=attr_name,
            class_name=sub.__class__.__name__,
            cfg_layer_type=cfg_type,
        ))

    n_attn = sum(1 for li in layers if li.kind == "softmax_attn")
    n_dn = sum(1 for li in layers if li.kind == "deltanet")
    n_other = sum(1 for li in layers if li.kind == "other")
    print(f"Classified {len(layers)} layers: "
          f"{n_attn} softmax_attn, {n_dn} deltanet, {n_other} other")

    return ProbedModel(
        model=model,
        tokenizer=tok,
        config=config,
        decoder_layers=decoder_layers,
        decoder_path=decoder_path,
        layers=layers,
    )


def register_output_zero_hooks(probed: ProbedModel, kind: str) -> None:
    """Zero the output of every sublayer matching `kind`. Residual stream
    still flows through the block (MLP + other-kind layer), so an ablation
    that preserves prediction quality means this pathway was redundant.
    """
    targets = [li for li in probed.layers if li.kind == kind]
    if not targets:
        print(f"[hooks] no layers with kind={kind}")
        return

    def _make_hook():
        def hook(module, inputs, output):
            # HF attention submodules return either a bare tensor or a tuple
            # whose first element is the hidden-state contribution.
            if isinstance(output, tuple):
                zeroed = torch.zeros_like(output[0])
                return (zeroed,) + output[1:]
            return torch.zeros_like(output)
        return hook

    for li in targets:
        sub = probed.get_sublayer(li.index)
        handle = sub.register_forward_hook(_make_hook())
        probed._hook_handles.append(handle)
    print(f"[hooks] registered {len(targets)} zero-output hooks on kind={kind}")


def capture_sequence_logits(
    probed: ProbedModel,
    input_ids: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Forward pass; return logits [batch, seq, vocab]. Text-only inputs for
    the VLM — Qwen3_5 handles text-only when no vision tokens are present.

    Robust to both CausalLM-style output (has `.logits`) and bare-AutoModel
    output (has `.last_hidden_state`, in which case we apply lm_head).
    """
    dev = probed.device
    with torch.no_grad():
        out = probed.model(
            input_ids=input_ids.to(dev),
            attention_mask=attention_mask.to(dev) if attention_mask is not None else None,
            use_cache=False,
            return_dict=True,
        )
    if getattr(out, "logits", None) is not None:
        return out.logits.float().cpu()
    # Fallback: apply lm_head manually
    hidden = getattr(out, "last_hidden_state", None)
    if hidden is None:
        raise RuntimeError(f"Model output has neither logits nor last_hidden_state; fields: {list(out.keys())}")
    lm_head = getattr(probed.model, "lm_head", None)
    if lm_head is None:
        # Some VLMs expose lm_head under the language_model submodule
        for path in (("language_model", "lm_head"), ("model", "lm_head")):
            obj = probed.model
            for attr in path:
                obj = getattr(obj, attr, None)
                if obj is None: break
            if obj is not None:
                lm_head = obj
                break
    if lm_head is None:
        raise RuntimeError("Could not locate lm_head to produce logits from hidden states")
    with torch.no_grad():
        logits = lm_head(hidden)
    return logits.float().cpu()
