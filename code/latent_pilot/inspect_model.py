"""Step 0: inspect Qwen3.5-9B module structure.

Dumps layer types, attribute paths, and cache structure to inform
subsequent tests.

Outputs:
  {RESULTS}/model_structure.json
  stdout — human-readable summary
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
from latent_pilot.model_probe import load_model  # noqa: E402
from latent_pilot.utils import results_root, save_json  # noqa: E402


def describe_cache(probed) -> dict:
    """Run a short forward with use_cache=True and describe the cache object."""
    ids = probed.tokenizer("Hello, world.", return_tensors="pt").input_ids.to(probed.device)
    with torch.no_grad():
        out = probed.model(input_ids=ids, use_cache=True, return_dict=True)
    pkv = out.past_key_values
    info = {"cache_class": pkv.__class__.__name__}
    try:
        if hasattr(pkv, "key_cache"):
            info["key_cache_len"] = len(pkv.key_cache)
            info["per_layer_key_shapes"] = []
            for i, k in enumerate(pkv.key_cache):
                if k is not None and hasattr(k, "shape"):
                    info["per_layer_key_shapes"].append({"layer": i, "shape": list(k.shape), "dtype": str(k.dtype)})
                else:
                    info["per_layer_key_shapes"].append({"layer": i, "shape": None})
        if hasattr(pkv, "value_cache"):
            info["value_cache_len"] = len(pkv.value_cache)
        # Qwen3_5 / hybrid caches often expose additional state buffers
        for extra in ("conv_states", "ssm_states", "linear_attn_states", "recurrent_states"):
            if hasattr(pkv, extra):
                buf = getattr(pkv, extra)
                info[f"{extra}_len"] = len(buf) if hasattr(buf, "__len__") else "<scalar>"
        info["cache_attrs"] = sorted([a for a in dir(pkv) if not a.startswith("_")])[:40]
        if hasattr(pkv, "get_seq_length"):
            info["get_seq_length"] = pkv.get_seq_length()
    except Exception as e:
        info["describe_error"] = str(e)
    return info


def sample_module_attrs(probed, index: int) -> dict:
    sub = probed.get_sublayer(index)
    params = {}
    for n, p in sub.named_parameters():
        params[n] = list(p.shape)
    return {
        "layer_index": index,
        "class_name": sub.__class__.__name__,
        "public_attrs": sorted([a for a in dir(sub) if not a.startswith("_")])[:40],
        "param_shapes": params,
    }


def main():
    probed = load_model()

    rows = [{
        "index": li.index,
        "kind": li.kind,
        "attr_name": li.attr_name,
        "class_name": li.class_name,
        "cfg_layer_type": li.cfg_layer_type,
    } for li in probed.layers]

    attn_idx = probed.attn_layer_indices()
    dn_idx = probed.deltanet_layer_indices()

    attn_sample = sample_module_attrs(probed, attn_idx[0]) if attn_idx else None
    dn_sample = sample_module_attrs(probed, dn_idx[0]) if dn_idx else None

    cache_info = describe_cache(probed)

    # Surface RoPE config since Test 3 depends on it
    text_cfg = getattr(probed.config, "text_config", probed.config)
    rope_info = getattr(text_cfg, "rope_parameters", None) or getattr(text_cfg, "rope_scaling", None)

    report = {
        "model_id": "Qwen/Qwen3.5-9B",
        "architectures": getattr(probed.config, "architectures", []),
        "decoder_path": probed.decoder_path,
        "n_layers": len(probed.layers),
        "n_softmax_attn": len(attn_idx),
        "n_deltanet": len(dn_idx),
        "softmax_attn_indices": attn_idx,
        "deltanet_indices": dn_idx,
        "layers": rows,
        "sample_attn_module": attn_sample,
        "sample_deltanet_module": dn_sample,
        "cache_info": cache_info,
        "rope_info": str(rope_info),
    }

    out_path = results_root() / "model_structure.json"
    save_json(report, out_path)

    print("\n" + "=" * 60)
    print(f"Model: {report['model_id']} (arch: {report['architectures']})")
    print(f"Decoder path: {report['decoder_path']}")
    print(f"Layers: {report['n_layers']} "
          f"({report['n_softmax_attn']} softmax_attn, {report['n_deltanet']} deltanet)")
    print(f"Softmax-attn layer indices: {attn_idx}")
    print(f"Cache: {cache_info.get('cache_class')}")
    print(f"RoPE: {report['rope_info']}")
    print(f"\nFull report → {out_path}")


if __name__ == "__main__":
    main()
