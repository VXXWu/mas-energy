"""Summarize the model_structure.json produced by latent_pilot.inspect_model.

Prints cache class, attention module shape info, and RoPE config — the
three pieces needed before filling in Test 2's ARCH_SPECIFIC sections.

Run:
  python mas-energy/scripts/summarize_model_structure.py
"""
import json
import os
from pathlib import Path


def main() -> int:
    user = os.environ.get("USER", "unknown")
    path = Path(f"/atlas2/u/{user}/mas_project/mas-energy/results/latent_pilot/model_structure.json")
    if not path.exists():
        print(f"not found: {path}")
        print("run latent_pilot_inspect.sbatch first.")
        return 1

    d = json.load(open(path))
    ci = d.get("cache_info", {})

    print(f"model: {d.get('model_id')}")
    print(f"decoder_path: {d.get('decoder_path')}")
    print(f"n_layers: {d.get('n_layers')} "
          f"({d.get('n_softmax_attn')} softmax_attn, {d.get('n_deltanet')} deltanet)")
    print(f"softmax_attn_indices: {d.get('softmax_attn_indices')}")
    print()
    print(f"cache_class: {ci.get('cache_class')}")
    print(f"cache_attrs: {ci.get('cache_attrs')}")
    for key in ("key_cache_len", "value_cache_len",
                "conv_states_len", "ssm_states_len",
                "linear_attn_states_len", "recurrent_states_len",
                "get_seq_length"):
        if key in ci:
            print(f"{key}: {ci[key]}")
    print()

    shapes = ci.get("per_layer_key_shapes") or []
    if shapes:
        print("per-layer key cache shapes (first 4 + last 1):")
        for row in shapes[:4] + ([shapes[-1]] if len(shapes) > 4 else []):
            print(f"  layer {row.get('layer')}: shape={row.get('shape')} dtype={row.get('dtype')}")
    print()

    attn = d.get("sample_attn_module") or {}
    print(f"sample attention module class: {attn.get('class_name')}")
    print("  param shapes:")
    for k, v in (attn.get("param_shapes") or {}).items():
        print(f"    {k}: {v}")
    print()

    dn = d.get("sample_deltanet_module") or {}
    print(f"sample deltanet module class: {dn.get('class_name')}")
    dn_params = dn.get("param_shapes") or {}
    print(f"  ({len(dn_params)} parameters; showing up to 10)")
    for k, v in list(dn_params.items())[:10]:
        print(f"    {k}: {v}")
    print()

    print(f"rope_info: {d.get('rope_info')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
