"""Validate mas-energy env for the latent_pilot work.

Confirms torch + torch._dynamo + sympy + transformers all import. The
latent pilot uses HuggingFace transformers directly (not sglang), so sglang
is not checked here — it's validated implicitly when its own sbatch scripts
launch it on a GPU node. Running this validator on the login node is safe.

Run:
  conda activate mas-energy
  python mas-energy/scripts/validate_env.py

Exit 0 if all imports succeed, 1 otherwise. Prints per-step progress so a
hang can be localized.
"""
import sys
import time


def step(name: str):
    print(f"[{time.strftime('%H:%M:%S')}] {name} ...", flush=True)


def main() -> int:
    t0 = time.time()
    try:
        step("import torch")
        import torch
        step(f"  torch {torch.__version__} ok")

        step("import torch._dynamo (needs sympy)")
        import torch._dynamo  # noqa: F401
        step("  dynamo ok")

        step("import sympy")
        import sympy
        step(f"  sympy {sympy.__version__} ok")

        step("import transformers.AutoModelForCausalLM + AutoTokenizer + AutoConfig")
        from transformers import (  # noqa: F401
            AutoConfig,
            AutoModelForCausalLM,
            AutoTokenizer,
        )
        step("  transformers ok")

    except Exception as e:
        print(f"FAILED: {type(e).__name__}: {e}", flush=True)
        return 1

    print(f"VALIDATED: torch={torch.__version__} sympy={sympy.__version__} "
          f"({time.time()-t0:.1f}s)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
