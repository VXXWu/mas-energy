"""Shared helpers for latent_pilot tests.

Task loading, metric computation, result I/O. Kept minimal so each test
file stays self-contained for later debugging on cluster.
"""
from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Iterator


def results_root() -> Path:
    """Canonical results dir for latent pilot on cluster."""
    user = os.environ.get("USER", "unknown")
    return Path(f"/atlas2/u/{user}/mas_project/mas-energy/results/latent_pilot")


def save_json(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(obj, f, indent=2, default=str)
    print(f"Wrote {path}")


def append_jsonl(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(obj, default=str) + "\n")


def iter_jsonl(path: Path) -> Iterator[dict]:
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def load_existing_traces(benchmark: str, n: int, seed: int = 42) -> list[dict]:
    """Pull n prompt/response records from existing Qwen3.5-9B SAS or Decentralized runs.

    Searches the canonical results dirs. We only need (prompt, response) pairs
    where transcripts were saved (SAVE_TRANSCRIPTS=1). Falls back to fresh
    prompts if none are found.
    """
    user = os.environ.get("USER", "unknown")
    base = Path(f"/atlas2/u/{user}/mas_project/mas-energy/results")
    patterns = {
        "qampari": ["a5000_qampari_v4", "a5000_transcripts_qampari", "qampari_v4"],
        "workbench": ["a5000_workbench_v2", "workbench_v2", "workbench_v1"],
    }
    candidate_dirs = [base / d for d in patterns.get(benchmark, [])]
    candidate_files: list[Path] = []
    for d in candidate_dirs:
        if d.exists():
            candidate_files.extend(sorted(d.glob("**/*.jsonl")))

    recs = []
    for f in candidate_files:
        try:
            for rec in iter_jsonl(f):
                msg = rec.get("messages") or rec.get("request_messages") or rec.get("prompt")
                resp = rec.get("response") or rec.get("completion") or rec.get("output")
                if msg and resp:
                    recs.append({"messages": msg, "response": resp, "source": str(f)})
                    if len(recs) >= n * 4:  # collect 4x budget, subsample later
                        break
        except Exception as e:
            print(f"Skipping {f}: {e}")
            continue
        if len(recs) >= n * 4:
            break

    if not recs:
        print(f"WARNING: no cached traces found for {benchmark}; tests will need fresh generation.")
        return []

    rng = random.Random(seed)
    rng.shuffle(recs)
    return recs[:n]


def top1_agreement(logits_a, logits_b) -> float:
    """Fraction of positions where argmax(logits_a) == argmax(logits_b).

    Accepts torch tensors or numpy arrays of shape [batch, seq, vocab] or
    [seq, vocab]. Returns a scalar in [0, 1].
    """
    import torch

    if not isinstance(logits_a, torch.Tensor):
        logits_a = torch.as_tensor(logits_a)
        logits_b = torch.as_tensor(logits_b)
    a = logits_a.argmax(dim=-1)
    b = logits_b.argmax(dim=-1)
    return (a == b).float().mean().item()
