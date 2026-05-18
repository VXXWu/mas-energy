"""Compare eval KL across encoder checkpoints from the k-sweep.

Walks results/latent_encoder/k{K}_epochs{N}/encoder_epoch{i}.pt for all K, N, i
and reports the held-out KL stored in each checkpoint. Plots KL-vs-k to find
the bandwidth-vs-fidelity frontier.

Usage:
  python mas-energy/code/latent_pilot/analyze_k_sweep.py
  python mas-energy/code/latent_pilot/analyze_k_sweep.py --out-png k_sweep.png
"""
import argparse
import re
from pathlib import Path

import torch
try:
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


def parse_dir_name(name):
    """k8_epochs2 -> (k=8, epochs=2). Returns None if non-conforming."""
    m = re.match(r'k(\d+)_epochs(\d+)', name)
    if not m: return None
    return int(m.group(1)), int(m.group(2))


def parse_ckpt_name(name):
    m = re.match(r'encoder_epoch(\d+)\.pt', name)
    return int(m.group(1)) if m else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default='mas-energy/results/latent_encoder',
                    help='Directory containing kN_epochsM subdirs')
    ap.add_argument('--out-png', default='figures/k_sweep.png',
                    help='Output figure path')
    args = ap.parse_args()

    root = Path(args.root)
    if not root.exists():
        print(f"No checkpoint root at {root}")
        return

    rows = []
    for sub in sorted(root.iterdir()):
        if not sub.is_dir(): continue
        parsed = parse_dir_name(sub.name)
        if not parsed: continue
        k, epochs = parsed
        for ckpt in sorted(sub.glob('encoder_epoch*.pt')):
            ep = parse_ckpt_name(ckpt.name)
            if ep is None: continue
            try:
                d = torch.load(ckpt, map_location='cpu', weights_only=False)
            except Exception as e:
                print(f"  (could not load {ckpt}: {type(e).__name__})")
                continue
            rows.append({
                'k': k,
                'configured_epochs': epochs,
                'ckpt_epoch': ep,
                'step': d.get('step'),
                'eval_kl': d.get('eval_kl'),
                'mlp_adapter': d.get('config', {}).get('mlp_adapter'),
                'path': str(ckpt),
            })

    if not rows:
        print(f"No checkpoints found under {root}")
        return

    rows.sort(key=lambda r: (r['k'], r['ckpt_epoch']))
    print(f"{'k':>4} {'ep':>3} {'step':>6} {'eval_kl':>10} {'mlp':>5}  ckpt")
    print("-" * 78)
    for r in rows:
        ev = f"{r['eval_kl']:.4f}" if r['eval_kl'] is not None else "n/a"
        mlp = "yes" if r['mlp_adapter'] else "no"
        print(f"{r['k']:>4} {r['ckpt_epoch']:>3} {r['step'] or '?':>6} {ev:>10} {mlp:>5}  "
              f"{Path(r['path']).parent.name}/{Path(r['path']).name}")

    # Final-epoch KL per k for the frontier curve
    final = {}
    for r in rows:
        if r['eval_kl'] is None: continue
        prev = final.get(r['k'])
        if prev is None or r['ckpt_epoch'] > prev['ckpt_epoch']:
            final[r['k']] = r

    if final and HAS_MPL:
        ks = sorted(final)
        kls = [final[k]['eval_kl'] for k in ks]
        plt.figure(figsize=(7, 5))
        plt.plot(ks, kls, 'o-', linewidth=2, markersize=10, color='#2c7fb8')
        for k, v in zip(ks, kls):
            plt.annotate(f'{v:.3f}', (k, v), xytext=(8, 6),
                         textcoords='offset points', fontsize=9)
        plt.xscale('log', base=2)
        plt.xlabel('k (number of latent vectors)', fontsize=11)
        plt.ylabel('Held-out KL (nats/token)', fontsize=11)
        plt.title('Latent encoder: bandwidth-vs-fidelity frontier', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        out = Path(args.out_png)
        out.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out, dpi=150, bbox_inches='tight')
        print(f"\nSaved {out}")
    elif not HAS_MPL:
        print("\n(matplotlib not available; skipping plot)")


if __name__ == "__main__":
    main()
