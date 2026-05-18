"""Phase B prep: extract (peer_text, target_response) training pairs from
Decentralized transcripts for latent compression distillation.

Each Decentralized debate-round LLM call has structure:
    request_messages = [system, user_question, ..., debate_user_msg]
    response = B's output text

The "debate_user_msg" content is built by format_debate_prompt() and contains
the peer agents' summaries — exactly the text we want to compress into a small
latent vector z. The B response under that peer text is the target distribution
the latent-fed B should match (via KL distillation).

Output: jsonl with one record per debate-round LLM call:
    {
      "task_id": ...,
      "benchmark": ...,
      "agent_id": "debater_1_r0",
      "round": 0,
      "peer_text":   <the format_debate_prompt output that was fed to B>,
      "history_len_chars": <length of pre-debate context, for cost analysis>,
      "target_response": <B's full response>,
      "target_completion_tokens": <token count>,
    }

Use this jsonl to train an encoder (peer_text -> z) + adapter (z -> input_embeds)
with KL distillation loss against B's text-conditioned output distribution.
"""
import argparse
import json
import re
from pathlib import Path
from collections import Counter


DEBATE_PROMPT_MARKER = "These are the approaches and results from other agents:"


def extract_peer_text(request_messages):
    """Pull the format_debate_prompt() output out of the request messages.

    The debate prompt is the LAST user message and starts with the canonical
    marker string. Returns (peer_text, history_chars_before).
    """
    if not request_messages:
        return None, 0
    history_chars = 0
    for i, m in enumerate(request_messages):
        if (m.get('role') == 'user'
                and DEBATE_PROMPT_MARKER in (m.get('content') or '')):
            return m['content'], history_chars
        history_chars += len(m.get('content') or '')
    return None, history_chars


def is_debate_round_call(call_record):
    """True if this is a debate-round (not init, not synth) LLM call."""
    aid = (call_record.get('agent_id') or '').lower()
    return bool(re.search(r'_r\d+', aid))


def round_from_agent_id(aid):
    m = re.search(r'_r(\d+)', aid or '')
    return int(m.group(1)) if m else None


def extract_pairs(jsonl_path, benchmark_label):
    """Walk one jsonl, yield training pairs."""
    pairs = []
    skipped = Counter()
    with open(jsonl_path) as f:
        for lineno, line in enumerate(f, 1):
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as e:
                skipped['malformed_json'] += 1
                continue
            for cr in (rec.get('call_records') or []):
                if cr.get('call_type') == 'tool_execution':
                    continue
                if not is_debate_round_call(cr):
                    skipped['not_debate'] += 1
                    continue
                resp = cr.get('response') or ''
                msgs = cr.get('request_messages') or []
                if not resp or not msgs:
                    skipped['missing_text'] += 1
                    continue
                peer_text, history_chars = extract_peer_text(msgs)
                if peer_text is None:
                    skipped['no_peer_marker'] += 1
                    continue
                pairs.append({
                    'task_id': rec.get('task_id'),
                    'benchmark': benchmark_label,
                    'topology': rec.get('topology'),
                    'agent_id': cr.get('agent_id'),
                    'round': round_from_agent_id(cr.get('agent_id', '')),
                    'peer_text': peer_text,
                    'peer_text_chars': len(peer_text),
                    'history_len_chars': history_chars,
                    'target_response': resp if isinstance(resp, str) else str(resp),
                    'target_completion_tokens': cr.get('completion_tokens', 0) or 0,
                    'gpu_dynamic_energy_joules': cr.get('gpu_dynamic_energy_joules', 0.0) or 0.0,
                    'task_correct': rec.get('correct'),
                    'task_loose_accuracy': rec.get('loose_accuracy'),
                })
    return pairs, skipped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--source-roots', nargs='+', default=[
        'mas-energy/results/a5000_latent_transcripts',
        'mas-energy/results/a5000_transcripts_qampari',
        'mas-energy/results/a5000_math_pilot',
    ], help='Top-level dirs to scan for *_decentralized_*.jsonl')
    ap.add_argument('--output', default='mas-energy/results/latent_pilot/training_pairs.jsonl')
    ap.add_argument('--min-target-chars', type=int, default=20,
                    help='Drop pairs whose target response is shorter than N chars')
    args = ap.parse_args()

    total_pairs = []
    total_skipped = Counter()
    sources_seen = []

    for root in args.source_roots:
        root_p = Path(root)
        if not root_p.exists():
            print(f"  (missing root: {root})")
            continue
        for jsonl in sorted(root_p.rglob('*decentralized*.jsonl')):
            # Heuristic benchmark label = parent dir name OR root
            bench_label = jsonl.parent.name
            if bench_label == 'a5000_math_pilot':
                bench_label = 'math'
            pairs, skipped = extract_pairs(jsonl, bench_label)
            pairs = [p for p in pairs if len(p['target_response']) >= args.min_target_chars]
            print(f"  {jsonl.relative_to('.')}: {len(pairs)} pairs  skipped={dict(skipped)}")
            total_pairs.extend(pairs)
            total_skipped.update(skipped)
            sources_seen.append(str(jsonl))

    out_p = Path(args.output)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    with open(out_p, 'w') as f:
        for p in total_pairs:
            f.write(json.dumps(p) + '\n')

    by_bench = Counter(p['benchmark'] for p in total_pairs)
    avg_peer = sum(p['peer_text_chars'] for p in total_pairs) / len(total_pairs) if total_pairs else 0
    avg_resp = sum(len(p['target_response']) for p in total_pairs) / len(total_pairs) if total_pairs else 0
    avg_tok = sum(p['target_completion_tokens'] for p in total_pairs) / len(total_pairs) if total_pairs else 0

    print("\n=== Summary ===")
    print(f"Total pairs:  {len(total_pairs)}")
    print(f"By benchmark: {dict(by_bench)}")
    print(f"Avg peer_text_chars:        {avg_peer:.0f}")
    print(f"Avg target_response_chars:  {avg_resp:.0f}")
    print(f"Avg target_completion_tok:  {avg_tok:.0f}")
    print(f"Skipped:      {dict(total_skipped)}")
    print(f"Output:       {out_p}")
    if not total_pairs:
        print("\nNo pairs extracted — check that source dirs contain "
              "decentralized jsonls with --save-transcripts text.")


if __name__ == "__main__":
    main()
