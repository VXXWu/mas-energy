"""N-gram redundancy audit: how much of agent A's output overlaps with sibling
agents' outputs (or with the question/prompt the receiver already has)?

Three measurements per debate-round response:
1. Self-redundancy with question prompt (restating what's already given)
2. Sibling-redundancy with other agents' outputs in the same round (parallel debate)
3. Predecessor-redundancy with prior-round outputs the receiver has already seen

Compression ceiling = total - novel_content.
"""
import json
import re
from pathlib import Path
from collections import defaultdict
from statistics import mean, median


def shingles(text, k=3):
    """Word-level k-shingles, lowercased, alphanumeric only."""
    words = re.findall(r"[a-z0-9]+", (text or "").lower())
    return set(zip(*[words[i:] for i in range(k)]))


def jaccard(a, b):
    if not a and not b: return 0.0
    if not a or not b: return 0.0
    return len(a & b) / len(a | b)


def overlap_frac(a, b):
    """Fraction of a covered by b."""
    if not a: return 0.0
    return len(a & b) / len(a)


def role_round(aid):
    if not aid: return ('unknown', None)
    aid = aid.lower()
    m = re.search(r'_r(\d+)', aid)
    if m: return ('debate', int(m.group(1)))
    if 'init' in aid: return ('init', 0)
    if 'synth' in aid: return ('synth', None)
    return ('other', None)


def agent_index(aid):
    """Pull numeric agent index from agent_id like 'debater_2_r1'."""
    m = re.search(r'(?:debater|agent|worker)_(\d+)', (aid or '').lower())
    return int(m.group(1)) if m else -1


def question_text_from_messages(msgs):
    """Pull the user-question portion from request_messages."""
    if not msgs: return ''
    out = []
    for m in msgs:
        if m.get('role') == 'user':
            out.append(m.get('content') or '')
    return '\n'.join(out)


def audit_task(rec):
    """Return one task's per-call analysis."""
    crs = rec.get('call_records') or []
    rows = []
    for cr in crs:
        ct = cr.get('call_type', '')
        if ct == 'tool_execution': continue
        resp = cr.get('response') or ''
        if not isinstance(resp, str): resp = str(resp)
        msgs = cr.get('request_messages') or []
        q_text = question_text_from_messages(msgs)
        role, rd = role_round(cr.get('agent_id', ''))
        rows.append({
            'task_id': rec.get('task_id'),
            'agent_id': cr.get('agent_id', ''),
            'agent_idx': agent_index(cr.get('agent_id', '')),
            'role': role,
            'round': rd,
            'completion_tokens': cr.get('completion_tokens', 0) or 0,
            'response': resp,
            'question_in_prompt': q_text,
            'shingles': shingles(resp),
        })
    return rows


def analyze_jsonl(path, label):
    """For decent jsonls: compute redundancy metrics per debate-round response."""
    by_task = defaultdict(list)
    n_malformed = 0
    with open(path) as f:
        for line in f:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                n_malformed += 1
                continue
            rows = audit_task(rec)
            by_task[rec.get('task_id')].extend(rows)
    if n_malformed:
        print(f"  ({label}: skipped {n_malformed} malformed json lines)")

    summary = []
    for tid, rows in by_task.items():
        # Index by (role, round, idx)
        debate_rows = [r for r in rows if r['role'] == 'debate']
        init_rows = [r for r in rows if r['role'] == 'init']
        for dr in debate_rows:
            shg = dr['shingles']
            qshg = shingles(dr['question_in_prompt'])
            # Sibling: same round, different agent_idx
            siblings = [r for r in debate_rows
                        if r['round'] == dr['round'] and r['agent_idx'] != dr['agent_idx']]
            sib_shg = set().union(*[r['shingles'] for r in siblings]) if siblings else set()
            # Prior: lower round number, any agent
            priors = [r for r in (debate_rows + init_rows)
                      if (r['role'] == 'init' or (r['role'] == 'debate' and r['round'] < dr['round']))]
            prior_shg = set().union(*[r['shingles'] for r in priors]) if priors else set()
            # All info already available to receiver: question + priors (NOT siblings, those arrive together)
            already_known = qshg | prior_shg
            novel = shg - already_known
            summary.append(dict(
                task_id=tid, agent_id=dr['agent_id'], round=dr['round'],
                completion_tokens=dr['completion_tokens'],
                shingles_total=len(shg),
                shingles_in_question=len(shg & qshg),
                shingles_in_siblings=len(shg & sib_shg),
                shingles_in_priors=len(shg & prior_shg),
                shingles_novel=len(novel),
                pct_in_question=100*len(shg & qshg)/len(shg) if shg else 0,
                pct_in_siblings=100*len(shg & sib_shg)/len(shg) if shg else 0,
                pct_in_priors=100*len(shg & prior_shg)/len(shg) if shg else 0,
                pct_novel=100*len(novel)/len(shg) if shg else 0,
            ))
    return summary


def report(label, summary):
    if not summary:
        print(f"\n=== {label}: no debate-round data ===")
        return
    print(f"\n=== {label} (n={len(summary)} debate-round responses) ===")
    keys = ['pct_in_question', 'pct_in_priors', 'pct_in_siblings', 'pct_novel']
    print(f"{'metric':<22} {'mean':>7} {'median':>7}")
    for k in keys:
        vals = [s[k] for s in summary]
        print(f"{k:<22} {mean(vals):>7.1f} {median(vals):>7.1f}")
    total_toks = sum(s['completion_tokens'] for s in summary)
    novel_toks = sum(s['completion_tokens'] * s['pct_novel']/100 for s in summary)
    redund_toks = total_toks - novel_toks
    print(f"\n  total decode tokens (debate rounds): {total_toks:,}")
    print(f"  estimated novel-content tokens:      {int(novel_toks):,} ({100*novel_toks/total_toks:.1f}%)")
    print(f"  estimated redundant tokens:          {int(redund_toks):,} ({100*redund_toks/total_toks:.1f}%)")
    print(f"  -> compression ceiling: ~{100*redund_toks/total_toks:.0f}% of debate-round decode")


def main():
    # Auto-discover all *decentralized*.jsonl under known result roots
    roots = [
        Path('mas-energy/results/a5000_latent_transcripts'),
        Path('mas-energy/results/a5000_math_pilot'),
        Path('mas-energy/results/a5000_transcripts_qampari'),
    ]
    sources = []
    for root in roots:
        if not root.exists(): continue
        for jsonl in sorted(root.rglob('*decentralized*.jsonl')):
            label = f"{jsonl.parent.name}-decent"
            sources.append((str(jsonl), label))

    all_summary = []
    for path, label in sources:
        try:
            s = analyze_jsonl(path, label)
        except Exception as e:
            print(f"  (error reading {label}: {type(e).__name__})")
            continue
        report(label, s)
        all_summary.extend(s)

    if all_summary:
        report('AGGREGATE (all benchmarks)', all_summary)

if __name__ == "__main__":
    main()
