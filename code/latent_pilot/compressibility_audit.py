"""Preliminary compressibility audit for latent inter-agent communication.

Goal: estimate what fraction of decode tokens in debate-round agent outputs
are "communication overhead" vs "essential content". Uses heuristic
classification + structural position-based decomposition. No training.

Outputs per-segment counts and aggregate compression ceiling estimate.
"""
import json
import re
from pathlib import Path
from collections import defaultdict, Counter
from statistics import mean, median


# Heuristic patterns for token-role classification.
REFERENCE_RE = re.compile(
    r'\b('
    r'agent\s*\d|agents?\b.*\bsaid|previous\s+(response|answer|agent)|'
    r'looking\s+at\s+the|based\s+on\s+the\s+(prior|previous|other)|'
    r'as\s+(mentioned|noted|the\s+other|previously)|'
    r'(i|we)\s+(agree|disagree)\s+with|'
    r"the\s+other\s+agent|the\s+responses\s+above|"
    r'building\s+on|in\s+light\s+of|'
    r'reviewing\s+the|considering\s+the\s+responses'
    r')\b',
    re.IGNORECASE,
)

RESTATE_RE = re.compile(
    r'\b('
    r'the\s+question\s+(asks|is)|'
    r'we\s+are\s+asked|'
    r'to\s+(find|determine|compute|solve)\s+the|'
    r"let'?s\s+(re)?(consider|examine|look\s+at)"
    r')\b',
    re.IGNORECASE,
)

ANSWER_RE = re.compile(
    r'(\\boxed\s*\{|the\s+(final\s+)?answer\s+is|therefore\s*,?\s+the|'
    r'final\s+answer|hence\s*,?\s+the\s+answer)',
    re.IGNORECASE,
)

TOOL_RE = re.compile(r'<tool_call>|<function=|"function":\s*\{', re.IGNORECASE)


def classify_line(line):
    """Classify a single line into one of: ref, restate, tool, answer, reason, blank."""
    s = line.strip()
    if not s:
        return 'blank'
    if TOOL_RE.search(s):
        return 'tool'
    if ANSWER_RE.search(s):
        return 'answer'
    if REFERENCE_RE.search(s):
        return 'ref'
    if RESTATE_RE.search(s):
        return 'restate'
    return 'reason'


def estimate_tokens(text):
    """Rough token estimate: ~4 chars/token (works for English+code mix)."""
    return max(1, len(text) // 4)


def analyze_response(resp_text):
    """Decompose a response into tokens by role."""
    if not resp_text:
        return {'total_chars': 0, 'total_tokens_est': 0, 'lines_by_class': {}, 'chars_by_class': {}}
    counts = Counter()
    chars_by_class = Counter()
    lines = resp_text.split('\n')
    for line in lines:
        cls = classify_line(line)
        counts[cls] += 1
        chars_by_class[cls] += len(line)
    return {
        'total_chars': len(resp_text),
        'total_tokens_est': estimate_tokens(resp_text),
        'lines_by_class': dict(counts),
        'chars_by_class': dict(chars_by_class),
    }


def role_from_agent_id(aid):
    """Map call_records.agent_id to a structural role.
    init: round 0 (no peer context); rN: debate round N.
    """
    if not aid: return 'unknown'
    aid = aid.lower()
    m = re.search(r'_r(\d+)', aid)
    if m: return f'r{m.group(1)}'
    if 'init' in aid: return 'init'
    if 'synth' in aid: return 'synth'
    if 'orchestr' in aid: return 'orch'
    if 'review' in aid: return 'review'
    if 'work' in aid: return 'worker'
    return 'other'


def audit_jsonl(path, topology_filter=None):
    """Walk one jsonl, return per-call decomposition rows."""
    rows = []
    for line in open(path):
        rec = json.loads(line)
        topo = rec.get('topology', '')
        if topology_filter and topo != topology_filter:
            continue
        for cr in (rec.get('call_records') or []):
            ct = cr.get('call_type', '')
            if ct == 'tool_execution': continue
            resp = cr.get('response') or ''
            if not isinstance(resp, str): resp = str(resp)
            agent_id = cr.get('agent_id', '')
            role = role_from_agent_id(agent_id)
            decomp = analyze_response(resp)
            decomp.update({
                'task_id': rec.get('task_id'),
                'topology': topo,
                'call_type': ct,
                'agent_id': agent_id,
                'role': role,
                'completion_tokens': cr.get('completion_tokens', 0),
                'prompt_tokens': cr.get('prompt_tokens', 0),
                'gpu_dynamic_energy_joules': cr.get('gpu_dynamic_energy_joules', 0.0),
                'response_text': resp,
            })
            rows.append(decomp)
    return rows


def aggregate(rows):
    """Aggregate stats by role."""
    by_role = defaultdict(list)
    for r in rows:
        by_role[r['role']].append(r)
    out = {}
    for role, rs in sorted(by_role.items()):
        n = len(rs)
        if n == 0: continue
        toks = [r['completion_tokens'] for r in rs if r['completion_tokens']]
        chars_total = sum(r['total_chars'] for r in rs)
        chars_ref = sum(r['chars_by_class'].get('ref',0) for r in rs)
        chars_restate = sum(r['chars_by_class'].get('restate',0) for r in rs)
        chars_reason = sum(r['chars_by_class'].get('reason',0) for r in rs)
        chars_answer = sum(r['chars_by_class'].get('answer',0) for r in rs)
        chars_tool = sum(r['chars_by_class'].get('tool',0) for r in rs)
        out[role] = dict(
            n=n,
            mean_completion_tokens=mean(toks) if toks else 0,
            median_completion_tokens=median(toks) if toks else 0,
            chars_total=chars_total,
            pct_ref=100*chars_ref/chars_total if chars_total else 0,
            pct_restate=100*chars_restate/chars_total if chars_total else 0,
            pct_reason=100*chars_reason/chars_total if chars_total else 0,
            pct_answer=100*chars_answer/chars_total if chars_total else 0,
            pct_tool=100*chars_tool/chars_total if chars_total else 0,
        )
    return out


def print_table(agg, label):
    print(f"\n=== {label} ===")
    print(f"{'role':<8} {'n':>4} {'tok_med':>8} {'tok_mean':>9} {'%ref':>6} {'%rest':>6} {'%rsn':>6} {'%ans':>6} {'%tool':>6}")
    for role, s in agg.items():
        print(f"{role:<8} {s['n']:>4} {s['median_completion_tokens']:>8.0f} {s['mean_completion_tokens']:>9.1f} "
              f"{s['pct_ref']:>6.1f} {s['pct_restate']:>6.1f} {s['pct_reason']:>6.1f} "
              f"{s['pct_answer']:>6.1f} {s['pct_tool']:>6.1f}")


def main():
    sources = [
        ('mas-energy/results/a5000_math_pilot/Qwen_Qwen3.5-9B_decentralized_k10.jsonl', 'MATH-decent'),
        ('mas-energy/results/a5000_math_pilot/Qwen_Qwen3.5-9B_independent_k10.jsonl', 'MATH-indep'),
        ('mas-energy/results/a5000_transcripts_qampari/Qwen_Qwen3.5-9B_decentralized_k5.jsonl', 'QAMPARI-decent'),
        ('mas-energy/results/a5000_transcripts_qampari/Qwen_Qwen3.5-9B_centralized_k5.jsonl', 'QAMPARI-cent'),
        ('mas-energy/results/a5000_transcripts_qampari/Qwen_Qwen3.5-9B_independent_k5.jsonl', 'QAMPARI-indep'),
    ]
    all_rows = []
    for path, label in sources:
        p = Path(path)
        if not p.exists():
            print(f"  (missing: {path})"); continue
        rows = audit_jsonl(p)
        agg = aggregate(rows)
        print_table(agg, label)
        all_rows.extend([(label, r) for r in rows])

    # Cross-cut: comparison of init vs r0 vs r1 outputs in decent (where available)
    print("\n\n=== Init vs Debate-round comparison (decent only) ===")
    decent = [r for label, r in all_rows if 'decent' in label.lower()]
    by = defaultdict(list)
    for r in decent: by[r['role']].append(r)
    print(f"{'role':<8} {'n':>4} {'tok_med':>8} {'tok_mean':>9} {'pct_comm':>9}")
    for role in ['init', 'r0', 'r1', 'r2']:
        rs = by.get(role, [])
        if not rs: continue
        toks = [r['completion_tokens'] for r in rs if r['completion_tokens']]
        ct = sum(r['total_chars'] for r in rs)
        comm = sum(r['chars_by_class'].get('ref',0)+r['chars_by_class'].get('restate',0) for r in rs)
        print(f"{role:<8} {len(rs):>4} {median(toks) if toks else 0:>8.0f} {mean(toks) if toks else 0:>9.1f} {100*comm/ct if ct else 0:>9.1f}")

    # Compression ceiling estimate: assume ref+restate is fully replaceable with latent
    print("\n\n=== Compression ceiling estimate (decent debate rounds) ===")
    debate_rows = [r for r in decent if r['role'].startswith('r')]
    if debate_rows:
        total_chars = sum(r['total_chars'] for r in debate_rows)
        comm_chars = sum(r['chars_by_class'].get('ref',0)+r['chars_by_class'].get('restate',0) for r in debate_rows)
        total_completion_toks = sum(r['completion_tokens'] for r in debate_rows)
        print(f"  debate-round LLM calls: {len(debate_rows)}")
        print(f"  total decode tokens (debate rounds only): {total_completion_toks:,}")
        print(f"  total response chars: {total_chars:,}")
        print(f"  comm-marker chars (ref+restate): {comm_chars:,} ({100*comm_chars/total_chars:.1f}%)")
        print(f"  estimated decode tokens replaceable: {int(total_completion_toks * comm_chars/total_chars):,}")

if __name__ == "__main__":
    main()
