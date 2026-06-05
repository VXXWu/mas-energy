"""Shared accuracy extraction.

For SWE-bench records, the `correct` field is a patch-presence proxy
(diff header exists + modifies a .py file). After the local harness run,
real pytest results are stored in `real_correct`. The cross-cell Spearman
between these two is ~0.015, so patch-presence is unreliable for ranking
SWE-bench cells. This helper enforces:

  SWE-bench resolution logic:
    - real_correct present  → use it (True/False)
    - real_correct absent BUT patch_presence (`correct`) was False
       → this means the agent produced no valid diff. It cannot have
         resolved the bug. → real_correct = False (counts in denominator)
    - real_correct absent AND patch_presence was True
       → the patch passed the proxy filter but wasn't evaluated by
         harness (e.g., not yet run, or pulled out of scope). SKIP this
         record (don't count in numerator OR denominator), because we
         cannot decide.

  This logic ensures we don't inflate accuracy by silently dropping
  empty-diff records (which would clearly count as 0 toward a real-eval
  accuracy) while still being conservative about un-evaluated valid
  patches whose true status is unknown.

  All other benchmarks: use `loose_accuracy` if present, else `correct`.

`record_accuracy(d)` returns (acc_float, included_flag).
  acc_float       : 0.0 / 1.0 / fractional for loose_accuracy
  included_flag   : False means "skip this record entirely"
"""

def record_accuracy(d):
    """Return (acc, included) for a single JSONL record."""
    if d.get("benchmark") == "swebench":
        if "real_correct" in d:
            return (1.0 if d["real_correct"] else 0.0, True)
        # No real_correct: decide based on patch-presence
        if d.get("correct") is False:
            # Empty / invalid diff → definitively did not fix
            return (0.0, True)
        # Patch passed proxy but harness never ran on it → unknown, skip
        return (0.0, False)
    if d.get("loose_accuracy") is not None:
        return (float(d["loose_accuracy"]), True)
    return (1.0 if d.get("correct") else 0.0, True)
