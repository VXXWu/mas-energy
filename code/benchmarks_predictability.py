"""Predictability-axis synthetic benchmark.

Direct test of the mechanism in `analysis/predictability_axis_test_design.md`:
*does the iteration-required claim depend on tool-output unpredictability*,
or is it a post-hoc explanation of cross-benchmark variation?

Task: N items with hidden integer labels; find all items whose label
satisfies a fixed predicate ("divisible by 3"). One tool: `query(item_id)`
returns the item's label.

Two variants — same task, same tool, same protocol — toggle ONLY whether
the prompt contains the labels:
  - 'P' (predictable):   prompt lists all labels verbatim. Tool calls are
                          redundant; agent could answer from prompt alone.
  - 'U' (unpredictable): prompt omits labels. Tool calls are the only way
                          to learn them.

Eval: F1 over predicted item indices vs ground-truth set.

Follows: load_tasks, get_tools, make_executor, evaluate.
"""
from __future__ import annotations

import random
import re
from typing import Any


PREDICTABILITY_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "query",
            "description": (
                "Return the integer label of an item by its ID. "
                "Item IDs are 0-indexed."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "item_id": {
                        "type": "integer",
                        "description": "Item ID in range [0, N).",
                    },
                },
                "required": ["item_id"],
            },
        },
    },
]


# Default task parameters
# Increased from N=30 to N=200 to ensure the U-variant cannot saturate at
# F1=1.0 within typical MAS compute budgets (k=5, M=3, parallel tool calls
# per step → ~15-50 queries total). At N=200 the budget covers <25% of
# the array, leaving substantial uncertainty for U-variant agents.
N_ITEMS = 200
LABEL_RANGE = (1, 999)
PREDICATE_DIVISOR = 3  # "label divisible by PREDICATE_DIVISOR"


def _generate_array(seed: int, n: int = N_ITEMS,
                    lo: int = LABEL_RANGE[0], hi: int = LABEL_RANGE[1]) -> list[int]:
    rng = random.Random(seed)
    return [rng.randint(lo, hi) for _ in range(n)]


def _gold_indices(arr: list[int], divisor: int = PREDICATE_DIVISOR) -> list[int]:
    return [i for i, v in enumerate(arr) if v % divisor == 0]


def _build_question(arr: list[int], variant: str, divisor: int = PREDICATE_DIVISOR) -> str:
    """Build the user-facing task prompt for a given variant."""
    n = len(arr)
    common = (
        f"There are {n} items, indexed 0 through {n-1}. Each item has a hidden "
        f"integer label between {LABEL_RANGE[0]} and {LABEL_RANGE[1]}. "
        f"Find ALL item IDs whose label is divisible by {divisor}.\n\n"
        f"Use the `query` tool to inspect an item's label by its ID. "
        f"When done, output the answer as a JSON list of item IDs, e.g. "
        f"`[2, 7, 14]`. Output ONLY the JSON list as your final answer."
    )
    if variant == "P":
        labels_block = "\n".join(f"  Item {i}: {v}" for i, v in enumerate(arr))
        return (
            f"{common}\n\n"
            f"For convenience, here are all the labels:\n{labels_block}"
        )
    # 'U'
    return common


_LIST_PAT = re.compile(r"\[(?:\s*\d+\s*,?)*\s*\]")


def _parse_index_list(text: str, n_items: int) -> list[int]:
    """Extract a list of integer indices from arbitrary text. Take the LAST
    bracketed list found (agents tend to refine before committing)."""
    if not text:
        return []
    matches = _LIST_PAT.findall(text)
    if not matches:
        return []
    last = matches[-1]
    nums = re.findall(r"\d+", last)
    out = []
    seen = set()
    for s in nums:
        try:
            i = int(s)
        except ValueError:
            continue
        if 0 <= i < n_items and i not in seen:
            out.append(i)
            seen.add(i)
    return out


def _f1(pred: list[int], gold: list[int]) -> tuple[float, float, float]:
    if not pred and not gold:
        return 1.0, 1.0, 1.0
    pset, gset = set(pred), set(gold)
    if not pset or not gset:
        return 0.0, 0.0, 0.0
    tp = len(pset & gset)
    p = tp / len(pset)
    r = tp / len(gset)
    f = (2 * p * r / (p + r)) if (p + r) > 0 else 0.0
    return p, r, f


class PredictabilityExecutor:
    """Per-task executor: holds the array, services `query` calls, tracks count."""

    def __init__(self, arr: list[int]):
        self.arr = arr
        self.n_calls = 0
        self.queried: set[int] = set()

    def __call__(self, tool_name: str, args: dict) -> str:
        if tool_name != "query":
            return f"Unknown tool: {tool_name}"
        try:
            i = int(args.get("item_id"))
        except (TypeError, ValueError):
            return "Error: 'item_id' must be an integer."
        if not (0 <= i < len(self.arr)):
            return f"Error: item_id {i} out of range [0, {len(self.arr)})."
        self.n_calls += 1
        self.queried.add(i)
        return f"Item {i} label = {self.arr[i]}"


class PredictabilityBenchmark:
    """Synthetic predictability-axis benchmark. Tasks are deterministic from
    seed; each (seed, variant) pair is a distinct task. Pairing across
    variants is by seed (paired analysis: P-task[s] vs U-task[s])."""

    def __init__(self, n_items: int = N_ITEMS,
                 predicate_divisor: int = PREDICATE_DIVISOR,
                 variants: tuple[str, ...] = ("P", "U")):
        self.n_items = n_items
        self.predicate_divisor = predicate_divisor
        self.variants = tuple(variants)

    def load_tasks(self, n_tasks: int = 100, seed: int = 42) -> list[dict]:
        """Generate n_tasks tasks per variant. Total = n_tasks × len(variants).
        Each task seeded deterministically by (variant, task_index)."""
        out = []
        rng = random.Random(seed)
        # Use distinct random task seeds (so pairing is by base_seed but task seeds vary)
        base_seeds = [rng.randint(0, 10**9) for _ in range(n_tasks)]
        for variant in self.variants:
            for k, base_seed in enumerate(base_seeds):
                arr = _generate_array(base_seed, self.n_items)
                gold = _gold_indices(arr, self.predicate_divisor)
                qid = f"pred-{variant}-{k:04d}"
                out.append({
                    "id": qid,
                    "qid": qid,
                    "variant": variant,
                    "base_seed": base_seed,
                    "task_index": k,  # use as paired key across variants
                    "task_id": k,     # so analyzer can pair P/U task[k]
                    "array": arr,
                    "gold_indices": gold,
                    "n_items": self.n_items,
                    "question": _build_question(arr, variant, self.predicate_divisor),
                })
        return out

    def get_tools(self) -> list[dict]:
        return PREDICTABILITY_TOOLS

    def make_executor(self, task=None):
        if task is None:
            executor = PredictabilityExecutor([])
            return executor, lambda: None
        executor = PredictabilityExecutor(task["array"])
        return executor, lambda: None

    def evaluate(self, task: dict, recorder=None, final_answer: str = "") -> dict:
        pred = _parse_index_list(final_answer or "", task["n_items"])
        gold = task["gold_indices"]
        precision, recall, f1 = _f1(pred, gold)
        return {
            "correct": (set(pred) == set(gold)),
            "loose_accuracy": f1,
            "precision": precision,
            "recall": recall,
            "n_pred": len(pred),
            "n_gold": len(gold),
        }
