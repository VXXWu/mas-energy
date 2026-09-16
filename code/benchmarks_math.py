"""MATH benchmark with stateful Python interpreter as the agent tool.

Reasoning-heavy adapter for the Hendrycks competition_math dataset
(hendrycks/competition_math on HuggingFace), filtered to Level 5 problems
where single-LLM accuracy is bounded and inter-agent debate is theoretically
most valuable (Du et al. 2023).

Design choices for Kim-et-al-compatible "agentic" framing:
  - Tool calling: Python REPL is the agent's only tool
  - Multi-step ReAct: agents typically call Python multiple times to
    explore, verify intermediate results, simplify expressions
  - STATEFULNESS: each task gets a persistent Python namespace. Variables,
    imports, and helper functions defined in one tool call persist to the
    next. This matches the statefulness profile of Kim et al.'s WorkBench
    (calendar/email databases persist) — a property QAMPARI/BrowseComp
    lack.

Why this benchmark fills a gap in your matrix:
  - Existing matrix is retrieval-heavy (QAMPARI, FanOutQA, BrowseComp)
    or procedural (WorkBench, SWE-bench)
  - Missing: REASONING-HEAVY agentic tasks where Du et al. showed debate
    helps. This benchmark fills that exact slot.

Evaluation: extract \\boxed{} from final answer, normalize via SymPy
(simplify both predicted and gold; equality check). Falls back to string
match when SymPy can't parse.

Follows the four-method pattern: load_tasks, get_tools, make_executor, evaluate.
"""
from __future__ import annotations

import io
import json
import os
import random
import re
import signal
import sys
from contextlib import redirect_stdout, redirect_stderr
from pathlib import Path


# ─────────────────────────────────────────────────────────
# Tool schema
# ─────────────────────────────────────────────────────────

MATH_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "python",
            "description": (
                "Execute Python code in a persistent REPL. Variables, "
                "imports, and function definitions persist across calls. "
                "Use sympy, math, numpy as needed for symbolic and "
                "numeric computation. Output is captured stdout."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "Python code to execute. Stdout is returned to the agent.",
                    },
                },
                "required": ["code"],
            },
        },
    },
]


# ─────────────────────────────────────────────────────────
# Evaluation helpers
# ─────────────────────────────────────────────────────────

_BOXED_PATTERN = re.compile(r"\\boxed\s*\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}")


def extract_boxed(text: str) -> str | None:
    """Extract the LAST \\boxed{...} content from text. MATH gold answers
    use this convention; agents are instructed to use it too. Returns None
    if no boxed expression is found.

    Handles single-level nesting (e.g. \\boxed{\\frac{1}{2}}).
    """
    if not text:
        return None
    matches = _BOXED_PATTERN.findall(text)
    if matches:
        return matches[-1].strip()
    # Fallback: look for "the answer is X" patterns
    m = re.search(r"(?:final answer|answer is|equals?)[:\s]+\$?([^\.\n$]+)", text, re.IGNORECASE)
    if m:
        return m.group(1).strip().rstrip(".")
    return None


def _normalize_for_compare(s: str) -> str:
    """Light string normalization before SymPy comparison."""
    if s is None:
        return ""
    s = s.strip()
    # Strip outer LaTeX delimiters
    s = re.sub(r"^\$+|\$+$", "", s).strip()
    # Common LaTeX → SymPy translation hints
    s = s.replace("\\dfrac", "\\frac")
    s = s.replace("\\tfrac", "\\frac")
    s = s.replace("\\cdot", "*")
    s = s.replace("\\,", "")
    s = s.replace("\\!", "")
    s = s.replace("^{\\circ}", "*pi/180")  # degree symbol
    s = s.replace("\\pi", "pi")
    s = s.replace("\\sqrt", "sqrt")
    return s


def _sympy_equal(pred: str, gold: str, timeout_s: float = 2.0) -> bool:
    """Try SymPy-based equality. Returns False on parse error or timeout."""
    try:
        from sympy import simplify, Eq, sympify, Rational
        from sympy.parsing.latex import parse_latex
    except ImportError:
        return False

    p = _normalize_for_compare(pred)
    g = _normalize_for_compare(gold)
    if not p or not g:
        return False

    def _parse(s):
        # Try LaTeX parser first; fall back to sympify
        try:
            return parse_latex(s)
        except Exception:
            try:
                return sympify(s, rational=True)
            except Exception:
                return None

    try:
        p_expr = _parse(p)
        g_expr = _parse(g)
        if p_expr is None or g_expr is None:
            return False
        diff = simplify(p_expr - g_expr)
        return diff == 0
    except Exception:
        return False


def evaluate_math(task: dict, predicted_text: str) -> dict:
    """Evaluate a MATH problem prediction.

    Signature matches the QAMPARI/BrowseComp pattern: (task_dict, predicted_text) → dict.
    Returns dict with: correct (bool), loose_accuracy (1.0 or 0.0),
    pred_boxed, gold_boxed.
    """
    pred = extract_boxed(predicted_text)
    gold = task.get("gold_boxed") or extract_boxed(task.get("solution", ""))

    correct = False
    if pred is not None and gold is not None:
        # Exact string match (after normalization)
        p_norm = _normalize_for_compare(pred)
        g_norm = _normalize_for_compare(gold)
        if p_norm == g_norm:
            correct = True
        else:
            # SymPy equality
            correct = _sympy_equal(pred, gold)

    return {
        "correct": correct,
        "loose_accuracy": 1.0 if correct else 0.0,
        "pred_boxed": pred,
        "gold_boxed": gold,
    }


# ─────────────────────────────────────────────────────────
# Stateful Python executor
# ─────────────────────────────────────────────────────────


class _TimeoutError(Exception):
    pass


def _timeout_handler(signum, frame):
    raise _TimeoutError("execution exceeded time limit")


class StatefulPythonExecutor:
    """Persistent Python REPL for one task (one agent's perspective).

    Each call executes code in the same namespace. Variables, imports, and
    function definitions persist across calls. Stdout is captured and
    returned. Hard time limit per call (default 5 s) prevents runaway.

    Bootstrap imports `sympy`, `math`, `numpy` (best-effort) so agents can
    use them without re-importing. fractions.Fraction is also available.
    """

    BOOTSTRAP = (
        "import math\n"
        "from fractions import Fraction\n"
        "try:\n"
        "    import sympy\n"
        "    from sympy import simplify, sympify, Rational, sqrt, Symbol, solve, Eq\n"
        "except ImportError: pass\n"
        "try:\n"
        "    import numpy as np\n"
        "except ImportError: pass\n"
    )

    def __init__(self, time_limit_s: float = 5.0, output_char_limit: int = 4000):
        self.time_limit_s = time_limit_s
        self.output_char_limit = output_char_limit
        self.namespace: dict = {"__name__": "__agent__"}
        # Run bootstrap imports in the namespace
        try:
            exec(self.BOOTSTRAP, self.namespace)
        except Exception as e:
            # Bootstrap failure is non-fatal; agents can still import manually
            pass

    def __call__(self, tool_name: str, args: dict) -> str:
        if tool_name != "python":
            return f"Unknown tool: {tool_name}"
        code = args.get("code", "")
        if not code:
            return "Error: code is required."
        return self._exec(code)

    def _exec(self, code: str) -> str:
        out = io.StringIO()
        err = io.StringIO()
        # Set timeout via SIGALRM (POSIX only). On Windows, fallback to no timeout.
        old_handler = None
        timed_out = False
        try:
            if hasattr(signal, "SIGALRM"):
                old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
                signal.setitimer(signal.ITIMER_REAL, self.time_limit_s)
            with redirect_stdout(out), redirect_stderr(err):
                exec(code, self.namespace)
        except _TimeoutError:
            timed_out = True
        except Exception as e:
            err.write(f"{type(e).__name__}: {e}\n")
        finally:
            if hasattr(signal, "SIGALRM"):
                signal.setitimer(signal.ITIMER_REAL, 0)
                if old_handler is not None:
                    signal.signal(signal.SIGALRM, old_handler)

        out_s = out.getvalue()
        err_s = err.getvalue()
        result_parts = []
        if out_s:
            result_parts.append(out_s.rstrip())
        if err_s:
            result_parts.append("[stderr] " + err_s.rstrip())
        if timed_out:
            result_parts.append(f"[TIMEOUT after {self.time_limit_s}s]")
        if not result_parts:
            return "(no output)"
        result = "\n".join(result_parts)
        # Truncate if too long
        if len(result) > self.output_char_limit:
            result = result[:self.output_char_limit] + "\n[output truncated]"
        return result


# ─────────────────────────────────────────────────────────
# Benchmark class
# ─────────────────────────────────────────────────────────

# MATH levels we want by default — reasoning-heavy (Level 5)
DEFAULT_LEVELS = ("Level 5",)


class MATHBenchmark:
    """Adapter for the MATH (hendrycks/competition_math) dataset, filtered
    by difficulty to focus on tasks where single-LLM accuracy is bounded.

    OFFLINE-FIRST loading order (cluster compute nodes lack internet):
      1. Local canonical Hendrycks-format directory: <data_dir>/MATH/<split>/
         (JSON files organized by subject; one problem per file). The
         canonical Hendrycks MATH tarball extracts to this layout.
      2. Pre-staged HuggingFace cache (HF_HOME). load_dataset() finds it
         without network access.
      3. HF Hub download (only works on nodes with internet).

    Tool: stateful Python REPL. Each task gets its own executor instance.
    """

    def __init__(self, levels: tuple[str, ...] = DEFAULT_LEVELS,
                 split: str = "test", time_limit_s: float = 5.0,
                 data_dir: str | None = None):
        self.levels = tuple(levels)
        self.split = split
        self.time_limit_s = time_limit_s
        if data_dir is None:
            data_dir = str(Path(__file__).parent.parent / "data" / "math")
        self.data_dir = Path(data_dir)
        self._all_tasks: list[dict] | None = None

    def _try_local_canonical(self) -> list[dict] | None:
        """Try loading from <data_dir>/MATH/<split>/<subject>/<n>.json.

        This is the canonical Hendrycks MATH layout (after extracting the
        official tarball). Each JSON file has: problem, level, type, solution.
        """
        canonical_root = self.data_dir / "MATH" / self.split
        if not canonical_root.is_dir():
            return None

        rows = []
        for json_path in sorted(canonical_root.rglob("*.json")):
            try:
                with open(json_path) as f:
                    rows.append(json.load(f))
            except Exception:
                continue
        if not rows:
            return None
        print(f"  [MATH] Loaded {len(rows)} problems from local canonical dir: {canonical_root}")
        return rows

    def _try_huggingface(self) -> list[dict] | None:
        """Try HF datasets — works if pre-cached, fails offline otherwise."""
        try:
            from datasets import load_dataset
        except ImportError:
            return None

        for repo in ("lighteval/MATH", "hendrycks/competition_math"):
            try:
                cfg = "all" if repo == "lighteval/MATH" else None
                ds = load_dataset(repo, cfg, split=self.split) if cfg \
                     else load_dataset(repo, split=self.split)
                rows = [dict(r) for r in ds]
                print(f"  [MATH] Loaded {len(rows)} problems from HuggingFace: {repo}")
                return rows
            except Exception as e:
                last_err = e
                continue
        return None

    def _load_all_tasks(self) -> list[dict]:
        if self._all_tasks is not None:
            return self._all_tasks

        rows = self._try_local_canonical()
        if rows is None:
            rows = self._try_huggingface()
        if rows is None:
            raise FileNotFoundError(
                f"MATH dataset not found. Tried:\n"
                f"  1. {self.data_dir / 'MATH' / self.split} (canonical Hendrycks layout)\n"
                f"  2. HuggingFace 'lighteval/MATH' and 'hendrycks/competition_math'\n"
                f"To fix offline: download MATH.tar from "
                f"https://people.eecs.berkeley.edu/~hendrycks/MATH.tar "
                f"and extract under {self.data_dir}/ such that "
                f"{self.data_dir}/MATH/{self.split}/<subject>/*.json exists."
            )

        out = []
        for i, row in enumerate(rows):
            level = row.get("level", "")
            if self.levels and level not in self.levels:
                continue
            problem = row.get("problem", "")
            solution = row.get("solution", "")
            gold_boxed = extract_boxed(solution)
            qid = f"math-{level.replace(' ', '')}-{i}"
            out.append({
                "id": qid,
                "qid": qid,
                "question": (
                    f"{problem}\n\n"
                    f"Solve this problem step by step. Use the python tool "
                    f"to compute and verify intermediate results when "
                    f"helpful. Provide your final numerical or expression "
                    f"answer in the form \\boxed{{answer}}."
                ),
                "question_text": problem,
                "level": level,
                "type": row.get("type", ""),
                "solution": solution,
                "gold_boxed": gold_boxed,
            })
        self._all_tasks = out
        return out

    def load_tasks(self, n_tasks: int = None, seed: int = 42) -> list[dict]:
        all_tasks = self._load_all_tasks()
        tasks = list(all_tasks)
        if n_tasks and n_tasks < len(tasks):
            rng = random.Random(seed)
            rng.shuffle(tasks)
            tasks = tasks[:n_tasks]
        return tasks

    def get_tools(self) -> list[dict]:
        return MATH_TOOLS

    def make_executor(self, task=None):
        """One stateful REPL per task. The cleanup callback resets the
        namespace (it's instance-level so just dropping the reference
        suffices)."""
        executor = StatefulPythonExecutor(time_limit_s=self.time_limit_s)
        return executor, lambda: None

    def evaluate(self, task: dict, recorder=None, final_answer: str = "") -> dict:
        return evaluate_math(task, final_answer)
