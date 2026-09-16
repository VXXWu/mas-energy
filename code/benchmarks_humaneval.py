"""HumanEval+ benchmark: write→test→fix code-completion adapter.

Tests the iteration-required claim on a code distribution distinct from
SWE-bench (atomic functions, no multi-file edits, no repo dependencies).

Tool: `run_tests(code)` runs the candidate against the hidden test suite
and returns pass/fail/errors. The agent iterates: write → test → revise.

Evaluation: re-run the agent's last submitted code against the full test
suite at task end. Falls back to extracting a function definition from the
agent's final response if no run_tests call was made.

Follows the four-method pattern: load_tasks, get_tools, make_executor, evaluate.
"""
from __future__ import annotations

import multiprocessing
import random
import re
import signal
from pathlib import Path


HUMANEVAL_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "run_tests",
            "description": (
                "Run the candidate Python code against the hidden test suite. "
                "Returns pass/fail status and any error messages. Use this to "
                "verify your implementation and iterate when tests fail."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": (
                            "Complete Python source defining the target function "
                            "(include the def line and full body). Will be exec'd "
                            "before running the test suite."
                        ),
                    },
                },
                "required": ["code"],
            },
        },
    },
]


def extract_function(text: str, entry_point: str) -> str:
    """Best-effort extraction of the function definition from arbitrary text.
    Strips markdown fences, then finds `def entry_point(`. Returns the function
    block (signature + body until next non-indented line)."""
    if not text:
        return ""
    # Strip markdown fences if present
    fence = re.search(r"```(?:python)?\s*\n(.*?)```", text, re.DOTALL)
    if fence:
        text = fence.group(1)
    pat = re.compile(rf"^def\s+{re.escape(entry_point)}\s*\(", re.MULTILINE)
    m = pat.search(text)
    if not m:
        return text  # Hope the whole thing is a function body
    start = m.start()
    lines = text[start:].split("\n")
    out = [lines[0]]
    for line in lines[1:]:
        if line and not line.startswith((" ", "\t")) and not line.startswith("#"):
            break
        out.append(line)
    return "\n".join(out)


def _run_check_worker(prompt, code, test, entry_point, q):
    """Subprocess: exec code (or prompt+code), then test, then check(fn)."""
    try:
        ns = {"__name__": "__test__"}
        try:
            exec(code, ns)
            if entry_point not in ns:
                ns = {"__name__": "__test__"}
                exec(prompt + "\n" + code, ns)
        except Exception:
            ns = {"__name__": "__test__"}
            exec(prompt + "\n" + code, ns)
        if entry_point not in ns:
            q.put(("fail", f"entry point '{entry_point}' not defined"))
            return
        exec(test, ns)
        if "check" not in ns:
            q.put(("fail", "test did not define check()"))
            return
        ns["check"](ns[entry_point])
        q.put(("pass", ""))
    except Exception as e:
        q.put(("fail", f"{type(e).__name__}: {e}"))


def run_tests_isolated(prompt: str, code: str, test: str, entry_point: str,
                       time_limit_s: float = 10.0) -> tuple[bool, str]:
    """Run code+test in a subprocess with a hard timeout."""
    code = extract_function(code, entry_point) or code
    q = multiprocessing.Queue()
    p = multiprocessing.Process(
        target=_run_check_worker,
        args=(prompt, code, test, entry_point, q),
    )
    p.start()
    p.join(timeout=time_limit_s)
    if p.is_alive():
        p.terminate()
        p.join()
        return False, f"timeout after {time_limit_s}s"
    if q.empty():
        return False, "subprocess crashed"
    status, msg = q.get()
    return status == "pass", msg


class HumanEvalExecutor:
    """Stateful per-task executor. Tracks the most recent code submitted via
    run_tests so the benchmark's evaluate() can re-run it at task end without
    parsing the agent's final response."""

    def __init__(self, task: dict, time_limit_s: float = 10.0):
        self.task = task
        self.time_limit_s = time_limit_s
        self.last_code = ""
        self.last_passed = False
        self.last_error = ""
        self.n_calls = 0

    def __call__(self, tool_name: str, args: dict) -> str:
        if tool_name != "run_tests":
            return f"Unknown tool: {tool_name}"
        code = args.get("code", "")
        if not code:
            return "Error: 'code' is required."
        self.n_calls += 1
        self.last_code = code
        passed, msg = run_tests_isolated(
            self.task["prompt"], code, self.task["test"],
            self.task["entry_point"], self.time_limit_s,
        )
        self.last_passed = passed
        self.last_error = msg
        if passed:
            return "PASS: all tests passed."
        return f"FAIL: {msg}"


class HumanEvalBenchmark:
    """HumanEval+ adapter. Loads from HuggingFace evalplus/humanevalplus.

    Tasks: 164 atomic Python function-completion problems with extended
    test suites (Liu et al., NeurIPS 2023).
    """

    def __init__(self, time_limit_s: float = 10.0):
        self.time_limit_s = time_limit_s
        self._all_tasks: list[dict] | None = None
        self._executors: dict = {}

    def _load_all_tasks(self) -> list[dict]:
        if self._all_tasks is not None:
            return self._all_tasks
        from datasets import load_dataset
        ds = load_dataset("evalplus/humanevalplus", split="test")
        rows = [dict(r) for r in ds]
        out = []
        for r in rows:
            qid = r["task_id"].replace("/", "_")
            question = (
                f"Implement the following Python function. You have access to "
                f"a `run_tests` tool that runs your code against a hidden test "
                f"suite and reports pass/fail with error messages. Use it to "
                f"verify your implementation; iterate by calling it again with "
                f"a revised version when tests fail. When all tests pass (or "
                f"you've exhausted attempts), provide the final function "
                f"definition as your answer.\n\n"
                f"```python\n{r['prompt']}\n```"
            )
            out.append({
                "id": qid,
                "qid": qid,
                "question": question,
                "question_text": r["prompt"],
                "task_id": r["task_id"],
                "prompt": r["prompt"],
                "test": r["test"],
                "entry_point": r["entry_point"],
                "canonical_solution": r.get("canonical_solution", ""),
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
        return HUMANEVAL_TOOLS

    def make_executor(self, task=None):
        executor = HumanEvalExecutor(task, time_limit_s=self.time_limit_s)
        if task is not None:
            self._executors[task["id"]] = executor

        def cleanup():
            self._executors.pop(task["id"], None) if task is not None else None

        return executor, cleanup

    def evaluate(self, task: dict, recorder=None, final_answer: str = "") -> dict:
        """Re-run the agent's last submitted code against the test suite.

        Order of preference for the code under test:
          1. Function extracted from agent's final_answer (the agent's stated
             "this is my answer" code)
          2. Last code submitted via run_tests tool (executor.last_code)
        """
        entry_point = task["entry_point"]
        code = ""
        if final_answer:
            code = extract_function(final_answer, entry_point)
        executor = self._executors.get(task["id"])
        if not code or len(code.split("\n")) < 2:
            if executor and executor.last_code:
                code = executor.last_code
        if not code:
            return {
                "correct": False,
                "loose_accuracy": 0.0,
                "error": "no code submitted",
                "n_test_calls": executor.n_calls if executor else 0,
            }
        passed, msg = run_tests_isolated(
            task["prompt"], code, task["test"], entry_point, self.time_limit_s,
        )
        return {
            "correct": passed,
            "loose_accuracy": 1.0 if passed else 0.0,
            "error": msg if not passed else "",
            "n_test_calls": executor.n_calls if executor else 0,
        }
