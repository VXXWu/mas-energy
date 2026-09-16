"""BigCodeBench benchmark adapter.

Function-calling code benchmark of 1,140 problems, designed to be
substantially harder than HumanEval+ (Zhuo et al. 2024). Each task ships
with a unittest-based test suite covering correctness AND library-usage
semantics. Expected pass@1 ~25-50% for Qwen3.5-9B class.

Available as parquet on HuggingFace (`bigcode/bigcodebench`, no loading
script issues). Standard schema: task_id, code_prompt (signature + imports),
canonical_solution, test (unittest.TestCase string), entry_point ('task_func').

Same `run_tests(code)` tool interface as HumanEval+/LCB. The agent submits
a `task_func` implementation, gets per-test pass/fail feedback, iterates.

Eval: re-runs full unittest suite on the agent's last submitted code.
Returns pass-rate (proportion of test_* methods passing) as loose accuracy
and exact-pass as strict correctness.

Follows: load_tasks, get_tools, make_executor, evaluate.
"""
from __future__ import annotations

import multiprocessing
import random
import re
from pathlib import Path


BIGCODEBENCH_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "run_tests",
            "description": (
                "Run the candidate Python code against the unittest test "
                "suite. Returns per-test pass/fail status and error "
                "messages. Use this to iterate on your implementation."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": (
                            "Complete Python source defining `task_func` "
                            "(include any necessary imports plus the `def` "
                            "and full body). Will be exec'd before the "
                            "unittest suite runs."
                        ),
                    },
                },
                "required": ["code"],
            },
        },
    },
]


def _extract_task_func(text: str) -> str:
    """Extract the task_func definition (and any preceding imports) from
    arbitrary text. Strips markdown fences. Returns the body verbatim if no
    fence/def pattern is found."""
    if not text:
        return ""
    fence = re.search(r"```(?:python)?\s*\n(.*?)```", text, re.DOTALL)
    if fence:
        text = fence.group(1)
    return text


def _run_unittest_worker(code: str, test_src: str, q):
    """Subprocess: exec code (defining task_func), then exec test_src (defining
    TestCases), then run the unittest suite. Reports n_passed, n_total, errors."""
    import unittest
    import io
    from contextlib import redirect_stdout, redirect_stderr
    try:
        ns = {"__name__": "__test__"}
        exec(code, ns)
        if "task_func" not in ns:
            q.put((0, 0, ["task_func not defined"]))
            return
        exec(test_src, ns)
        if "TestCases" not in ns:
            q.put((0, 0, ["TestCases class not defined in test source"]))
            return
        loader = unittest.TestLoader()
        suite = loader.loadTestsFromTestCase(ns["TestCases"])
        n_total = suite.countTestCases()
        # Run suite, capture results
        buf_out, buf_err = io.StringIO(), io.StringIO()
        with redirect_stdout(buf_out), redirect_stderr(buf_err):
            runner = unittest.TextTestRunner(stream=io.StringIO(), verbosity=0)
            result = runner.run(suite)
        n_failed = len(result.failures) + len(result.errors)
        n_passed = n_total - n_failed
        # Collect first few error messages
        errors = []
        for kind, lst in (("FAIL", result.failures), ("ERROR", result.errors)):
            for tc, tb in lst:
                if len(errors) >= 3:
                    break
                # Trim traceback to last line
                last = tb.strip().split("\n")[-1] if tb else ""
                errors.append(f"{kind} {tc}: {last[:160]}")
        q.put((n_passed, n_total, errors))
    except Exception as e:
        q.put((0, 0, [f"Setup error: {type(e).__name__}: {e}"]))


def run_tests_isolated(code: str, test_src: str,
                       time_limit_s: float = 30.0) -> tuple[int, int, list[str]]:
    """Run candidate code+test in subprocess with hard timeout. Returns
    (n_passed, n_total, error_messages)."""
    code = _extract_task_func(code)
    if not code:
        return 0, 0, ["empty code"]
    q = multiprocessing.Queue()
    p = multiprocessing.Process(
        target=_run_unittest_worker, args=(code, test_src, q),
    )
    p.start()
    p.join(timeout=time_limit_s)
    if p.is_alive():
        p.terminate()
        p.join()
        return 0, 0, [f"timeout after {time_limit_s}s"]
    if q.empty():
        return 0, 0, ["subprocess crashed"]
    return q.get()


class BigCodeBenchExecutor:
    """Per-task stateful executor. Tracks last submitted code so eval() can
    re-run it against the test suite at task end."""

    def __init__(self, task: dict, time_limit_s: float = 30.0):
        self.task = task
        self.time_limit_s = time_limit_s
        self.last_code = ""
        self.n_calls = 0

    def __call__(self, tool_name: str, args: dict) -> str:
        if tool_name != "run_tests":
            return f"Unknown tool: {tool_name}"
        code = args.get("code", "")
        if not code:
            return "Error: 'code' is required."
        self.n_calls += 1
        self.last_code = code
        passed, total, errors = run_tests_isolated(
            code, self.task["test"], self.time_limit_s,
        )
        if total > 0 and passed == total:
            return f"PASS: all {total} tests passed."
        msg = f"FAIL: {passed}/{total} tests passed."
        if errors:
            msg += "\n" + "\n".join(errors[:3])
        return msg


class BigCodeBenchBenchmark:
    """BigCodeBench adapter. Loads from HuggingFace bigcode/bigcodebench
    parquet. Uses the latest version by default (v0.1.4)."""

    def __init__(self, version: str = "v0.1.4", time_limit_s: float = 30.0):
        self.version = version
        self.time_limit_s = time_limit_s
        self._all_tasks: list[dict] | None = None
        self._executors: dict = {}

    def _load_all_tasks(self) -> list[dict]:
        if self._all_tasks is not None:
            return self._all_tasks
        from huggingface_hub import hf_hub_download
        fname = f"data/{self.version}-00000-of-00001.parquet"
        local_path = hf_hub_download(
            repo_id="bigcode/bigcodebench",
            filename=fname,
            repo_type="dataset",
        )
        print(f"  [BCB] reading {fname} ({Path(local_path).stat().st_size:,} bytes)")
        try:
            import pyarrow.parquet as pq
            rows = pq.read_table(local_path).to_pylist()
        except ImportError:
            import pandas as pd
            rows = pd.read_parquet(local_path).to_dict("records")
        print(f"  [BCB] loaded {len(rows)} tasks")

        out = []
        for r in rows:
            task_id = r.get("task_id", "")
            qid = re.sub(r"[^A-Za-z0-9_-]", "_", str(task_id))[:80]
            entry_point = r.get("entry_point", "task_func") or "task_func"
            code_prompt = r.get("code_prompt", "") or ""
            instruct = r.get("instruct_prompt") or r.get("complete_prompt") or ""
            test_src = r.get("test", "") or ""
            if not test_src:
                continue
            question = (
                f"{instruct}\n\n"
                f"Implement the function `{entry_point}` per the specification above. "
                f"You have a `run_tests` tool that runs your code against a unittest "
                f"suite covering correctness, edge cases, and library semantics. "
                f"It reports per-test pass/fail. Iterate by calling `run_tests` "
                f"with revised code when tests fail. When you're done, provide your "
                f"final implementation as your answer (include necessary imports + "
                f"the full `def {entry_point}(...)` block)."
            )
            out.append({
                "id": qid,
                "qid": qid,
                "question": question,
                "task_id": task_id,
                "entry_point": entry_point,
                "code_prompt": code_prompt,
                "test": test_src,
                "canonical_solution": r.get("canonical_solution", "") or "",
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
        return BIGCODEBENCH_TOOLS

    def make_executor(self, task=None):
        executor = BigCodeBenchExecutor(task, time_limit_s=self.time_limit_s)
        if task is not None:
            self._executors[task["id"]] = executor

        def cleanup():
            self._executors.pop(task["id"], None) if task is not None else None

        return executor, cleanup

    def evaluate(self, task: dict, recorder=None, final_answer: str = "") -> dict:
        executor = self._executors.get(task["id"])
        code = ""
        if final_answer:
            code = _extract_task_func(final_answer)
            if "def " not in code and executor and executor.last_code:
                code = executor.last_code
        elif executor and executor.last_code:
            code = executor.last_code
        if not code:
            return {
                "correct": False,
                "loose_accuracy": 0.0,
                "error": "no code submitted",
                "n_test_calls": executor.n_calls if executor else 0,
            }
        passed, total, errors = run_tests_isolated(
            code, task["test"], self.time_limit_s,
        )
        all_pass = (total > 0 and passed == total)
        return {
            "correct": all_pass,
            "loose_accuracy": (passed / total) if total > 0 else 0.0,
            "error": ("; ".join(errors[:2])) if errors and not all_pass else "",
            "n_test_calls": executor.n_calls if executor else 0,
            "n_pass": passed,
            "n_total": total,
        }
