"""LiveCodeBench benchmark adapter.

Fallback for the second code benchmark if HumanEval+ saturates. LiveCodeBench
sources contest problems from LeetCode/AtCoder/CodeForces with a "live" cutoff
to minimize training-data contamination. Substantially harder than HumanEval+
(expected pass@1 ~15-35% for Qwen3.5-9B on LeetCode subset).

Dataset: livecodebench/code_generation_lite on HuggingFace.
This adapter filters to LeetCode platform and "functional" test type
(class-method tests, stable JSON args/return), avoiding the more fragile
stdin/stdout subset of contest problems.

Tool: same `run_tests(code)` interface as HumanEval+, returns pass/fail
per test case. Agent iterates write→test→fix.

Eval: agent's final code is run against a held-out portion of the same
test bundle (public/private split is available in the dataset; we hold
out 50% of public tests for eval; private tests are gated by harness
auth and not used here).

Follows: load_tasks, get_tools, make_executor, evaluate.
"""
from __future__ import annotations

import base64
import io
import json
import multiprocessing
import pickle
import random
import re
import zlib
from contextlib import redirect_stdout, redirect_stderr
from pathlib import Path


LIVECODEBENCH_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "run_tests",
            "description": (
                "Run the candidate Python code against the public test cases. "
                "Returns pass/fail status and error messages per failing test. "
                "Use this to iterate on your implementation when tests fail."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": (
                            "Complete Python source. For LeetCode problems, "
                            "must define `class Solution` with the requested "
                            "method. Will be exec'd before tests run."
                        ),
                    },
                },
                "required": ["code"],
            },
        },
    },
]


def _decode_test_cases(raw):
    """LiveCodeBench test cases come in several formats. Decode defensively:
      - public_test_cases: plain JSON string of a list
      - private_test_cases: base64 → zlib → pickle → JSON string → list (LCB's
        actual quadruple encoding, verified against the dataset)
    """
    if not raw:
        return []
    if isinstance(raw, list):
        return raw
    if not isinstance(raw, str):
        return []
    # Plain JSON list
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    # base64 + zlib + something
    try:
        decompressed = zlib.decompress(base64.b64decode(raw))
    except Exception:
        return []
    # Try pickle (LCB private_test_cases)
    try:
        obj = pickle.loads(decompressed)
        if isinstance(obj, str):
            return json.loads(obj)
        if isinstance(obj, list):
            return obj
    except Exception:
        pass
    # Try utf-8 + json (alternative encoding)
    try:
        s = decompressed.decode("utf-8")
        x = json.loads(s)
        if isinstance(x, str):
            x = json.loads(x)
        if isinstance(x, list):
            return x
    except Exception:
        pass
    return []


def _parse_metadata(meta_str):
    if not meta_str:
        return {}
    try:
        return json.loads(meta_str) if isinstance(meta_str, str) else meta_str
    except Exception:
        return {}


def _run_functional_tests_worker(code, tests, func_name, q):
    """Subprocess: exec code, instantiate Solution, call func_name on each test."""
    try:
        ns = {"__name__": "__test__"}
        exec(code, ns)
        if "Solution" not in ns:
            q.put((0, len(tests), [f"Solution class not defined"]))
            return
        sol = ns["Solution"]()
        if not hasattr(sol, func_name):
            q.put((0, len(tests), [f"Method '{func_name}' not on Solution"]))
            return
        method = getattr(sol, func_name)
        passed = 0
        failures = []
        for i, t in enumerate(tests):
            try:
                inp_raw = t.get("input", "")
                # Each line is a JSON-serialized argument
                if isinstance(inp_raw, str):
                    arg_lines = [ln for ln in inp_raw.split("\n") if ln.strip()]
                    args = [json.loads(ln) for ln in arg_lines]
                else:
                    args = inp_raw if isinstance(inp_raw, list) else [inp_raw]
                exp_raw = t.get("output", "")
                expected = json.loads(exp_raw) if isinstance(exp_raw, str) else exp_raw
                actual = method(*args)
                if actual == expected:
                    passed += 1
                elif len(failures) < 3:
                    failures.append(
                        f"Test {i+1}: input={args} expected={expected} got={actual}"
                    )
            except Exception as e:
                if len(failures) < 3:
                    failures.append(f"Test {i+1}: {type(e).__name__}: {e}")
        q.put((passed, len(tests), failures))
    except Exception as e:
        q.put((0, len(tests), [f"Setup error: {type(e).__name__}: {e}"]))


def run_tests_isolated(code: str, tests: list, func_name: str,
                       time_limit_s: float = 10.0) -> tuple[int, int, list[str]]:
    """Run candidate code against tests in a subprocess. Returns (passed, total, failures)."""
    if not tests:
        return 0, 0, ["no tests"]
    q = multiprocessing.Queue()
    p = multiprocessing.Process(
        target=_run_functional_tests_worker,
        args=(code, tests, func_name, q),
    )
    p.start()
    p.join(timeout=time_limit_s)
    if p.is_alive():
        p.terminate()
        p.join()
        return 0, len(tests), [f"timeout after {time_limit_s}s"]
    if q.empty():
        return 0, len(tests), ["subprocess crashed"]
    return q.get()


class LiveCodeBenchExecutor:
    """Stateful per-task executor. Tracks last submitted code and runs it
    against the visible-to-agent test set. Hidden eval set is reserved for
    final scoring."""

    def __init__(self, task: dict, time_limit_s: float = 10.0):
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
        passed, total, failures = run_tests_isolated(
            code, self.task["visible_tests"], self.task["func_name"], self.time_limit_s,
        )
        if passed == total:
            return f"PASS: all {total} tests passed."
        msg = f"FAIL: {passed}/{total} tests passed."
        if failures:
            msg += "\n" + "\n".join(failures[:3])
        return msg


class LiveCodeBenchBenchmark:
    """LiveCodeBench code-generation adapter. Filters to LeetCode functional
    problems for stable evaluation. Holds out 50% of public tests for final
    scoring (visible to the agent's tool: 50% of public tests; held-out: the
    other 50%). This isolates a write→test→fix iteration property without
    relying on private_test_cases (which are gated and base64-encoded)."""

    def __init__(self, version_tag: str = "release_v6", time_limit_s: float = 10.0,
                 difficulties: tuple[str, ...] = ("easy", "medium", "hard")):
        self.version_tag = version_tag
        self.time_limit_s = time_limit_s
        self.difficulties = difficulties
        self._all_tasks: list[dict] | None = None
        self._executors: dict = {}

    def _load_raw_rows(self) -> list[dict]:
        """Load LiveCodeBench dataset rows, bypassing `datasets.load_dataset`
        (which rejects loading scripts in datasets>=3.0).

        Strategy: directly fetch the version-specific jsonl files via
        huggingface_hub.hf_hub_download (avoids the 1.25GB-per-file
        cumulative-download cost of snapshot_download).

        Versions are cumulative per the official LCB schema:
          release_v1 = test.jsonl
          release_v2 = test.jsonl + test2.jsonl
          ...
          release_v6 = test.jsonl + test2.jsonl + ... + test6.jsonl
        """
        from huggingface_hub import hf_hub_download
        version_files = {
            "release_v1": ["test.jsonl"],
            "release_v2": ["test.jsonl", "test2.jsonl"],
            "release_v3": ["test.jsonl", "test2.jsonl", "test3.jsonl"],
            "release_v4": ["test.jsonl", "test2.jsonl", "test3.jsonl", "test4.jsonl"],
            "release_v5": ["test.jsonl", "test2.jsonl", "test3.jsonl", "test4.jsonl", "test5.jsonl"],
            "release_v6": ["test.jsonl", "test2.jsonl", "test3.jsonl", "test4.jsonl", "test5.jsonl", "test6.jsonl"],
            "release_latest": ["test.jsonl", "test2.jsonl", "test3.jsonl", "test4.jsonl", "test5.jsonl", "test6.jsonl"],
        }
        files = version_files.get(self.version_tag, ["test.jsonl"])
        print(f"  [LCB] version_tag={self.version_tag} → loading {files}")
        rows: list[dict] = []
        for fname in files:
            local_path = hf_hub_download(
                repo_id="livecodebench/code_generation_lite",
                filename=fname,
                repo_type="dataset",
            )
            print(f"  [LCB] reading {fname} ({Path(local_path).stat().st_size:,} bytes)")
            with open(local_path) as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rows.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        print(f"  [LCB] loaded {len(rows)} rows total")
        return rows

    def _load_all_tasks(self) -> list[dict]:
        if self._all_tasks is not None:
            return self._all_tasks
        rows = self._load_raw_rows()
        out = []
        for row in rows:
            platform = row.get("platform", "")
            if platform != "leetcode":
                continue
            difficulty = row.get("difficulty", "")
            if self.difficulties and difficulty not in self.difficulties:
                continue
            meta = _parse_metadata(row.get("metadata"))
            func_name = meta.get("func_name")
            if not func_name:
                continue  # Need function name for functional tests
            public_tests = _decode_test_cases(row.get("public_test_cases"))
            private_tests = _decode_test_cases(row.get("private_test_cases"))
            # Filter to functional tests only
            public_tests = [t for t in public_tests if t.get("testtype", "functional") == "functional"]
            private_tests = [t for t in private_tests if t.get("testtype", "functional") == "functional"]
            if not public_tests or not private_tests:
                continue

            # Visible to agent's run_tests tool: ALL public tests (typically 2).
            # Held-out for final eval: private tests (typically 10+). This gives
            # the agent feedback during iteration AND a rigorous held-out
            # evaluation, mirroring HumanEval+'s extended-test methodology.
            visible = public_tests
            held_out = private_tests

            qid = row.get("question_id", "") or row.get("question_title", "")
            qid = re.sub(r"[^A-Za-z0-9_-]", "_", str(qid))[:80]
            starter = row.get("starter_code", "") or ""
            question = (
                f"{row.get('question_content', '')}\n\n"
                f"Starter code:\n```python\n{starter}\n```\n\n"
                f"Implement the `{func_name}` method in `class Solution`. "
                f"You have a `run_tests` tool that runs your code against "
                f"sample tests and reports pass/fail. Iterate by calling it "
                f"again with revised code when tests fail. When you're done, "
                f"provide the final `class Solution` definition as your answer."
            )
            out.append({
                "id": qid,
                "qid": qid,
                "question": question,
                "func_name": func_name,
                "starter_code": starter,
                "visible_tests": visible,
                "held_out_tests": held_out,
                "difficulty": difficulty,
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
        return LIVECODEBENCH_TOOLS

    def make_executor(self, task=None):
        executor = LiveCodeBenchExecutor(task, time_limit_s=self.time_limit_s)
        if task is not None:
            self._executors[task["id"]] = executor

        def cleanup():
            self._executors.pop(task["id"], None) if task is not None else None

        return executor, cleanup

    def evaluate(self, task: dict, recorder=None, final_answer: str = "") -> dict:
        """Re-run the agent's last code against the held-out tests."""
        executor = self._executors.get(task["id"])
        # Try final_answer first (extract Solution class), fall back to executor.last_code
        code = ""
        if final_answer:
            # Strip markdown fences
            m = re.search(r"```(?:python)?\s*\n(.*?)```", final_answer, re.DOTALL)
            code = m.group(1) if m else final_answer
            if "class Solution" not in code and executor and executor.last_code:
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
        passed, total, failures = run_tests_isolated(
            code, task["held_out_tests"], task["func_name"], self.time_limit_s,
        )
        all_pass = (passed == total) and total > 0
        return {
            "correct": all_pass,
            "loose_accuracy": (passed / total) if total else 0.0,
            "error": ("; ".join(failures[:2])) if failures and not all_pass else "",
            "n_test_calls": executor.n_calls if executor else 0,
            "n_pass": passed,
            "n_total": total,
        }
