"""SWE-bench-Lite benchmark adapter.

Agent receives a GitHub issue description and must produce a code fix
in a checked-out repository. Tool suite matches Kim et al.
agent_scaling/env/swebench.py:113-562 (bash + read_file + edit_file with
Python syntax lint + run_tests inline + find_file + search_dir +
submit_patch with FAIL_TO_PASS pytest).

Tools:
  - bash(command)                              — execute bash in repo work_dir
  - read_file(file_path, start_line, end_line) — read with line range
  - edit_file(file_path, old_content, new_content) — exact-match replace
                                                     with auto-revert on syntax error
  - run_tests(test_names="")                   — runs FAIL_TO_PASS pytest /
                                                  Django runtests, returns pass/fail
  - find_file(file_name, directory=".")        — name pattern search
  - search_dir(search_term, directory=".")     — grep -rn in directory
  - submit_patch(reasoning)                    — captures patch, applies test_patch,
                                                  runs FAIL_TO_PASS, sets is_done

Evaluation reads from executor state (patch + is_done flag set by submit_patch),
falling back to git diff of work_dir if no submit was called. Full SWE-bench
harness eval (pytest on the patch in Docker per the official harness pipeline)
is still run offline via eval_swebench.py for paper-quality numbers.

Setup (run once on cluster):
  pip install datasets
  # Clone repos needed for selected tasks:
  git clone https://github.com/sympy/sympy.git /atlas2/u/$USER/mas_project/swebench_repos/sympy
  git clone https://github.com/django/django.git /atlas2/u/$USER/mas_project/swebench_repos/django
  # etc. for repos in your task subset
"""

import fcntl
import json
import logging
import os
import random
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────
# Tool schema
# ─────────────────────────────────────────────────────────

SWEBENCH_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "bash",
            "description": (
                "Execute a bash command in the repository working directory. "
                "Use for exploration, custom edits, or running commands not "
                "covered by the dedicated tools. Each command runs independently."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "The bash command to execute"},
                },
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": (
                "Read a file from the repository, optionally restricting to a "
                "line range. Returns file contents with line numbers."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {"type": "string", "description": "Path relative to repo root"},
                    "start_line": {"type": "integer", "description": "1-indexed start (default 1)", "default": 1},
                    "end_line": {"type": "integer", "description": "Inclusive end, -1 = end of file (default -1)", "default": -1},
                },
                "required": ["file_path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "edit_file",
            "description": (
                "Replace exact text content in a file. old_content must match "
                "EXACTLY ONE location in the file (include surrounding context "
                "if needed for uniqueness). For Python files, edits that produce "
                "syntax errors are automatically reverted."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {"type": "string", "description": "Path relative to repo root"},
                    "old_content": {"type": "string", "description": "Exact text to find"},
                    "new_content": {"type": "string", "description": "Replacement text"},
                },
                "required": ["file_path", "old_content", "new_content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_tests",
            "description": (
                "Run tests to verify your changes BEFORE submitting. If no "
                "test_names provided, runs the fail-to-pass tests from the "
                "issue. Returns per-test pass/fail. This does NOT submit your "
                "patch — use submit_patch when you are ready to finalize."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "test_names": {"type": "string", "description": "Space-separated test names; empty = FAIL_TO_PASS tests", "default": ""},
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "find_file",
            "description": "Find files matching a name pattern (supports wildcards like '*.py').",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_name": {"type": "string", "description": "Name or wildcard pattern"},
                    "directory": {"type": "string", "description": "Search root relative to repo (default '.')", "default": "."},
                },
                "required": ["file_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_dir",
            "description": "Search for a term across .py files in a directory (grep -rn equivalent).",
            "parameters": {
                "type": "object",
                "properties": {
                    "search_term": {"type": "string", "description": "Text or regex pattern"},
                    "directory": {"type": "string", "description": "Directory relative to repo (default '.')", "default": "."},
                },
                "required": ["search_term"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "submit_patch",
            "description": (
                "Submit your current changes as a patch and run evaluation. "
                "Generates a git diff, applies the test_patch, and runs the "
                "FAIL_TO_PASS test suite. After calling this the task is "
                "considered complete and further work has no effect."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "reasoning": {"type": "string", "description": "Brief explanation of the fix"},
                },
                "required": ["reasoning"],
            },
        },
    },
]

# ─────────────────────────────────────────────────────────
# Executor
# ─────────────────────────────────────────────────────────

MAX_OUTPUT_CHARS = 8000
COMMAND_TIMEOUT = 60


class SWEBenchExecutor:
    """Executes Kim et al.-aligned tool suite in a repo working directory.

    Owns per-instance state:
      - work_dir            : git worktree path (set in __init__)
      - task                : SWE-bench task dict (set via attach_task before use)
      - is_done             : True after submit_patch called
      - success             : True after submit_patch + FAIL_TO_PASS all passed
      - submitted_patch     : git diff captured at submit time (before test_patch)
      - submit_reasoning    : the 'reasoning' arg from the agent's submit_patch call
    """

    def __init__(self, work_dir):
        self.work_dir = work_dir
        self.task = None
        self.is_done = False
        self.success = False
        self.submitted_patch = ""
        self.submit_reasoning = ""
        # MAS_TEST_FEEDBACK — verifiable-feedback ablation (2026-07-13):
        #   ""    (default) : current behavior — run_tests in the agent worktree
        #                     (test_patch absent → mostly uninformative), raw output.
        #   "off" : same worktree as default, but counts-only return format.
        #           Format-normalized control arm.
        #   "on"  : run_tests executes in a SHADOW worktree (base_commit +
        #           test_patch, never exposed to any other tool), with the
        #           agent's current diff applied — returns counts only
        #           (no test names/tracebacks → no reward-hacking surface).
        #           This is the martingale-break arm: a genuine, private,
        #           correctness-correlated signal via the SAME tool schema.
        #   "inject" (2026-07-14): forced-uptake arm. Same shadow counts as
        #           "on", but the agent does not have to ask: after any tool
        #           call that changed the working diff (edit_file or a
        #           mutating bash), the counts line is APPENDED to that tool's
        #           result. Motivated by the "on"-arm forensics: only ~22% of
        #           tasks ever called run_tests (Decent 6-14%), so the pooled
        #           null is an availability-null, not a verification-null.
        #           Diff-hash gating: tests run only when the diff actually
        #           changed, so read-only calls cost nothing extra.
        self.test_feedback_mode = os.environ.get("MAS_TEST_FEEDBACK", "").strip().lower()
        self._shadow_dir = None      # lazily created on first run_tests ("on" mode)
        self._shadow_error = None    # sticky setup failure message
        self._last_feedback_diff = None   # inject mode: hash of diff at last feedback

    def attach_task(self, task):
        """Attach the SWE-bench task dict so run_tests/submit_patch can read
        FAIL_TO_PASS and test_patch. Called by make_executor."""
        self.task = task

    def __call__(self, tool_name, args):
        if self.is_done and tool_name != "bash":
            # After submit_patch, ignore further state-changing calls. Allow
            # bash for trivial inspection (cheap, no commit-to-answer).
            return "Task already submitted; further tool calls have no effect on the patch."
        dispatch = {
            "bash":         self._bash,
            "read_file":    self._read_file,
            "edit_file":    self._edit_file,
            "run_tests":    self._run_tests,
            "find_file":    self._find_file,
            "search_dir":   self._search_dir,
            "submit_patch": self._submit_patch,
        }
        fn = dispatch.get(tool_name)
        if fn is None:
            return f"Unknown tool: {tool_name}"
        try:
            result = fn(args or {})
        except Exception as e:
            return f"Error executing {tool_name}: {e}"
        # Forced-uptake feedback arm: append shadow test counts to the result
        # of any call that changed the working diff (agent need not ask).
        if (self.test_feedback_mode == "inject" and not self.is_done
                and tool_name in ("bash", "edit_file")):
            try:
                fb = self._maybe_inject_feedback()
            except Exception as e:
                fb = None
                print(f"  inject-feedback error (non-fatal): {e}")
            if fb:
                result = f"{result}\n\n{fb}"
        return result

    def _maybe_inject_feedback(self):
        """inject mode: if the agent's working diff changed since the last
        feedback, run the shadow counts and return the counts line."""
        import hashlib
        diff = self.get_patch()
        if not diff.strip():
            return None
        h = hashlib.md5(diff.encode()).hexdigest()
        if h == self._last_feedback_diff:
            return None
        self._last_feedback_diff = h
        tests = self._fail_to_pass_tests()
        if not tests:
            return None
        return self._run_tests_counts(tests)

    # ─── bash ───
    def _bash(self, args):
        command = args.get("command", "")
        if not command:
            return "Error: command is required."
        try:
            result = subprocess.run(
                command, shell=True, cwd=self.work_dir,
                capture_output=True, text=True, timeout=COMMAND_TIMEOUT,
                env={**os.environ, "PAGER": "cat", "GIT_PAGER": "cat"},
            )
            output = result.stdout
            if result.stderr:
                output = output + "\nSTDERR:\n" + result.stderr if output else result.stderr
            if not output.strip():
                output = f"(exit code {result.returncode})"
            return self._truncate_output(output)
        except subprocess.TimeoutExpired:
            return f"Error: command timed out after {COMMAND_TIMEOUT} seconds."

    # ─── read_file ───
    def _read_file(self, args):
        file_path = args.get("file_path", "")
        if not file_path:
            return "Error: file_path is required."
        start_line = int(args.get("start_line", 1) or 1)
        end_line = int(args.get("end_line", -1) or -1)
        abs_path = os.path.join(self.work_dir, file_path)
        if not os.path.isfile(abs_path):
            return f"Error: file not found: {file_path}"
        try:
            with open(abs_path) as f:
                lines = f.readlines()
        except Exception as e:
            return f"Error reading file: {e}"
        n = len(lines)
        if end_line == -1 or end_line > n:
            end_line = n
        if start_line < 1:
            start_line = 1
        if start_line > n:
            return f"(file has {n} lines; start_line {start_line} beyond end)"
        selected = lines[start_line - 1:end_line]
        output = "".join(f"{start_line + i:6d}\t{ln}" for i, ln in enumerate(selected))
        if not output.endswith("\n"):
            output += "\n"
        return self._truncate_output(output)

    # ─── edit_file ───
    def _edit_file(self, args):
        file_path = args.get("file_path", "")
        old_content = args.get("old_content", "")
        new_content = args.get("new_content", "")
        if not file_path or old_content is None or new_content is None:
            return "Error: file_path, old_content, and new_content are all required."
        abs_path = os.path.join(self.work_dir, file_path)
        if not os.path.isfile(abs_path):
            return f"Error: file not found: {file_path}"
        try:
            with open(abs_path) as f:
                current = f.read()
        except Exception as e:
            return f"Error reading file: {e}"
        if old_content not in current:
            return (f"Error: old_content was not found in {file_path}. "
                    "Make sure to include exact text including whitespace and indentation.")
        count = current.count(old_content)
        if count > 1:
            return (f"Error: old_content matches {count} locations in {file_path}. "
                    "Provide more surrounding context to make the match unique.")
        new_file = current.replace(old_content, new_content, 1)
        try:
            with open(abs_path, "w") as f:
                f.write(new_file)
        except Exception as e:
            return f"Error writing file: {e}"
        # Syntax lint for Python — revert on syntax error
        if file_path.endswith(".py"):
            lint = subprocess.run(
                ["python3", "-c",
                 f"compile(open({abs_path!r}).read(), {abs_path!r}, 'exec')"],
                capture_output=True, text=True, timeout=10,
            )
            if lint.returncode != 0:
                try:
                    with open(abs_path, "w") as f:
                        f.write(current)
                except Exception:
                    pass
                err = (lint.stderr or lint.stdout or "").strip().split("\n")[-1]
                return (f"Edit reverted — syntax error in {file_path}: {err}\n"
                        "Please fix the syntax and try again.")
        return f"Successfully edited {file_path}."

    # ─── run_tests ───
    def _is_django_repo(self):
        if not self.task:
            return False
        return "django" in self.task.get("repo", "").lower()

    def _django_test_label(self, test_str):
        """Convert Django 'test_method (module.Class)' format to 'module.Class.test_method'."""
        m = re.match(r"^(\w+)\s+\(([^)]+)\)", test_str.strip())
        if m:
            return f"{m.group(2)}.{m.group(1)}"
        return test_str.strip()

    def _django_module_labels(self, tests):
        labels = set()
        for t in tests:
            m = re.match(r"^\w+\s+\(([^)]+)\)", t.strip())
            if m:
                module_class = m.group(1)
                parts = module_class.rsplit(".", 1)
                labels.add(parts[0] if len(parts) == 2 else module_class)
            else:
                labels.add(t.strip())
        return sorted(labels)

    def _run_test_subprocess(self, cmd, timeout=180, cwd=None):
        try:
            r = subprocess.run(cmd, shell=True, cwd=cwd or self.work_dir,
                               capture_output=True, text=True, timeout=timeout,
                               env={**os.environ, "PAGER": "cat", "GIT_PAGER": "cat"})
            return r.returncode, r.stdout, r.stderr
        except subprocess.TimeoutExpired:
            return -1, "", f"Test execution timed out after {timeout}s"

    # ─── verifiable-feedback ablation: shadow worktree (MAS_TEST_FEEDBACK=on) ───
    def _ensure_shadow(self):
        """Create {work_dir}_shadow: worktree at base_commit + test_patch,
        committed so each run_tests call can reset --hard. Never surfaced to
        bash/read_file/find_file (those are cwd=self.work_dir only)."""
        if self._shadow_dir or self._shadow_error:
            return
        base_commit = self.task.get("base_commit", "")
        shadow = f"{self.work_dir}_shadow"
        # serialize worktree bookkeeping like make_executor does (shared NFS repos)
        rc, common, _ = self._run_test_subprocess(
            "git rev-parse --path-format=absolute --git-common-dir", timeout=15)
        repo_root = os.path.dirname((common or "").strip()) or self.work_dir
        try:
            with open(os.path.join(repo_root, ".mas_energy_worktree.lock"), "w") as fh:
                fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
                try:
                    rc, _, err = self._run_test_subprocess(
                        f"git worktree add --detach {shadow} {base_commit}", timeout=180)
                finally:
                    fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
        except OSError as e:
            rc, err = 1, str(e)
        if rc != 0:
            self._shadow_error = f"shadow worktree setup failed: {(err or '').strip()[:200]}"
            return
        test_patch = self.task.get("test_patch") or ""
        if test_patch:
            tmp = os.path.join(tempfile.gettempdir(), f"shadow_tp_{os.getpid()}.diff")
            with open(tmp, "w") as f:
                f.write(test_patch)
            rc, _, err = self._run_test_subprocess(f"git apply {tmp}", timeout=30, cwd=shadow)
            if rc != 0:
                rc, _, err = self._run_test_subprocess(
                    f"git apply --3way {tmp}", timeout=30, cwd=shadow)
            try:
                os.unlink(tmp)
            except OSError:
                pass
            if rc != 0:
                self._shadow_error = f"test_patch failed to apply in shadow: {(err or '').strip()[:200]}"
                shutil.rmtree(shadow, ignore_errors=True)
                return
        # commit so per-call reset --hard restores base_commit + test_patch exactly
        self._run_test_subprocess(
            "git -c user.email=mas@energy -c user.name=mas add -A && "
            "git -c user.email=mas@energy -c user.name=mas commit -qm shadow-baseline",
            timeout=60, cwd=shadow)
        self._shadow_dir = shadow

    def _sync_shadow_to_agent_state(self):
        """Reset shadow to base+test_patch, then apply the agent's current diff.
        Returns (ok, note)."""
        self._run_test_subprocess("git reset --hard -q && git clean -fdq",
                                  timeout=60, cwd=self._shadow_dir)
        rc, diff_out, _ = self._run_test_subprocess("git diff", timeout=15)
        agent_diff = (diff_out or "").strip()
        if not agent_diff:
            return True, "(no changes made yet — testing the unmodified repo)"
        tmp = os.path.join(tempfile.gettempdir(), f"shadow_ad_{os.getpid()}.diff")
        with open(tmp, "w") as f:
            f.write(agent_diff + "\n")
        rc, _, err = self._run_test_subprocess(f"git apply {tmp}", timeout=30,
                                               cwd=self._shadow_dir)
        if rc != 0:
            rc, _, err = self._run_test_subprocess(f"git apply --3way {tmp}", timeout=30,
                                                   cwd=self._shadow_dir)
        try:
            os.unlink(tmp)
        except OSError:
            pass
        if rc != 0:
            return False, "(warning: your current changes could not be applied to the test environment; results reflect the unmodified repo)"
        return True, ""

    def _run_tests_counts(self, tests):
        """Counts-only run_tests for the MAS_TEST_FEEDBACK ablation arms.
        'on'/'inject' → run in the shadow worktree (base+test_patch+agent diff).
        'off'         → run in the agent worktree (format-normalized control)."""
        note = ""
        if self.test_feedback_mode in ("on", "inject"):
            self._ensure_shadow()
            if self._shadow_error:
                return (f"## Test Results: unavailable ({self._shadow_error})\n\n"
                        "Use your own judgment to fix issues before calling submit_patch.")
            ok, note = self._sync_shadow_to_agent_state()
            run_cwd = self._shadow_dir
        else:
            run_cwd = self.work_dir
        n_pass = 0
        if self._is_django_repo():
            labels = " ".join(self._django_module_labels(tests))
            rc, _, _ = self._run_test_subprocess(
                f"python tests/runtests.py --parallel 1 {labels} 2>&1",
                timeout=300, cwd=run_cwd)
            n_pass = len(tests) if rc == 0 else 0
        else:
            for t in tests:
                rc, _, _ = self._run_test_subprocess(
                    f"python -m pytest {t} -x 2>&1", timeout=180, cwd=run_cwd)
                if rc == 0:
                    n_pass += 1
        line = f"## Test Results: {n_pass}/{len(tests)} FAIL_TO_PASS tests passing."
        if note:
            line += f"\n{note}"
        return line + "\n\nUse this to fix issues before calling submit_patch."

    def _fail_to_pass_tests(self):
        """Parse the task's FAIL_TO_PASS list (dataset stores it JSON-encoded)."""
        if not self.task:
            return []
        tests = self.task.get("fail_to_pass") or []
        if isinstance(tests, str):
            try:
                tests = json.loads(tests)
            except json.JSONDecodeError:
                tests = tests.split()
        return tests

    def _run_tests(self, args):
        if not self.task:
            return "Error: no task attached to executor (cannot run tests)."
        names = (args.get("test_names") or "").strip()
        tests = names.split() if names else self._fail_to_pass_tests()
        if not tests:
            return "No tests to run (FAIL_TO_PASS empty or missing)."
        # Verifiable-feedback ablation arms: counts-only format (same tool
        # name/schema/description; only the backend + return format change).
        if self.test_feedback_mode in ("on", "off", "inject"):
            return self._run_tests_counts(tests)
        results = []
        all_pass = True
        if self._is_django_repo():
            labels = " ".join(self._django_module_labels(tests))
            cmd = f"python tests/runtests.py --parallel 1 {labels} 2>&1"
            rc, stdout, stderr = self._run_test_subprocess(cmd, timeout=300)
            combined = (stdout or "") + (stderr or "")
            if rc == 0:
                for t in tests:
                    results.append(f"  PASS: {t}")
            else:
                all_pass = False
                for t in tests:
                    results.append(f"  FAIL: {t}")
                tail = combined[-2000:] if len(combined) > 2000 else combined
                results.append(f"\n[output tail]:\n{tail}")
        else:
            for t in tests:
                cmd = f"python -m pytest {t} -x 2>&1"
                rc, stdout, stderr = self._run_test_subprocess(cmd, timeout=180)
                passed = (rc == 0)
                if not passed:
                    all_pass = False
                    tail = ((stdout or "") + (stderr or ""))[-1000:]
                    results.append(f"  FAIL: {t}\n    {tail}")
                else:
                    results.append(f"  PASS: {t}")
        status = "ALL PASSED" if all_pass else "SOME FAILED"
        return (f"## Test Results: {status}\n\n" + "\n".join(results)
                + "\n\nUse this to fix issues before calling submit_patch.")

    # ─── find_file ───
    def _find_file(self, args):
        name = args.get("file_name", "")
        directory = args.get("directory", ".") or "."
        if not name:
            return "Error: file_name is required."
        cmd = f"find {directory} -type f -name '{name}' 2>/dev/null | head -50"
        rc, stdout, _ = self._run_test_subprocess(cmd, timeout=30)
        if not (stdout or "").strip():
            return f"No files matching '{name}' found in {directory}."
        lines = (stdout or "").strip().split("\n")
        out = f"Found {len(lines)} file(s):\n" + "\n".join(lines)
        if len(lines) == 50:
            out += "\n(Results truncated to 50 files. Narrow your search.)"
        return out

    # ─── search_dir ───
    def _search_dir(self, args):
        term = args.get("search_term", "")
        directory = args.get("directory", ".") or "."
        if not term:
            return "Error: search_term is required."
        escaped = term.replace("'", "'\\''")
        cmd = f"grep -rn --include='*.py' '{escaped}' {directory} 2>/dev/null | head -30"
        rc, stdout, _ = self._run_test_subprocess(cmd, timeout=30)
        if not (stdout or "").strip():
            return f"No matches for '{term}' in {directory}."
        lines = (stdout or "").strip().split("\n")
        out = f"Found matches ({len(lines)} shown):\n" + "\n".join(lines)
        if len(lines) == 30:
            out += "\n(Results truncated to 30 lines. Narrow your search.)"
        return out

    # ─── submit_patch ───
    def _submit_patch(self, args):
        if not self.task:
            return "Error: no task attached to executor (cannot submit)."
        if self.is_done:
            return "Task already submitted."
        reasoning = (args.get("reasoning") or "")[:1000]
        self.submit_reasoning = reasoning
        # 1. Capture the model patch BEFORE applying test_patch
        patch_rc, patch_out, patch_err = self._run_test_subprocess("git diff", timeout=15)
        self.submitted_patch = (patch_out or "").strip()
        if not self.submitted_patch:
            self.is_done = True
            self.success = False
            return "No changes detected. You must modify files before submitting."
        # 2. Apply test_patch
        test_patch = self.task.get("test_patch") or ""
        if test_patch:
            tmp = os.path.join(tempfile.gettempdir(), f"test_patch_{os.getpid()}.diff")
            try:
                with open(tmp, "w") as f:
                    f.write(test_patch)
                apply_rc, _, apply_err = self._run_test_subprocess(
                    f"git apply {tmp}", timeout=30)
                if apply_rc != 0:
                    # fallback with --3way
                    self._run_test_subprocess(f"git apply --3way {tmp}", timeout=30)
            finally:
                try:
                    os.unlink(tmp)
                except OSError:
                    pass
        # 3. Run FAIL_TO_PASS
        test_result = self._run_tests({"test_names": ""})
        self.is_done = True
        self.success = "ALL PASSED" in test_result
        status = "RESOLVED" if self.success else "FAILED"
        return (f"## Submit Result: {status}\n\n"
                f"**Reasoning:** {reasoning}\n\n"
                f"**Patch size:** {len(self.submitted_patch)} chars\n\n"
                f"{test_result}")

    # ─── helpers ───
    def _truncate_output(self, output):
        if len(output) > MAX_OUTPUT_CHARS:
            half = MAX_OUTPUT_CHARS // 2
            output = output[:half] + "\n... (truncated) ...\n" + output[-half:]
        return output

    def get_patch(self):
        """Get the captured submit patch if submitted, else git diff fallback."""
        if self.submitted_patch:
            return self.submitted_patch
        try:
            result = subprocess.run(
                "git diff", shell=True, cwd=self.work_dir,
                capture_output=True, text=True, timeout=10,
            )
            return result.stdout.strip()
        except Exception:
            return ""

    def env_done(self):
        return self.is_done


# ─────────────────────────────────────────────────────────
# Evaluation (lightweight, deferred full eval to SWE-bench harness)
# ─────────────────────────────────────────────────────────

def evaluate_swebench(patch, task):
    """Lightweight evaluation: did the agent produce a non-empty patch?

    Full evaluation (run pytest in Docker) is deferred. For now, we check:
    - patch is non-empty
    - patch contains valid diff headers
    - patch modifies .py files (not just config/docs)
    """
    if not patch:
        return {
            "correct": False,
            "has_patch": False,
            "patch_lines": 0,
            "modifies_python": False,
            "patch": "",
        }

    lines = patch.split("\n")
    has_diff_header = any(l.startswith("diff --git") for l in lines)
    modifies_py = any(l.startswith("diff --git") and ".py" in l for l in lines)
    additions = sum(1 for l in lines if l.startswith("+") and not l.startswith("+++"))
    deletions = sum(1 for l in lines if l.startswith("-") and not l.startswith("---"))

    return {
        "correct": has_diff_header and modifies_py,
        "has_patch": bool(patch),
        "has_diff_header": has_diff_header,
        "modifies_python": modifies_py,
        "patch_lines": len(lines),
        "additions": additions,
        "deletions": deletions,
        "patch": patch,
    }


# ─────────────────────────────────────────────────────────
# Benchmark class
# ─────────────────────────────────────────────────────────

class SWEBenchBenchmark:
    """Adapter for SWE-bench-Lite.

    Follows the four-method pattern:
        load_tasks, get_tools, make_executor, evaluate

    Requires repos pre-cloned to repos_dir. Per task, creates a git
    worktree at the task's base_commit for isolated execution.
    """

    def __init__(self, repos_dir=None, data_dir=None):
        if repos_dir is None:
            repos_dir = os.environ.get(
                "SWEBENCH_REPOS",
                os.path.expanduser("/atlas2/u/{}/mas_project/swebench_repos".format(
                    os.environ.get("USER", ""))),
            )
        self.repos_dir = Path(repos_dir)
        self.data_dir = Path(data_dir) if data_dir else None
        self._dataset = None
        # _work_dirs maps instance_id -> either:
        #   shared mode  : (executor, work_dir)   [legacy/default]
        #   isolated mode: {"isolated": True, "agents": [(executor, work_dir), ...], "dispatcher": executor}
        self._work_dirs = {}
        # MAS_SWEBENCH_ISOLATED_WORKSPACES=1 → per-agent worktrees (ablates the
        # hidden filesystem communication channel that exists when all M agents
        # in a MAS topology share one work_dir).
        self._isolated_workspaces = os.environ.get(
            "MAS_SWEBENCH_ISOLATED_WORKSPACES") == "1"
        # MAS_SWEBENCH_LEAD_CURATION=1 → workers still SHARE one repo (realistic
        # MAS-on-code), but the centralized orchestrator curates the final patch
        # in its own clean lead worktree instead of shipping the raw union git
        # diff. Independent/decentralized keep the shared-union answer unchanged.
        self._lead_curation = (
            os.environ.get("MAS_SWEBENCH_LEAD_CURATION") == "1"
            and not self._isolated_workspaces
        )

    def _load_dataset(self):
        if self._dataset is not None:
            return self._dataset
        from datasets import load_dataset
        self._dataset = load_dataset("princeton-nlp/SWE-bench_Lite", split="test")
        return self._dataset

    def load_tasks(self, n_tasks=None, seed=42):
        ds = self._load_dataset()

        # Filter to repos we have cloned
        available_repos = set()
        if self.repos_dir.exists():
            for d in self.repos_dir.iterdir():
                if d.is_dir() and (d / ".git").exists():
                    available_repos.add(d.name)

        tasks = []
        for row in ds:
            repo_name = row["repo"].split("/")[-1]
            if available_repos and repo_name not in available_repos:
                continue

            problem = row["problem_statement"]
            formatted_q = (
                f"You are working on the repository at the current directory. "
                f"Please solve the following GitHub issue by modifying the source code.\n\n"
                f"## Important: Repository Location\n\n"
                f"The repository is at your **current working directory** (use `pwd` "
                f"to confirm). DO NOT use `/testbed` paths — that directory does NOT "
                f"exist in this environment. All paths must be either relative to the "
                f"current directory (e.g., `./django/contrib/admin/`) or absolute paths "
                f"starting from `pwd`. Start by running `pwd` and `ls` to orient yourself.\n\n"
                f"## Issue\n\n{problem}\n\n"
                f"## Instructions\n\n"
                f"1. Run `pwd` and `ls` to see your current directory\n"
                f"2. Explore the repository structure from the current directory\n"
                f"3. Find the relevant source files\n"
                f"4. Understand the bug or feature request\n"
                f"5. Make the necessary code changes using bash commands\n"
                f"6. Verify your changes work correctly\n"
                f"7. Call `submit_patch` when your fix is ready\n\n"
                f"Make minimal, targeted changes. Do not modify test files."
            )

            tasks.append({
                "id": row["instance_id"],
                "instance_id": row["instance_id"],
                "question": formatted_q,
                "question_text": problem,
                "repo": row["repo"],
                "repo_name": repo_name,
                "base_commit": row["base_commit"],
                "patch": row["patch"],
                "test_patch": row["test_patch"],
                "fail_to_pass": row["FAIL_TO_PASS"],
                "pass_to_pass": row["PASS_TO_PASS"],
            })

        if n_tasks and n_tasks < len(tasks):
            rng = random.Random(seed)
            rng.shuffle(tasks)
            tasks = tasks[:n_tasks]

        log.info(f"Loaded {len(tasks)} SWE-bench tasks (repos available: {available_repos})")
        return tasks

    def get_tools(self):
        return SWEBENCH_TOOLS

    def _create_worktree(self, repo_path, base_commit, instance_id, suffix=""):
        """Create a working tree at base_commit, return (executor, work_dir).

        Uses `git worktree add --detach` to match Kim et al.'s default
        deployment. The worktree shares the parent repo's .git directory, so
        the agent has access to the full commit history (including future
        fix commits) via `git log --all`. This matches the standard SWE-agent
        / Kim et al. setup; we previously tried scrubbing history but
        empirically rejected the leakage hypothesis (similarity-to-gold
        decreases with compute, opposite of what leakage would predict).
        """
        tmp_base = self.repos_dir.parent / "swebench_worktrees"
        tmp_base.mkdir(exist_ok=True)
        prefix = f"swe_{instance_id}{suffix}_"
        work_dir = Path(tempfile.mkdtemp(prefix=prefix, dir=tmp_base))
        try:
            # 900s (was 180): with two model grids running SWE cells against the
            # SAME shared repos, worktree adds queue behind the per-repo flock —
            # the 2026-07-16 MoE pilot lost 9/10 tasks to 180s timeouts while the
            # Gemma chain held the locks. Contention is expected steady-state now.
            subprocess.run(
                f"git worktree add --detach {work_dir} {base_commit}",
                shell=True, cwd=str(repo_path),
                capture_output=True, text=True, check=True, timeout=900,
            )
        except subprocess.CalledProcessError as e:
            shutil.rmtree(work_dir, ignore_errors=True)
            raise RuntimeError(
                f"Failed to create worktree for {instance_id}{suffix} at {base_commit}: {e.stderr}"
            )
        return SWEBenchExecutor(str(work_dir)), work_dir

    def _scrub_stale_pth_files(self, work_dir):
        """Scrub site-packages of .pth/.egg-link entries pointing into a
        removed worktree. See cleanup() rationale below."""
        try:
            import site
            for sp_str in site.getsitepackages():
                sp = Path(sp_str)
                for pattern in ("*.pth", "*.egg-link"):
                    for f in sp.glob(pattern):
                        try:
                            content = f.read_text()
                            if "swebench_worktrees" in content or str(work_dir) in content:
                                f.unlink()
                        except Exception:
                            pass
        except Exception:
            pass

    def make_executor(self, task=None):
        """Create a working directory (or N work_dirs, if isolated mode).

        Default (shared): one work_dir per task. All M MAS agents share it.
        Isolated (env MAS_SWEBENCH_ISOLATED_WORKSPACES=1): one work_dir per
        agent (M=config.N_AGENTS). A dispatcher executor reads the current
        agent_id contextvar and routes each tool call to the right work_dir's
        executor. Ablates the hidden filesystem communication channel.
        """
        repo_name = task["repo_name"]
        base_commit = task["base_commit"]
        instance_id = task["instance_id"]

        repo_path = self.repos_dir / repo_name
        if not repo_path.exists():
            raise FileNotFoundError(
                f"Repository {repo_name} not found at {repo_path}. "
                f"Clone it: git clone https://github.com/{task['repo']}.git {repo_path}"
            )

        # Serialize git worktree ops across concurrent array tasks. `git worktree
        # prune` + `git worktree add --detach` both mutate .git/worktrees/
        # bookkeeping and grab .git/index.lock. When multiple SLURM array tasks
        # run concurrently on the same shared SWEBENCH_REPOS (atlas' /atlas2 NFS),
        # they contend for these locks and time out at 30s. Observed rate: up to
        # 80% failure at Ind k=7 with array throttle 5.
        # Fix: flock a per-repo lock file so worktree ops are serialized. The
        # actual model inference (which is what we're timing for energy) is NOT
        # inside this lock — only the git bookkeeping is serialized.
        lock_path = repo_path / ".mas_energy_worktree.lock"
        with open(lock_path, "w") as _lock_fh:
            fcntl.flock(_lock_fh.fileno(), fcntl.LOCK_EX)
            try:
                subprocess.run(
                    "git worktree prune",
                    shell=True, cwd=repo_path,
                    # 30s was too short for the k1R30 refill on /atlas2 NFS
                    # (14/21 tasks died on prune timeout, 2026-07-25). NFS
                    # worktree-list scans are slow; node-local /tmp repos avoid
                    # it entirely, but raise the ceiling too as belt-and-braces.
                    # 120->300 (2026-08-03): low-k cross-model cent cells on
                    # atlas29 lost samples to prune timeouts when moe+gemma grids
                    # shared the repos (gemma k1 6/50 valid, moe k1 32/50).
                    capture_output=True, timeout=300,
                )
                if not self._isolated_workspaces:
                    # Legacy shared-workspace path (unchanged behaviour).
                    executor, work_dir = self._create_worktree(repo_path, base_commit, instance_id)
                    executor.attach_task(task)
                else:
                    # Isolated path — handled below, still needs the lock
                    executor = work_dir = None
            finally:
                fcntl.flock(_lock_fh.fileno(), fcntl.LOCK_UN)

        # NOTE: for backward compatibility, non-isolated path returns here with
        # (executor, work_dir); isolated path continues below.
        if not self._isolated_workspaces:
            pass  # executor/work_dir set inside lock above

            # Lead-curation mode: workers share `executor`/`work_dir` (realistic);
            # the orchestrator (agent_id="synthesizer") is routed to a separate
            # clean lead worktree where it writes the curated final patch.
            if self._lead_curation:
                from agent_context import current_agent_id
                lead_exec, lead_wd = self._create_worktree(
                    repo_path, base_commit, instance_id, suffix="_lead")
                lead_exec.attach_task(task)

                def dispatcher(tool_name, args):
                    aid = current_agent_id.get()
                    if aid and ("synth" in str(aid) or "lead" in str(aid)):
                        return lead_exec(tool_name, args)
                    return executor(tool_name, args)

                def cleanup():
                    patches_dir = self.repos_dir.parent / "swebench_patches"
                    patches_dir.mkdir(exist_ok=True)
                    for tag, ex, wd in (("", executor, work_dir),
                                        ("_lead", lead_exec, lead_wd)):
                        try:
                            (patches_dir / f"{instance_id}{tag}.patch").write_text(
                                ex.get_patch() or "")
                        except Exception as e:
                            log.warning(f"Failed to save patch for {instance_id}{tag}: {e}")
                        for w in (wd, f"{wd}_shadow"):
                            try:
                                subprocess.run(
                                    f"git worktree remove --force {w}",
                                    shell=True, cwd=repo_path,
                                    capture_output=True, timeout=30,
                                )
                            except Exception:
                                pass
                            shutil.rmtree(w, ignore_errors=True)
                        self._scrub_stale_pth_files(wd)
                    self._work_dirs.pop(instance_id, None)

                self._work_dirs[instance_id] = {
                    "lead_curation": True,
                    "shared": (executor, work_dir),
                    "lead": (lead_exec, lead_wd),
                }
                return dispatcher, cleanup

            def cleanup():
                try:
                    patch = executor.get_patch()
                    patches_dir = self.repos_dir.parent / "swebench_patches"
                    patches_dir.mkdir(exist_ok=True)
                    (patches_dir / f"{instance_id}.patch").write_text(patch)
                except Exception as e:
                    log.warning(f"Failed to save patch for {instance_id}: {e}")
                for wd in (work_dir, f"{work_dir}_shadow"):
                    try:
                        subprocess.run(
                            f"git worktree remove --force {wd}",
                            shell=True, cwd=repo_path,
                            capture_output=True, timeout=30,
                        )
                    except Exception:
                        pass
                    shutil.rmtree(wd, ignore_errors=True)
                self._scrub_stale_pth_files(work_dir)
                self._work_dirs.pop(instance_id, None)

            self._work_dirs[instance_id] = (executor, work_dir)
            return executor, cleanup

        # Isolated mode: create N_AGENTS worktrees + 1 synthesizer worktree,
        # dispatcher routes by contextvar. The extra worktree (index N_AGENTS) is
        # a clean checkout the centralized orchestrator uses in its synthesis
        # react loop (agent_id="synthesizer") to write the integrated final fix.
        # Independent/decentralized never populate it, so it stays empty and
        # evaluate() falls back to best-of-N over the worker worktrees.
        from config import N_AGENTS
        from agent_context import current_agent_id, extract_agent_idx

        SYNTH_IDX = N_AGENTS
        agent_execs = []
        for i in range(N_AGENTS):
            exec_i, wd_i = self._create_worktree(repo_path, base_commit, instance_id,
                                                 suffix=f"_a{i}")
            exec_i.attach_task(task)
            agent_execs.append((exec_i, wd_i))
        synth_exec, synth_wd = self._create_worktree(repo_path, base_commit,
                                                     instance_id, suffix="_synth")
        synth_exec.attach_task(task)
        agent_execs.append((synth_exec, synth_wd))

        def dispatcher(tool_name, args):
            aid = current_agent_id.get()
            if aid and "synth" in str(aid):
                idx = SYNTH_IDX
            else:
                idx = extract_agent_idx(aid) % N_AGENTS
            executor_i, _ = agent_execs[idx]
            return executor_i(tool_name, args)

        def cleanup():
            patches_dir = self.repos_dir.parent / "swebench_patches"
            patches_dir.mkdir(exist_ok=True)
            for i, (exec_i, wd_i) in enumerate(agent_execs):
                try:
                    patch = exec_i.get_patch()
                    (patches_dir / f"{instance_id}_agent_{i}.patch").write_text(patch)
                except Exception as e:
                    log.warning(f"Failed to save isolated patch for {instance_id}/agent_{i}: {e}")
                for wd in (wd_i, f"{wd_i}_shadow"):
                    try:
                        subprocess.run(
                            f"git worktree remove --force {wd}",
                            shell=True, cwd=repo_path,
                            capture_output=True, timeout=30,
                        )
                    except Exception:
                        pass
                    shutil.rmtree(wd, ignore_errors=True)
                self._scrub_stale_pth_files(wd_i)
            self._work_dirs.pop(instance_id, None)

        self._work_dirs[instance_id] = {
            "isolated": True,
            "agents": agent_execs,
            "n_workers": N_AGENTS,
            "synth_idx": SYNTH_IDX,
            "dispatcher": dispatcher,
        }
        return dispatcher, cleanup

    def seed_lead_worktree(self, instance_id, patch):
        """Apply a candidate patch to the lead worktree so the orchestrator's
        synthesis starts from the team's combined work instead of a blank
        checkout. Without this, the synthesizer usually explores (bash/read_file)
        without editing, leaving the lead patch empty -> evaluate() falls back to
        shared_union. Seeding makes lead-curation always >= union quality.
        Returns True if the patch applied.
        """
        entry = self._work_dirs.get(instance_id)
        if not isinstance(entry, dict) or not (patch or "").strip():
            return False
        if entry.get("lead_curation"):
            _, lead_wd = entry["lead"]
        elif entry.get("isolated"):
            lead_wd = entry["agents"][entry["synth_idx"]][1]
        else:
            return False
        try:
            pf = os.path.join(lead_wd, ".seed.patch")
            with open(pf, "w") as fh:
                fh.write(patch if patch.endswith("\n") else patch + "\n")
            r = subprocess.run(
                "git apply --3way .seed.patch 2>/dev/null || git apply .seed.patch",
                shell=True, cwd=lead_wd, capture_output=True, text=True, timeout=30,
            )
            try:
                os.remove(pf)
            except OSError:
                pass
            return r.returncode == 0
        except Exception:
            return False

    def evaluate(self, task, recorder, final_answer=""):
        """Evaluate the agent's patch.

        Shared mode  : evaluate the single shared work_dir's patch.
        Isolated mode: pick the agent with the LONGEST non-empty patch (proxy
                       for "agent that did the most work"; deterministic ties
                       broken by lowest agent_idx). Mirrors Kim et al.'s
                       auto-submit pattern (first agent to do meaningful work
                       wins), without requiring an explicit submit tool.
        Full evaluation (pytest in Docker) is deferred to the harness pipeline.
        """
        instance_id = task["instance_id"]
        entry = self._work_dirs.get(instance_id)
        if not entry:
            return evaluate_swebench("", task)
        if isinstance(entry, dict) and entry.get("isolated"):
            agents = entry["agents"]
            n_workers = entry.get("n_workers", len(agents))
            synth_idx = entry.get("synth_idx")

            def _patch_lines(p):
                return sum(1 for ln in p.splitlines() if ln.strip())

            # Centralized: the orchestrator wrote the integrated final fix in the
            # synthesizer worktree. Prefer it when non-empty (this IS the topology's
            # aggregation). Independent/decentralized leave it empty -> fall through.
            if synth_idx is not None and synth_idx < len(agents):
                try:
                    synth_patch = agents[synth_idx][0].get_patch() or ""
                except Exception:
                    synth_patch = ""
                if _patch_lines(synth_patch) > 0:
                    result = evaluate_swebench(synth_patch, task)
                    if isinstance(result, dict):
                        result["aggregation"] = "orchestrator_synthesis"
                        result["synth_patch_lines"] = _patch_lines(synth_patch)
                    return result

            # Fallback: best-of-N by length over the WORKER worktrees only.
            best_patch = ""
            best_len = -1
            best_idx = -1
            for i, (exec_i, _wd_i) in enumerate(agents[:n_workers]):
                try:
                    p = exec_i.get_patch() or ""
                except Exception:
                    p = ""
                non_blank_lines = _patch_lines(p)
                if non_blank_lines > best_len:
                    best_len = non_blank_lines
                    best_patch = p
                    best_idx = i
            result = evaluate_swebench(best_patch, task)
            if isinstance(result, dict):
                result["aggregation"] = "best_of_n"
                result["isolated_winning_agent_idx"] = best_idx
                result["isolated_winning_patch_lines"] = best_len
            return result
        if isinstance(entry, dict) and entry.get("lead_curation"):
            shared_exec, _ = entry["shared"]
            lead_exec, _ = entry["lead"]
            try:
                lead_patch = lead_exec.get_patch() or ""
            except Exception:
                lead_patch = ""
            # Centralized populates the lead worktree -> use the curated patch.
            # Independent/decentralized leave it empty -> fall back to the shared
            # union git diff (unchanged behaviour for those topologies).
            if sum(1 for ln in lead_patch.splitlines() if ln.strip()) > 0:
                result = evaluate_swebench(lead_patch, task)
                if isinstance(result, dict):
                    result["aggregation"] = "lead_curation"
                return result
            union_patch = shared_exec.get_patch() or ""
            result = evaluate_swebench(union_patch, task)
            if isinstance(result, dict):
                result["aggregation"] = "shared_union"
            return result

        # Shared mode (legacy tuple)
        executor, _work_dir = entry
        patch = executor.get_patch()
        return evaluate_swebench(patch, task)
