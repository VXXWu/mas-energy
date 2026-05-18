"""SWE-bench-Lite benchmark adapter.

Agent receives a GitHub issue description and must produce a code fix
in a checked-out repository. Uses a single bash tool for all interactions
(explore, read, edit, test) -- same approach as mini-SWE-agent.

Tools:
  - bash(command) -> execute bash command in the repo directory

Evaluation: deferred to SWE-bench harness (requires Docker). During
energy measurement, we save git diffs as patches. For the pilot, we
check whether a non-empty, syntactically valid patch was produced.

Setup (run once on cluster):
  pip install datasets
  # Clone repos needed for selected tasks:
  git clone https://github.com/sympy/sympy.git /atlas2/u/$USER/mas_project/swebench_repos/sympy
  git clone https://github.com/django/django.git /atlas2/u/$USER/mas_project/swebench_repos/django
  # etc. for repos in your task subset
"""

import json
import logging
import os
import random
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
                "Execute a bash command in the repository directory. "
                "Use this to explore the codebase (find, grep), read files "
                "(cat, head), edit files (sed, patch, or echo/cat with "
                "redirection), and run tests (python -m pytest). "
                "Each command runs independently (no persistent shell state)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "The bash command to execute",
                    },
                },
                "required": ["command"],
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
    """Executes bash commands in a repo working directory."""

    def __init__(self, work_dir):
        self.work_dir = work_dir

    def __call__(self, tool_name, args):
        if tool_name == "bash":
            return self._bash(args)
        return f"Unknown tool: {tool_name}"

    def _bash(self, args):
        command = args.get("command", "")
        if not command:
            return "Error: command is required."
        try:
            result = subprocess.run(
                command,
                shell=True,
                cwd=self.work_dir,
                capture_output=True,
                text=True,
                timeout=COMMAND_TIMEOUT,
                env={**os.environ, "PAGER": "cat", "GIT_PAGER": "cat"},
            )
            output = result.stdout
            if result.stderr:
                output = output + "\nSTDERR:\n" + result.stderr if output else result.stderr
            if not output.strip():
                output = f"(exit code {result.returncode})"
            if len(output) > MAX_OUTPUT_CHARS:
                half = MAX_OUTPUT_CHARS // 2
                output = output[:half] + "\n... (truncated) ...\n" + output[-half:]
            return output
        except subprocess.TimeoutExpired:
            return f"Error: command timed out after {COMMAND_TIMEOUT} seconds."
        except Exception as e:
            return f"Error: {e}"

    def get_patch(self):
        """Get the git diff of changes made by the agent."""
        try:
            result = subprocess.run(
                "git diff",
                shell=True, cwd=self.work_dir,
                capture_output=True, text=True, timeout=10,
            )
            return result.stdout.strip()
        except Exception:
            return ""


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
        self._work_dirs = {}

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
                f"## Issue\n\n{problem}\n\n"
                f"## Instructions\n\n"
                f"1. Explore the repository structure to understand the codebase\n"
                f"2. Find the relevant source files\n"
                f"3. Understand the bug or feature request\n"
                f"4. Make the necessary code changes using bash commands\n"
                f"5. Verify your changes work correctly\n\n"
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

    def make_executor(self, task=None):
        """Create a working directory with the repo at the right commit."""
        repo_name = task["repo_name"]
        base_commit = task["base_commit"]
        instance_id = task["instance_id"]

        repo_path = self.repos_dir / repo_name
        if not repo_path.exists():
            raise FileNotFoundError(
                f"Repository {repo_name} not found at {repo_path}. "
                f"Clone it: git clone https://github.com/{task['repo']}.git {repo_path}"
            )

        # Prune stale worktrees from prior crashed runs
        subprocess.run(
            "git worktree prune",
            shell=True, cwd=repo_path,
            capture_output=True, timeout=30,
        )

        # Use a dedicated temp dir on atlas2 (not /tmp which may be small)
        tmp_base = self.repos_dir.parent / "swebench_worktrees"
        tmp_base.mkdir(exist_ok=True)
        work_dir = Path(tempfile.mkdtemp(prefix=f"swe_{instance_id}_", dir=tmp_base))

        try:
            result = subprocess.run(
                f"git worktree add --detach {work_dir} {base_commit}",
                shell=True, cwd=repo_path,
                capture_output=True, text=True, check=True, timeout=120,
            )
        except subprocess.CalledProcessError as e:
            shutil.rmtree(work_dir, ignore_errors=True)
            raise RuntimeError(
                f"Failed to create worktree for {instance_id} at {base_commit}: {e.stderr}"
            )

        executor = SWEBenchExecutor(str(work_dir))

        def cleanup():
            try:
                # Get patch before cleanup
                patch = executor.get_patch()
                # Save patch for later evaluation
                patches_dir = self.repos_dir.parent / "swebench_patches"
                patches_dir.mkdir(exist_ok=True)
                patch_file = patches_dir / f"{instance_id}.patch"
                patch_file.write_text(patch)
            except Exception as e:
                log.warning(f"Failed to save patch for {instance_id}: {e}")

            try:
                subprocess.run(
                    f"git worktree remove --force {work_dir}",
                    shell=True, cwd=repo_path,
                    capture_output=True, timeout=30,
                )
            except Exception:
                pass
            shutil.rmtree(work_dir, ignore_errors=True)

            # Remove from tracking dict
            self._work_dirs.pop(instance_id, None)

        self._work_dirs[instance_id] = (executor, work_dir)
        return executor, cleanup

    def evaluate(self, task, recorder, final_answer=""):
        """Evaluate the agent's patch.

        For now, just checks if a valid patch was produced.
        Full evaluation with pytest requires the SWE-bench harness.
        """
        instance_id = task["instance_id"]
        executor_info = self._work_dirs.get(instance_id)
        patch = ""
        if executor_info:
            executor, work_dir = executor_info
            patch = executor.get_patch()
        return evaluate_swebench(patch, task)
