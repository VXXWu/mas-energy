"""SWE-bench batched benchmark: 3 independent bugs as one compound task.

Tests whether Centralized topology benefits from genuinely independent,
parallelizable subtasks. Each compound task bundles 3 SWE-bench bugs from
different repos/files. The agent gets 3 separate bash tools, each operating
in a different worktree.

For SAS: the agent must solve all 3 sequentially using 3 different tools.
For Centralized: the orchestrator assigns each worker a different bug/tool.
For Independent/Decentralized: each agent attempts all 3 bugs.

This is a controlled experiment isolating the parallelism variable.
"""

import json
import logging
import os
import random
import shutil
import subprocess
import tempfile
from pathlib import Path

from benchmarks_swebench import (
    SWEBenchExecutor, evaluate_swebench, COMMAND_TIMEOUT, MAX_OUTPUT_CHARS,
)

log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────
# Tool schemas (3 separate bash tools, one per repo)
# ─────────────────────────────────────────────────────────

def make_batched_tools(labels):
    """Create N bash tools, one per sub-task."""
    tools = []
    for i, label in enumerate(labels):
        tools.append({
            "type": "function",
            "function": {
                "name": f"bash_task{i+1}",
                "description": (
                    f"Execute a bash command in the repository for Task {i+1} ({label}). "
                    f"Use this to explore, read, edit, and test code for Task {i+1} ONLY. "
                    f"Each task has its own separate repository checkout."
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
        })
    return tools


# ─────────────────────────────────────────────────────────
# Executor (dispatches to per-task worktrees)
# ─────────────────────────────────────────────────────────

class BatchedSWEBenchExecutor:
    """Routes bash_task1/bash_task2/bash_task3 to separate worktrees."""

    def __init__(self, executors):
        """executors: list of SWEBenchExecutor, one per sub-task."""
        self.executors = executors
        self.tool_names = [f"bash_task{i+1}" for i in range(len(executors))]

    def __call__(self, tool_name, args):
        for i, name in enumerate(self.tool_names):
            if tool_name == name:
                return self.executors[i](tool_name="bash", args=args)
        return f"Unknown tool: {tool_name}. Available: {', '.join(self.tool_names)}"

    def get_patches(self):
        """Get git diff from each sub-task worktree."""
        return [ex.get_patch() for ex in self.executors]


# ─────────────────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────────────────

def evaluate_batched(patches, sub_tasks):
    """Evaluate each sub-task's patch independently."""
    results = []
    for patch, task in zip(patches, sub_tasks):
        results.append(evaluate_swebench(patch, task))

    n_patched = sum(1 for r in results if r["has_patch"])
    n_correct = sum(1 for r in results if r["correct"])

    return {
        "correct": n_correct == len(sub_tasks),
        "has_patch": n_patched > 0,
        "n_patched": n_patched,
        "n_correct": n_correct,
        "n_total": len(sub_tasks),
        "partial_score": n_correct / len(sub_tasks),
        "sub_results": results,
        "patches": {t["instance_id"]: p for t, p in zip(sub_tasks, patches)},
    }


# ─────────────────────────────────────────────────────────
# Benchmark class
# ─────────────────────────────────────────────────────────

BATCH_SIZE = 3  # Number of bugs per compound task


class SWEBenchBatchedBenchmark:
    """SWE-bench with batched independent tasks.

    Groups BATCH_SIZE independent bugs into one compound task.
    Each sub-task gets its own bash tool and worktree.
    """

    def __init__(self, repos_dir=None):
        if repos_dir is None:
            repos_dir = os.environ.get(
                "SWEBENCH_REPOS",
                os.path.expanduser("/atlas2/u/{}/mas_project/swebench_repos".format(
                    os.environ.get("USER", ""))),
            )
        self.repos_dir = Path(repos_dir)
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

        # Filter to available repos
        available_repos = set()
        if self.repos_dir.exists():
            for d in self.repos_dir.iterdir():
                if d.is_dir() and (d / ".git").exists():
                    available_repos.add(d.name)

        all_tasks = []
        for row in ds:
            repo_name = row["repo"].split("/")[-1]
            if available_repos and repo_name not in available_repos:
                continue
            all_tasks.append({
                "instance_id": row["instance_id"],
                "repo": row["repo"],
                "repo_name": repo_name,
                "base_commit": row["base_commit"],
                "problem_statement": row["problem_statement"],
                "patch": row["patch"],
                "test_patch": row["test_patch"],
                "fail_to_pass": row["FAIL_TO_PASS"],
                "pass_to_pass": row["PASS_TO_PASS"],
            })

        # Shuffle and group into batches of BATCH_SIZE
        rng = random.Random(seed)
        rng.shuffle(all_tasks)

        batched_tasks = []
        for i in range(0, len(all_tasks) - BATCH_SIZE + 1, BATCH_SIZE):
            batch = all_tasks[i:i + BATCH_SIZE]

            # Build compound question
            parts = []
            for j, t in enumerate(batch):
                parts.append(
                    f"### Task {j+1}: {t['instance_id']}\n"
                    f"Repository: {t['repo']}\n\n"
                    f"{t['problem_statement']}\n\n"
                    f"Use the `bash_task{j+1}` tool to work on this task. "
                    f"Each task has its own separate repository checkout."
                )

            compound_q = (
                f"You have {BATCH_SIZE} independent bug fixes to complete. "
                f"Each task is in a DIFFERENT repository with its own bash tool. "
                f"Solve all {BATCH_SIZE} tasks.\n\n"
                + "\n\n---\n\n".join(parts)
                + "\n\nMake minimal, targeted changes for each task. "
                "Do not modify test files."
            )

            # Raw question for orchestrator decomposition.
            # Include full bug descriptions so workers get the complete
            # issue text, not just the orchestrator's summary.
            raw_q = (
                f"There are {BATCH_SIZE} independent bugs to fix. "
                f"Each has its own bash tool. "
                f"Assign each worker to one task.\n\n"
            )
            for j, t in enumerate(batch):
                raw_q += (
                    f"### Task {j+1} (use bash_task{j+1}): {t['instance_id']}\n"
                    f"{t['problem_statement']}\n\n"
                )

            labels = [t["instance_id"] for t in batch]
            batched_tasks.append({
                "id": f"batch_{i//BATCH_SIZE}",
                "question": compound_q,
                "question_text": raw_q,
                "sub_tasks": batch,
                "labels": labels,
            })

        if n_tasks and n_tasks < len(batched_tasks):
            batched_tasks = batched_tasks[:n_tasks]

        log.info(f"Loaded {len(batched_tasks)} batched tasks "
                 f"({len(batched_tasks) * BATCH_SIZE} sub-tasks)")
        return batched_tasks

    def get_tools(self):
        # Return generic 3-tool schema; actual labels set per task
        return make_batched_tools(["bug_1", "bug_2", "bug_3"])

    def make_executor(self, task=None):
        """Create worktrees for each sub-task in the batch."""
        sub_tasks = task["sub_tasks"]
        executors = []
        work_dirs = []
        cleanups = []

        tmp_base = self.repos_dir.parent / "swebench_worktrees"
        tmp_base.mkdir(exist_ok=True)

        for t in sub_tasks:
            repo_path = self.repos_dir / t["repo_name"]
            if not repo_path.exists():
                raise FileNotFoundError(f"Repo {t['repo_name']} not at {repo_path}")

            subprocess.run("git worktree prune", shell=True,
                           cwd=repo_path, capture_output=True, timeout=30)

            work_dir = Path(tempfile.mkdtemp(
                prefix=f"swe_{t['instance_id']}_", dir=tmp_base))

            subprocess.run(
                f"git worktree add --detach {work_dir} {t['base_commit']}",
                shell=True, cwd=repo_path,
                capture_output=True, text=True, check=True, timeout=120,
            )

            executors.append(SWEBenchExecutor(str(work_dir)))
            work_dirs.append((work_dir, repo_path))

        batched_executor = BatchedSWEBenchExecutor(executors)
        self._work_dirs[task["id"]] = batched_executor

        def cleanup():
            try:
                patches = batched_executor.get_patches()
                patches_dir = self.repos_dir.parent / "swebench_patches"
                patches_dir.mkdir(exist_ok=True)
                for t, patch in zip(sub_tasks, patches):
                    (patches_dir / f"{t['instance_id']}.patch").write_text(patch)
            except Exception as e:
                log.warning(f"Failed to save patches: {e}")

            for work_dir, repo_path in work_dirs:
                try:
                    subprocess.run(f"git worktree remove --force {work_dir}",
                                   shell=True, cwd=repo_path,
                                   capture_output=True, timeout=30)
                except Exception:
                    pass
                shutil.rmtree(work_dir, ignore_errors=True)

            self._work_dirs.pop(task["id"], None)

        return batched_executor, cleanup

    def evaluate(self, task, recorder, final_answer=""):
        batch_exec = self._work_dirs.get(task["id"])
        patches = batch_exec.get_patches() if batch_exec else [""] * BATCH_SIZE
        return evaluate_batched(patches, task["sub_tasks"])
