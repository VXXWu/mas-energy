"""Test three decomposition strategies on FanOutQA tasks.

Runs against a live SGLang server. Saves raw orchestrator outputs
for qualitative comparison -- no worker execution, no evaluation.

Strategies:
  vanilla   - Current: task + tool names only, single chat() call
  option_a  - Rich: task + full tool schemas + CoT scratchpad, single chat() call
  option_b  - Explore-then-decompose: orchestrator runs ReAct (k_explore steps)
              with tools, then decomposes with discovered context via chat()

Usage:
  python test_decompose.py                     # default 10 tasks
  python test_decompose.py --n_tasks 50
  python test_decompose.py --port 30000        # custom SGLang port
"""

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from llm import make_client, chat, react_loop
from prompts import ORCHESTRATOR_PROMPT, parse_subtasks, _summarize_trajectory
from config import ORCHESTRATOR_TEMP, BASE_SEED, N_AGENTS, SGLANG_URL
from benchmarks_fanoutqa import FanOutQABenchmark


# ── Decomposition prompts ──────────────────────────────────────────


def prompt_vanilla(question, tools, n_workers=3):
    """Current implementation: tool names only."""
    tool_names = [t["function"]["name"] for t in tools]
    return (
        f"Task: {question}\n\n"
        f"Available tools: {', '.join(tool_names)}\n\n"
        f"Decompose this task into {n_workers} subtasks, one per worker. "
        f"Each worker can make function calls using the available tools.\n"
        f"Format each subtask on a separate line starting with 'SUBTASK:'"
    )


def prompt_option_a(question, tools, n_workers=3):
    """Option A: full tool schemas + CoT analysis scratchpad."""
    tool_lines = []
    for t in tools:
        fn = t["function"]
        desc = fn.get("description", "No description")
        params = fn.get("parameters", {}).get("properties", {})
        tool_lines.append(f"- {fn['name']}: {desc}")
        for pname, pspec in params.items():
            pdesc = pspec.get("description", pspec.get("type", ""))
            tool_lines.append(f"    - {pname}: {pdesc}")
    return (
        f"Task: {question}\n\n"
        f"Available tools:\n" + "\n".join(tool_lines) + "\n\n"
        f"Before decomposing, analyze the task:\n"
        f"ANALYSIS: Identify what specific entities or information must be "
        f"gathered, what dependencies exist between steps, and how to "
        f"divide the work so each worker can operate independently.\n\n"
        f"Then decompose into exactly {n_workers} subtasks. "
        f"Each worker can call any of the above tools.\n"
        f"Format each subtask on a separate line starting with 'SUBTASK:'"
    )


def prompt_option_b(question, tools, explore_summary, n_workers=3):
    """Option B: decompose after exploration, conditioned on findings."""
    tool_lines = []
    for t in tools:
        fn = t["function"]
        desc = fn.get("description", "No description")
        tool_lines.append(f"- {fn['name']}: {desc}")
    return (
        f"Task: {question}\n\n"
        f"Available tools:\n" + "\n".join(tool_lines) + "\n\n"
        f"Your preliminary research found:\n{explore_summary}\n\n"
        f"Based on these findings, decompose into exactly {n_workers} "
        f"subtasks so each worker can operate independently. "
        f"Be specific -- assign concrete entities or sub-questions to each worker.\n"
        f"Format each subtask on a separate line starting with 'SUBTASK:'"
    )


EXPLORE_SYSTEM = (
    "You are a research assistant preparing to coordinate a team of workers. "
    "Use the available tools to investigate the task below. Your goal is to "
    "identify the key entities, facts, or list items needed to answer the "
    "question. Do NOT try to fully answer the task -- just gather enough "
    "context to understand its structure and what needs to be looked up."
)

K_EXPLORE = 3  # max ReAct steps for exploration phase


# ── Null energy monitor (no GPU measurement needed) ────────────────


class NullMonitor:
    """Drop-in for EnergyMonitor that records nothing."""
    def start(self): pass
    def stop(self, metadata=None):
        return {"gpu_energy_joules": 0, "wall_seconds": 0, **(metadata or {})}


# ── Run one task through all three strategies ──────────────────────


def run_task(client, model, task, tools, executor):
    question = task["question"]
    monitor = NullMonitor()
    results = {}

    # ── Vanilla ──
    t0 = time.time()
    text_v, usage_v = chat(
        client, model,
        messages=[
            {"role": "system", "content": ORCHESTRATOR_PROMPT},
            {"role": "user", "content": prompt_vanilla(question, tools)},
        ],
        temperature=ORCHESTRATOR_TEMP, seed=BASE_SEED,
    )
    results["vanilla"] = {
        "raw_output": text_v,
        "subtasks": parse_subtasks(text_v, N_AGENTS, question),
        "usage": usage_v,
        "wall_s": time.time() - t0,
    }

    # ── Option A ──
    t0 = time.time()
    text_a, usage_a = chat(
        client, model,
        messages=[
            {"role": "system", "content": ORCHESTRATOR_PROMPT},
            {"role": "user", "content": prompt_option_a(question, tools)},
        ],
        temperature=ORCHESTRATOR_TEMP, seed=BASE_SEED,
    )
    results["option_a"] = {
        "raw_output": text_a,
        "subtasks": parse_subtasks(text_a, N_AGENTS, question),
        "usage": usage_a,
        "wall_s": time.time() - t0,
    }

    # ── Option B: explore then decompose ──
    t0 = time.time()
    explore_result = react_loop(
        client=client, model=model,
        messages=[
            {"role": "system", "content": EXPLORE_SYSTEM},
            {"role": "user", "content": f"Task: {question}"},
        ],
        tools=tools, execute_tool=executor,
        energy_monitor=monitor,
        max_steps=K_EXPLORE,
        temperature=0.0, seed=BASE_SEED,
        agent_id="orchestrator_explore",
    )
    explore_summary = _summarize_trajectory(explore_result["messages"])
    if explore_result.get("final_response"):
        explore_summary += "\nFindings: " + explore_result["final_response"][:500]

    text_b, usage_b = chat(
        client, model,
        messages=[
            {"role": "system", "content": ORCHESTRATOR_PROMPT},
            {"role": "user", "content": prompt_option_b(
                question, tools, explore_summary
            )},
        ],
        temperature=ORCHESTRATOR_TEMP, seed=BASE_SEED,
    )
    results["option_b"] = {
        "raw_output": text_b,
        "subtasks": parse_subtasks(text_b, N_AGENTS, question),
        "explore_trajectory": explore_summary,
        "explore_response": explore_result.get("final_response", ""),
        "explore_steps": explore_result.get("steps", 0),
        "usage_explore": explore_result.get("total_usage", {}),
        "usage_decompose": usage_b,
        "wall_s": time.time() - t0,
    }

    return results


# ── Main ───────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Test decomposition strategies")
    parser.add_argument("--n_tasks", type=int, default=10)
    parser.add_argument("--port", type=int, default=30000)
    parser.add_argument("--model", type=str, default=None,
                        help="Model path (auto-detected from server if omitted)")
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    base_url = f"http://localhost:{args.port}/v1"
    client = make_client(base_url=base_url)

    # Auto-detect model
    if args.model:
        model = args.model
    else:
        models = client.models.list()
        model = models.data[0].id
        print(f"Auto-detected model: {model}")

    # Load benchmark
    bench = FanOutQABenchmark()
    tasks = bench.load_tasks(n_tasks=args.n_tasks, seed=42)
    tools = bench.get_tools()
    print(f"Loaded {len(tasks)} FanOutQA tasks, {len(tools)} tools")

    output_path = args.output or os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..", "results", "decompose_test.jsonl",
    )
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    print(f"Output: {output_path}")
    print(f"Strategies: vanilla, option_a (rich prompt), option_b (explore k={K_EXPLORE})")
    print("=" * 70)

    with open(output_path, "w") as fout:
        for idx, task in enumerate(tasks):
            task_id = task["id"]
            question = task["question"]
            decomp_gt = task.get("decomposition", [])
            evidence_gt = task.get("necessary_evidence", [])

            print(f"\n[{idx+1}/{len(tasks)}] {task_id}: {question[:80]}...")

            # Fresh executor per task (stateless for fanoutqa, but correct pattern)
            executor, cleanup = bench.make_executor(task)
            try:
                results = run_task(client, model, task, tools, executor)
            finally:
                cleanup()

            record = {
                "task_id": task_id,
                "question": question,
                "ground_truth_n_subqs": len(decomp_gt),
                "ground_truth_decomposition": [d["question"] for d in decomp_gt],
                "ground_truth_articles": [e["title"] for e in evidence_gt],
            }
            for strategy, res in results.items():
                record[strategy] = res

            fout.write(json.dumps(record) + "\n")
            fout.flush()

            # Print comparison
            for strategy in ["vanilla", "option_a", "option_b"]:
                subtasks = results[strategy]["subtasks"]
                print(f"  {strategy:>10}: {' | '.join(s[:60] for s in subtasks)}")
            gt_strs = [d["question"][:60] for d in decomp_gt]
            print(f"  {'GT':>10}: {' | '.join(gt_strs)}")

    print(f"\nDone. Results saved to {output_path}")

    # Summary stats
    print("\n" + "=" * 70)
    print("TOKEN USAGE SUMMARY")
    print("=" * 70)
    with open(output_path) as f:
        records = [json.loads(line) for line in f]
    for strategy in ["vanilla", "option_a", "option_b"]:
        totals = []
        for r in records:
            s = r[strategy]
            if strategy == "option_b":
                t = (s.get("usage_explore", {}).get("total_tokens", 0)
                     + s.get("usage_decompose", {}).get("total_tokens", 0))
            else:
                t = s.get("usage", {}).get("total_tokens", 0)
            totals.append(t)
        avg = sum(totals) / len(totals) if totals else 0
        print(f"  {strategy:>10}: {avg:,.0f} avg tokens/task ({sum(totals):,} total)")


if __name__ == "__main__":
    main()
