"""Quick pilot test for QAMPARI benchmark.

Tests the benchmark adapter locally WITHOUT SGLang/GPU -- just verifies
that tasks load, search works, and evaluation produces sane results.
Simulates what an agent would do by searching for each answer entity.

Usage:
    cd mas-energy/code
    python pilot_qampari.py [--n-tasks 20]
"""

import argparse
import json
import time
from pathlib import Path

from benchmarks_qampari import QampariBenchmark, evaluate_qampari


def main():
    parser = argparse.ArgumentParser(description="QAMPARI Pilot Test (no GPU)")
    parser.add_argument("--n-tasks", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print("=" * 60)
    print("QAMPARI Pilot Test (offline, no SGLang needed)")
    print("=" * 60)

    bench = QampariBenchmark()
    tasks = bench.load_tasks(n_tasks=args.n_tasks, seed=args.seed)
    tools = bench.get_tools()
    print(f"Loaded {len(tasks)} tasks, {len(tools)} tool(s)")
    print(f"Tool: {tools[0]['function']['name']}")
    print()

    results = []
    for i, task in enumerate(tasks):
        executor, cleanup = bench.make_executor(task)
        n_answers = task["num_answers"]

        # Simulate agent: search for the question, then search for hints
        t0 = time.time()

        # Search 1: the question itself
        r1 = executor("search", {"query": task["question"]})

        # Search 2-4: search for entity names from the question
        entities = task.get("entities", [])
        for ent in entities[:2]:
            executor("search", {"query": ent.get("entity_text", "")})

        elapsed = time.time() - t0

        # Simulate "found answers" by checking which gold answers appear
        # in any search result (this approximates what a real agent would find)
        all_search_text = r1.lower()
        found_answers = []
        for ans in task["answer_list"]:
            if ans["answer_text"].lower() in all_search_text:
                found_answers.append(ans["answer_text"])

        # Build a simulated model output
        simulated_output = "\n".join(f"{j+1}. {a}" for j, a in enumerate(found_answers))

        # Evaluate
        eval_result = bench.evaluate(task, None, final_answer=simulated_output)

        status = "OK" if eval_result["f1"] > 0 else "MISS"
        print(
            f"  [{i+1}/{len(tasks)}] {task['id'][:30]:30s} | "
            f"gold={n_answers:3d} found={eval_result['num_correct']:2d} "
            f"pred={eval_result['num_predicted']:2d} | "
            f"F1={eval_result['f1']:.3f} R={eval_result['recall']:.3f} "
            f"P={eval_result['precision']:.3f} | "
            f"{elapsed:.2f}s | {status}"
        )
        results.append(eval_result)
        cleanup()

    # Summary
    print()
    print("=" * 60)
    avg_f1 = sum(r["f1"] for r in results) / len(results)
    avg_recall = sum(r["recall"] for r in results) / len(results)
    avg_precision = sum(r["precision"] for r in results) / len(results)
    pct_f1_05 = sum(1 for r in results if r["f1"] >= 0.5) / len(results)
    print(f"Mean F1:        {avg_f1:.3f}")
    print(f"Mean Recall:    {avg_recall:.3f}")
    print(f"Mean Precision: {avg_precision:.3f}")
    print(f"%F1>=0.5:       {pct_f1_05:.1%}")
    print(f"Tasks:          {len(results)}")
    print()
    print("This simulates a naive agent that only searches the question text.")
    print("A real agent with multiple search calls should achieve higher recall.")
    print("With MAS (3 workers), expect ~3x the search coverage -> higher recall.")
    print("=" * 60)


if __name__ == "__main__":
    main()
