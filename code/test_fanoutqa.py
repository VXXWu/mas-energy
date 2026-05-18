"""Smoke test for FanOutQA benchmark adapter.

Verifies:
1. Tasks load correctly from dev set
2. Single search tool schema is correct
3. Executor returns BM25+-ranked chunks (exact title match)
4. Executor returns suggestions (no exact title match)
5. Evaluation produces correct scores
"""

import json

from benchmarks_fanoutqa import FanOutQABenchmark


def test_load_tasks():
    bench = FanOutQABenchmark()
    tasks = bench.load_tasks(n_tasks=5, seed=42)
    assert len(tasks) == 5, f"Expected 5 tasks, got {len(tasks)}"
    for t in tasks:
        assert "id" in t
        assert "question" in t
        assert "ground_truth_answer" in t
        assert "decomposition" in t
        assert "necessary_evidence" in t
        print(f"  Task {t['id']}: {t['question'][:70]}...")
        print(f"    Answer type: {type(t['ground_truth_answer']).__name__}")
        print(f"    Subquestions: {len(t['decomposition'])}")
        print(f"    Evidence pages: {len(t['necessary_evidence'])}")
    print("PASS: load_tasks\n")
    return tasks


def test_tools():
    bench = FanOutQABenchmark()
    tools = bench.get_tools()
    assert len(tools) == 1, f"Expected 1 tool, got {len(tools)}"
    assert tools[0]["function"]["name"] == "search"
    assert "query" in tools[0]["function"]["parameters"]["properties"]
    print("PASS: get_tools (single 'search' tool)\n")


def test_executor_exact_match(task):
    """Test that an exact title match returns BM25+-ranked chunks."""
    bench = FanOutQABenchmark()
    executor, cleanup = bench.make_executor(task)

    # Use a known Wikipedia title from the task's evidence
    if task["necessary_evidence"]:
        title = task["necessary_evidence"][0]["title"]
    else:
        title = "United States"

    print(f"  Searching exact title: '{title}'")
    result = executor("search", {"query": title})
    assert isinstance(result, str), f"Expected string, got {type(result)}"
    assert "<document>" in result, f"Expected XML document format, got: {result[:200]}"
    assert "<title>" in result
    assert "<fragment>" in result
    n_fragments = result.count("<fragment>")
    print(f"  Got {n_fragments} BM25+-ranked fragments")
    print(f"  Total result length: {len(result)} chars")
    print(f"  Preview: {result[:200]}...")
    cleanup()
    print("PASS: executor exact match\n")


def test_executor_no_match(task):
    """Test that a non-existent title returns suggestions."""
    bench = FanOutQABenchmark()
    executor, cleanup = bench.make_executor(task)

    result = executor("search", {"query": "Xyzzy Nonexistent Page 12345"})
    assert isinstance(result, str)
    assert "No page" in result or "similar" in result.lower() or "No Wikipedia" in result
    print(f"  Non-existent title response: {result[:200]}...")
    cleanup()
    print("PASS: executor no match\n")


def test_evaluate():
    bench = FanOutQABenchmark()
    tasks = bench.load_tasks(n_tasks=2, seed=42)
    task = tasks[0]

    gt = task["ground_truth_answer"]
    print(f"  Ground truth: {json.dumps(gt)[:120]}...")

    # Perfect answer: stringify the ground truth
    if isinstance(gt, dict):
        perfect = ", ".join(f"{k}: {v}" for k, v in gt.items())
    elif isinstance(gt, list):
        perfect = ", ".join(str(x) for x in gt)
    else:
        perfect = str(gt)

    result = bench.evaluate(task, None, final_answer=perfect)
    print(f"  Perfect answer score: {result}")
    assert result["loose_accuracy"] > 0.5, f"Perfect answer should score >0.5, got {result}"

    # Empty answer
    result2 = bench.evaluate(task, None, final_answer="")
    print(f"  Empty answer score: {result2}")
    assert result2["correct"] is False

    # Wrong answer
    result3 = bench.evaluate(task, None, final_answer="I don't know anything about this topic.")
    print(f"  Wrong answer score: {result3}")

    print("PASS: evaluate\n")


if __name__ == "__main__":
    print("=" * 50)
    print("FanOutQA Smoke Test")
    print("=" * 50)

    print("\n1. Testing load_tasks...")
    tasks = test_load_tasks()

    print("2. Testing get_tools...")
    test_tools()

    print("3. Testing executor - exact title match (requires network)...")
    test_executor_exact_match(tasks[0])

    print("4. Testing executor - no match...")
    test_executor_no_match(tasks[0])

    print("5. Testing evaluation...")
    test_evaluate()

    print("=" * 50)
    print("ALL TESTS PASSED")
    print("=" * 50)
