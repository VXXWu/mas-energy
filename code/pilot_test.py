"""Quick pilot test: validate the full pipeline before batch jobs.

Tests SGLang connection, tool calling, energy measurement, and all 4
topologies on a small WorkBench sample.

Usage (from compute node with SGLang running):
    cd /atlas2/u/$USER/mas-energy/code
    python pilot_test.py --model Qwen/Qwen3.5-9B \
        --workbench-path /atlas2/u/$USER/WorkBench

Or with model key:
    python pilot_test.py --model toy \
        --workbench-path /atlas2/u/$USER/WorkBench
"""

import argparse
import json
import os
import sys
from datetime import datetime

from config import SGLANG_URL, MODELS, TOY_MODEL
from energy import EnergyMonitor
from llm import make_client, warmup, chat, react_loop
from benchmarks import load_benchmark, WORKBENCH_TOOLS
from topologies import TOPOLOGY_RUNNERS


def get_model_path(model_key):
    if model_key == "toy":
        return TOY_MODEL["model_path"], TOY_MODEL.get("extra_body")
    if model_key in MODELS:
        cfg = MODELS[model_key]
        return cfg["model_path"], cfg.get("extra_body")
    return model_key, None


def main():
    parser = argparse.ArgumentParser(description="MAS Energy Pilot Test")
    parser.add_argument("--model", default="toy")
    parser.add_argument("--workbench-path", default=None)
    parser.add_argument("--sglang-url", default=None)
    parser.add_argument("--n-tasks", type=int, default=5)
    parser.add_argument("--gpu-index", type=int, default=0)
    args = parser.parse_args()

    model_path, extra_body = get_model_path(args.model)
    sglang_url = args.sglang_url or SGLANG_URL

    print("=" * 60)
    print("MAS Energy Pilot Test")
    print("=" * 60)
    print(f"Model: {model_path}")
    print(f"SGLang URL: {sglang_url}")
    print(f"Tasks: {args.n_tasks}")
    print()

    # --- Phase A: SGLang connection ---
    print("[Phase A] Testing SGLang connection...")
    client = make_client(base_url=sglang_url)
    try:
        text, usage = chat(
            client, model_path,
            [{"role": "user", "content": "What is 2+2?"}],
            max_tokens=50, extra_body=extra_body,
        )
        print(f"  OK: '{text[:80]}'")
        print(f"  Tokens: {usage}")
    except Exception as e:
        print(f"  FAILED: {e}")
        print(f"  Is SGLang running at {sglang_url}?")
        sys.exit(1)

    # --- Phase B: Tool calling ---
    print("\n[Phase B] Testing tool calling...")
    dummy_tools = [{
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string", "description": "City name"},
                },
                "required": ["location"],
            },
        },
    }]

    def dummy_executor(name, args):
        return {"temperature": 72, "condition": "sunny"}

    monitor = EnergyMonitor(gpu_index=args.gpu_index)

    result = react_loop(
        client=client, model=model_path,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What's the weather in San Francisco?"},
        ],
        tools=dummy_tools,
        execute_tool=dummy_executor,
        energy_monitor=monitor,
        max_steps=5,
        temperature=0.0,
        agent_id="test_agent",
        extra_body=extra_body,
    )
    print(f"  Steps: {result['steps']}")
    print(f"  Final response: {(result['final_response'] or '')[:100]}")
    print(f"  Energy records: {len(result['call_records'])}")
    has_tool_calls = any(
        r.get("call_type") == "tool_execution" for r in result["call_records"]
    )
    print(f"  Tool calls made: {has_tool_calls}")
    if not has_tool_calls:
        print("  WARNING: Model did not make any tool calls!")

    # --- Phase C: Energy measurement ---
    print("\n[Phase C] Testing energy measurement...")
    print(f"  GPU: {monitor.gpu_name}")

    idle_w = monitor.measure_idle(duration=5)
    print(f"  Idle power: {idle_w:.1f} W")

    monitor.start()
    chat(client, model_path,
         [{"role": "user", "content": "Hello"}],
         max_tokens=10, extra_body=extra_body)
    record = monitor.stop()
    print(f"  GPU energy: {record['gpu_energy_joules']:.4f} J")
    print(f"  GPU dynamic: {record['gpu_dynamic_energy_joules']:.4f} J")
    print(f"  Wall time: {record['wall_seconds']:.3f} s")
    print(f"  Avg power: {record['avg_gpu_power_watts']:.0f} W")

    if record["gpu_energy_joules"] <= 0:
        print("  ERROR: GPU energy is 0 or negative. NVML may not work.")
        sys.exit(1)

    # --- Phase D: Warmup ---
    print("\n[Phase D] Warming up (5 calls)...")
    warmup(client, model_path, n=5, extra_body=extra_body)
    print("  Done")

    # --- Phase E: WorkBench topology test ---
    print(f"\n[Phase E] Testing topologies on WorkBench ({args.n_tasks} tasks)...")

    wb_path = args.workbench_path or os.environ.get(
        "WORKBENCH_PATH", os.path.expanduser("~/WorkBench")
    )
    try:
        bench = load_benchmark("workbench", repo_path=wb_path)
        tasks = bench.load_tasks(args.n_tasks)
        tools = bench.get_tools()
        print(f"  Loaded {len(tasks)} tasks, {len(tools)} tools")
    except Exception as e:
        print(f"  WorkBench load failed: {e}")
        print(f"  Skipping topology tests. Clone WorkBench to {wb_path}")
        tasks = []

    output_dir = os.path.join(os.path.dirname(__file__), "..", "results", "pilot")
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "pilot_results.jsonl")

    topologies = ["sas", "independent", "centralized", "decentralized"]

    for topo_name in topologies:
        if not tasks:
            break
        print(f"\n  --- Topology: {topo_name} ---")
        runner = TOPOLOGY_RUNNERS[topo_name]

        correct = 0
        total_energy = 0.0
        total_tokens = 0
        total_calls = 0

        for task in tasks[:args.n_tasks]:
            monitor.call_log = []
            executor = None
            cleanup = None
            try:
                executor, cleanup = bench.make_executor(task)

                task_question = task["question"]
                if bench.TIME_CONTEXT:
                    task_question = bench.TIME_CONTEXT + "\n\n" + task_question

                result = runner(
                    client=client,
                    model=model_path,
                    task_question=task_question,
                    tools=tools,
                    execute_tool=executor,
                    energy_monitor=monitor,
                    extra_body=extra_body,
                )

                is_correct = bench.evaluate(task, executor)

                records = result["call_records"]
                run_gpu_dyn = sum(r.get("gpu_dynamic_energy_joules", 0) for r in records)
                run_tokens = result["total_usage"]["total_tokens"]
                n_llm = len([r for r in records if "react_step" in r.get("call_type", "") or "synthesis" in r.get("call_type", "") or "decompose" in r.get("call_type", "")])
                n_tool = len([r for r in records if r.get("call_type") == "tool_execution"])

                correct += int(is_correct)
                total_energy += run_gpu_dyn
                total_tokens += run_tokens
                total_calls += n_llm

                print(
                    f"    {task['id']}: {'OK' if is_correct else 'WRONG'} | "
                    f"gpu_dyn={run_gpu_dyn:.2f}J | tokens={run_tokens} | "
                    f"llm_calls={n_llm} | tool_calls={n_tool}"
                )

                record = {
                    "model": model_path,
                    "benchmark": "workbench",
                    "topology": topo_name,
                    "task_id": task["id"],
                    "correct": is_correct,
                    "gpu_dynamic_energy_joules": run_gpu_dyn,
                    "total_tokens": run_tokens,
                    "n_llm_calls": n_llm,
                    "n_tool_calls": n_tool,
                    "timestamp": datetime.now().isoformat(),
                }
                with open(output_file, "a") as f:
                    f.write(json.dumps(record) + "\n")

            except Exception as e:
                print(f"    {task['id']}: ERROR - {e}")
                import traceback
                traceback.print_exc()

            finally:
                if cleanup:
                    cleanup()

        n = min(len(tasks), args.n_tasks)
        if n > 0:
            print(
                f"  SUMMARY [{topo_name}]: acc={correct}/{n}, "
                f"energy={total_energy:.1f}J (avg {total_energy/n:.2f}J/task), "
                f"tokens={total_tokens} (avg {total_tokens/n:.0f}/task), "
                f"llm_calls={total_calls}"
            )

    # --- Summary ---
    print(f"\n{'='*60}")
    print("Pilot Validation Checklist")
    print("=" * 60)
    print("[ ] GPU energy values are non-zero and reasonable (1-50 J per call)")
    print("[ ] Tool calling works (model makes tool calls)")
    print("[ ] All 4 topologies produced results")
    print("[ ] SAS has fewest calls, Decentralized has most")
    print("[ ] Token counts are reasonable")
    print("[ ] Energy ordering: SAS < Independent < Centralized ~ Decentralized")
    if tasks:
        print(f"\nResults saved to: {output_file}")
    monitor.shutdown()


if __name__ == "__main__":
    main()
