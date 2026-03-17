"""Main experiment runner for MAS energy benchmarking.

Usage:
    python run_experiments.py --model qwen35-27b \
        --benchmarks workbench \
        --topologies sas independent centralized decentralized \
        --output-dir /atlas2/u/$USER/mas-energy/results/pilot \
        --n-tasks 10 --n-reps 1

    # Toy experiment with Qwen3.5-9B
    python run_experiments.py --model toy \
        --benchmarks workbench \
        --topologies sas independent \
        --output-dir /atlas2/u/$USER/mas-energy/results/toy \
        --n-tasks 5 --n-reps 1 \
        --workbench-path /atlas2/u/$USER/WorkBench
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime

from config import (
    MODELS, TOY_MODEL, TOPOLOGIES, BENCHMARKS,
    N_REPS, N_WARMUP, SGLANG_URL,
)
from energy import EnergyMonitor
from llm import make_client, warmup
from benchmarks import load_benchmark, ToolCallRecorder, evaluate_sequence
from topologies import TOPOLOGY_RUNNERS


def get_model_config(model_key):
    """Resolve model key to config dict."""
    if model_key == "toy":
        return TOY_MODEL
    if model_key in MODELS:
        return MODELS[model_key]
    # Allow raw model path
    return {
        "model_path": model_key,
        "architecture": "unknown",
        "quantization": None,
        "tool_call_parser": "qwen3_coder",
        "extra_body": None,
        "sglang_extra_args": [],
    }


def load_completed(output_file):
    """Load already-completed (benchmark, topology, task_id, rep, max_react_steps) tuples."""
    completed = set()
    if os.path.exists(output_file):
        with open(output_file, "r") as f:
            for line in f:
                try:
                    d = json.loads(line)
                    if "error" not in d or "answer" in d:
                        key = (d["benchmark"], d["topology"], d["task_id"],
                               d["rep"], d.get("max_react_steps"))
                        completed.add(key)
                except (json.JSONDecodeError, KeyError):
                    continue
    return completed


def run(args):
    model_cfg = get_model_config(args.model)
    model_path = model_cfg["model_path"]
    extra_body = model_cfg.get("extra_body")

    client = make_client(
        base_url=args.sglang_url or SGLANG_URL,
    )
    monitor = EnergyMonitor(gpu_index=args.gpu_index)

    os.makedirs(args.output_dir, exist_ok=True)
    safe_name = model_path.replace("/", "_")
    output_file = os.path.join(args.output_dir, f"{safe_name}.jsonl")

    # Session metadata
    meta_file = os.path.join(args.output_dir, f"{safe_name}_meta.json")
    session_meta = {
        "model": model_path,
        "model_key": args.model,
        "architecture": model_cfg.get("architecture"),
        "gpu_name": monitor.gpu_name,
        "start_time": datetime.now().isoformat(),
        "args": vars(args),
    }

    # Warmup first: ensures model is fully loaded in VRAM before idle measurement.
    # SGLang health check can pass before model loading completes, so measuring
    # idle before warmup risks capturing bare GPU idle (~25W) instead of
    # model-loaded idle (~65W).
    print(f"Warming up with {N_WARMUP} calls...")
    warmup(client, model_path, n=N_WARMUP, extra_body=extra_body)

    # Brief cooldown so GPU settles to idle after warmup inference
    time.sleep(15)

    # Measure idle power (model in VRAM, no active inference)
    print("Measuring idle GPU power (10s)...")
    idle_power = monitor.measure_idle(duration=10)
    session_meta["idle_power_watts"] = idle_power
    print(f"Idle power: {idle_power:.1f} W")

    with open(meta_file, "w") as f:
        json.dump(session_meta, f, indent=2)

    # Resume support
    completed = load_completed(output_file)
    if completed:
        print(f"Resuming: {len(completed)} runs already completed")

    completed_count = 0
    skipped_count = 0

    for benchmark_name in args.benchmarks:
        print(f"\n{'='*60}")
        print(f"Benchmark: {benchmark_name}")
        print(f"{'='*60}")

        bench = load_benchmark(
            benchmark_name,
            repo_path=args.workbench_path if benchmark_name == "workbench" else None,
        )
        tasks = bench.load_tasks(args.n_tasks)
        tools = bench.get_tools()
        print(f"Loaded {len(tasks)} tasks, {len(tools)} tools")

        for topology_name in args.topologies:
            print(f"\n--- Topology: {topology_name} ---")
            runner = TOPOLOGY_RUNNERS[topology_name]

            correct = 0
            total = 0
            topo_energy = 0.0

            for rep in range(args.n_reps):
                for task in tasks:
                    key = (benchmark_name, topology_name, task["id"], rep,
                           args.max_react_steps)
                    if key in completed:
                        skipped_count += 1
                        continue

                    monitor.call_log = []

                    executor = None
                    cleanup = None
                    try:
                        executor, cleanup = bench.make_executor(task)
                        recorder = ToolCallRecorder(executor)

                        # Prepend time context for workbench tasks
                        task_question = task["question"]
                        if benchmark_name == "workbench":
                            task_question = bench.TIME_CONTEXT + "\n\n" + task_question

                        extra_kwargs = {}
                        if args.max_react_steps is not None:
                            extra_kwargs["max_react_steps"] = args.max_react_steps

                        result = runner(
                            client=client,
                            model=model_path,
                            task_question=task_question,
                            tools=tools,
                            execute_tool=recorder,
                            energy_monitor=monitor,
                            extra_body=extra_body,
                            **extra_kwargs,
                        )

                        # Sequence-matching eval (Kim et al.): gold ⊆ predicted
                        is_correct = evaluate_sequence(task, recorder)

                        # Aggregate energy from call records
                        records = result["call_records"]
                        run_gpu = sum(r.get("gpu_energy_joules", 0) for r in records)
                        run_gpu_dynamic = sum(r.get("gpu_dynamic_energy_joules", 0) for r in records)
                        run_total = sum(r.get("total_energy_joules", 0) for r in records)
                        run_wall = sum(r.get("wall_seconds", 0) for r in records)

                        # Separate inference vs tool execution energy
                        inference_records = [r for r in records if "react_step" in r.get("call_type", "")]
                        tool_records = [r for r in records if r.get("call_type") == "tool_execution"]
                        other_records = [r for r in records
                                         if "react_step" not in r.get("call_type", "")
                                         and r.get("call_type") != "tool_execution"]

                        record = {
                            "model": model_path,
                            "model_key": args.model,
                            "architecture": model_cfg.get("architecture"),
                            "benchmark": benchmark_name,
                            "topology": topology_name,
                            "task_id": task["id"],
                            "rep": rep,
                            "max_react_steps": args.max_react_steps,
                            "answer": result.get("answer", ""),
                            "correct": is_correct,
                            # Energy (primary: GPU dynamic)
                            "gpu_energy_joules": run_gpu,
                            "gpu_dynamic_energy_joules": run_gpu_dynamic,
                            "total_energy_joules": run_total,
                            "total_wall_seconds": run_wall,
                            # Energy decomposition
                            "inference_gpu_energy_joules": sum(
                                r.get("gpu_dynamic_energy_joules", 0) for r in inference_records
                            ),
                            "tool_exec_gpu_energy_joules": sum(
                                r.get("gpu_dynamic_energy_joules", 0) for r in tool_records
                            ),
                            "coordination_gpu_energy_joules": sum(
                                r.get("gpu_dynamic_energy_joules", 0) for r in other_records
                            ),
                            # Tokens
                            "total_prompt_tokens": result["total_usage"]["prompt_tokens"],
                            "total_completion_tokens": result["total_usage"]["completion_tokens"],
                            "total_tokens": result["total_usage"]["total_tokens"],
                            # Calls
                            "n_llm_calls": len(inference_records) + len(other_records),
                            "n_tool_calls": len(tool_records),
                            "n_react_steps": result.get("steps", 0),
                            "call_records": records,
                            "timestamp": datetime.now().isoformat(),
                        }

                        with open(output_file, "a") as f:
                            f.write(json.dumps(record) + "\n")

                        correct += int(is_correct)
                        total += 1
                        topo_energy += run_gpu_dynamic
                        completed_count += 1

                        if total % 5 == 0:
                            print(
                                f"  [{benchmark_name}/{topology_name}] "
                                f"{total} done, acc={correct/total:.1%}, "
                                f"avg_gpu_dynamic={topo_energy/total:.2f}J"
                            )

                    except Exception as e:
                        print(f"  ERROR on {task['id']} rep={rep}: {e}")
                        import traceback
                        traceback.print_exc()
                        error_record = {
                            "model": model_path,
                            "benchmark": benchmark_name,
                            "topology": topology_name,
                            "task_id": task["id"],
                            "rep": rep,
                            "max_react_steps": args.max_react_steps,
                            "error": str(e),
                            "timestamp": datetime.now().isoformat(),
                        }
                        with open(output_file, "a") as f:
                            f.write(json.dumps(error_record) + "\n")
                        completed_count += 1

                    finally:
                        if cleanup:
                            cleanup()

            if total > 0:
                print(
                    f"  SUMMARY [{benchmark_name}/{topology_name}]: "
                    f"acc={correct/total:.1%}, "
                    f"avg_gpu_dynamic={topo_energy/total:.2f}J, "
                    f"total_gpu_dynamic={topo_energy:.1f}J"
                )

    print(f"\nDone. {completed_count} completed, {skipped_count} skipped (resume).")
    print(f"Results: {output_file}")
    monitor.shutdown()


def main():
    parser = argparse.ArgumentParser(description="MAS Energy Benchmarking")
    parser.add_argument("--model", required=True,
                        help="Model key (qwen35b-a3b, glm47-flash, qwen35-27b, toy) "
                             "or HuggingFace model path")
    parser.add_argument("--benchmarks", nargs="+", default=BENCHMARKS)
    parser.add_argument("--topologies", nargs="+", default=TOPOLOGIES)
    parser.add_argument("--n-tasks", type=int, default=None,
                        help="Tasks per benchmark (None = all)")
    parser.add_argument("--n-reps", type=int, default=N_REPS)
    parser.add_argument("--max-react-steps", type=int, default=None,
                        help="Override MAX_REACT_STEPS for Pareto sweep")
    parser.add_argument("--gpu-index", type=int, default=0)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--sglang-url", default=None, help="Override SGLang URL")
    parser.add_argument("--workbench-path", default=None,
                        help="Path to cloned WorkBench repo")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
