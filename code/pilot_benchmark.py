"""Pilot test for any benchmark on cluster.

Records energy (GPU dynamic + total), accuracy, tokens, latency, and
VRAM usage. Runs specified topologies.

Usage (from compute node with SGLang running):
    cd /atlas2/u/$USER/mas_project/mas-energy/code
    python pilot_benchmark.py --benchmark qampari --model toy --n-tasks 10
    python pilot_benchmark.py --benchmark fanoutqa --model toy --n-tasks 10
    python pilot_benchmark.py --benchmark qampari --model toy --n-tasks 5 --topologies sas independent centralized
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime

import pynvml

from config import SGLANG_URL, MODELS, TOY_MODEL, TOPOLOGIES, N_WARMUP
from energy import EnergyMonitor
from llm import make_client, warmup, chat
from benchmarks import load_benchmark
from topologies import TOPOLOGY_RUNNERS


def get_model_path(model_key):
    if model_key == "toy":
        return TOY_MODEL["model_path"], TOY_MODEL.get("extra_body")
    if model_key in MODELS:
        cfg = MODELS[model_key]
        return cfg["model_path"], cfg.get("extra_body")
    return model_key, None


def get_vram_usage_gb(gpu_handle):
    info = pynvml.nvmlDeviceGetMemoryInfo(gpu_handle)
    return info.used / 1e9


def main():
    parser = argparse.ArgumentParser(description="Benchmark Pilot Test")
    parser.add_argument("--benchmark", required=True,
                        help="Benchmark name (qampari, fanoutqa, workbench, etc.)")
    parser.add_argument("--model", default="toy")
    parser.add_argument("--sglang-url", default=None)
    parser.add_argument("--n-tasks", type=int, default=10)
    parser.add_argument("--max-react-steps", type=int, default=10)
    parser.add_argument("--topologies", nargs="+", default=["sas", "independent", "centralized"])
    parser.add_argument("--gpu-index", type=int, default=0)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    benchmark_name = args.benchmark
    model_path, extra_body = get_model_path(args.model)
    sglang_url = args.sglang_url or SGLANG_URL

    print("=" * 60)
    print(f"Pilot Test: {benchmark_name}")
    print("=" * 60)
    print(f"Model: {model_path}")
    print(f"SGLang URL: {sglang_url}")
    print(f"Tasks: {args.n_tasks}")
    print(f"Max ReAct steps: {args.max_react_steps}")
    print(f"Topologies: {args.topologies}")
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

    # --- Phase B: Energy monitor + warmup ---
    print("\n[Phase B] Initializing energy monitor...")
    monitor = EnergyMonitor(gpu_index=args.gpu_index)
    vram_baseline = get_vram_usage_gb(monitor.gpu_handle)
    print(f"  VRAM baseline: {vram_baseline:.1f} GB")

    print(f"\n[Phase C] Warming up ({N_WARMUP} calls)...")
    warmup(client, model_path, n=N_WARMUP, extra_body=extra_body)
    time.sleep(15)

    print("\n[Phase D] Measuring idle power (10s)...")
    idle_power = monitor.measure_idle(duration=10)
    print(f"  Idle power: {idle_power:.1f} W")

    # --- Phase E: Load benchmark ---
    print(f"\n[Phase E] Loading {benchmark_name} ({args.n_tasks} tasks)...")
    bench = load_benchmark(benchmark_name)
    tasks = bench.load_tasks(args.n_tasks)
    tools = bench.get_tools()
    print(f"  Loaded {len(tasks)} tasks, {len(tools)} tool(s)")
    for t in tasks[:3]:
        q = t.get("question", t.get("question_text", ""))
        print(f"    {t.get('id', t.get('qid', '?'))}: {q[:70]}...")

    # --- Output setup ---
    output_dir = args.output_dir or os.path.join(
        os.path.dirname(__file__), "..", "results", "pilot"
    )
    os.makedirs(output_dir, exist_ok=True)
    safe_name = model_path.replace("/", "_")
    output_file = os.path.join(output_dir, f"{benchmark_name}_pilot_{safe_name}.jsonl")

    meta = {
        "model": model_path,
        "model_key": args.model,
        "benchmark": benchmark_name,
        "gpu_name": monitor.gpu_name,
        "vram_baseline_gb": vram_baseline,
        "idle_power_watts": idle_power,
        "n_tasks": args.n_tasks,
        "max_react_steps": args.max_react_steps,
        "topologies": args.topologies,
        "start_time": datetime.now().isoformat(),
        "args": vars(args),
    }
    meta_file = os.path.join(output_dir, f"{benchmark_name}_pilot_{safe_name}_meta.json")
    with open(meta_file, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"\n  Meta: {meta_file}")
    print(f"  Results: {output_file}")

    # --- Phase F: Topology runs ---
    for topo_name in args.topologies:
        print(f"\n{'='*60}")
        print(f"Topology: {topo_name}")
        print(f"{'='*60}")
        runner = TOPOLOGY_RUNNERS[topo_name]

        correct = 0
        total_f1 = 0.0
        total_tokens = 0
        total_wall = 0.0
        total_gpu_dyn = 0.0

        for task in tasks:
            monitor.call_log = []
            executor = None
            cleanup = None
            try:
                executor, cleanup = bench.make_executor(task)
                vram_before = get_vram_usage_gb(monitor.gpu_handle)

                result = runner(
                    client=client,
                    model=model_path,
                    task_question=task.get("question", task.get("question_text", "")),
                    tools=tools,
                    execute_tool=executor,
                    energy_monitor=monitor,
                    extra_body=extra_body,
                    max_react_steps=args.max_react_steps,
                )

                vram_after = get_vram_usage_gb(monitor.gpu_handle)

                eval_result = bench.evaluate(
                    task, None,
                    final_answer=result.get("answer", ""),
                )
                is_correct = eval_result.get("correct", False)
                # Use f1 for QAMPARI, loose_accuracy for FanOutQA, correct for WorkBench
                score = eval_result.get("f1", eval_result.get("loose_accuracy", float(is_correct)))

                records = result["call_records"]
                run_wall = sum(r.get("wall_seconds", 0) for r in records)
                run_gpu = sum(r.get("gpu_energy_joules", 0) for r in records)
                run_gpu_dyn = sum(r.get("gpu_dynamic_energy_joules", 0) for r in records)
                run_total_energy = sum(r.get("total_energy_joules", 0) for r in records)
                run_tokens = result["total_usage"]["total_tokens"]
                run_prompt = result["total_usage"]["prompt_tokens"]
                run_completion = result["total_usage"]["completion_tokens"]

                inference_records = [r for r in records if "react_step" in r.get("call_type", "")]
                tool_records = [r for r in records if r.get("call_type") == "tool_execution"]
                other_records = [r for r in records
                                 if "react_step" not in r.get("call_type", "")
                                 and r.get("call_type") != "tool_execution"]

                n_llm = len(inference_records) + len(other_records)
                n_tool = len(tool_records)

                correct += int(is_correct)
                total_f1 += score
                total_tokens += run_tokens
                total_wall += run_wall
                total_gpu_dyn += run_gpu_dyn

                task_id = task.get("id", task.get("qid", "?"))
                print(
                    f"  {task_id}: "
                    f"{'OK' if is_correct else 'NO'} "
                    f"(score={score:.2f}) | "
                    f"gpu_dyn={run_gpu_dyn:.2f}J | "
                    f"tokens={run_tokens} | "
                    f"llm={n_llm} tool={n_tool} | "
                    f"{run_wall:.1f}s | "
                    f"vram={vram_after:.1f}GB"
                )

                record = {
                    "model": model_path,
                    "model_key": args.model,
                    "benchmark": benchmark_name,
                    "topology": topo_name,
                    "task_id": task_id,
                    "max_react_steps": args.max_react_steps,
                    "correct": is_correct,
                    "score": score,
                    "answer": result.get("answer", ""),
                    "eval_result": eval_result,
                    "gpu_energy_joules": run_gpu,
                    "gpu_dynamic_energy_joules": run_gpu_dyn,
                    "total_energy_joules": run_total_energy,
                    "idle_power_watts": idle_power,
                    "inference_gpu_dynamic_joules": sum(
                        r.get("gpu_dynamic_energy_joules", 0) for r in inference_records
                    ),
                    "tool_exec_gpu_dynamic_joules": sum(
                        r.get("gpu_dynamic_energy_joules", 0) for r in tool_records
                    ),
                    "coordination_gpu_dynamic_joules": sum(
                        r.get("gpu_dynamic_energy_joules", 0) for r in other_records
                    ),
                    "prompt_tokens": run_prompt,
                    "completion_tokens": run_completion,
                    "total_tokens": run_tokens,
                    "n_llm_calls": n_llm,
                    "n_tool_calls": n_tool,
                    "n_react_steps": result.get("steps", 0),
                    "wall_seconds": run_wall,
                    "vram_before_gb": vram_before,
                    "vram_after_gb": vram_after,
                    "vram_peak_gb": max(vram_before, vram_after),
                    "gpu_name": monitor.gpu_name,
                    "timestamp": datetime.now().isoformat(),
                }
                with open(output_file, "a") as f:
                    f.write(json.dumps(record) + "\n")

            except Exception as e:
                task_id = task.get("id", task.get("qid", "?"))
                print(f"  {task_id}: ERROR - {e}")
                import traceback
                traceback.print_exc()
                error_record = {
                    "model": model_path,
                    "benchmark": benchmark_name,
                    "topology": topo_name,
                    "task_id": task_id,
                    "max_react_steps": args.max_react_steps,
                    "error": str(e),
                    "timestamp": datetime.now().isoformat(),
                }
                with open(output_file, "a") as f:
                    f.write(json.dumps(error_record) + "\n")

            finally:
                if cleanup:
                    cleanup()

        n = len(tasks)
        if n > 0:
            print(
                f"\n  SUMMARY [{topo_name}]: "
                f"correct={correct}/{n} ({correct/n:.0%}), "
                f"avg_score={total_f1/n:.3f}, "
                f"gpu_dyn={total_gpu_dyn:.1f}J (avg {total_gpu_dyn/n:.2f}J/task), "
                f"tokens={total_tokens} (avg {total_tokens/n:.0f}/task), "
                f"wall={total_wall:.1f}s (avg {total_wall/n:.1f}s/task)"
            )

    print(f"\n{'='*60}")
    print(f"Results saved to: {output_file}")
    print("=" * 60)
    monitor.shutdown()


if __name__ == "__main__":
    main()
