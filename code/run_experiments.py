"""Main experiment runner for MAS energy benchmarking.

Usage:
    python run_experiments.py --model toy \
        --benchmarks fanoutqa \
        --topologies sas \
        --max-react-steps 2 5 10 \
        --output-dir /atlas2/u/$USER/mas-energy/results/fanoutqa_v4 \
        --n-tasks 50 --n-reps 1

Each (topology, k) combination writes to a separate JSONL:
    {output_dir}/{model}_{topology}_k{k}.jsonl
"""

import argparse
import json
import os
import signal
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


# Graceful shutdown on SIGTERM: finish current task, then exit
_shutdown_requested = False

def _sigterm_handler(signum, frame):
    global _shutdown_requested
    _shutdown_requested = True
    print("\nSIGTERM received — will exit after current task completes.", flush=True)

signal.signal(signal.SIGTERM, _sigterm_handler)


def get_model_config(model_key):
    """Resolve model key to config dict."""
    if model_key == "toy":
        return TOY_MODEL
    if model_key in MODELS:
        return MODELS[model_key]
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


def output_filename(output_dir, model_path, topology, k, n_rounds=None, n_agents=None, thinking=False):
    """Per-(topology, k, rounds, agents) output file for clean separation."""
    safe_name = model_path.replace("/", "_")
    k_str = f"k{k}" if k is not None else "kdefault"
    r_str = f"_R{n_rounds}" if n_rounds is not None else ""
    m_str = f"_M{n_agents}" if n_agents is not None else ""
    t_str = "_thinking" if thinking else ""
    return os.path.join(output_dir, f"{safe_name}_{topology}_{k_str}{r_str}{m_str}{t_str}.jsonl")


def run(args):
    global _shutdown_requested

    model_cfg = get_model_config(args.model)
    model_path = model_cfg["model_path"]
    extra_body = model_cfg.get("extra_body")

    if args.thinking:
        extra_body = {"chat_template_kwargs": {"enable_thinking": True}}

    if args.n_agents is not None:
        import config as _cfg
        _cfg.N_AGENTS = args.n_agents
        # topologies.py imports N_AGENTS at module load; patch it there too
        import topologies as _topo
        _topo.N_AGENTS = args.n_agents
        print(f"[INFO] Overriding N_AGENTS = {args.n_agents}", flush=True)

    client = make_client(
        base_url=args.sglang_url or SGLANG_URL,
    )
    monitor = EnergyMonitor(gpu_index=args.gpu_index)
    os.makedirs(args.output_dir, exist_ok=True)

    # Session metadata (one per invocation)
    safe_name = model_path.replace("/", "_")
    meta_file = os.path.join(args.output_dir, f"{safe_name}_meta.json")
    session_meta = {
        "model": model_path,
        "model_key": args.model,
        "architecture": model_cfg.get("architecture"),
        "gpu_name": monitor.gpu_name,
        "start_time": datetime.now().isoformat(),
        "args": vars(args),
    }

    print(f"Warming up with {N_WARMUP} calls...", flush=True)
    warmup(client, model_path, n=N_WARMUP, extra_body=extra_body)

    time.sleep(15)

    print("Measuring idle GPU power (10s)...", flush=True)
    idle_power = monitor.measure_idle(duration=10)
    session_meta["idle_power_watts"] = idle_power
    print(f"Idle power: {idle_power:.1f} W", flush=True)

    with open(meta_file, "w") as f:
        json.dump(session_meta, f, indent=2)

    # Sweep k-values (or just [None] if not specified)
    k_values = args.max_react_steps if args.max_react_steps else [None]

    for benchmark_name in args.benchmarks:
        print(f"\n{'='*60}", flush=True)
        print(f"Benchmark: {benchmark_name}", flush=True)
        print(f"{'='*60}", flush=True)

        bench = load_benchmark(
            benchmark_name,
            repo_path=args.workbench_path if benchmark_name == "workbench" else None,
        )
        tasks = bench.load_tasks(args.n_tasks)
        tools = bench.get_tools()
        print(f"Loaded {len(tasks)} tasks, {len(tools)} tools", flush=True)

        for topology_name in args.topologies:
            for k in k_values:
                if _shutdown_requested:
                    print("Shutdown requested, exiting.", flush=True)
                    monitor.shutdown()
                    return

                out_file = output_filename(args.output_dir, model_path, topology_name, k,
                                           n_rounds=args.rounds, n_agents=args.n_agents,
                                           thinking=args.thinking)
                completed = load_completed(out_file)

                k_label = f"k={k}" if k is not None else "k=default"
                r_label = f",R={args.rounds}" if args.rounds is not None else ""
                m_label = f",M={args.n_agents}" if args.n_agents is not None else ""
                print(f"\n--- {topology_name} / {k_label}{r_label}{m_label} → {os.path.basename(out_file)} ---",
                      flush=True)
                if completed:
                    print(f"  Resuming: {len(completed)} already done", flush=True)

                runner = TOPOLOGY_RUNNERS[topology_name]
                correct = 0
                total = 0
                topo_energy = 0.0

                for rep in range(args.n_reps):
                    for task in tasks:
                        if _shutdown_requested:
                            print(f"Shutdown requested after {total} tasks.", flush=True)
                            monitor.shutdown()
                            return

                        run_key = (benchmark_name, topology_name, task["id"], rep, k)
                        if run_key in completed:
                            continue

                        monitor.call_log = []
                        executor = None
                        cleanup = None
                        try:
                            executor, cleanup = bench.make_executor(task)
                            recorder = ToolCallRecorder(executor)

                            task_question = task["question"]
                            if benchmark_name == "workbench":
                                task_question = bench.TIME_CONTEXT + "\n\n" + task_question

                            extra_kwargs = {}
                            if k is not None:
                                extra_kwargs["max_react_steps"] = k
                            if args.rounds is not None:
                                extra_kwargs["n_rounds"] = args.rounds
                            # Raw question for orchestrator decomposition
                            # (without benchmark-specific format instructions)
                            raw_q = task.get("question_text", task.get("question", ""))
                            extra_kwargs["raw_question"] = raw_q

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

                            # Benchmark-specific evaluation
                            if benchmark_name == "fanoutqa":
                                from benchmarks_fanoutqa import evaluate_answer
                                is_correct, loose_accuracy = evaluate_answer(
                                    task, result.get("answer", ""))
                            elif benchmark_name == "qampari":
                                from benchmarks_qampari import evaluate_qampari
                                qampari_result = evaluate_qampari(
                                    task["answer_list"], result.get("answer", ""))
                                is_correct = qampari_result["correct"]
                                loose_accuracy = qampari_result["recall_substr"]
                            elif benchmark_name == "maslegalbench":
                                from benchmarks_maslegalbench import evaluate_answer as legal_eval
                                is_correct, extracted = legal_eval(
                                    task, result.get("answer", ""))
                                loose_accuracy = float(is_correct)
                            elif benchmark_name == "browsecomp_plus":
                                from benchmarks_browsecomp import evaluate_browsecomp
                                bc_result = evaluate_browsecomp(
                                    task["answer"], result.get("answer", ""))
                                is_correct = bc_result["correct"]
                                loose_accuracy = float(bc_result["substring_match"])
                            elif benchmark_name == "swebench":
                                swe_result = bench.evaluate(task, recorder)
                                is_correct = swe_result["correct"]
                                loose_accuracy = float(swe_result["has_patch"])
                                result["answer"] = swe_result.get("patch", "")
                            elif benchmark_name == "swebench_batched":
                                swe_result = bench.evaluate(task, recorder)
                                is_correct = swe_result["correct"]
                                loose_accuracy = swe_result["partial_score"]
                                result["answer"] = json.dumps(swe_result.get("patches", {}))
                            elif benchmark_name == "workbench":
                                is_correct = evaluate_sequence(task, recorder)
                                loose_accuracy = None
                            elif benchmark_name == "math":
                                from benchmarks_math import evaluate_math
                                math_result = evaluate_math(task, result.get("answer", ""))
                                is_correct = math_result["correct"]
                                loose_accuracy = math_result["loose_accuracy"]
                            elif benchmark_name == "humaneval":
                                he_result = bench.evaluate(task, recorder, result.get("answer", ""))
                                is_correct = he_result["correct"]
                                loose_accuracy = he_result["loose_accuracy"]
                            elif benchmark_name == "livecodebench":
                                lcb_result = bench.evaluate(task, recorder, result.get("answer", ""))
                                is_correct = lcb_result["correct"]
                                loose_accuracy = lcb_result["loose_accuracy"]
                            elif benchmark_name == "predictability":
                                pred_result = bench.evaluate(task, recorder, result.get("answer", ""))
                                is_correct = pred_result["correct"]
                                loose_accuracy = pred_result["loose_accuracy"]
                            elif benchmark_name == "bigcodebench":
                                bcb_result = bench.evaluate(task, recorder, result.get("answer", ""))
                                is_correct = bcb_result["correct"]
                                loose_accuracy = bcb_result["loose_accuracy"]
                            else:
                                is_correct = evaluate_sequence(task, recorder)
                                loose_accuracy = None

                            records = result["call_records"]
                            run_gpu = sum(r.get("gpu_energy_joules", 0) for r in records)
                            run_gpu_dynamic = sum(r.get("gpu_dynamic_energy_joules", 0) for r in records)
                            run_total = sum(r.get("total_energy_joules", 0) for r in records)
                            run_wall = sum(r.get("wall_seconds", 0) for r in records)

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
                                "max_react_steps": k,
                                "n_rounds_override": args.rounds,
                                "n_agents_override": args.n_agents,
                                "answer": result.get("answer", ""),
                                "correct": is_correct,
                                "loose_accuracy": loose_accuracy,
                                "gpu_energy_joules": run_gpu,
                                "gpu_dynamic_energy_joules": run_gpu_dynamic,
                                "total_energy_joules": run_total,
                                "total_wall_seconds": run_wall,
                                "inference_gpu_energy_joules": sum(
                                    r.get("gpu_dynamic_energy_joules", 0) for r in inference_records
                                ),
                                "tool_exec_gpu_energy_joules": sum(
                                    r.get("gpu_dynamic_energy_joules", 0) for r in tool_records
                                ),
                                "coordination_gpu_energy_joules": sum(
                                    r.get("gpu_dynamic_energy_joules", 0) for r in other_records
                                ),
                                "total_prompt_tokens": result["total_usage"]["prompt_tokens"],
                                "total_completion_tokens": result["total_usage"]["completion_tokens"],
                                "total_tokens": result["total_usage"]["total_tokens"],
                                "n_llm_calls": len(inference_records) + len(other_records),
                                "n_tool_calls": len(tool_records),
                                "n_react_steps": result.get("steps", 0),
                                "rounds_used": result.get("rounds_used"),
                                "subtasks": result.get("subtasks"),
                                "call_records": records,
                                "timestamp": datetime.now().isoformat(),
                            }

                            with open(out_file, "a") as f:
                                f.write(json.dumps(record) + "\n")
                                f.flush()

                            correct += int(is_correct)
                            total += 1
                            topo_energy += run_gpu_dynamic

                            if total % 5 == 0:
                                print(
                                    f"  [{topology_name}/{k_label}] "
                                    f"{total} done, acc={correct/total:.1%}, "
                                    f"avg_gpu_dynamic={topo_energy/total:.2f}J",
                                    flush=True,
                                )

                        except Exception as e:
                            print(f"  ERROR on {task['id']} rep={rep}: {e}", flush=True)
                            import traceback
                            traceback.print_exc()
                            error_record = {
                                "model": model_path,
                                "benchmark": benchmark_name,
                                "topology": topology_name,
                                "task_id": task["id"],
                                "rep": rep,
                                "max_react_steps": k,
                                "error": str(e),
                                "timestamp": datetime.now().isoformat(),
                            }
                            with open(out_file, "a") as f:
                                f.write(json.dumps(error_record) + "\n")
                                f.flush()

                        finally:
                            if cleanup:
                                cleanup()

                if total > 0:
                    print(
                        f"  SUMMARY [{topology_name}/{k_label}]: "
                        f"acc={correct/total:.1%}, "
                        f"avg_gpu_dynamic={topo_energy/total:.2f}J, "
                        f"total_gpu_dynamic={topo_energy:.1f}J",
                        flush=True,
                    )

    print(f"\nDone.", flush=True)
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
    parser.add_argument("--max-react-steps", type=int, nargs="+", default=None,
                        help="k values for Pareto sweep (e.g. 2 5 10)")
    parser.add_argument("--rounds", type=int, default=None,
                        help="Override n_rounds for multi-round topologies (e.g. 1 for single-round)")
    parser.add_argument("--n-agents", type=int, default=None,
                        help="Override N_AGENTS (e.g. 2, 4, 5 for agent count ablation)")
    parser.add_argument("--thinking", action="store_true",
                        help="Enable thinking mode (overrides model config)")
    parser.add_argument("--gpu-index", type=int, default=0)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--sglang-url", default=None, help="Override SGLang URL")
    parser.add_argument("--workbench-path", default=None,
                        help="Path to cloned WorkBench repo")
    parser.add_argument("--save-transcripts", action="store_true",
                        help="Save full LLM request/response text into call_records "
                             "(grows result file size ~10x; intended for spot-check runs only)")
    args = parser.parse_args()
    if args.save_transcripts:
        # Set the env var BEFORE any module that reads it has been imported in a
        # way that captured the value. config.py reads at import time, so we also
        # poke the value directly into the already-imported module.
        import os as _os
        _os.environ["MAS_SAVE_TRANSCRIPTS"] = "1"
        import config as _cfg
        _cfg.SAVE_TRANSCRIPTS = True
        # llm.py imported `SAVE_TRANSCRIPTS` by name, so update it there too
        import llm as _llm
        _llm.SAVE_TRANSCRIPTS = True
        print("[INFO] --save-transcripts ENABLED: full request/response text will be "
              "saved into call_records (this grows file size by ~10x)")
    run(args)


if __name__ == "__main__":
    main()
