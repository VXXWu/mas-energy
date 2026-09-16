"""Self-batching experiment: measure the within-task parallel-agent energy
discount for MAS topologies (2026-07-22).

The batch-coefficient analysis predicts (analytically) that a parallel
topology's agents self-batch to B~N in dedicated serving, discounting the
decode term b(1)->b(N) — a discount SAS cannot access. This runner measures
it: the SAME topology cell runs in serial mode (canonical protocol) and
parallel mode (MAS_PARALLEL_AGENTS=1 phase-level agent concurrency in
topologies.py), both under the SAME instrument.

Instrument: tasks run strictly one at a time, so a task-level NVML window
(counter delta, idle-subtracted) is valid in BOTH modes; per-call energy is
zeroed+flagged in both modes for symmetry (parallel-safe monitor). Cell-level
AggregateMeter adds CPU/RAM (CodeCarbon) and a 1 Hz power trace.

Benchmark: FanOutQA (stateless search tool — concurrent agents are
semantically identical to serial; SWE-bench is excluded because agents share
a mutable repo executor and would race). k=10, R=2 canonical.

One invocation = one (topology, N, mode) cell. Resume: a cell whose output
file already contains its aggregate row is skipped.

Usage (per cell):
  python run_selfbatch_experiment.py --model toy --topology decentralized \
      --n-agents 3 --parallel 1 --k 10 --n-tasks 40 --output-dir <dir>
"""

import argparse
import json
import os
import time
from datetime import datetime


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="toy")
    ap.add_argument("--topology", required=True,
                    choices=["sas", "independent", "centralized", "decentralized"])
    ap.add_argument("--parallel", type=int, default=0, choices=[0, 1])
    ap.add_argument("--n-agents", type=int, default=3)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--n-rounds", type=int, default=2)
    ap.add_argument("--n-tasks", type=int, default=40)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--sglang-url", default=None)
    args = ap.parse_args()

    # MUST precede the topologies import (flag is read at module load).
    os.environ["MAS_PARALLEL_AGENTS"] = str(args.parallel)

    import pynvml
    import config as _cfg
    _cfg.N_AGENTS = args.n_agents
    from llm import make_client, warmup
    from benchmarks import load_benchmark, ToolCallRecorder
    import topologies as _topo
    _topo.N_AGENTS = args.n_agents
    from run_latency_experiment import ThreadLocalMonitor, AggregateMeter

    model_cfg = _cfg.TOY_MODEL if args.model == "toy" else _cfg.MODELS[args.model]
    model_path = model_cfg["model_path"]
    extra_body = model_cfg.get("extra_body")

    runner = {
        "sas": _topo.run_sas,
        "independent": _topo.run_independent,
        "centralized": _topo.run_centralized,
        "decentralized": _topo.run_decentralized,
    }[args.topology]

    mode = "par" if args.parallel else "ser"
    safe_name = model_path.replace("/", "_")
    os.makedirs(args.output_dir, exist_ok=True)
    out_file = os.path.join(
        args.output_dir,
        f"{safe_name}_selfbatch_{args.topology}_k{args.k}"
        f"_N{args.n_agents}_R{args.n_rounds}_{mode}.jsonl")
    if os.path.exists(out_file):
        with open(out_file) as f:
            if any('"aggregate"' in line for line in f):
                print(f"Cell already complete: {out_file} — skipping.", flush=True)
                return

    class ParallelSafeMonitor(ThreadLocalMonitor):
        parallel_safe = True   # same zeroed instrument in BOTH modes

        def stop(self, metadata=None):
            t_end = time.monotonic()
            t0 = self._t0 or t_end
            rec = super().stop(metadata)
            rec["t_start"], rec["t_end"] = t0, t_end
            return rec

    def peak_concurrency(call_log):
        # max overlapping LLM-call intervals (tool_execution excluded): the
        # ACHIEVED concurrency. In parallel mode, peak << N ⇒ pool-limited
        # staggering, not real self-batching. Serial mode ⇒ ~1.
        iv = [(r["t_start"], r["t_end"]) for r in call_log
              if r.get("call_type") != "tool_execution"
              and r.get("t_start") is not None]
        if not iv:
            return 0
        ev = sorted([(s, 1) for s, _ in iv] + [(e, -1) for _, e in iv])
        cur = peak = 0
        for _, d in ev:
            cur += d
            peak = max(peak, cur)
        return peak

    bench = load_benchmark("fanoutqa")
    tasks = bench.load_tasks(args.n_tasks)
    tools = bench.get_tools()
    print(f"Cell: {args.topology} N={args.n_agents} k={args.k} "
          f"R={args.n_rounds} mode={mode}, {len(tasks)} tasks "
          f"-> {os.path.basename(out_file)}", flush=True)

    client = make_client(base_url=args.sglang_url, timeout=1800)
    meter = AggregateMeter()
    print("Warming up (5 calls)...", flush=True)
    warmup(client, model_path, n=5, extra_body=extra_body)
    time.sleep(15)
    idle_w = meter.measure_idle(duration=10)
    print(f"Idle power: {idle_w:.1f} W", flush=True)

    handle = meter.handle
    rows = []
    meter.start()
    for t_idx, task in enumerate(tasks):
        monitor = ParallelSafeMonitor()
        executor, cleanup = None, None
        e0 = pynvml.nvmlDeviceGetTotalEnergyConsumption(handle)
        t0 = time.monotonic()
        try:
            executor, cleanup = bench.make_executor(task)
            recorder = ToolCallRecorder(executor)
            kwargs = dict(max_react_steps=args.k,
                          raw_question=task.get("question_text",
                                                task.get("question", "")))
            if args.topology in ("centralized", "decentralized"):
                kwargs["n_rounds"] = args.n_rounds
            result = runner(
                client=client, model=model_path,
                task_question=task["question"], tools=tools,
                execute_tool=recorder, energy_monitor=monitor,
                extra_body=extra_body, **kwargs)
            wall_s = time.monotonic() - t0
            e1 = pynvml.nvmlDeviceGetTotalEnergyConsumption(handle)
            gpu_j = (e1 - e0) / 1000.0
            ev = bench.evaluate(task, recorder, result.get("answer", ""))
            usage = result.get("total_usage", {})
            row = {
                "task_id": task["id"], "status": "ok",
                "correct": ev.get("correct"),
                "loose_accuracy": ev.get("loose_accuracy"),
                "task_gpu_energy_joules": gpu_j,
                "task_gpu_dynamic_joules": max(0.0, gpu_j - idle_w * wall_s),
                "task_wall_seconds": wall_s,
                "prompt_tokens": usage.get("prompt_tokens", 0),
                "completion_tokens": usage.get("completion_tokens", 0),
                "n_llm_calls": sum(1 for r in monitor.call_log
                                   if r.get("call_type") != "tool_execution"),
                "n_tool_calls": sum(1 for r in monitor.call_log
                                    if r.get("call_type") == "tool_execution"),
                "peak_llm_concurrency": peak_concurrency(monitor.call_log),
                "answer": result.get("answer", ""),
                "call_records": monitor.call_log,   # transcripts ride here
            }
        except Exception as e:
            row = {"task_id": task.get("id"), "status": f"error: {e}",
                   "task_wall_seconds": time.monotonic() - t0}
        finally:
            if cleanup:
                try:
                    cleanup()
                except Exception:
                    pass
        rows.append(row)
        print(f"  [{t_idx + 1}/{len(tasks)}] {row.get('task_id')} "
              f"{row['status']} {row.get('task_gpu_dynamic_joules', 0):.0f} J "
              f"{row.get('task_wall_seconds', 0):.0f}s", flush=True)
    agg = meter.stop()

    ok = [r for r in rows if r.get("status") == "ok"]
    aggregate = {
        "record_type": "aggregate",
        "model": model_path, "model_key": args.model,
        "benchmark": "fanoutqa", "topology": args.topology,
        "k": args.k, "n_agents": args.n_agents, "n_rounds": args.n_rounds,
        "parallel_agents": bool(args.parallel),
        "n_tasks_requested": len(tasks), "n_tasks_ok": len(ok),
        "n_tasks_error": len(rows) - len(ok),
        "mean_task_gpu_dynamic_joules":
            sum(r["task_gpu_dynamic_joules"] for r in ok) / len(ok) if ok else None,
        "mean_task_wall_seconds":
            sum(r["task_wall_seconds"] for r in ok) / len(ok) if ok else None,
        "mean_loose_accuracy":
            sum(r["loose_accuracy"] or 0 for r in ok) / len(ok) if ok else None,
        # median achieved LLM concurrency: in parallel mode, << n_agents
        # flags pool-limited staggering (result is a memory ceiling, not a
        # self-batch null). Serial mode ≈ 1 (validates the instrument).
        "median_peak_concurrency":
            sorted(r["peak_llm_concurrency"] for r in ok)[len(ok) // 2] if ok else None,
        "gpu_name": meter.gpu_name,
        "timestamp": datetime.now().isoformat(),
        **agg,
    }
    with open(out_file, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
        f.write(json.dumps(aggregate) + "\n")
    meter.shutdown()
    print(f"\nCell done: {len(ok)}/{len(tasks)} ok, "
          f"{aggregate['mean_task_gpu_dynamic_joules'] and round(aggregate['mean_task_gpu_dynamic_joules'])} J/task dynamic "
          f"-> {out_file}", flush=True)


if __name__ == "__main__":
    main()
