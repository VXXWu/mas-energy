"""Arm B of the tool-latency experiment (deep-research energy side study):
concurrent SAS sessions with injected tool latency, AGGREGATE energy metering.

Why a separate runner: per-call NVML windows overlap under concurrency and
double-count, so per-call energy attribution is invalid here. This runner
meters the whole cell at the server level (NVML counter start/end + 1 Hz
power sampling + one CodeCarbon task for CPU/RAM) and reports J per
completed task. Per-task records keep wall/tokens/tool-calls only, with
energy fields zeroed and flagged.

One invocation = ONE (concurrency, tool_latency) cell. Tool latency is set
via the MAS_TOOL_LATENCY_S env var (read by config.py/llm.py at import), so
the sbatch script loops over cells with one python invocation each.

Design: difference-in-differences. For each concurrency N, run tau=0 and
tau>0 cells; the idle-absorption estimate is the per-task energy DIFFERENCE
at fixed N, which cancels batching-efficiency gains on the LLM side.

Usage (per cell):
  MAS_TOOL_LATENCY_S=5 python run_latency_experiment.py --model toy \
      --k 10 --concurrency 8 --n-tasks 48 --output-dir <dir>
"""

import argparse
import json
import os
import queue
import threading
import time
from datetime import datetime

import pynvml
from codecarbon import EmissionsTracker

from config import TOOL_LATENCY_S, TOOL_LATENCY_JITTER, MODELS, TOY_MODEL
from llm import make_client, warmup
from benchmarks import load_benchmark, ToolCallRecorder
from benchmarks_browsecomp import evaluate_browsecomp
from topologies import run_sas


class ThreadLocalMonitor:
    """Same start()/stop() interface as EnergyMonitor, but records wall time
    and metadata only. Energy fields are zeroed: per-call energy is not
    attributable when calls from concurrent sessions overlap on one GPU."""

    def __init__(self):
        self.call_log = []
        self._t0 = None

    def start(self):
        self._t0 = time.monotonic()

    def stop(self, metadata=None):
        wall_s = time.monotonic() - (self._t0 or time.monotonic())
        record = {
            "gpu_energy_joules": 0.0,
            "gpu_dynamic_energy_joules": 0.0,
            "gpu_idle_energy_joules": 0.0,
            "cpu_energy_joules": 0.0,
            "ram_energy_joules": 0.0,
            "total_energy_joules": 0.0,
            "wall_seconds": wall_s,
            "avg_gpu_power_watts": 0.0,
            "emissions_kg_co2": 0.0,
            "per_call_energy": "not_attributable_concurrent",
        }
        if metadata:
            record.update(metadata)
        self.call_log.append(record)
        return record


class AggregateMeter:
    """Server-level metering for one cell: NVML energy counter at start/end,
    1 Hz power sampling for the utilization trace, CodeCarbon for CPU/RAM."""

    def __init__(self):
        pynvml.nvmlInit()
        self.handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        self.gpu_name = pynvml.nvmlDeviceGetName(self.handle)
        self._tracker = EmissionsTracker(
            log_level="error", tracking_mode="process", allow_multiple_runs=True)
        self._tracker.start()
        self.idle_power_watts = None
        self._sampling = False

    def measure_idle(self, duration=10):
        e0 = pynvml.nvmlDeviceGetTotalEnergyConsumption(self.handle)
        time.sleep(duration)
        e1 = pynvml.nvmlDeviceGetTotalEnergyConsumption(self.handle)
        self.idle_power_watts = (e1 - e0) / 1000.0 / duration
        return self.idle_power_watts

    def _sample_loop(self):
        while self._sampling:
            try:
                mw = pynvml.nvmlDeviceGetPowerUsage(self.handle)
                self.power_samples.append(mw / 1000.0)
            except pynvml.NVMLError:
                pass
            time.sleep(1.0)

    def start(self):
        self.power_samples = []
        self._t0 = time.monotonic()
        self._e0 = pynvml.nvmlDeviceGetTotalEnergyConsumption(self.handle)
        self._tracker.start_task("cell")
        self._sampling = True
        self._sampler = threading.Thread(target=self._sample_loop, daemon=True)
        self._sampler.start()

    def stop(self):
        self._sampling = False
        self._sampler.join(timeout=3)
        e1 = pynvml.nvmlDeviceGetTotalEnergyConsumption(self.handle)
        emissions = self._tracker.stop_task("cell")
        wall_s = time.monotonic() - self._t0
        gpu_j = (e1 - self._e0) / 1000.0
        KWH_TO_J = 3_600_000
        cpu_j = emissions.cpu_energy * KWH_TO_J if emissions else 0.0
        ram_j = emissions.ram_energy * KWH_TO_J if emissions else 0.0
        ps = sorted(self.power_samples)
        def pct(p):
            return ps[min(len(ps) - 1, int(p / 100 * len(ps)))] if ps else None
        return {
            "gpu_energy_joules": gpu_j,
            "cpu_energy_joules": cpu_j,
            "ram_energy_joules": ram_j,
            "total_energy_joules": gpu_j + cpu_j + ram_j,
            "wall_seconds": wall_s,
            "avg_gpu_power_watts": gpu_j / wall_s if wall_s > 0 else 0.0,
            "idle_power_watts": self.idle_power_watts,
            "power_trace": {
                "n_samples": len(ps), "mean_w": sum(ps) / len(ps) if ps else None,
                "p10_w": pct(10), "p50_w": pct(50), "p90_w": pct(90),
            },
        }

    def shutdown(self):
        self._tracker.stop()
        pynvml.nvmlShutdown()


def worker_loop(wq, results, lock, client, model_path, tools, bench, k, extra_body):
    while True:
        try:
            task, rep = wq.get_nowait()
        except queue.Empty:
            return
        monitor = ThreadLocalMonitor()
        executor, cleanup = None, None
        t0 = time.monotonic()
        try:
            executor, cleanup = bench.make_executor(task)
            recorder = ToolCallRecorder(executor)
            result = run_sas(
                client=client, model=model_path,
                task_question=task["question"], tools=tools,
                execute_tool=recorder, energy_monitor=monitor,
                extra_body=extra_body, max_react_steps=k,
                raw_question=task.get("question_text", task.get("question", "")),
            )
            bc = evaluate_browsecomp(task["answer"], result.get("answer", ""))
            usage = result.get("total_usage", {})
            injected = sum(r.get("injected_latency_s", 0) for r in monitor.call_log)
            row = {
                "task_id": task["id"], "rep": rep,
                "correct": bc["correct"],
                "substring_match": bc["substring_match"],
                "task_wall_seconds": time.monotonic() - t0,
                "prompt_tokens": usage.get("prompt_tokens", 0),
                "completion_tokens": usage.get("completion_tokens", 0),
                "n_llm_calls": sum(1 for r in monitor.call_log
                                   if r.get("call_type") != "tool_execution"),
                "n_tool_calls": sum(1 for r in monitor.call_log
                                    if r.get("call_type") == "tool_execution"),
                "tool_wall_seconds": sum(r["wall_seconds"] for r in monitor.call_log
                                         if r.get("call_type") == "tool_execution"),
                "injected_latency_seconds": injected,
                "status": "ok",
            }
        except Exception as e:
            row = {"task_id": task.get("id"), "rep": rep, "status": f"error: {e}"}
        finally:
            if cleanup:
                try:
                    cleanup()
                except Exception:
                    pass
        with lock:
            results.append(row)
            done = len(results)
        print(f"  [{done}] task {row.get('task_id')} "
              f"{row.get('status')} wall={row.get('task_wall_seconds', 0):.0f}s",
              flush=True)
        wq.task_done()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="toy")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--concurrency", type=int, required=True)
    ap.add_argument("--n-tasks", type=int, default=48)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--sglang-url", default=None)
    args = ap.parse_args()

    model_cfg = TOY_MODEL if args.model == "toy" else MODELS[args.model]
    model_path = model_cfg["model_path"]
    extra_body = model_cfg.get("extra_body")

    tau_label = (f"{TOOL_LATENCY_S:g}".replace(".", "p"))
    safe_name = model_path.replace("/", "_")
    os.makedirs(args.output_dir, exist_ok=True)
    out_file = os.path.join(
        args.output_dir,
        f"{safe_name}_latency_sas_k{args.k}_N{args.concurrency}_tau{tau_label}.jsonl")
    if os.path.exists(out_file):
        with open(out_file) as f:
            if any('"aggregate"' in line for line in f):
                print(f"Cell already complete: {out_file} — skipping.", flush=True)
                return

    bench = load_benchmark("browsecomp_plus")
    tasks = bench.load_tasks(args.n_tasks)
    tools = bench.get_tools()
    print(f"Cell: N={args.concurrency}, tau={TOOL_LATENCY_S}s "
          f"(jitter {TOOL_LATENCY_JITTER}), k={args.k}, "
          f"{len(tasks)} tasks -> {os.path.basename(out_file)}", flush=True)

    client = make_client(base_url=args.sglang_url, timeout=600)
    meter = AggregateMeter()

    print("Warming up (5 calls)...", flush=True)
    warmup(client, model_path, n=5, extra_body=extra_body)
    time.sleep(15)
    idle_w = meter.measure_idle(duration=10)
    print(f"Idle power: {idle_w:.1f} W", flush=True)

    wq = queue.Queue()
    for task in tasks:
        wq.put((task, 0))
    results, lock = [], threading.Lock()

    meter.start()
    threads = [
        threading.Thread(
            target=worker_loop,
            args=(wq, results, lock, client, model_path, tools, bench, args.k,
                  extra_body),
            daemon=True)
        for _ in range(args.concurrency)
    ]
    t_start = time.monotonic()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    agg = meter.stop()

    n_ok = sum(1 for r in results if r.get("status") == "ok")
    aggregate = {
        "record_type": "aggregate",
        "model": model_path, "model_key": args.model,
        "benchmark": "browsecomp_plus", "topology": "sas",
        "k": args.k, "concurrency": args.concurrency,
        "tool_latency_s": TOOL_LATENCY_S, "tool_latency_jitter": TOOL_LATENCY_JITTER,
        "n_tasks_requested": len(tasks), "n_tasks_ok": n_ok,
        "n_tasks_error": len(results) - n_ok,
        "joules_per_task_gpu": agg["gpu_energy_joules"] / n_ok if n_ok else None,
        "joules_per_task_total": agg["total_energy_joules"] / n_ok if n_ok else None,
        "throughput_tasks_per_hour": n_ok / (agg["wall_seconds"] / 3600)
                                     if agg["wall_seconds"] > 0 else None,
        "gpu_name": meter.gpu_name,
        "timestamp": datetime.now().isoformat(),
        **agg,
    }
    with open(out_file, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
        f.write(json.dumps(aggregate) + "\n")
    meter.shutdown()
    print(f"\nCell done: {n_ok}/{len(tasks)} ok, "
          f"{aggregate['joules_per_task_gpu'] and round(aggregate['joules_per_task_gpu'])} J/task GPU, "
          f"mean power {agg['power_trace']['mean_w'] and round(agg['power_trace']['mean_w'])} W "
          f"-> {out_file}", flush=True)


if __name__ == "__main__":
    main()
