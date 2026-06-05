"""Standalone HumanEval+ saturation pilot.

Runs SAS-style single-agent code completion on N HumanEval+ problems and
reports pass@1. Used to decide whether HumanEval+ is too saturated for
Qwen3.5-9B (in which case we'd pivot to LiveCodeBench for the second
code benchmark).

Decision rule:
  pass@1 < 0.70  → run full ablation suite on HumanEval+ (best case)
  pass@1 0.70-0.85 → usable; run with awareness CIs will be tighter
  pass@1 > 0.85  → saturated; pivot to LiveCodeBench

This script does NOT integrate with the MAS topology runner. It only
calls the SGLang server directly with a single agent prompt per problem.
The point is a fast saturation check, not a full ablation.

Run: python pilot_humaneval_saturation.py --sglang-url http://localhost:PORT/v1 --n-tasks 20
"""
import argparse
import json
import multiprocessing
import re
import sys
import time
from pathlib import Path

from openai import OpenAI


def load_humaneval_plus(n_tasks: int, seed: int = 42):
    """Load N HumanEval+ tasks from HuggingFace cache."""
    from datasets import load_dataset
    ds = load_dataset("evalplus/humanevalplus", split="test")
    rows = [dict(r) for r in ds]
    # Stable shuffle so the same N tasks are sampled across pilots
    import random
    rng = random.Random(seed)
    rng.shuffle(rows)
    return rows[:n_tasks] if n_tasks else rows


def extract_function(completion: str, entry_point: str) -> str:
    """Best-effort extraction of the function definition from a completion.

    Strategy: find the def for entry_point, return everything from that line
    onward until a non-indented line that isn't blank/comment. Falls back to
    the entire completion if no def is found.
    """
    pattern = re.compile(rf"^def\s+{re.escape(entry_point)}\s*\(", re.MULTILINE)
    m = pattern.search(completion)
    if not m:
        # Maybe the model returned just the body (continuation of prompt)
        return completion
    start = m.start()
    # Find end: first non-indented line after start that isn't blank
    lines = completion[start:].split("\n")
    out_lines = [lines[0]]
    for line in lines[1:]:
        if line and not line.startswith((" ", "\t")) and not line.startswith("#"):
            break
        out_lines.append(line)
    return "\n".join(out_lines)


def _run_check(prompt, completion, test, entry_point, time_limit_s, q):
    """Worker: exec prompt+completion, then exec test, then call check(fn)."""
    try:
        full_code = prompt + "\n" + completion
        # Try: assume the model returned the full function (with signature)
        ns = {"__name__": "__test__"}
        try:
            exec(completion, ns)
            if entry_point not in ns:
                # Fall back: prompt+completion form
                ns = {"__name__": "__test__"}
                exec(full_code, ns)
        except Exception:
            ns = {"__name__": "__test__"}
            exec(full_code, ns)

        if entry_point not in ns:
            q.put(("fail", f"entry point '{entry_point}' not defined"))
            return
        # Now exec the test and call check
        exec(test, ns)
        if "check" not in ns:
            q.put(("fail", "test did not define check()"))
            return
        ns["check"](ns[entry_point])
        q.put(("pass", ""))
    except Exception as e:
        q.put(("fail", f"{type(e).__name__}: {e}"))


def evaluate_one(prompt, completion, test, entry_point, time_limit_s=10.0):
    """Run a single problem in a subprocess so timeouts are enforced."""
    completion = extract_function(completion, entry_point)
    q = multiprocessing.Queue()
    p = multiprocessing.Process(
        target=_run_check,
        args=(prompt, completion, test, entry_point, time_limit_s, q),
    )
    p.start()
    p.join(timeout=time_limit_s)
    if p.is_alive():
        p.terminate()
        p.join()
        return False, "timeout"
    if q.empty():
        return False, "subprocess crashed without result"
    status, msg = q.get()
    return status == "pass", msg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sglang-url", required=True,
                    help="SGLang OpenAI-compatible URL, e.g. http://localhost:30000/v1")
    ap.add_argument("--model", default="Qwen/Qwen3.5-9B")
    ap.add_argument("--n-tasks", type=int, default=20)
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--out-jsonl", default="humaneval_pilot_sas.jsonl")
    ap.add_argument("--time-limit-s", type=float, default=10.0,
                    help="Per-problem test execution timeout")
    args = ap.parse_args()

    print(f"Loading HumanEval+ ({args.n_tasks} tasks)...", flush=True)
    tasks = load_humaneval_plus(args.n_tasks)
    print(f"Loaded {len(tasks)} tasks. Connecting to {args.sglang_url}...", flush=True)

    client = OpenAI(base_url=args.sglang_url, api_key="local")

    out_path = Path(args.out_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out = open(out_path, "w")

    n_pass = 0
    t0 = time.time()
    for i, task in enumerate(tasks):
        prompt = task["prompt"]
        entry_point = task["entry_point"]
        test = task["test"]
        task_id = task["task_id"]

        # SAS prompt: ask for the function implementation directly
        user_msg = (
            f"Complete the following Python function. Return ONLY the function "
            f"definition (with signature and body). No explanation, no test code, "
            f"no markdown formatting.\n\n```python\n{prompt}\n```"
        )

        gen_t0 = time.time()
        try:
            resp = client.chat.completions.create(
                model=args.model,
                messages=[{"role": "user", "content": user_msg}],
                temperature=args.temperature,
                max_tokens=args.max_tokens,
            )
            completion = resp.choices[0].message.content or ""
        except Exception as e:
            completion = ""
            print(f"  [{i+1}/{len(tasks)}] {task_id} GEN_ERROR: {e}", flush=True)
        gen_secs = time.time() - gen_t0

        # Strip markdown fences if model added them despite instructions
        m = re.search(r"```(?:python)?\s*\n(.*?)```", completion, re.DOTALL)
        if m:
            completion = m.group(1)

        passed, err = evaluate_one(
            prompt, completion, test, entry_point, args.time_limit_s)
        if passed:
            n_pass += 1
        print(f"  [{i+1}/{len(tasks)}] {task_id} {'PASS' if passed else 'FAIL'} "
              f"(gen {gen_secs:.1f}s){' err='+err if not passed else ''}", flush=True)

        out.write(json.dumps({
            "task_id": task_id,
            "entry_point": entry_point,
            "passed": passed,
            "error": err if not passed else "",
            "gen_secs": gen_secs,
            "completion": completion,
        }) + "\n")
        out.flush()

    out.close()
    elapsed = time.time() - t0
    rate = n_pass / len(tasks)
    print(f"\n=== Pilot summary ===", flush=True)
    print(f"  n = {len(tasks)}, pass@1 = {rate:.3f} ({n_pass}/{len(tasks)})", flush=True)
    print(f"  total wall = {elapsed:.0f}s", flush=True)
    if rate < 0.70:
        verdict = "BELOW SATURATION — proceed with HumanEval+ full ablation"
    elif rate <= 0.85:
        verdict = "USABLE — proceed with HumanEval+ but expect tighter CIs"
    else:
        verdict = "SATURATED — pivot to LiveCodeBench for second code benchmark"
    print(f"  verdict: {verdict}", flush=True)
    print(f"  output:  {out_path}", flush=True)


if __name__ == "__main__":
    main()
