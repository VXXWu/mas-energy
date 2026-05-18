"""Evaluate the trained topology classifier as a runtime orchestrator.

Loads the classifier from train_topology_classifier.py and wires it into
the agentic_latentmas pipeline. For each task:

  1. Run Phase 1 (N parallel agents).
  2. Compute features (agreement, tool-call stats, phase1_energy).
  3. Classifier predicts P(debate_helps).
  4. If p >= threshold: run Phase 2 debate.
  5. Else: skip to synthesis (saving debate energy).

Compares to:
  - Always-debate baseline (standard Decentralized)
  - Never-debate baseline (Independent + synthesis)
  - Simple agreement-threshold gate (our earlier heuristic)

Reports:
  - Accuracy and energy for each policy on the common task set
  - Orchestrator's decision correctness (relative to actual debate-helps labels)

Usage:
    python run_orchestrator_eval.py \
        --classifier orchestrator_classifier.json \
        --benchmark qampari \
        --n-tasks 30 \
        --model-name Qwen/Qwen3-8B \
        --data-dir /path/to/qampari_data \
        --out-path eval_orchestrator.jsonl
"""
import argparse
import json
import math
import os
import statistics
import sys
import time
from pathlib import Path

CODE_DIR = os.environ.get(
    "MAS_ENERGY_CODE",
    f"/atlas2/u/{os.environ.get('USER', 'vincewu8')}/mas_project/mas-energy/code",
)
sys.path.insert(0, CODE_DIR)


def load_classifier(path):
    with open(path) as f:
        m = json.load(f)
    return m


def predict_debate_helps(model_dict, features):
    """Apply the classifier to feature dict."""
    x_raw = [features[f] for f in model_dict["feature_names"]]
    means = model_dict["standardize_means"]
    stds = model_dict["standardize_stds"]
    x = [(x_raw[k] - means[k]) / stds[k] for k in range(len(x_raw))]
    z = sum(model_dict["weights"][k] * x[k] for k in range(len(x))) + model_dict["bias"]
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    e = math.exp(z)
    return e / (1.0 + e)


def compute_features(phase1_responses, phase1_tool_calls, phase1_energy_j):
    """Compute runtime features from Phase 1 output."""
    from latent_pilot.agentic_latentmas import _compute_semantic_agreement
    agreement = _compute_semantic_agreement(phase1_responses)
    calls_per_agent = [len(c) for c in phase1_tool_calls] if phase1_tool_calls else []
    mean_calls = statistics.mean(calls_per_agent) if calls_per_agent else 0.0

    def norm(c):
        args = c.get("args", {}) or {}
        q = args.get("query", "") if isinstance(args, dict) else ""
        return f"{c.get('name', '')}({q.strip().lower()})" if isinstance(q, str) else str(c)
    flat = [norm(c) for calls in (phase1_tool_calls or []) for c in calls]
    dup_rate = 1.0 - (len(set(flat)) / max(len(flat), 1)) if flat else 0.0

    lens = []
    for calls in (phase1_tool_calls or []):
        total = 0
        for call in calls:
            args = call.get("args", {})
            if isinstance(args, dict):
                for v in args.values():
                    if isinstance(v, str):
                        total += len(v)
        lens.append(total)
    length_var = statistics.variance(lens) if len(lens) > 1 else 0.0

    return {
        "agreement": agreement,
        "mean_calls": mean_calls,
        "dup_rate": dup_rate,
        "length_var": length_var,
        "phase1_energy": phase1_energy_j,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--classifier", type=str, required=True)
    ap.add_argument("--benchmark", type=str, required=True, choices=["qampari", "fanoutqa"])
    ap.add_argument("--n-tasks", type=int, default=30)
    ap.add_argument("--n-agents", type=int, default=3)
    ap.add_argument("--max-react-steps", type=int, default=10)
    ap.add_argument("--model-name", type=str, default="Qwen/Qwen3-8B")
    ap.add_argument("--data-dir", type=str, default=None)
    ap.add_argument("--threshold", type=float, default=0.5,
                    help="Predicted debate-helps probability above which to run Phase 2.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-path", type=str, required=True)
    args = ap.parse_args()

    model_dict = load_classifier(args.classifier)
    print(f"Loaded classifier with {len(model_dict['feature_names'])} features, "
          f"bias={model_dict['bias']:+.3f}")

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from latent_pilot.agentic_latentmas import (
        build_react_prompt, TOOL_INSTRUCTION, text_react_loop,
        _extract_tool_summary, _extract_tool_queries,
        _compute_semantic_agreement, _format_debate_prompt, _format_synthesis,
        DEBATE_SYNTHESIZER_PROMPT, generate_text, build_sas_prompt,
        load_benchmark,
    )
    from energy import EnergyMonitor

    print(f"Loading {args.model_name}...")
    tok = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, torch_dtype=torch.bfloat16, trust_remote_code=True,
    ).to("cuda")
    model.eval()

    class MW:
        tokenizer = tok
        model = model
    mw = MW()

    bench, evaluate_fn, question_key = load_benchmark(args.benchmark,
                                                       data_dir=args.data_dir)
    tasks = bench.load_tasks(n_tasks=args.n_tasks, seed=args.seed)
    print(f"Loaded {len(tasks)} {args.benchmark} tasks")

    em = EnergyMonitor()
    idle = em.measure_idle(duration=5)
    print(f"Idle power: {idle:.1f} W")

    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        ts = time.strftime("%Y%m%d_%H%M%S")
        out_path.rename(out_path.with_suffix(out_path.suffix + f".{ts}.bak"))

    agg = {"orch_acc": 0, "orch_e": 0, "ran_debate": 0, "n": 0,
           "always_acc": 0, "always_e": 0, "never_acc": 0, "never_e": 0}

    for i, task in enumerate(tasks):
        q = task[question_key]
        executor, cleanup = bench.make_executor(task)
        torch.manual_seed(args.seed * 10000 + i * 10)

        try:
            # ====== Policy 1: orchestrator (our learned classifier) ======
            em.start()
            # Phase 1: parallel text ReAct (shared across policies that need it)
            trajectories = []
            for agent_idx in range(args.n_agents):
                msgs = build_react_prompt(q, TOOL_INSTRUCTION)
                resp = text_react_loop(mw, msgs, executor,
                                       max_steps=args.max_react_steps, temperature=0.5)
                trajectories.append({
                    "final_response": resp,
                    "tool_summary": _extract_tool_summary(msgs),
                    "messages": msgs,
                })
            phase1_rec = em.stop(metadata={"stage": "phase1"})
            phase1_e = phase1_rec.get("gpu_dynamic_energy_joules", 0.0)

            # Compute features + classifier decision
            p1_calls = [_extract_tool_queries(t["messages"]) for t in trajectories]
            feats = compute_features(
                [t["final_response"] for t in trajectories],
                p1_calls, phase1_e,
            )
            p_helps = predict_debate_helps(model_dict, feats)
            should_debate = p_helps >= args.threshold

            # Phase 2: debate if classifier says yes
            em.start()
            if should_debate:
                new_traj = []
                for a in range(args.n_agents):
                    dmsg = _format_debate_prompt(trajectories, exclude_idx=a)
                    msgs = trajectories[a]["messages"] + [{"role": "user", "content": dmsg}]
                    resp = text_react_loop(mw, msgs, executor,
                                           max_steps=args.max_react_steps, temperature=0.5)
                    new_traj.append({
                        "final_response": resp,
                        "tool_summary": _extract_tool_summary(msgs),
                        "messages": msgs,
                    })
                orch_traj = new_traj
            else:
                orch_traj = trajectories
            # Phase 3: synthesis
            synth_msgs = [
                {"role": "system", "content": DEBATE_SYNTHESIZER_PROMPT},
                {"role": "user", "content": _format_synthesis(q, orch_traj)},
            ]
            orch_ans = generate_text(mw, synth_msgs, temperature=0.0)
            phase23_rec = em.stop(metadata={"stage": "phase23"})
            orch_e = phase1_e + phase23_rec.get("gpu_dynamic_energy_joules", 0.0)
            orch_acc, orch_f1, _ = evaluate_fn(task, orch_ans)

            # ====== Policy 2: always debate (for comparison) ======
            # Reuse Phase 1, run debate forcefully
            em.start()
            new_traj = []
            for a in range(args.n_agents):
                dmsg = _format_debate_prompt(trajectories, exclude_idx=a)
                msgs = trajectories[a]["messages"] + [{"role": "user", "content": dmsg}]
                resp = text_react_loop(mw, msgs, executor,
                                       max_steps=args.max_react_steps, temperature=0.5)
                new_traj.append({
                    "final_response": resp,
                    "tool_summary": _extract_tool_summary(msgs),
                    "messages": msgs,
                })
            synth_msgs = [
                {"role": "system", "content": DEBATE_SYNTHESIZER_PROMPT},
                {"role": "user", "content": _format_synthesis(q, new_traj)},
            ]
            always_ans = generate_text(mw, synth_msgs, temperature=0.0)
            always_rec = em.stop(metadata={"stage": "always_debate"})
            always_e = phase1_e + always_rec.get("gpu_dynamic_energy_joules", 0.0)
            always_acc, always_f1, _ = evaluate_fn(task, always_ans)

            # ====== Policy 3: never debate (synthesize from Phase 1 only) ======
            em.start()
            synth_msgs = [
                {"role": "system", "content": DEBATE_SYNTHESIZER_PROMPT},
                {"role": "user", "content": _format_synthesis(q, trajectories)},
            ]
            never_ans = generate_text(mw, synth_msgs, temperature=0.0)
            never_rec = em.stop(metadata={"stage": "never_debate"})
            never_e = phase1_e + never_rec.get("gpu_dynamic_energy_joules", 0.0)
            never_acc, never_f1, _ = evaluate_fn(task, never_ans)

            cleanup()
        except Exception as e:
            print(f"[{i}] error: {e}")
            import traceback; traceback.print_exc()
            cleanup()
            continue

        rec = {
            "i": i, "task_id": task.get("qid", task.get("id", i)),
            "features": feats,
            "p_debate_helps": p_helps,
            "should_debate": should_debate,
            "orch_loose_accuracy": orch_acc, "orch_f1": orch_f1, "orch_energy_j": orch_e,
            "always_loose_accuracy": always_acc, "always_f1": always_f1, "always_energy_j": always_e,
            "never_loose_accuracy": never_acc, "never_f1": never_f1, "never_energy_j": never_e,
            "phase1_energy_j": phase1_e,
        }
        with open(out_path, "a") as f:
            f.write(json.dumps(rec) + "\n")

        agg["orch_acc"] += orch_acc; agg["orch_e"] += orch_e
        agg["always_acc"] += always_acc; agg["always_e"] += always_e
        agg["never_acc"] += never_acc; agg["never_e"] += never_e
        agg["ran_debate"] += int(should_debate)
        agg["n"] += 1

        if (i + 1) % 5 == 0:
            n = agg["n"]
            print(f"[{i+1}/{len(tasks)}] "
                  f"orch={agg['orch_acc']/n:.3f}({agg['orch_e']/n:.0f}J) "
                  f"always={agg['always_acc']/n:.3f}({agg['always_e']/n:.0f}J) "
                  f"never={agg['never_acc']/n:.3f}({agg['never_e']/n:.0f}J) "
                  f"ran_debate={agg['ran_debate']}/{n}")

    n = agg["n"]
    if n == 0:
        print("No tasks completed.")
        return
    print(f"\n{'='*70}")
    print(f"  Policy           Accuracy       Energy      vs Always-Debate")
    print(f"  Orchestrator     {agg['orch_acc']/n:10.3f} {agg['orch_e']/n:10.0f}J  "
          f"{(agg['orch_e']-agg['always_e'])/max(agg['always_e'],1)*100:+.1f}%")
    print(f"  Always Debate    {agg['always_acc']/n:10.3f} {agg['always_e']/n:10.0f}J  baseline")
    print(f"  Never Debate     {agg['never_acc']/n:10.3f} {agg['never_e']/n:10.0f}J  "
          f"{(agg['never_e']-agg['always_e'])/max(agg['always_e'],1)*100:+.1f}%")
    print(f"  Orchestrator ran debate on {agg['ran_debate']}/{n} tasks "
          f"({agg['ran_debate']/n:.0%})")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
