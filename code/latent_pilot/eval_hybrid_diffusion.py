"""Pilot evaluation: TextMAS vs LatentMAS-W_e vs LatentMAS-Diffusion.

Compares three inter-agent communication mechanisms in a Decentralized M=3,
R=1 debate on QAMPARI (tool-use benchmark, list-answer QA via BM25 search):

  1. TEXT      — sender decodes text, receiver prefills text. Existing
                 agentic_latentmas.run_text_parallel_decentralized.
  2. LATENT_WE — sender does m latent CoT, KV cache concatenated to receiver.
                 Existing agentic_latentmas.run_latent_parallel_decentralized.
  3. LATENT_DB — sender does m latent CoT, multi-layer hidden states extracted
                 + diffusion bridge → k soft prompt embeddings injected at
                 receiver's input embedding layer (NEW; this script).

Tool calls + final answers stay decoded text in all conditions. Only the
inter-agent reasoning channel differs.

Metrics (per task, per condition):
  - accuracy: QAMPARI strict + loose F1
  - energy: GPU dynamic + total (J), via EnergyMonitor
  - tokens: prompt + completion
  - latency: wall seconds

Usage:
  python eval_hybrid_diffusion.py \
      --bridge-ckpt mas-energy/results/diffusion_bridge/run1/bridge_epoch1.pt \
      --qampari-data /atlas2/u/$USER/mas_project/data/qampari \
      --output mas-energy/results/diffusion_pilot/results.jsonl \
      --n-tasks 50 --m-latent 20 --max-react-steps 5 --K-sample 20
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
import traceback
from pathlib import Path

import torch

# Project paths — match agentic_latentmas.py conventions
_USER = os.environ.get("USER", "vincewu8")
REPO_DIR = os.environ.get("LATENTMAS_REPO",
    str(Path(f"/atlas2/u/{_USER}/mas_project/LatentMAS")))
CODE_DIR = os.environ.get("MAS_ENERGY_CODE",
    str(Path(f"/atlas2/u/{_USER}/mas_project/mas-energy/code")))
sys.path.insert(0, REPO_DIR)
sys.path.insert(0, CODE_DIR)
sys.path.insert(0, str(Path(__file__).parent.parent))

from energy import EnergyMonitor  # noqa: E402
from latent_pilot.diffusion_bridge import (  # noqa: E402
    BridgeConfig, DiffusionBridge,
    extract_source_layers,
)
from latent_pilot.latentmas_baseline import compute_alignment  # noqa: E402


# ────────────────────────────────────────────────────────────────────
# Diffusion-bridge inter-agent communication
# ────────────────────────────────────────────────────────────────────

@torch.no_grad()
def sender_text_to_source(model_wrapper, messages, m_positions, layer_indices,
                          max_prompt_tokens=4096):
    """Re-encode the sender's message trajectory (including its decoded
    response) through the frozen backbone, return multi-layer hidden states
    at the last m_positions token positions.

    This matches the training distribution of the bridge: train pairs were
    constructed by running the backbone over decoded peer_text and capturing
    hidden states. For Phase 2 (when training data includes latent-CoT-source
    pairs), this can be swapped for `sender_latent_thoughts` to skip the
    decode step entirely.

    Returns: source (1, m_positions, L, d).
    """
    tokenizer = model_wrapper.tokenizer
    model = model_wrapper.model
    dev = model.device

    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False,
        enable_thinking=False,
    )
    enc = tokenizer(prompt, return_tensors="pt", truncation=True,
                    max_length=max_prompt_tokens)
    input_ids = enc.input_ids.to(dev)

    out = model(input_ids=input_ids, output_hidden_states=True,
                use_cache=False, return_dict=True)
    source = extract_source_layers(out.hidden_states, layer_indices,
                                   positions=m_positions)
    if source.size(1) < m_positions:
        # Left-pad by repeating the first available position
        deficit = m_positions - source.size(1)
        pad_h = source[:, :1, :, :].expand(-1, deficit, -1, -1)
        source = torch.cat([pad_h, source], dim=1)
    return source


@torch.no_grad()
def task_representation(model_wrapper, question, max_tokens=512):
    """Pooled (last-position) hidden state of the question — the bridge's
    task condition vector."""
    tok = model_wrapper.tokenizer
    model = model_wrapper.model
    enc = tok(question, return_tensors="pt", truncation=True, max_length=max_tokens)
    ids = enc.input_ids.to(model.device)
    mask = enc.attention_mask.to(model.device)
    out = model(input_ids=ids, attention_mask=mask,
                output_hidden_states=True, use_cache=False, return_dict=True)
    last_pos = mask.sum(dim=1) - 1
    return out.hidden_states[-1][torch.arange(ids.size(0)), last_pos, :]


@torch.no_grad()
def receiver_with_soft_prompt(model_wrapper, messages, soft_prompt,
                              executor, max_steps, m_latent,
                              max_prompt_tokens=4096):
    """Receiver agent that consumes a (1, k, d) soft-prompt prefix instead of
    the sender's text. After the soft prompt, the receiver's own ReAct loop
    runs normally (latent CoT for thinking, text decode for tool calls/answer).

    Returns the decoded final response string.
    """
    from latent_pilot.latentmas_baseline import apply_alignment
    from agentic_latentmas import _get_we_alignment, detect_tool_call

    tokenizer = model_wrapper.tokenizer
    model = model_wrapper.model
    dev = model.device
    eos = tokenizer.eos_token_id
    MAX_RESP = 4096

    # Step 1: prefill the receiver's own messages (question + role prompts), but
    # WITHOUT the peer text. The bridge soft prompt replaces what would have
    # been the peer text in the receiver's context.
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False,
        enable_thinking=False,
    )
    enc = tokenizer(prompt, return_tensors="pt", truncation=True,
                    max_length=max_prompt_tokens)
    own_ids = enc.input_ids.to(dev)
    own_embeds = model.get_input_embeddings()(own_ids)              # (1, T_own, d)

    # Concatenate: [own context embeds] + [bridge soft prompt embeds]
    sp = soft_prompt.to(own_embeds.dtype).to(dev)
    full_embeds = torch.cat([own_embeds, sp], dim=1)                # (1, T_own+k, d)

    out = model(inputs_embeds=full_embeds, use_cache=True,
                output_hidden_states=True, return_dict=True)
    kv = out.past_key_values
    h_t = out.hidden_states[-1][:, -1:, :]

    W_e, target_norm = _get_we_alignment(model)

    for step in range(max_steps):
        # m latent CoT steps before each text decode
        for _ in range(m_latent):
            e_next = apply_alignment(h_t, W_e, target_norm)
            step_out = model(inputs_embeds=e_next, past_key_values=kv,
                             use_cache=True, output_hidden_states=True,
                             return_dict=True)
            kv = step_out.past_key_values
            h_t = step_out.hidden_states[-1][:, -1:, :]

        # Text decode (greedy for determinism in the pilot)
        logits = model.lm_head(h_t)[:, -1, :]
        next_token = logits.argmax(dim=-1, keepdim=True)
        generated = [next_token[0, 0].item()]
        for _ in range(MAX_RESP - 1):
            step_out = model(input_ids=next_token, past_key_values=kv,
                             use_cache=True, return_dict=True)
            kv = step_out.past_key_values
            logits = step_out.logits[:, -1, :]
            next_token = logits.argmax(dim=-1, keepdim=True)
            tid = next_token[0, 0].item()
            if tid == eos:
                break
            generated.append(tid)

        response = tokenizer.decode(generated, skip_special_tokens=True)
        tool = detect_tool_call(response)
        if tool is None:
            return response

        tool_name, tool_args = tool
        if isinstance(tool_args, str):
            try:
                tool_args = json.loads(tool_args)
            except json.JSONDecodeError:
                tool_args = {"query": tool_args}
        result = executor(tool_name, tool_args)

        # Append [response, tool_result] to KV via re-prefill of just those tokens
        followup = (
            f"\n<|im_start|>assistant\n{response}<|im_end|>"
            f"\n<|im_start|>user\nTool result:\n{result}\n\nContinue.<|im_end|>"
            f"\n<|im_start|>assistant\n"
        )
        f_ids = tokenizer(followup, return_tensors="pt",
                          add_special_tokens=False).input_ids.to(dev)
        out = model(input_ids=f_ids, past_key_values=kv,
                    use_cache=True, output_hidden_states=True, return_dict=True)
        kv = out.past_key_values
        h_t = out.hidden_states[-1][:, -1:, :]

    return ""


def run_diffusion_parallel_decentralized(
    model_wrapper, question, executor, bridge,
    n_agents=3, m_latent=20, max_react_steps=5, layer_indices=None,
    K_sample=20, m_source_positions=32,
):
    """Decentralized M-agent debate with diffusion-bridge inter-agent transfer.

    Round 0 (initial): each agent does ReAct (text decode) on the question.
                       Their decoded responses are CONVERTED via:
                       (a) re-encode response as the agent's "what I'd say" prompt,
                       (b) extract sender's m latent thoughts via latent CoT,
                       (c) bridge → k soft prompts.
    Round 1 (debate):  each agent receives soft prompts from peer agents +
                       its own role messages, does latent CoT + text decode.
    Final: synthesizer aggregates.

    Returns: dict with per-agent responses, final answer, call records.
    """
    from agentic_latentmas import (
        build_react_prompt, text_react_loop, _format_synthesis,
        DEBATE_AGENT_PROMPT, DEBATE_SYNTHESIZER_PROMPT, generate_text,
        TOOL_FORMAT_INSTRUCTION,
    )

    if layer_indices is None:
        layer_indices = list(range(1, model_wrapper.model.config.num_hidden_layers + 1))[-3:]

    # ---- Round 0: each agent does its own ReAct on the question ----
    initial_responses = []
    initial_messages = []
    for i in range(n_agents):
        msg = build_react_prompt(question)
        resp = text_react_loop(model_wrapper, msg, executor,
                               max_steps=max_react_steps, temperature=0.5)
        initial_responses.append(resp)
        initial_messages.append(msg)

    # ---- Compute the diffusion-bridge soft prompts FROM each agent ----
    # Re-encode each agent's response through the backbone, take last
    # m_source_positions hidden states at L layers as the bridge input.
    # This matches the bridge's training distribution (re-encoded text).
    task_vec = task_representation(model_wrapper, question)
    soft_prompts = []
    for i in range(n_agents):
        sender_msg = build_react_prompt(question) + [
            {"role": "assistant", "content": initial_responses[i]},
        ]
        source = sender_text_to_source(
            model_wrapper, sender_msg, m_source_positions, layer_indices,
        )
        sp = bridge.sample(source, task_vec, K=K_sample)         # (1, k, d)
        soft_prompts.append(sp)

    # ---- Round 1: each agent receives concatenated soft prompts from peers ----
    final_responses = []
    for i in range(n_agents):
        peer_sps = [soft_prompts[j] for j in range(n_agents) if j != i]
        # Concatenate peer soft prompts along sequence dimension
        combined_sp = torch.cat(peer_sps, dim=1)                  # (1, (M-1)*k, d)
        receiver_msg = [
            {"role": "system",
             "content": f"{DEBATE_AGENT_PROMPT}\n\n{TOOL_FORMAT_INSTRUCTION}"},
            {"role": "user", "content": question},
            {"role": "user",
             "content": "Peer agents' approaches and findings follow as latent context. "
                        "Use them to refine your reasoning, then provide your final answer."},
        ]
        resp = receiver_with_soft_prompt(
            model_wrapper, receiver_msg, combined_sp, executor,
            max_steps=max_react_steps, m_latent=m_latent,
        )
        final_responses.append(resp)

    # ---- Synthesis (text, same as text/latent baselines) ----
    synth_prompt = _format_synthesis(question, [
        {"final": fr} for fr in final_responses
    ])
    synth_msg = [
        {"role": "system", "content": DEBATE_SYNTHESIZER_PROMPT},
        {"role": "user", "content": synth_prompt},
    ]
    final = generate_text(model_wrapper, synth_msg, temperature=0.0)

    return {
        "initial_responses": initial_responses,
        "final_responses": final_responses,
        "synthesized_answer": final,
    }


# ────────────────────────────────────────────────────────────────────
# Bridge loading
# ────────────────────────────────────────────────────────────────────

def load_bridge(ckpt_path, model, device, dtype):
    """Load a trained DiffusionBridge from a training checkpoint."""
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg_dict = state["config"]
    cfg = BridgeConfig(**cfg_dict)
    bridge = DiffusionBridge(cfg).to(device).to(dtype)

    # Re-attach W_e from the loaded model (not stored in checkpoint)
    W_e, target_norm = compute_alignment(model)
    W_e = W_e.to(device, dtype=dtype)
    bridge.attach_w_e(W_e, target_norm)

    bridge.load_state_dict(state["bridge_state_dict"], strict=False)
    bridge.eval()
    layer_indices = state.get("layer_indices",
                              list(range(model.config.num_hidden_layers - cfg.n_source_layers + 1,
                                         model.config.num_hidden_layers + 1)))
    return bridge, cfg, layer_indices


# ────────────────────────────────────────────────────────────────────
# Main eval loop
# ────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-name", default="Qwen/Qwen3-8B")
    ap.add_argument("--bridge-ckpt", required=True, type=Path)
    ap.add_argument("--qampari-data", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--n-tasks", type=int, default=50)
    ap.add_argument("--n-agents", type=int, default=3)
    ap.add_argument("--m-latent", type=int, default=20)
    ap.add_argument("--max-react-steps", type=int, default=5)
    ap.add_argument("--K-sample", type=int, default=20)
    ap.add_argument("--m-source-positions", type=int, default=32,
                    help="Last N hidden positions per layer fed to bridge")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--conditions", nargs="+",
                    default=["text", "latent_we", "latent_diffusion"],
                    choices=["text", "latent_we", "latent_diffusion"])
    ap.add_argument("--dtype", default="bfloat16",
                    choices=["bfloat16", "float16"])
    args = ap.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    # ---- Energy ----
    monitor = EnergyMonitor()
    print("Measuring idle GPU power (10s)...")
    idle_w = monitor.measure_idle(duration=10)
    print(f"  P_idle = {idle_w:.2f} W")

    # ---- Backbone via LatentMAS ModelWrapper (matches agentic_latentmas.py) ----
    from models import ModelWrapper
    print(f"Loading backbone via ModelWrapper: {args.model_name}")
    model_wrapper = ModelWrapper(args.model_name, dtype=args.dtype)
    model = model_wrapper.model
    device = model.device
    dtype = next(model.parameters()).dtype

    # ---- Bridge ----
    print(f"Loading bridge from {args.bridge_ckpt}")
    bridge, bridge_cfg, layer_indices = load_bridge(
        args.bridge_ckpt, model, device, dtype,
    )
    print(f"  Bridge: k={bridge_cfg.k_soft_prompts}, "
          f"layers={layer_indices}, K_sample={args.K_sample}")

    # ---- Benchmark + executor ----
    from benchmarks_qampari import QampariBenchmark, evaluate_qampari
    bench = QampariBenchmark(data_dir=str(args.qampari_data))
    tasks = bench.load_tasks(n_tasks=args.n_tasks, seed=args.seed)
    print(f"Loaded {len(tasks)} QAMPARI tasks")
    QUESTION_KEY = "question"  # ALCE-style formatted version (matches main study)

    # ---- Session metadata header (per project rules) ----
    import pynvml
    info = pynvml.nvmlDeviceGetMemoryInfo(monitor.gpu_handle)
    session_meta = {
        "_session": True,
        "gpu_name": monitor.gpu_name,
        "idle_power_watts": idle_w,
        "vram_baseline_gb": info.used / 1e9,
        "model_name": args.model_name,
        "bridge_ckpt": str(args.bridge_ckpt),
        "bridge_config": bridge_cfg.to_dict(),
        "layer_indices": layer_indices,
        "args": vars(args) | {"output": str(args.output),
                              "qampari_data": str(args.qampari_data),
                              "bridge_ckpt": str(args.bridge_ckpt)},
        "n_tasks": len(tasks),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    with open(args.output, "w") as f:
        f.write(json.dumps(session_meta) + "\n")

    # ---- Conditions ----
    from agentic_latentmas import (
        run_text_parallel_decentralized,
        run_latent_parallel_decentralized,
        _CachingExecutor,
    )

    def run_one(condition, task):
        """Returns the per-task result record for this condition."""
        raw_executor, cleanup = bench.make_executor(task)
        executor = _CachingExecutor(raw_executor)
        question = task[QUESTION_KEY]

        monitor.start()
        try:
            if condition == "text":
                out = run_text_parallel_decentralized(
                    model_wrapper, question, executor,
                    n_agents=args.n_agents,
                    max_react_steps=args.max_react_steps,
                )
            elif condition == "latent_we":
                out = run_latent_parallel_decentralized(
                    model_wrapper, question, executor,
                    n_agents=args.n_agents,
                    m_latent=args.m_latent,
                    max_react_steps=args.max_react_steps,
                )
            elif condition == "latent_diffusion":
                out = run_diffusion_parallel_decentralized(
                    model_wrapper, question, executor, bridge,
                    n_agents=args.n_agents,
                    m_latent=args.m_latent,
                    max_react_steps=args.max_react_steps,
                    layer_indices=layer_indices,
                    K_sample=args.K_sample,
                    m_source_positions=args.m_source_positions,
                )
            else:
                raise ValueError(condition)
            err = None
        except Exception as e:
            out = {"synthesized_answer": "", "error": str(e),
                   "traceback": traceback.format_exc()}
            err = str(e)
        energy_rec = monitor.stop(metadata={"condition": condition,
                                            "task_id": task.get("id")})

        # Accuracy
        ans = out.get("synthesized_answer", "")
        eval_res = evaluate_qampari(task["answer_list"], ans) if ans else \
            {"recall_substr": 0.0, "f1": 0.0, "precision": 0.0}

        try:
            cleanup()
        except Exception:
            pass

        return {
            "condition": condition,
            "task_id": task.get("id"),
            "question": question[:200],
            "synthesized_answer": ans[:500],
            "f1": eval_res.get("f1", 0.0),
            "recall": eval_res.get("recall_substr", 0.0),
            "evaluation": eval_res,
            "energy": energy_rec,
            "error": err,
        }

    # ---- Main loop ----
    n_done = 0
    t0 = time.time()
    with open(args.output, "a") as out_f:
        for ti, task in enumerate(tasks):
            for cond in args.conditions:
                rec = run_one(cond, task)
                out_f.write(json.dumps(rec, default=str) + "\n")
                out_f.flush()
                e = rec["energy"]["gpu_dynamic_energy_joules"]
                print(f"  task {ti+1}/{len(tasks)} [{cond:>17}] "
                      f"f1={rec['f1']:.3f} energy={e:.1f}J "
                      f"wall={rec['energy']['wall_seconds']:.1f}s")
            n_done += 1
            elapsed = time.time() - t0
            eta = elapsed / n_done * (len(tasks) - n_done)
            print(f"  [progress] {n_done}/{len(tasks)} elapsed={elapsed/60:.1f}min "
                  f"eta={eta/60:.1f}min")

    monitor.shutdown()
    print(f"Done. Output → {args.output}")


if __name__ == "__main__":
    main()
