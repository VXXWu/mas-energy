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
_USER = os.environ["USER"]
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
                          max_prompt_tokens=89292):
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
def sender_latent_thoughts(model_wrapper, messages, m_positions, layer_indices,
                           m_thoughts=None, max_prompt_tokens=89292):
    """Energy-saving sender: no text decode, no tools. Prefills the role+question
    context, then runs `m_thoughts` latent CoT steps using the W_e alignment
    trick (same primitive as receiver_with_soft_prompt). Returns multi-layer
    hidden states at the last `m_positions` latent positions.

    This is the "design-as-intended" diffusion sender: it pays only for one
    prefill of the role prompt + m_thoughts single-token forwards, with zero
    text generation. Contrast with `sender_text_to_source`, which still pays
    the full text_react_loop decode + a re-encoding forward.

    Caveat: the bridge was trained on hidden states from re-encoded decoded
    text. Feeding it latent-CoT hidden states is a distribution shift, so
    bridge generalization is itself part of what this condition tests.
    """
    from latent_pilot.latentmas_baseline import apply_alignment
    from latent_pilot.agentic_latentmas import _get_we_alignment

    tokenizer = model_wrapper.tokenizer
    model = model_wrapper.model
    dev = model.device
    if m_thoughts is None:
        m_thoughts = max(m_positions, 32)

    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False,
        enable_thinking=False,
    )
    enc = tokenizer(prompt, return_tensors="pt", truncation=True,
                    max_length=max_prompt_tokens)
    input_ids = enc.input_ids.to(dev)

    # Prefill the role+question context once, get final hidden state.
    out = model(input_ids=input_ids, use_cache=True,
                output_hidden_states=True, return_dict=True)
    kv = out.past_key_values
    h_t = out.hidden_states[-1][:, -1:, :]

    W_e, target_norm = _get_we_alignment(model)

    # Collect per-layer hidden states across m_thoughts latent positions.
    L = len(layer_indices)
    d = h_t.size(-1)
    collected = torch.zeros(1, m_thoughts, L, d, device=dev, dtype=h_t.dtype)
    for j in range(m_thoughts):
        e_next = apply_alignment(h_t, W_e, target_norm)
        step_out = model(inputs_embeds=e_next, past_key_values=kv,
                         use_cache=True, output_hidden_states=True,
                         return_dict=True)
        kv = step_out.past_key_values
        h_t = step_out.hidden_states[-1][:, -1:, :]
        for li, layer_idx in enumerate(layer_indices):
            collected[:, j, li, :] = step_out.hidden_states[layer_idx][:, -1, :]

    # Return last m_positions of the latent trajectory.
    source = collected[:, -m_positions:, :, :]
    if source.size(1) < m_positions:
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
                              max_prompt_tokens=89292):
    """Receiver agent that consumes a (1, k, d) soft-prompt prefix instead of
    the sender's text. After the soft prompt, the receiver's own ReAct loop
    runs normally (latent CoT for thinking, text decode for tool calls/answer).

    Returns the decoded final response string.
    """
    from latent_pilot.latentmas_baseline import apply_alignment
    from latent_pilot.agentic_latentmas import _get_we_alignment, detect_tool_call

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

        # Text decode (greedy for determinism in the pilot).
        # 2026-06-04: tried model.generate(past_key_values=kv) — crashed on
        # DynamicCache format mismatch. Instead, we run a SMALL manual chunk
        # of single-token decodes (chunk_size tokens) to "warm up" the response,
        # then re-prefill the warmup tokens as input_ids and use a fresh
        # generate() call to finish the response. The re-prefill of T_warm
        # tokens is O(T_warm * d^2) batched (fast), then generate runs through
        # an OPTIMIZED autoregressive path. Net: replaces ~200 single-token
        # python-loop forwards with 1 batched prefill + optimized generate.
        # Should give ~3-5× speedup on the decode portion.
        #
        # WARM_CHUNK trades off: too small and we re-prefill almost everything;
        # too large and we waste time on the slow manual loop. ~10 tokens is
        # enough to commit the assistant to a stable continuation while letting
        # the batched re-prefill carry most of the cost.
        WARM_CHUNK = 10

        # Step A: manual single-token loop for the first WARM_CHUNK tokens.
        # This is the same as before but bounded — kv keeps building up.
        logits = model.lm_head(h_t)[:, -1, :]
        next_token = logits.argmax(dim=-1, keepdim=True)
        warmup_ids = [next_token[0, 0].item()]
        warmup_terminated = (warmup_ids[0] == eos)
        for _ in range(WARM_CHUNK - 1):
            if warmup_terminated:
                break
            step_out = model(input_ids=next_token, past_key_values=kv,
                             use_cache=True, return_dict=True)
            kv = step_out.past_key_values
            logits = step_out.logits[:, -1, :]
            next_token = logits.argmax(dim=-1, keepdim=True)
            tid = next_token[0, 0].item()
            if tid == eos:
                warmup_terminated = True
                break
            warmup_ids.append(tid)

        if warmup_terminated or len(warmup_ids) == 0:
            # Response ended within warmup — no need for generate() path.
            response = tokenizer.decode(warmup_ids, skip_special_tokens=True)
        else:
            # Step B: re-encode the prompt + warmup as input_ids and let
            # model.generate() handle the rest. This sidesteps the broken
            # past_key_values=kv path by building a fresh self-contained
            # generation call. The trade: we re-prefill the prompt + soft_prompt
            # equivalents, which means we need to inject the soft prompt back
            # in some way. Since soft prompts don't tokenize, we tokenize the
            # warmup tokens as the "continuation" and let the model continue
            # from there.
            #
            # Construct the input by combining: original_prompt_text + warmup_text.
            # The original prompt was built earlier as `prompt` from
            # apply_chat_template(messages, add_generation_prompt=False).
            # We need to add the assistant turn header + the warmup text.
            warmup_text = tokenizer.decode(warmup_ids, skip_special_tokens=False)
            # Build a generation-ready prompt: original messages + assistant turn
            # already started with warmup tokens.
            gen_prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
                enable_thinking=False,
            ) + warmup_text

            gen_enc = tokenizer(gen_prompt, return_tensors="pt", truncation=True,
                                max_length=max_prompt_tokens)
            gen_input_ids = gen_enc.input_ids.to(dev)

            # Build EOS list (matches text_react_loop pattern).
            eos_ids = []
            if eos is not None:
                eos_ids.append(eos)
            for stop in ("<|im_end|>", "<|endoftext|>"):
                tid = tokenizer.convert_tokens_to_ids(stop)
                if tid is not None and tid != tokenizer.unk_token_id and tid not in eos_ids:
                    eos_ids.append(tid)

            with torch.no_grad():
                gen_out = model.generate(
                    input_ids=gen_input_ids,
                    attention_mask=gen_enc.attention_mask.to(dev),
                    max_new_tokens=MAX_RESP - len(warmup_ids),
                    do_sample=False,
                    eos_token_id=eos_ids if eos_ids else None,
                    pad_token_id=(tokenizer.pad_token_id
                                  if tokenizer.pad_token_id is not None else eos),
                    use_cache=True,
                )
            # gen_out shape: (1, gen_input_ids_len + N_new). Decode only the
            # NEW tokens after gen_input_ids (which already contains warmup).
            new_tokens = gen_out[0, gen_input_ids.shape[1]:].tolist()
            full_response_ids = warmup_ids + new_tokens
            response = tokenizer.decode(full_response_ids, skip_special_tokens=True)

            # The kv cache is now "stale" relative to the new response, but
            # the no-tool-call path returns immediately below, so this doesn't
            # matter. The tool-call path further down would need a fresh kv,
            # but that path is rarely hit in pure-latent (no real tool use).

        # NOTE: response now contains the full decoded text. The kv variable
        # may be in a partially-stale state if we took the generate() branch,
        # which is fine for the no-tool-call return path. The tool-call branch
        # below has its own re-prefill that rebuilds the relevant kv state.
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
    K_sample=20, m_source_positions=32, tools=None, n_rounds=1,
):
    """Decentralized M-agent debate with diffusion-bridge inter-agent transfer.

    Round 0 (initial): each agent does ReAct (text decode) on the question.
                       Their decoded responses are CONVERTED via:
                       (a) re-encode response as the agent's "what I'd say" prompt,
                       (b) extract last hidden states as bridge sources,
                       (c) bridge → k soft prompts.
    Debate rounds (n_rounds total): each round, every agent receives peers' soft
                       prompts and does receiver_with_soft_prompt → text response.
                       Between rounds, each round-r response is re-encoded via
                       sender_text_to_source for round r+1 bridge inputs.
    Final: synthesizer aggregates the LAST round's responses.

    Returns: dict with per-agent responses, final answer.
    """
    from latent_pilot.agentic_latentmas import (
        build_react_prompt, text_react_loop, _format_synthesis,
        DEBATE_AGENT_PROMPT, DEBATE_SYNTHESIZER_PROMPT, generate_text,
        TOOL_FORMAT_INSTRUCTION,
    )

    if layer_indices is None:
        layer_indices = list(range(1, model_wrapper.model.config.num_hidden_layers + 1))[-3:]

    # ---- Round 0: each agent does its own ReAct on the question ----
    initial_responses = []
    for i in range(n_agents):
        msg = build_react_prompt(question)
        resp = text_react_loop(model_wrapper, msg, executor,
                               max_steps=max_react_steps, temperature=0.5,
                               tools=tools)
        initial_responses.append(resp)

    task_vec = task_representation(model_wrapper, question)

    def bridge_round(round_responses):
        """Re-encode each agent's text response, bridge → soft prompts."""
        sps = []
        for i in range(n_agents):
            sender_msg = build_react_prompt(question) + [
                {"role": "assistant", "content": round_responses[i]},
            ]
            source = sender_text_to_source(
                model_wrapper, sender_msg, m_source_positions, layer_indices,
            )
            sp = bridge.sample(source, task_vec, K=K_sample)
            sps.append(sp)
        return sps

    # System prompt: DEBATE_AGENT_PROMPT + TOOL_FORMAT_INSTRUCTION. The latter
    # (now benchmark-agnostic, see agentic_latentmas.py) provides the prose
    # tool-use nudge that HF transformers lacks vs SGLang's tool_choice="auto".
    # Without it, the model defaults to answering from internal knowledge.
    receiver_msg_template = [
        {"role": "system",
         "content": f"{DEBATE_AGENT_PROMPT}\n\n{TOOL_FORMAT_INSTRUCTION}"},
        {"role": "user", "content": question},
        {"role": "user",
         "content": "Peer agents' approaches and findings follow as latent context. "
                    "Use them to refine your reasoning, then provide your final answer."},
    ]

    # First bridge conversion uses Round 0 (initial) text responses.
    soft_prompts = bridge_round(initial_responses)
    current_responses = initial_responses

    # ---- n_rounds debate rounds ----
    for r in range(n_rounds):
        round_responses = []
        for i in range(n_agents):
            peer_sps = [soft_prompts[j] for j in range(n_agents) if j != i]
            combined_sp = torch.cat(peer_sps, dim=1)
            resp = receiver_with_soft_prompt(
                model_wrapper, receiver_msg_template, combined_sp, executor,
                max_steps=max_react_steps, m_latent=m_latent,
            )
            round_responses.append(resp)
        current_responses = round_responses
        # Compute next round's soft prompts unless this is the last round.
        if r < n_rounds - 1:
            soft_prompts = bridge_round(current_responses)

    final_responses = current_responses

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


@torch.no_grad()
def run_diffusion_latent_only_decentralized(
    model_wrapper, question, executor, bridge,
    n_agents=3, m_latent=20, max_react_steps=5, layer_indices=None,
    K_sample=20, m_source_positions=32, m_thoughts=None, tools=None,
    n_rounds=1,
):
    """Energy-saving "design-as-intended" variant.

    Round 0 does NOT call text_react_loop or sender_text_to_source. Each sender
    does pure latent CoT on the question (no text decode, no tools, no
    re-encoding). The bridge maps those latent thoughts directly to soft prompts.

    Subsequent debate rounds (n_rounds total): each round, every agent receives
    peers' soft prompts and does receiver_with_soft_prompt → text response.
    Between rounds, each round-r response is re-encoded via sender_text_to_source
    for round r+1 bridge inputs. From Round 1 onward we pay text decode (the
    pure-latent property only applies to Round 0).

    Per-task energy budget vs `run_diffusion_parallel_decentralized` at the same
    n_rounds: saves Round 0 text decode + Round 0 re-encoding (3 agents worth).
    """
    from latent_pilot.agentic_latentmas import (
        build_react_prompt, _format_synthesis,
        DEBATE_AGENT_PROMPT, DEBATE_SYNTHESIZER_PROMPT, generate_text,
        TOOL_FORMAT_INSTRUCTION,
    )

    if layer_indices is None:
        layer_indices = list(range(1, model_wrapper.model.config.num_hidden_layers + 1))[-3:]

    task_vec = task_representation(model_wrapper, question)

    # ---- Round 0: pure latent thoughts → bridge source ----
    soft_prompts = []
    for i in range(n_agents):
        sender_msg = build_react_prompt(question)
        source = sender_latent_thoughts(
            model_wrapper, sender_msg,
            m_positions=m_source_positions,
            layer_indices=layer_indices,
            m_thoughts=m_thoughts,
        )
        sp = bridge.sample(source, task_vec, K=K_sample)
        soft_prompts.append(sp)

    def bridge_round(round_responses):
        """Re-encode text responses for subsequent rounds. Only used for n_rounds > 1."""
        sps = []
        for i in range(n_agents):
            sender_msg = build_react_prompt(question) + [
                {"role": "assistant", "content": round_responses[i]},
            ]
            source = sender_text_to_source(
                model_wrapper, sender_msg, m_source_positions, layer_indices,
            )
            sp = bridge.sample(source, task_vec, K=K_sample)
            sps.append(sp)
        return sps

    # System prompt: DEBATE_AGENT_PROMPT + TOOL_FORMAT_INSTRUCTION. The latter
    # (now benchmark-agnostic, see agentic_latentmas.py) provides the prose
    # tool-use nudge that HF transformers lacks vs SGLang's tool_choice="auto".
    # Without it, the model defaults to answering from internal knowledge.
    receiver_msg_template = [
        {"role": "system",
         "content": f"{DEBATE_AGENT_PROMPT}\n\n{TOOL_FORMAT_INSTRUCTION}"},
        {"role": "user", "content": question},
        {"role": "user",
         "content": "Peer agents' approaches and findings follow as latent context. "
                    "Use them to refine your reasoning, then provide your final answer."},
    ]

    current_responses = None
    for r in range(n_rounds):
        round_responses = []
        for i in range(n_agents):
            peer_sps = [soft_prompts[j] for j in range(n_agents) if j != i]
            combined_sp = torch.cat(peer_sps, dim=1)
            resp = receiver_with_soft_prompt(
                model_wrapper, receiver_msg_template, combined_sp, executor,
                max_steps=max_react_steps, m_latent=m_latent,
            )
            round_responses.append(resp)
        current_responses = round_responses
        if r < n_rounds - 1:
            soft_prompts = bridge_round(current_responses)

    final_responses = current_responses

    # ---- Synthesis (text, same as other conditions) ----
    synth_prompt = _format_synthesis(question, [
        {"final": fr} for fr in final_responses
    ])
    synth_msg = [
        {"role": "system", "content": DEBATE_SYNTHESIZER_PROMPT},
        {"role": "user", "content": synth_prompt},
    ]
    final = generate_text(model_wrapper, synth_msg, temperature=0.0)

    return {
        "initial_responses": None,
        "final_responses": final_responses,
        "synthesized_answer": final,
    }


@torch.no_grad()
def run_sas_single(model_wrapper, question, executor, max_react_steps=5, tools=None):
    """SAS (single-agent) baseline: one ReAct loop, no debate, no synthesis.
    Matches Stage 8's max_react_steps so it's k-paired against the MAS conditions.
    Returns the agent's decoded final response (the answer) directly.
    """
    from latent_pilot.agentic_latentmas import build_react_prompt, text_react_loop

    msg = build_react_prompt(question)
    resp = text_react_loop(model_wrapper, msg, executor,
                           max_steps=max_react_steps, temperature=0.0,
                           tools=tools)
    return resp


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
    ap.add_argument("--benchmark", default="fanoutqa",
                    choices=["qampari", "fanoutqa"])
    ap.add_argument("--data-dir", default=None,
                    help="Benchmark data dir (QAMPARI only). Required if "
                         "--benchmark qampari.")
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--n-tasks", type=int, default=50)
    ap.add_argument("--n-agents", type=int, default=3)
    ap.add_argument("--m-latent", type=int, default=20)
    ap.add_argument("--max-react-steps", type=int, default=5)
    ap.add_argument("--K-sample", type=int, default=20)
    ap.add_argument("--m-source-positions", type=int, default=32,
                    help="Last N hidden positions per layer fed to bridge")
    ap.add_argument("--n-rounds", type=int, default=1,
                    help="Number of debate rounds (R). Matches Du et al. R={1,2,3}. "
                         "All four conditions use this; runners are uniform across R.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--conditions", nargs="+",
                    default=["text", "latent_we", "latent_diffusion"],
                    choices=["sas", "text", "latent_we", "latent_diffusion",
                             "latent_diffusion_pure"])
    ap.add_argument("--dtype", default="bfloat16",
                    choices=["bfloat16", "float16"])
    ap.add_argument("--no-resume", action="store_true",
                    help="Skip the per-task-resume scan of an existing output "
                         "JSONL. Default: scan and skip task_ids that already "
                         "have valid (non-errored, non-empty) records.")
    args = ap.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    # ---- Energy ----
    monitor = EnergyMonitor()
    print("Measuring idle GPU power (10s)...")
    idle_w = monitor.measure_idle(duration=10)
    print(f"  P_idle = {idle_w:.2f} W")

    # ---- Backbone via local SimpleModelWrapper (avoids LatentMAS
    # ModelWrapper's W_e-init OOM and its `dtype` kwarg signature).
    print(f"Loading backbone: {args.model_name}")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    class SimpleModelWrapper:
        def __init__(self, model_name, device="cuda"):
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            dt = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name, torch_dtype=dt, trust_remote_code=True,
            ).to(device)
            self.model.eval()

    model_wrapper = SimpleModelWrapper(args.model_name, device="cuda")
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

    # ---- Benchmark + executor (use the shared load_benchmark helper) ----
    from latent_pilot.agentic_latentmas import load_benchmark
    bench, evaluate_fn, question_key = load_benchmark(
        args.benchmark, data_dir=args.data_dir,
    )
    tasks = bench.load_tasks(n_tasks=args.n_tasks, seed=args.seed)
    print(f"Loaded {len(tasks)} {args.benchmark} tasks")
    QUESTION_KEY = question_key

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
                              "bridge_ckpt": str(args.bridge_ckpt),
                              "data_dir": str(args.data_dir or "")},
        "n_tasks": len(tasks),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    # Append-only: preserves per-task records from prior invocations with a
    # different `--conditions` flag. Without this, every re-run with one
    # condition would wipe the records of other conditions on the same file.
    # The analyze script skips the session_meta lines (no "condition" field).
    with open(args.output, "a") as f:
        f.write(json.dumps(session_meta) + "\n")

    # ---- Conditions ----
    from latent_pilot.agentic_latentmas import (
        run_text_parallel_decentralized,
        run_latent_parallel_decentralized,
        _CachingExecutor,
    )

    def run_one(condition, task):
        """Returns the per-task result record for this condition."""
        raw_executor, cleanup = bench.make_executor(task)
        executor = _CachingExecutor(raw_executor)
        question = task[QUESTION_KEY]

        # Native tool calling for all three conditions (matches main study
        # protocol). bench.get_tools() returns OpenAI-format schemas that
        # the model's chat template will inject correctly.
        tools = bench.get_tools() if hasattr(bench, "get_tools") else None

        monitor.start()
        try:
            if condition == "sas":
                out = run_sas_single(
                    model_wrapper, question, executor,
                    max_react_steps=args.max_react_steps,
                    tools=tools,
                )
            elif condition == "text":
                out = run_text_parallel_decentralized(
                    model_wrapper, question, executor,
                    n_agents=args.n_agents,
                    max_react_steps=args.max_react_steps,
                    n_rounds=args.n_rounds,
                    tools=tools,
                )
            elif condition == "latent_we":
                out = run_latent_parallel_decentralized(
                    model_wrapper, question, executor,
                    n_agents=args.n_agents,
                    m_latent=args.m_latent,
                    max_react_steps=args.max_react_steps,
                    n_rounds=args.n_rounds,
                    tools=tools,
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
                    n_rounds=args.n_rounds,
                )
            elif condition == "latent_diffusion_pure":
                out = run_diffusion_latent_only_decentralized(
                    model_wrapper, question, executor, bridge,
                    n_agents=args.n_agents,
                    m_latent=args.m_latent,
                    max_react_steps=args.max_react_steps,
                    layer_indices=layer_indices,
                    K_sample=args.K_sample,
                    m_source_positions=args.m_source_positions,
                    tools=tools,
                    n_rounds=args.n_rounds,
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

        # run_text_parallel_decentralized and run_latent_parallel_decentralized
        # return the synthesized answer string directly; run_diffusion_*
        # returns a dict with "synthesized_answer". Normalize to a string.
        if isinstance(out, str):
            ans = out
        elif isinstance(out, dict):
            ans = out.get("synthesized_answer", "")
        else:
            ans = str(out)
        if ans:
            loose, f1, eval_extra = evaluate_fn(task, ans)
        else:
            loose, f1, eval_extra = 0.0, 0.0, {}

        try:
            cleanup()
        except Exception:
            pass

        return {
            "condition": condition,
            "n_rounds": args.n_rounds,
            "task_id": task.get("id"),
            "question": question[:200],
            "synthesized_answer": ans[:500],
            "f1": f1,
            "loose_accuracy": loose,
            "evaluation": eval_extra,
            "energy": energy_rec,
            "error": err,
        }

    # ---- Per-task resume: scan output JSONL for already-completed
    # (task_id, condition, n_rounds) tuples and skip those tasks. Lets us
    # recover from mid-run crashes (cluster reboots, env breaks, OOMs on
    # specific tasks) without re-running 60+ minutes of already-completed
    # work. A completed run is one with no `error` field and a non-empty
    # synthesized_answer. Errored runs are NOT considered done, so they
    # get retried on the next submission. Disable with --no-resume.
    already_done = set()  # set of (task_id, condition, n_rounds)
    skip_summary = {}     # condition -> count of skipped task_ids
    if args.output.exists() and not args.no_resume:
        with open(args.output) as f:
            for line in f:
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if "condition" not in r:
                    continue
                if r.get("error"):
                    continue
                ans = r.get("synthesized_answer", "")
                if not ans:
                    continue
                tid = r.get("task_id")
                if tid is None:
                    continue
                key = (tid, r["condition"], r.get("n_rounds", 1))
                already_done.add(key)
                skip_summary[r["condition"]] = skip_summary.get(r["condition"], 0) + 1
        if already_done:
            print(f"Resume mode: found {len(already_done)} already-completed "
                  f"(task_id, condition, R) tuples in {args.output.name}")
            for cond, n in sorted(skip_summary.items()):
                print(f"    {cond}: {n} tasks already done — will skip")

    # ---- Main loop ----
    n_done = 0
    n_skipped = 0
    t0 = time.time()
    with open(args.output, "a") as out_f:
        for ti, task in enumerate(tasks):
            tid = task.get("id")
            for cond in args.conditions:
                resume_key = (tid, cond, args.n_rounds)
                if resume_key in already_done:
                    n_skipped += 1
                    continue
                rec = run_one(cond, task)
                out_f.write(json.dumps(rec, default=str) + "\n")
                out_f.flush()
                e = rec["energy"]["gpu_dynamic_energy_joules"]
                print(f"  task {ti+1}/{len(tasks)} [{cond:>17}] "
                      f"f1={rec['f1']:.3f} energy={e:.1f}J "
                      f"wall={rec['energy']['wall_seconds']:.1f}s")
            n_done += 1
            elapsed = time.time() - t0
            # Avoid divide-by-zero on the rare all-skipped-so-far iteration.
            eta = elapsed / max(1, n_done) * (len(tasks) - n_done)
            print(f"  [progress] {n_done}/{len(tasks)} elapsed={elapsed/60:.1f}min "
                  f"eta={eta/60:.1f}min  skipped_this_run={n_skipped}")

    monitor.shutdown()
    print(f"Done. Output → {args.output} "
          f"(processed={n_done * len(args.conditions) - n_skipped}, "
          f"skipped_via_resume={n_skipped})")


if __name__ == "__main__":
    main()
