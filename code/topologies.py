"""5 MAS topologies (Kim et al. 2025) with agentic tool calling.

Each topology function takes:
    client, model, task_question, tools, execute_tool, energy_monitor,
    extra_body=None, **kwargs
and returns a dict with: answer, call_records, total_usage.

All LLM calls are serial for clean NVML energy attribution.
Tool calling is handled by react_loop() in llm.py.

Early stopping: multi-round topologies (Centralized, Decentralized, Hybrid)
stop when all agents converge (respond in 1 step with no tool calls in a round),
rather than running for a fixed number of rounds. The n_rounds parameter acts
as a maximum cap, not a fixed count.
"""

import os
from llm import chat, react_loop
from prompts import (
    SAS_PROMPT, INDEPENDENT_AGENT_PROMPT, INDEPENDENT_AGENT_PROMPT_MINIMAL,
    WORKER_PROMPT,
    ORCHESTRATOR_PROMPT, DEBATE_AGENT_PROMPT, DEBATE_AGENT_PROMPT_MINIMAL,
    SYNTHESIZER_PROMPT, DEBATE_SYNTHESIZER_PROMPT,
    HYBRID_WORKER_PROMPT,
    format_decompose_prompt, format_synthesis_prompt,
    format_review_prompt, format_review_prompt_routed, parse_routed_feedback,
    format_centralized_synthesis_prompt,
    format_centralized_code_synthesis_prompt,
    format_debate_prompt, format_debate_synthesis,
    format_peer_debate_prompt,
    parse_subtasks, _summarize_trajectory,
)
from config import (
    N_AGENTS, MAX_REACT_STEPS,
    CENTRALIZED_ROUNDS, CENTRALIZED_WORKER_STEPS,
    DECENTRALIZED_ROUNDS, DECENTRALIZED_DEBATE_STEPS,
    HYBRID_ROUNDS, HYBRID_WORKER_STEPS, N_PEER_ROUNDS,
    SAS_TEMP, INDEPENDENT_TEMP, DEBATE_TEMP, PEER_TEMP, ORCHESTRATOR_TEMP,
    BASE_SEED,
)


def _aggregate_usage(*usage_dicts):
    """Sum token usage across multiple dicts."""
    total = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    for u in usage_dicts:
        for k in total:
            total[k] += u.get(k, 0)
    return total


def _parse_hetero_k(n_agents):
    """C2: per-agent k list from MAS_HETERO_K env, or None if unset/mismatched."""
    raw = os.environ.get("MAS_HETERO_K", "").strip()
    if not raw:
        return None
    ks = [int(x) for x in raw.split(",") if x.strip()]
    if len(ks) != n_agents:
        raise ValueError(
            f"MAS_HETERO_K has {len(ks)} values but N_AGENTS={n_agents}")
    return ks


def _agents_converged(results):
    """True if all agents finished in 1 step (no tool calls -- just a text response).

    This means every agent already has its answer and additional rounds
    would only re-process the growing context for no benefit.
    """
    return all(r.get("steps", 0) == 1 for r in results)


# ─────────────────────────────────────────────────────────
# Parallel-agent phase execution (2026-07-22, self-batching experiment)
# ─────────────────────────────────────────────────────────
# MAS_PARALLEL_AGENTS=1 runs each topology's within-phase per-agent work
# concurrently (agents in a phase are data-independent: debate/feedback
# context always comes from the PREVIOUS round). This measures the
# self-batching deployment regime. Per-call NVML attribution is invalid
# under concurrency, so parallel mode refuses any monitor that reports
# real per-call energy — the caller must meter at the task level
# (see run_selfbatch_experiment.py) and pass a parallel-safe monitor.

MAS_PARALLEL_AGENTS = os.environ.get("MAS_PARALLEL_AGENTS", "0") == "1"


class _AgentCallMonitor:
    """Per-thread call monitor: wall time + metadata only, energy zeroed
    (same convention as run_latency_experiment.ThreadLocalMonitor)."""

    parallel_safe = True

    def __init__(self):
        import time as _time
        self._time = _time
        self.call_log = []
        self._t0 = None

    def start(self):
        self._t0 = self._time.monotonic()

    def stop(self, metadata=None):
        t_end = self._time.monotonic()
        t0 = self._t0 or t_end
        record = {
            "gpu_energy_joules": 0.0, "gpu_dynamic_energy_joules": 0.0,
            "gpu_idle_energy_joules": 0.0, "cpu_energy_joules": 0.0,
            "ram_energy_joules": 0.0, "total_energy_joules": 0.0,
            "wall_seconds": t_end - t0, "avg_gpu_power_watts": 0.0,
            "emissions_kg_co2": 0.0,
            "per_call_energy": "not_attributable_concurrent",
            # monotonic call interval — lets the runner reconstruct peak
            # achieved concurrency per task (real self-batch vs pool-limited
            # staggering), the key diagnostic for high-N cells.
            "t_start": t0, "t_end": t_end,
        }
        if metadata:
            record.update(metadata)
        self.call_log.append(record)
        return record


def _run_agent_phase(agent_fns, energy_monitor):
    """Execute one phase's per-agent closures fn(monitor) -> result.

    Serial by default (canonical protocol, per-call metering intact).
    MAS_PARALLEL_AGENTS=1: all closures run concurrently on per-thread
    zeroed monitors whose call records merge back in agent order.
    """
    if not MAS_PARALLEL_AGENTS:
        return [fn(energy_monitor) for fn in agent_fns]
    if not getattr(energy_monitor, "parallel_safe", False):
        raise RuntimeError(
            "MAS_PARALLEL_AGENTS=1 requires a parallel-safe (zeroed per-call) "
            "monitor with task-level aggregate metering — see "
            "run_selfbatch_experiment.py. Refusing to corrupt per-call energy.")
    import concurrent.futures
    subs = [_AgentCallMonitor() for _ in agent_fns]
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(agent_fns)) as ex:
        futures = [ex.submit(fn, m) for fn, m in zip(agent_fns, subs)]
        results = [f.result() for f in futures]
    for m in subs:
        energy_monitor.call_log.extend(m.call_log)
    return results


# ─────────────────────────────────────────────────────────
# Topology 1: SAS (Single-Agent System)
# ─────────────────────────────────────────────────────────

def run_sas(client, model, task_question, tools, execute_tool,
            energy_monitor, extra_body=None, **kwargs):
    """Single agent runs the full ReAct loop. LLM calls: S steps."""
    max_steps = kwargs.get("max_react_steps", MAX_REACT_STEPS)
    messages = [
        {"role": "system", "content": SAS_PROMPT},
        {"role": "user", "content": task_question},
    ]
    result = react_loop(
        client=client, model=model, messages=messages,
        tools=tools, execute_tool=execute_tool,
        energy_monitor=energy_monitor,
        max_steps=max_steps,
        temperature=SAS_TEMP, seed=BASE_SEED,
        agent_id="sas_agent",
        extra_body=extra_body,
    )
    return {
        "answer": result["final_response"] or "",
        "call_records": result["call_records"],
        "total_usage": result["total_usage"],
        "steps": result["steps"],
    }


# ─────────────────────────────────────────────────────────
# Topology 2: Independent (M parallel agents + synthesis)
# ─────────────────────────────────────────────────────────

def run_independent(client, model, task_question, tools, execute_tool,
                    energy_monitor, extra_body=None, **kwargs):
    """M agents independently solve the task; text-only synthesis.

    LLM calls: M * S_avg + 1 (synthesis).
    Per Kim et al., C = empty set (no communication). synthesis_only policy:
    aggregator concatenates without cross-validation.

    concat_synthesis=True (kwargs): match Kim et al.'s pure-concatenation
    Independent (no LLM synthesis call). Answer = "=== agent_i ===\n{answer}"
    per agent, joined with blank lines. Saves one LLM call per task.
    """
    max_steps = kwargs.get("max_react_steps", MAX_REACT_STEPS)
    # Default: LLM synthesis (handles grader-on-text benchmarks robustly).
    # concat_synthesis=True (Kim et al. _synthesize_only) is benchmark-dependent —
    # better for multi-component list answers (FanOutQA, QAMPARI) where union
    # recall helps, worse for single-answer-expected benchmarks (BrowseComp,
    # MASLegalBench) where the grader must extract one answer from M concatenated.
    # Kept as an opt-in variant rather than default since LLM synth is safer
    # across our benchmark mix.
    concat_synthesis = kwargs.get("concat_synthesis", False)
    # C2 (hetero-k, 2026-07-23): MAS_HETERO_K="k0,k1,..." gives agent i its own
    # depth k[i] instead of the uniform max_steps. Length must equal N_AGENTS;
    # unset => byte-identical to the uniform path.
    hetero_k = kwargs.get("hetero_k") or _parse_hetero_k(N_AGENTS)
    all_records = []

    def _agent(i):
        agent_steps = hetero_k[i] if hetero_k else max_steps

        def run(monitor):
            messages = [
                {"role": "system", "content": INDEPENDENT_AGENT_PROMPT},
                {"role": "user", "content": task_question},
            ]
            return react_loop(
                client=client, model=model, messages=messages,
                tools=tools, execute_tool=execute_tool,
                energy_monitor=monitor,
                max_steps=agent_steps,
                temperature=INDEPENDENT_TEMP,
                seed=BASE_SEED + i,
                agent_id=f"independent_{i}",
                extra_body=extra_body,
            )
        return run

    agent_results = _run_agent_phase(
        [_agent(i) for i in range(N_AGENTS)], energy_monitor)
    for result in agent_results:
        all_records.extend(result["call_records"])

    if concat_synthesis:
        # Kim et al. independent: no synthesis LLM call. Pure concatenation
        # of each agent's final answer, in deterministic agent order.
        parts = []
        for i, r in enumerate(agent_results):
            ans = r.get("final_response") or ""
            if ans:
                parts.append(f"=== independent_{i} ===\n{ans}")
        concat_text = "\n\n".join(parts)
        total_usage = _aggregate_usage(*[r["total_usage"] for r in agent_results])
        return {
            "answer": concat_text,
            "call_records": all_records,
            "total_usage": total_usage,
            "agent_results": agent_results,
            "hetero_k": hetero_k,
        }

    # Text-only synthesis (no tool access, per Kim et al. synthesis_only)
    synthesis_input = format_synthesis_prompt(task_question, agent_results)
    energy_monitor.start()
    synth_text, synth_usage = chat(
        client, model,
        messages=[
            {"role": "system", "content": SYNTHESIZER_PROMPT},
            {"role": "user", "content": synthesis_input},
        ],
        temperature=ORCHESTRATOR_TEMP, seed=BASE_SEED,
        extra_body=extra_body,
    )
    synth_record = energy_monitor.stop(metadata={
        "agent_id": "synthesizer",
        "call_type": "synthesis",
        **synth_usage,
    })
    all_records.append(synth_record)

    total_usage = _aggregate_usage(
        *[r["total_usage"] for r in agent_results], synth_usage
    )
    return {
        "answer": synth_text or "",
        "call_records": all_records,
        "total_usage": total_usage,
        "agent_results": agent_results,
        "hetero_k": hetero_k,
    }


# ─────────────────────────────────────────────────────────
# C1: adaptive confidence-gated escalation (2026-07-23)
# ─────────────────────────────────────────────────────────
# A cheap Independent gate runs first; a strong config always also runs, and
# BOTH answers/energies are logged. The escalation THRESHOLD is swept offline
# (analysis/orchestration_C1_adaptive_frontier.py), so one cell yields the whole
# adaptive frontier. The per-cell run cost here is gate + escalate (the cost of
# GENERATING the frontier data), NOT the deployed adaptive cost.
# Scope: answer-string-scored benchmarks only (FanOutQA, BrowseComp+). WorkBench
# (call-set) and SWE (shared worktree) have no per-tier answer to score.

def _norm_tokens(text):
    import re
    t = re.sub(r"[^\w\s]", " ", str(text).lower())
    return set(t.split())


def _mean_pairwise_jaccard(answers):
    toks = [_norm_tokens(a) for a in answers if str(a).strip()]
    if len(toks) < 2:
        return 1.0 if toks else 0.0
    sims, n = 0.0, 0
    for i in range(len(toks)):
        for j in range(i + 1, len(toks)):
            u = toks[i] | toks[j]
            sims += (len(toks[i] & toks[j]) / len(u)) if u else 1.0
            n += 1
    return sims / n if n else 1.0


def _gate_signals(agent_results, cheap_k):
    """Panel of FREE confidence signals from the cheap gate trace (no extra LLM
    call). The offline frontier sweep chooses whichever best predicts when
    escalation helps, and benchmarks all against the oracle. Higher = more
    confident (less need to escalate) for every signal.

      jaccard        mean pairwise token-Jaccard of cheap answers (agreement)
      exact_agree    fraction of agents on the plurality normalized answer
      answered_frac  fraction of agents that produced a non-empty answer
      early_frac     1 - mean(realized_steps / cheap_k): utilization-complement
                     (agents that exhausted their budget were still working ->
                     low confidence). This is the Fig A1 signal.
    """
    answers = [(ar.get("final_response") or "") for ar in agent_results]
    norm = [" ".join(sorted(_norm_tokens(a))) for a in answers if str(a).strip()]
    if norm:
        from collections import Counter
        top = Counter(norm).most_common(1)[0][1]
        exact_agree = top / len(answers)
    else:
        exact_agree = 0.0
    answered_frac = sum(1 for a in answers if str(a).strip()) / max(len(answers), 1)
    steps = [ar.get("steps", cheap_k) for ar in agent_results]
    early_frac = 1.0 - (sum(min(s, cheap_k) for s in steps) / (cheap_k * max(len(steps), 1)))
    return {
        "jaccard": _mean_pairwise_jaccard(answers),
        "exact_agree": exact_agree,
        "answered_frac": answered_frac,
        "early_frac": early_frac,
    }


def _widen_escalate(client, model, task_question, tools, execute_tool,
                    energy_monitor, cheap_agent_results, esc_n, esc_k,
                    cheap_n, extra_body=None):
    """MONOTONE-SAFE escalation: keep the cheap gate's agents as voting members,
    add esc_n fresh agents at depth esc_k, and re-synthesize over the UNION.

    Motivated by the C1 mechanism analysis (handoff 2026-07-24): replacement-
    escalation (SAS/Independent that discards the cheap answer) is a per-task
    COIN FLIP -- it breaks ~as many cheap-correct tasks as it rescues -- which is
    why no gating signal beat the static hull. Widening instead of replacing
    (a) never discards cheap's correct answers (the cheap agents still vote), and
    (b) reuses the already-spent cheap computation, so the marginal cost is only
    the esc_n new agents + one synthesis call, NOT a fresh N-agent ensemble.
    The escalate ENERGY logged here is that marginal cost by construction.
    """
    new_results = []

    def _agent(j):
        def run(monitor):
            messages = [
                {"role": "system", "content": INDEPENDENT_AGENT_PROMPT},
                {"role": "user", "content": task_question},
            ]
            return react_loop(
                client=client, model=model, messages=messages,
                tools=tools, execute_tool=execute_tool,
                energy_monitor=monitor,
                max_steps=esc_k,
                temperature=INDEPENDENT_TEMP,
                # seeds offset past the cheap agents -> genuinely new samples
                seed=BASE_SEED + cheap_n + j,
                agent_id=f"widen_{j}",
                extra_body=extra_body,
            )
        return run

    new_results = _run_agent_phase([_agent(j) for j in range(esc_n)], energy_monitor)
    new_records = []
    for r in new_results:
        new_records.extend(r["call_records"])

    # Synthesize over cheap agents (reused) + new agents (union vote).
    pool = list(cheap_agent_results) + list(new_results)
    synthesis_input = format_synthesis_prompt(task_question, pool)
    energy_monitor.start()
    synth_text, synth_usage = chat(
        client, model,
        messages=[
            {"role": "system", "content": SYNTHESIZER_PROMPT},
            {"role": "user", "content": synthesis_input},
        ],
        temperature=ORCHESTRATOR_TEMP, seed=BASE_SEED,
        extra_body=extra_body,
    )
    synth_record = energy_monitor.stop(metadata={
        "agent_id": "widen_synthesizer",
        "call_type": "synthesis",
        **synth_usage,
    })
    new_records.append(synth_record)
    total_usage = _aggregate_usage(
        *[r["total_usage"] for r in new_results], synth_usage)
    return {
        "answer": synth_text or "",
        "call_records": new_records,   # marginal cost only (cheap agents excluded)
        "total_usage": total_usage,
    }


def run_adaptive(client, model, task_question, tools, execute_tool,
                 energy_monitor, extra_body=None, **kwargs):
    """Cheap Independent gate + always-run strong escalate; log both tiers.

    Env: MAS_ADAPT_CHEAP_N (default 2), MAS_ADAPT_CHEAP_K (default 5),
         MAS_ADAPT_ESC_TOPO (default sas; also "independent" or "widen"),
         MAS_ADAPT_ESC_K (default 30), MAS_ADAPT_ESC_N (default 1).

    esc_topo="widen" (2026-07-24): monotone-safe escalation that reuses the cheap
    agents as voting members and only adds esc_n new agents (see _widen_escalate).
    The escalate tier's logged energy is then the true MARGINAL cost of widening.
    """
    cheap_n = int(os.environ.get("MAS_ADAPT_CHEAP_N", "2"))
    cheap_k = int(os.environ.get("MAS_ADAPT_CHEAP_K", "5"))
    esc_topo = os.environ.get("MAS_ADAPT_ESC_TOPO", "sas")
    esc_k = int(os.environ.get("MAS_ADAPT_ESC_K", "30"))
    esc_n = int(os.environ.get("MAS_ADAPT_ESC_N", "1"))

    global N_AGENTS

    # Tier 0: cheap Independent gate (its own answer + per-agent answers).
    saved_n = N_AGENTS
    N_AGENTS = cheap_n
    try:
        gate = run_independent(
            client, model, task_question, tools, execute_tool,
            energy_monitor, extra_body=extra_body, max_react_steps=cheap_k)
    finally:
        N_AGENTS = saved_n
    for r in gate["call_records"]:
        r["tier"] = "cheap"
    cheap_agent_answers = [(ar.get("final_response") or "")
                           for ar in gate.get("agent_results", [])]
    cheap_energy = sum(r.get("gpu_dynamic_energy_joules", 0)
                       for r in gate["call_records"])

    # Tier 1: strong escalate config (always run; offline threshold decides use).
    if esc_topo == "widen":
        esc = _widen_escalate(
            client, model, task_question, tools, execute_tool, energy_monitor,
            gate.get("agent_results", []), esc_n, esc_k, cheap_n,
            extra_body=extra_body)
    elif esc_topo == "sas":
        esc = run_sas(client, model, task_question, tools, execute_tool,
                      energy_monitor, extra_body=extra_body, max_react_steps=esc_k)
    else:
        N_AGENTS = esc_n
        try:
            esc = run_independent(
                client, model, task_question, tools, execute_tool,
                energy_monitor, extra_body=extra_body, max_react_steps=esc_k)
        finally:
            N_AGENTS = saved_n
    for r in esc["call_records"]:
        r["tier"] = "escalate"
    esc_energy = sum(r.get("gpu_dynamic_energy_joules", 0)
                     for r in esc["call_records"])

    all_records = gate["call_records"] + esc["call_records"]
    total_usage = _aggregate_usage(gate["total_usage"], esc["total_usage"])
    return {
        "answer": esc["answer"],                # default scoring scores escalate
        "call_records": all_records,
        "total_usage": total_usage,
        "adaptive": {
            "cheap_answer": gate["answer"],
            "cheap_agent_answers": cheap_agent_answers,
            "escalate_answer": esc["answer"],
            "gate_signals": _gate_signals(gate.get("agent_results", []), cheap_k),
            "cheap_energy_joules": cheap_energy,
            "escalate_energy_joules": esc_energy,
            "config": {"cheap_n": cheap_n, "cheap_k": cheap_k,
                       "esc_topo": esc_topo, "esc_k": esc_k, "esc_n": esc_n},
        },
    }


def run_independent_share(client, model, task_question, tools, execute_tool,
                          energy_monitor, extra_body=None, **kwargs):
    """Independent agents + one text-only refinement round (no tools, no full
    ReAct) where each agent sees peers' final answers and revises terse-only,
    then synthesize.

    Tests the hypothesis (motivated by Phase A's answer_only ≈ 0 ΔAcc result)
    that Decent's accuracy benefit over Independent comes from agents reading
    peer answers, not from receiver-side additional tool calls. If true, this
    matches Decent's accuracy at much lower cost: skips R*M*ReAct in favor of
    M one-shot terse chats.

    LLM calls: M init + M refine + 1 synth = 2M+1 (vs Decent's M + R*M + 1).
    For M=3, R=2: 7 calls (this) vs 10 calls (Decent), with init+synth shared.
    Crucial: refine step uses no tools, so it's pure short text decode + small
    prefill, very cheap relative to a full ReAct iteration.
    """
    max_steps = kwargs.get("max_react_steps", MAX_REACT_STEPS)
    minimal_output = kwargs.get("minimal_output", False)
    agent_results = []
    all_records = []
    all_usages = []

    sys_prompt = INDEPENDENT_AGENT_PROMPT_MINIMAL if minimal_output else INDEPENDENT_AGENT_PROMPT

    # Phase 1: Independent init (same as run_independent)
    for i in range(N_AGENTS):
        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": task_question},
        ]
        result = react_loop(
            client=client, model=model, messages=messages,
            tools=tools, execute_tool=execute_tool,
            energy_monitor=energy_monitor,
            max_steps=max_steps,
            temperature=INDEPENDENT_TEMP,
            seed=BASE_SEED + i,
            agent_id=f"share_init_{i}",
            extra_body=extra_body,
        )
        agent_results.append(result)
        all_records.extend(result["call_records"])
        all_usages.append(result["total_usage"])

    # Phase 2: Text-only refinement — each agent sees peers' final answers and
    # produces a terse revised answer. No ReAct loop, no tools, just one chat.
    refined_responses = []
    for i in range(N_AGENTS):
        peer_answers = []
        for j, res in enumerate(agent_results):
            if j == i: continue
            peer_answers.append(f"[Agent {j}] Final answer: {res.get('final_response') or 'No response'}")
        peer_text = "\n".join(peer_answers)
        refine_user = (
            f"You answered this question. Other agents' answers:\n\n{peer_text}\n\n"
            "Considering their answers, output ONLY your final answer. "
            "No explanation. No commentary. Just the answer."
        )
        # Single chat (no tools, no react). Encode the agent's prior turn so it
        # remembers its own context, then add the refine_user message.
        own_history = agent_results[i]["messages"]
        messages = own_history + [{"role": "user", "content": refine_user}]
        energy_monitor.start()
        refine_text, refine_usage = chat(
            client, model, messages=messages,
            temperature=DEBATE_TEMP, seed=BASE_SEED + i + N_AGENTS,
            extra_body=extra_body,
        )
        refine_record = energy_monitor.stop(metadata={
            "agent_id": f"share_refine_{i}",
            "call_type": "share_refine",
            **refine_usage,
        })
        all_records.append(refine_record)
        all_usages.append(refine_usage)
        refined_responses.append(refine_text or agent_results[i].get("final_response", ""))

    # Phase 3: Synthesis from refined responses
    synth_input_lines = [f"Task: {task_question}\n\nAgent responses after sharing:\n"]
    for i, resp in enumerate(refined_responses):
        synth_input_lines.append(f"[Agent {i}] Answer: {resp}\n")
    synth_input_lines.append("\nSynthesize the best final answer.")
    synthesis_input = "\n".join(synth_input_lines)

    energy_monitor.start()
    synth_text, synth_usage = chat(
        client, model,
        messages=[
            {"role": "system", "content": SYNTHESIZER_PROMPT},
            {"role": "user", "content": synthesis_input},
        ],
        temperature=ORCHESTRATOR_TEMP, seed=BASE_SEED,
        extra_body=extra_body,
    )
    synth_record = energy_monitor.stop(metadata={
        "agent_id": "synthesizer",
        "call_type": "share_synthesis",
        **synth_usage,
    })
    all_records.append(synth_record)
    all_usages.append(synth_usage)

    return {
        "answer": synth_text or "",
        "call_records": all_records,
        "total_usage": _aggregate_usage(*all_usages),
        "agent_results": agent_results,
        "refined_responses": refined_responses,
    }


# ─────────────────────────────────────────────────────────
# Topology 3: Centralized (orchestrator + M workers × R rounds)
# ─────────────────────────────────────────────────────────

def run_centralized(client, model, task_question, tools, execute_tool,
                    energy_monitor, extra_body=None, **kwargs):
    """Orchestrator decomposes task; workers execute with tools; orchestrator
    reviews and synthesizes.

    Kim et al.: "3 sub-agents with 1 orchestrator across maximum 5
    orchestration rounds, with 3 iterations per agent per round."

    LLM calls: 1 (decompose) + R * M * S_worker + (R-1) reviews + 1 (synthesis).

    Per Kim et al.: C = {(a_orch, ai) : ∀i} — workers communicate ONLY
    through the orchestrator, not with each other.
    """
    worker_steps = kwargs.get("max_react_steps", CENTRALIZED_WORKER_STEPS)
    n_rounds = kwargs.get("n_rounds", CENTRALIZED_ROUNDS)
    # Default protocol now matches Kim et al. mas_lead_agent.py:349-439:
    # M separate orchestrator LLM calls per round (one per worker), each
    # with team_context (peer last-round answers). This is the only
    # scientifically justifiable default — the prior single-call broadcast
    # design suffered from cross-worker context pollution + worker-identity
    # collisions + orchestrator pre-commitment leakage.
    # Legacy escape hatches preserved for reproducing pre-fix data only:
    #   broadcast_feedback=True: single LLM call broadcast (OLD default — broken)
    #   single_call_routed=True: single LLM call with [Worker i]: blocks parsed
    broadcast_feedback = kwargs.get("broadcast_feedback", False)
    single_call_routed = kwargs.get("single_call_routed", False)
    if broadcast_feedback:
        routed_feedback = False
        include_team_context = False
        per_worker_review_calls = False
    elif single_call_routed:
        routed_feedback = True
        include_team_context = False
        per_worker_review_calls = False
    else:
        # Justified default: full Kim et al. alignment
        routed_feedback = True
        include_team_context = True
        per_worker_review_calls = True
    all_records = []
    all_usages = []

    # Phase 1: Orchestrator decomposes (uses raw question without format instructions)
    decompose_question = kwargs.get("raw_question", task_question)
    energy_monitor.start()
    decompose_text, decompose_usage = chat(
        client, model,
        messages=[
            {"role": "system", "content": ORCHESTRATOR_PROMPT},
            {"role": "user", "content": format_decompose_prompt(
                decompose_question, tools, n_workers=N_AGENTS
            )},
        ],
        temperature=ORCHESTRATOR_TEMP, seed=BASE_SEED,
        extra_body=extra_body,
    )
    decompose_record = energy_monitor.stop(metadata={
        "agent_id": "orchestrator",
        "call_type": "decompose",
        **decompose_usage,
    })
    all_records.append(decompose_record)
    all_usages.append(decompose_usage)

    subtasks = parse_subtasks(decompose_text, N_AGENTS, task_question)

    # Phase 2: Up to R rounds of worker execution + orchestrator review.
    # Workers have PERSISTENT memory across rounds -- they accumulate tool
    # results and feedback, building on prior work each round.
    # Early stop: if all workers respond in 1 step (no tool calls), they've
    # converged and further rounds would just re-process context for no benefit.
    worker_results = [None] * N_AGENTS
    worker_histories = [None] * N_AGENTS
    feedback = None
    rounds_used = 0

    for round_idx in range(n_rounds):
        rounds_used = round_idx + 1

        def _worker(i, round_idx=round_idx, feedback=feedback):
          def run(monitor):
            if round_idx == 0:
                # Explicit worker-identity assertion. Disambiguates against
                # off-by-one collisions: the orchestrator's decomposition may
                # 1-index, or feedback may address "Worker N" without the
                # worker knowing which N they are. The "You are Worker {i}"
                # marker pins the worker's self-identity to the runtime index.
                worker_messages = [
                    {"role": "system", "content": WORKER_PROMPT},
                    {"role": "user",
                     "content": (
                         f"You are Worker {i} (0-indexed) on a team of {N_AGENTS} workers.\n\n"
                         f"Full task context:\n{task_question}\n\n"
                         f"Your assigned subtask: {subtasks[i]}"
                     )},
                ]
            else:
                # Continue from prior history with feedback appended.
                # feedback is either a string (broadcast) or list[str] (routed).
                #
                # Worker sees ONLY the orchestrator's coordination text — not a
                # separate verbatim peer-answer block. This matches Kim et al.
                # mas_lead_agent.py exactly: the coordination LLM receives
                # team_context as input and chooses what to relay to the worker
                # via its generated text. Per-round context pollution is
                # therefore bounded by the orchestrator's coord output length
                # (~hundreds of tokens), not by Σ peer_answer_lengths
                # (which can be thousands of tokens of raw peer text per round
                # accumulating across R rounds).
                # The `include_team_context` flag now controls whether the
                # ORCHESTRATOR sees peer findings when generating its coord text
                # (default: True), not whether the worker sees a verbatim block.
                worker_messages = list(worker_histories[i])
                if feedback is not None:
                    if isinstance(feedback, list):
                        worker_feedback = feedback[i]
                    else:
                        worker_feedback = feedback
                    worker_messages.append(
                        {"role": "user",
                         "content": (
                             f"Feedback from coordinator: {worker_feedback}\n\n"
                             f"Continue working on your subtask based on this feedback."
                         )}
                    )

            return react_loop(
                client=client, model=model, messages=worker_messages,
                tools=tools, execute_tool=execute_tool,
                energy_monitor=monitor,
                max_steps=worker_steps,
                temperature=DEBATE_TEMP, seed=BASE_SEED + i,
                agent_id=f"worker_{i}_r{round_idx}",
                extra_body=extra_body,
            )
          return run

        round_worker_results = _run_agent_phase(
            [_worker(i) for i in range(N_AGENTS)], energy_monitor)
        for i, result in enumerate(round_worker_results):
            worker_results[i] = result
            worker_histories[i] = result["messages"]
            all_records.extend(result["call_records"])
            all_usages.append(result["total_usage"])

        # Early stop disabled by default (Kim-aligned). Matches Decent default.
        # Re-enable via MAS_DISABLE_EARLY_STOP=0 only for legacy reproduction.
        # The old heuristic also broke trivially at k=1 (every round has zero
        # tool calls structurally), so disabling by default is robust.
        disable_early_stop_cent = os.environ.get("MAS_DISABLE_EARLY_STOP", "1") != "0"
        if round_idx > 0 and _agents_converged(worker_results) and not disable_early_stop_cent:
            break

        # Orchestrator review (except last round)
        if round_idx < n_rounds - 1:
            if per_worker_review_calls:
                # Kim et al.-strict: one orchestrator LLM call per worker per
                # round (M calls). Each call sees this worker's findings +
                # team_context (all OTHER workers' findings).
                def _review(i, round_idx=round_idx):
                    def run(monitor):
                        own_finding = (worker_results[i] or {}).get("final_response") or "(no response)"
                        peer_blocks = []
                        for j, wr in enumerate(worker_results):
                            if j == i or wr is None:
                                continue
                            peer_ans = (wr.get("final_response") or "").strip()
                            if peer_ans:
                                peer_blocks.append(f"[Worker {j}] {peer_ans}")
                        peer_team_context = "\n\n".join(peer_blocks) or "(no peer responses yet)"
                        per_worker_prompt = (
                            f"Original task: {task_question}\n\n"
                            f"You are coordinating Worker {i} (0-indexed), whose "
                            f"assigned subtask is:\n{subtasks[i]}\n\n"
                            f"Worker {i}'s most recent finding:\n{own_finding}\n\n"
                            f"Team context (other workers' last-round findings):\n"
                            f"{peer_team_context}\n\n"
                            f"CONSTRAINTS:\n"
                            f"- Provide specific feedback for Worker {i} for their next "
                            f"round of work. Focus only on Worker {i} — do not address "
                            f"other workers (they will not see this message).\n"
                            f"- DO NOT produce a final synthesized answer. The synthesizer "
                            f"will produce the final answer separately.\n"
                            f"- DO NOT include phrases like 'Final Answer:', 'Final Answer "
                            f"Formulation:', or any draft of the final answer.\n"
                            f"- Provide ONLY concrete revision guidance for Worker {i}."
                        )
                        monitor.start()
                        per_text, per_usage = chat(
                            client, model,
                            messages=[
                                {"role": "system", "content": ORCHESTRATOR_PROMPT},
                                {"role": "user", "content": per_worker_prompt},
                            ],
                            temperature=ORCHESTRATOR_TEMP, seed=BASE_SEED + i,
                            extra_body=extra_body,
                        )
                        per_record = monitor.stop(metadata={
                            "agent_id": "orchestrator",
                            "call_type": f"per_worker_review_w{i}_r{round_idx}",
                            **per_usage,
                        })
                        return per_text, per_usage, per_record
                    return run

                review_out = _run_agent_phase(
                    [_review(i) for i in range(N_AGENTS)], energy_monitor)
                feedback = []
                for per_text, per_usage, per_record in review_out:
                    all_records.append(per_record)
                    all_usages.append(per_usage)
                    feedback.append(per_text or "")
            else:
                if routed_feedback:
                    review_user_prompt = format_review_prompt_routed(
                        task_question, subtasks, worker_results
                    )
                else:
                    review_user_prompt = format_review_prompt(
                        task_question, subtasks, worker_results
                    )
                energy_monitor.start()
                review_text, review_usage = chat(
                    client, model,
                    messages=[
                        {"role": "system", "content": ORCHESTRATOR_PROMPT},
                        {"role": "user", "content": review_user_prompt},
                    ],
                    temperature=ORCHESTRATOR_TEMP, seed=BASE_SEED,
                    extra_body=extra_body,
                )
                review_record = energy_monitor.stop(metadata={
                    "agent_id": "orchestrator",
                    "call_type": f"review_r{round_idx}{'_routed' if routed_feedback else ''}",
                    **review_usage,
                })
                all_records.append(review_record)
                all_usages.append(review_usage)
                if routed_feedback:
                    feedback = parse_routed_feedback(review_text or "", N_AGENTS)
                else:
                    feedback = review_text

    # Phase 3: Orchestrator synthesis
    # For artifact-producing benchmarks (SWE), the final answer is a code patch,
    # not text: a text synthesis is discarded downstream (evaluate() reads the
    # workspace). synthesis_react=True (set by the runner for SWE isolated mode)
    # makes the orchestrator SYNTHESIZE by writing the fix itself in its own
    # clean worktree, integrating the workers' candidate patches — the faithful
    # orchestrator-produces-the-answer step, not best-of-N selection.
    synthesis_react = kwargs.get("synthesis_react", False)
    if synthesis_react:
        worker_artifacts_hook = kwargs.get("worker_artifacts_hook")
        try:
            worker_patches = list(worker_artifacts_hook()) if worker_artifacts_hook else []
        except Exception:
            worker_patches = []
        # Seed the lead worktree with the team's combined candidate so the
        # orchestrator refines it instead of usually leaving it empty (which
        # silently falls back to shared_union). Combined = the single shared-repo
        # diff, or the longest candidate in isolated mode.
        seed_lead_hook = kwargs.get("seed_lead_hook")
        if seed_lead_hook and worker_patches:
            seed = (worker_patches[0] if len(worker_patches) == 1
                    else max(worker_patches, key=lambda p: len((p or "").splitlines())))
            try:
                seed_lead_hook(seed)
            except Exception:
                pass
        synth_result = react_loop(
            client=client, model=model,
            messages=[
                {"role": "system", "content": WORKER_PROMPT},
                {"role": "user", "content": format_centralized_code_synthesis_prompt(
                    task_question, subtasks, worker_results, worker_patches
                )},
            ],
            tools=tools, execute_tool=execute_tool,
            energy_monitor=energy_monitor,
            max_steps=worker_steps,
            temperature=ORCHESTRATOR_TEMP, seed=BASE_SEED,
            agent_id="synthesizer",
            extra_body=extra_body,
        )
        all_records.extend(synth_result["call_records"])
        all_usages.append(synth_result["total_usage"])
        synth_text = synth_result["final_response"] or ""
    else:
        energy_monitor.start()
        synth_text, synth_usage = chat(
            client, model,
            messages=[
                {"role": "system", "content": ORCHESTRATOR_PROMPT},
                {"role": "user", "content": format_centralized_synthesis_prompt(
                    task_question, subtasks, worker_results
                )},
            ],
            temperature=ORCHESTRATOR_TEMP, seed=BASE_SEED,
            extra_body=extra_body,
        )
        synth_record = energy_monitor.stop(metadata={
            "agent_id": "orchestrator",
            "call_type": "synthesis",
            **synth_usage,
        })
        all_records.append(synth_record)
        all_usages.append(synth_usage)

    return {
        "answer": synth_text,
        "call_records": all_records,
        "total_usage": _aggregate_usage(*all_usages),
        "subtasks": subtasks,
        "worker_results": worker_results,
        "rounds_used": rounds_used,
    }


# ─────────────────────────────────────────────────────────
# Topology 4: Decentralized (M agents × R debate rounds)
# ─────────────────────────────────────────────────────────

def run_decentralized(client, model, task_question, tools, execute_tool,
                      energy_monitor, extra_body=None, **kwargs):
    """M agents each run ReAct, then debate for R rounds by seeing
    each other's final answers and revising.

    Peers exchange final answers only, NOT tool output: the default
    comm_mode is "answer_only" (see below). comm_mode="full" is the
    ablation that also shares tool-call summaries.

    Kim et al.: "3 agents through 3 debate rounds with 3 iterations
    per round." The initial independent phase counts as round 1 (no
    prior trajectories to debate), then DECENTRALIZED_ROUNDS debate
    rounds follow. Total = 1 + DECENTRALIZED_ROUNDS = 3 rounds.

    LLM calls: M*k (initial) + R*M*k (debate) + 1 (synthesis).
    """
    debate_steps = kwargs.get("max_react_steps", DECENTRALIZED_DEBATE_STEPS)
    n_debate_rounds = kwargs.get("n_rounds", DECENTRALIZED_ROUNDS)
    # comm_mode default = "answer_only" (peer final answers only, NO tool-call
    # summaries). Matches Kim et al. multiagent_decentralized.py:107-124 AND
    # provides cross-topology consistency: Cent's team_context also shares only
    # peer final_response, so both topologies share the same channel content
    # (one signal: each peer's final answer at the communication checkpoint).
    # Set comm_mode="full" only for ablations that explicitly test the value
    # of also sharing tool-call summaries.
    comm_mode = kwargs.get("comm_mode", "answer_only")
    terse_speaker = kwargs.get("terse_speaker", False)
    minimal_output = kwargs.get("minimal_output", False)
    # Default: LLM synthesis (DEBATE_SYNTHESIZER_PROMPT). Matches Indep choice:
    # safer for grader-on-text benchmarks (BrowseComp, MASLegalBench) than vote,
    # which requires agents to converge on exact-match strings to work well.
    # consensus_vote=True is Kim et al.'s _consensus_vote; opt-in for benchmarks
    # where short factoid agreement makes vote viable.
    consensus_vote = kwargs.get("consensus_vote", False)
    sys_prompt = DEBATE_AGENT_PROMPT_MINIMAL if minimal_output else DEBATE_AGENT_PROMPT
    # tool_partition: None | "round_robin" | "block" — when set, each agent
    # gets a disjoint subset of tools. Tests whether the channel-muting result
    # holds when agents have GENUINELY disjoint capabilities (not just
    # overlapping access to a shared tool environment).
    tool_partition = kwargs.get("tool_partition", None)
    all_records = []
    all_usages = []

    # Deployment-faithful categorical partition for WorkBench (3 specialist roles
    # mapping to common production patterns: scheduler, communicator, analytics).
    # Matches tool names by domain prefix (e.g., 'calendar.search_events').
    WORKBENCH_CATEGORIES = {
        0: ('calendar.', 'project_management.'),                  # Schedule/workflow specialist
        1: ('email.', 'company_directory.'),                      # Communications specialist
        2: ('customer_relationship_manager.', 'analytics.'),      # CRM/analytics specialist
    }

    def _agent_tools(agent_idx):
        """Return the tools available to agent_idx given the partition mode."""
        if tool_partition is None or len(tools) <= N_AGENTS:
            return tools
        if tool_partition == "round_robin":
            return [tools[j] for j in range(len(tools)) if j % N_AGENTS == agent_idx]
        if tool_partition == "block":
            n = len(tools)
            return tools[agent_idx * n // N_AGENTS : (agent_idx + 1) * n // N_AGENTS]
        if tool_partition == "categorical_workbench":
            prefixes = WORKBENCH_CATEGORIES.get(agent_idx, ())
            return [t for t in tools
                    if any(t.get('function', {}).get('name', '').startswith(p)
                           for p in prefixes)]
        return tools

    agent_histories = [None] * N_AGENTS
    agent_trajectories = [None] * N_AGENTS

    # Phase 1: Initial independent ReAct (uses same step budget as debate)
    def _init_agent(i):
        def run(monitor):
            messages = [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": task_question},
            ]
            return react_loop(
                client=client, model=model, messages=messages,
                tools=_agent_tools(i), execute_tool=execute_tool,
                energy_monitor=monitor,
                max_steps=debate_steps,
                temperature=DEBATE_TEMP,
                seed=BASE_SEED + i,
                agent_id=f"debater_{i}_init",
                extra_body=extra_body,
            )
        return run

    init_results = _run_agent_phase(
        [_init_agent(i) for i in range(N_AGENTS)], energy_monitor)
    for i, result in enumerate(init_results):
        agent_histories[i] = result["messages"]
        agent_trajectories[i] = {
            "final_response": result["final_response"],
            "steps": result["steps"],
            "tool_summary": _summarize_trajectory(result["messages"]),
        }
        all_records.extend(result["call_records"])
        all_usages.append(result["total_usage"])

    # Phase 2: Debate rounds.
    # Default (Kim et al. multiagent_decentralized.py:285): R is the TOTAL
    # number of react_loops per agent. Phase 1 ran 1 react_loop, so debate
    # runs R-1 additional loops. The previous default (R = debate rounds
    # AFTER initial → total = R+1) was a Du-et-al-faithful convention that
    # did NOT match Kim et al. and caused per-agent budget asymmetry with
    # Cent's "R = total rounds" semantic. Set legacy_du_r_semantic=True only
    # to reproduce pre-fix data.
    if kwargs.get("legacy_du_r_semantic", False):
        n_actual_debate_rounds = n_debate_rounds
    else:
        n_actual_debate_rounds = max(0, n_debate_rounds - 1)
    # Default: no early-stop. Kim et al. multiagent_decentralized.py:349-352
    # explicitly comments "we do NOT exit early when any single agent's env
    # is done. All d debate rounds run so the consensus aggregation has full
    # information." Set MAS_DISABLE_EARLY_STOP=0 only to re-enable the old
    # heuristic (kept for reproducing pre-fix data).
    disable_early_stop = os.environ.get("MAS_DISABLE_EARLY_STOP", "1") != "0"
    debate_rounds_used = 0
    for round_idx in range(n_actual_debate_rounds):
        debate_rounds_used = round_idx + 1
        new_trajectories = [None] * N_AGENTS

        def _debate_agent(i, round_idx=round_idx,
                          prev_trajectories=agent_trajectories):
            def run(monitor):
                debate_msg = {
                    "role": "user",
                    "content": format_debate_prompt(prev_trajectories, i,
                                                    comm_mode=comm_mode,
                                                    terse_speaker=terse_speaker),
                }
                messages = agent_histories[i] + [debate_msg]
                return react_loop(
                    client=client, model=model, messages=messages,
                    tools=_agent_tools(i), execute_tool=execute_tool,
                    energy_monitor=monitor,
                    max_steps=debate_steps,
                    temperature=DEBATE_TEMP,
                    seed=BASE_SEED + i + (round_idx + 1) * N_AGENTS,
                    agent_id=f"debater_{i}_r{round_idx}",
                    extra_body=extra_body,
                )
            return run

        round_results = _run_agent_phase(
            [_debate_agent(i) for i in range(N_AGENTS)], energy_monitor)
        for i, result in enumerate(round_results):
            agent_histories[i] = result["messages"]
            new_trajectories[i] = {
                "final_response": result["final_response"],
                "steps": result["steps"],
                "tool_summary": _summarize_trajectory(result["messages"]),
            }
            all_records.extend(result["call_records"])
            all_usages.append(result["total_usage"])

        agent_trajectories = new_trajectories

        # Early stop disabled by default (Kim-aligned). Re-enable via
        # MAS_DISABLE_EARLY_STOP=0 for legacy reproduction only.
        if _agents_converged(round_results) and not disable_early_stop:
            break

    # Phase 3: Final aggregation
    if consensus_vote:
        # Kim et al. Decentralized: no LLM synthesis. Majority vote over
        # final-round answers (Counter.most_common with stable first-seen
        # tie-break). Preserves the original answer text verbatim.
        from collections import Counter
        final_answers = [
            (traj or {}).get("final_response") or ""
            for traj in agent_trajectories
        ]
        non_empty = [a for a in final_answers if a]
        if non_empty:
            counts = Counter(non_empty)
            winning_answer, _vote_count = counts.most_common(1)[0]
        else:
            winning_answer = ""
        return {
            "answer": winning_answer,
            "call_records": all_records,
            "total_usage": _aggregate_usage(*all_usages),
            "agent_trajectories": agent_trajectories,
            "rounds_used": 1 + debate_rounds_used,
        }

    energy_monitor.start()
    synth_text, synth_usage = chat(
        client, model,
        messages=[
            {"role": "system", "content": DEBATE_SYNTHESIZER_PROMPT},
            {"role": "user", "content": format_debate_synthesis(
                task_question, agent_trajectories
            )},
        ],
        temperature=ORCHESTRATOR_TEMP, seed=BASE_SEED,
        extra_body=extra_body,
    )
    synth_record = energy_monitor.stop(metadata={
        "agent_id": "synthesizer",
        "call_type": "debate_synthesis",
        **synth_usage,
    })
    all_records.append(synth_record)
    all_usages.append(synth_usage)

    return {
        "answer": synth_text,
        "call_records": all_records,
        "total_usage": _aggregate_usage(*all_usages),
        "agent_trajectories": agent_trajectories,
        "rounds_used": 1 + debate_rounds_used,  # init + debate rounds
    }


# ─────────────────────────────────────────────────────────
# Topology 5: Hybrid (Centralized + peer communication)
# ─────────────────────────────────────────────────────────

def run_hybrid(client, model, task_question, tools, execute_tool,
               energy_monitor, extra_body=None, **kwargs):
    """Centralized orchestration with limited peer communication.

    Extends Centralized by inserting P text-only peer exchange rounds
    between worker execution and orchestrator review in each round.

    Kim et al.: "Hybrid systems combine centralized orchestration with
    limited peer communication phases." Uses same rounds/worker steps
    as Centralized, plus N_PEER_ROUNDS peer exchanges per round.

    LLM calls: 1 (decompose)
             + R * [M*S (execution) + P*M (peer exchange) + 1 (review)]
             + 1 (synthesis)
    """
    worker_steps = kwargs.get("max_react_steps", HYBRID_WORKER_STEPS)
    n_rounds = kwargs.get("n_rounds", HYBRID_ROUNDS)
    all_records = []
    all_usages = []

    # ── Phase 1: Orchestrator decomposes (uses raw question without format instructions) ──
    decompose_question = kwargs.get("raw_question", task_question)
    energy_monitor.start()
    decompose_text, decompose_usage = chat(
        client, model,
        messages=[
            {"role": "system", "content": ORCHESTRATOR_PROMPT},
            {"role": "user", "content": format_decompose_prompt(
                decompose_question, tools, n_workers=N_AGENTS
            )},
        ],
        temperature=ORCHESTRATOR_TEMP, seed=BASE_SEED,
        extra_body=extra_body,
    )
    decompose_record = energy_monitor.stop(metadata={
        "agent_id": "orchestrator",
        "call_type": "decompose",
        **decompose_usage,
    })
    all_records.append(decompose_record)
    all_usages.append(decompose_usage)

    subtasks = parse_subtasks(decompose_text, N_AGENTS, task_question)

    # ── Phase 2: Up to R rounds of (execution → peer debate → orchestrator review) ──
    # Workers have persistent memory across rounds (same as Centralized).
    # worker_histories also accumulates peer exchange context.
    # Early stop: if all workers converge (no tool calls), skip remaining rounds.
    worker_results = [None] * N_AGENTS
    worker_histories = [None] * N_AGENTS
    feedback = None
    hybrid_rounds_used = 0

    for round_idx in range(n_rounds):
        hybrid_rounds_used = round_idx + 1

        # ── 2a. Worker execution (persistent memory, same as Centralized) ──
        for i in range(N_AGENTS):
            if round_idx == 0:
                worker_messages = [
                    {"role": "system", "content": HYBRID_WORKER_PROMPT},
                    {"role": "user",
                     "content": (
                         f"Full task context:\n{task_question}\n\n"
                         f"Your assigned subtask: {subtasks[i]}"
                     )},
                ]
            else:
                # Continue from prior history (includes peer exchange)
                worker_messages = list(worker_histories[i])
                if feedback is not None:
                    worker_messages.append(
                        {"role": "user",
                         "content": f"Feedback from coordinator: {feedback}\n\n"
                                    f"Continue working on your subtask based on this feedback."}
                    )

            result = react_loop(
                client=client, model=model, messages=worker_messages,
                tools=tools, execute_tool=execute_tool,
                energy_monitor=energy_monitor,
                max_steps=worker_steps,
                temperature=DEBATE_TEMP, seed=BASE_SEED + i,
                agent_id=f"hybrid_worker_{i}_r{round_idx}",
                extra_body=extra_body,
            )
            worker_results[i] = result
            worker_histories[i] = result["messages"]
            all_records.extend(result["call_records"])
            all_usages.append(result["total_usage"])

        # Early stop: all workers converged (no tool calls this round)
        if round_idx > 0 and _agents_converged(worker_results):
            break

        # ── 2b. Peer debate rounds (text-only, no tools) ──
        # Kim et al.: "limited peer communication" — O(pn) not O(pnk).
        # Each worker sees peers' results and revises via a single LLM call.
        worker_trajectories = [
            {
                "final_response": worker_results[i].get("final_response", ""),
                "steps": worker_results[i].get("steps", 0),
                "tool_summary": _summarize_trajectory(
                    worker_results[i].get("messages", [])
                ),
            }
            for i in range(N_AGENTS)
        ]

        for peer_round in range(N_PEER_ROUNDS):
            new_trajectories = [None] * N_AGENTS
            for i in range(N_AGENTS):
                peer_prompt = format_peer_debate_prompt(
                    subtasks, worker_trajectories, exclude_idx=i
                )
                # Single chat call — no tool calling
                energy_monitor.start()
                peer_text, peer_usage = chat(
                    client, model,
                    messages=worker_histories[i] + [
                        {"role": "user", "content": peer_prompt},
                    ],
                    temperature=PEER_TEMP,
                    seed=BASE_SEED + i + (peer_round + 1) * 100,
                    extra_body=extra_body,
                )
                peer_record = energy_monitor.stop(metadata={
                    "agent_id": f"hybrid_peer_{i}_r{round_idx}_p{peer_round}",
                    "call_type": "peer_exchange",
                    **peer_usage,
                })
                all_records.append(peer_record)
                all_usages.append(peer_usage)

                # Update worker history with the peer exchange
                worker_histories[i] = worker_histories[i] + [
                    {"role": "user", "content": peer_prompt},
                    {"role": "assistant", "content": peer_text},
                ]
                # Update result so orchestrator review sees revised answers
                worker_results[i] = {
                    **worker_results[i],
                    "final_response": peer_text,
                }
                new_trajectories[i] = {
                    "final_response": peer_text,
                    "steps": worker_results[i].get("steps", 0),
                    "tool_summary": _summarize_trajectory(
                        worker_histories[i]
                    ),
                }

            worker_trajectories = new_trajectories

        # ── 2c. Orchestrator review (except last round) ──
        if round_idx < n_rounds - 1:
            energy_monitor.start()
            feedback, review_usage = chat(
                client, model,
                messages=[
                    {"role": "system", "content": ORCHESTRATOR_PROMPT},
                    {"role": "user", "content": format_review_prompt(
                        task_question, subtasks, worker_results
                    )},
                ],
                temperature=ORCHESTRATOR_TEMP, seed=BASE_SEED,
                extra_body=extra_body,
            )
            review_record = energy_monitor.stop(metadata={
                "agent_id": "orchestrator",
                "call_type": f"review_r{round_idx}",
                **review_usage,
            })
            all_records.append(review_record)
            all_usages.append(review_usage)

    # ── Phase 3: Orchestrator synthesis ──
    energy_monitor.start()
    synth_text, synth_usage = chat(
        client, model,
        messages=[
            {"role": "system", "content": ORCHESTRATOR_PROMPT},
            {"role": "user", "content": format_centralized_synthesis_prompt(
                task_question, subtasks, worker_results
            )},
        ],
        temperature=ORCHESTRATOR_TEMP, seed=BASE_SEED,
        extra_body=extra_body,
    )
    synth_record = energy_monitor.stop(metadata={
        "agent_id": "orchestrator",
        "call_type": "synthesis",
        **synth_usage,
    })
    all_records.append(synth_record)
    all_usages.append(synth_usage)

    return {
        "answer": synth_text,
        "call_records": all_records,
        "total_usage": _aggregate_usage(*all_usages),
        "subtasks": subtasks,
        "worker_results": worker_results,
        "worker_trajectories": worker_trajectories,
        "rounds_used": hybrid_rounds_used,
    }


from functools import partial


def _decent_variant(comm_mode="full", terse_speaker=False, tool_partition=None,
                    minimal_output=False):
    """Factory for Decentralized topology runners with custom comm/speaker/tool modes."""
    def _runner(*args, **kwargs):
        kwargs["comm_mode"] = comm_mode
        kwargs["terse_speaker"] = terse_speaker
        if tool_partition is not None:
            kwargs["tool_partition"] = tool_partition
        if minimal_output:
            kwargs["minimal_output"] = True
        return run_decentralized(*args, **kwargs)
    _runner.__name__ = (
        f"run_decentralized_cm{comm_mode}_terse{terse_speaker}"
        f"_tp{tool_partition}_min{minimal_output}"
    )
    return _runner


def _independent_share_minimal(*args, **kwargs):
    """independent_share + minimal-output system prompt during Phase-1 ReAct.
    Stacks structural and within-call decode reductions."""
    kwargs["minimal_output"] = True
    return run_independent_share(*args, **kwargs)
_independent_share_minimal.__name__ = "run_independent_share_minimal"


# ─────────────────────────────────────────────────────────
# Topology variant: Decentralized with mid-stream communication
# ─────────────────────────────────────────────────────────

def run_decentralized_midstream(client, model, task_question, tools, execute_tool,
                                energy_monitor, extra_body=None, **kwargs):
    """M agents each run a SINGLE react loop with total budget k.
    At configurable checkpoints (default: every k/R steps), each agent
    receives a mid-stream injection of other agents' progress so far
    (tool calls + partial results), WITHOUT being asked to commit to
    a final answer.

    Unlike standard decentralized (full react → commit → debate → full react),
    this keeps agents in a continuous reasoning flow. Peer info arrives
    as a user message mid-loop, similar to how tool results arrive.

    Tests whether the Du et al. premature-commitment problem is the reason
    R doesn't help: if mid-stream communication makes R meaningful,
    the standard debate protocol is structurally wasteful.
    """
    total_steps = kwargs.get("max_react_steps", DECENTRALIZED_DEBATE_STEPS)
    n_checkpoints = kwargs.get("n_rounds", DECENTRALIZED_ROUNDS)
    all_records = []
    all_usages = []

    # Compute injection points: evenly spaced through the budget
    # e.g., k=10 n_checkpoints=2 → inject at steps 3 and 6
    interval = total_steps // (n_checkpoints + 1)
    inject_steps = set(interval * (i + 1) for i in range(n_checkpoints))

    # Phase 1: Run all agents in parallel up to first checkpoint
    # We need to run step-by-step across agents to collect partial results
    # at each checkpoint. Use the mid_stream_injections mechanism.

    # Strategy: run agents sequentially but with injection callbacks.
    # At each checkpoint step, we pause, collect all agents' progress,
    # format peer summaries, then continue.

    # First, initialize all agents
    agent_messages = []
    agent_results = [None] * N_AGENTS
    sys_prompt = DEBATE_AGENT_PROMPT

    for i in range(N_AGENTS):
        agent_messages.append([
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": task_question},
        ])

    # Run in checkpoint segments
    steps_done = [0] * N_AGENTS
    agent_trajectories = [{"tool_summary": "No tool calls yet", "partial_response": ""} for _ in range(N_AGENTS)]
    checkpoints_hit = 0

    # Build the step segments: [0, first_inject), [first_inject, second_inject), ...
    sorted_injects = sorted(inject_steps)
    boundaries = [0] + sorted_injects + [total_steps]

    for seg_idx in range(len(boundaries) - 1):
        seg_start = boundaries[seg_idx]
        seg_end = boundaries[seg_idx + 1]
        seg_steps = seg_end - seg_start

        if seg_steps <= 0:
            continue

        # If not the first segment, inject peer info before running
        if seg_idx > 0:
            checkpoints_hit += 1
            for i in range(N_AGENTS):
                peer_parts = []
                for j in range(N_AGENTS):
                    if j == i:
                        continue
                    peer_parts.append(
                        f"[Agent {j}] Progress so far:\n"
                        f"  Tool calls: {agent_trajectories[j]['tool_summary']}\n"
                        f"  Working notes: {agent_trajectories[j]['partial_response'][:300]}"
                    )
                inject_msg = (
                    f"Mid-task update from peer agents (checkpoint {checkpoints_hit}/{n_checkpoints}):\n\n"
                    + "\n\n".join(peer_parts) + "\n\n"
                    "Continue working on the task. Use this information if helpful."
                )
                agent_messages[i].append({"role": "user", "content": inject_msg})

        # Run each agent for this segment
        for i in range(N_AGENTS):
            result = react_loop(
                client=client, model=model, messages=agent_messages[i],
                tools=tools, execute_tool=execute_tool,
                energy_monitor=energy_monitor,
                max_steps=seg_steps,
                temperature=DEBATE_TEMP,
                seed=BASE_SEED + i + seg_idx * N_AGENTS,
                agent_id=f"debater_{i}_seg{seg_idx}",
                extra_body=extra_body,
            )
            agent_messages[i] = result["messages"]
            agent_results[i] = result
            all_records.extend(result["call_records"])
            all_usages.append(result["total_usage"])
            steps_done[i] += result["steps"]

            # Update trajectory for peer sharing
            agent_trajectories[i] = {
                "tool_summary": _summarize_trajectory(result["messages"]),
                "partial_response": result.get("final_response", "") or "",
                "final_response": result.get("final_response", ""),
                "steps": steps_done[i],
            }

    # Phase 2: Final synthesis
    energy_monitor.start()
    synth_text, synth_usage = chat(
        client, model,
        messages=[
            {"role": "system", "content": DEBATE_SYNTHESIZER_PROMPT},
            {"role": "user", "content": format_debate_synthesis(
                task_question, agent_trajectories
            )},
        ],
        temperature=ORCHESTRATOR_TEMP, seed=BASE_SEED,
        extra_body=extra_body,
    )
    synth_record = energy_monitor.stop(metadata={
        "agent_id": "synthesizer",
        "call_type": "debate_synthesis",
        **synth_usage,
    })
    all_records.append(synth_record)
    all_usages.append(synth_usage)

    return {
        "answer": synth_text,
        "call_records": all_records,
        "total_usage": _aggregate_usage(*all_usages),
        "agent_trajectories": agent_trajectories,
        "rounds_used": n_checkpoints,
    }


def _centralized_legacy_broadcast(*args, **kwargs):
    """LEGACY: original broadcast-feedback Cent. Single LLM call per round
    produces one feedback string broadcast to all workers. Known confounds:
    cross-worker context pollution, 1-vs-0-indexed worker identity collision
    (pre-fix prompts), orchestrator pre-commitment leakage (pre-fix prompts).
    Kept ONLY to reproduce pre-fix data. New runs should use default Cent."""
    kwargs["broadcast_feedback"] = True
    return run_centralized(*args, **kwargs)
_centralized_legacy_broadcast.__name__ = "run_centralized_legacy_broadcast"


def _decentralized_legacy_r(*args, **kwargs):
    """LEGACY: Du-style R semantic (R = debate rounds AFTER initial → total
    react_loops = R+1) with the heuristic early-stop active. Aggregation +
    comm_mode kept at current defaults since those are defensible choices.
    Use only to reproduce pre-fix kbudget60/kbudget30 Decent R-sweep data."""
    kwargs["legacy_du_r_semantic"] = True
    import os as _os
    _os.environ["MAS_DISABLE_EARLY_STOP"] = "0"
    return run_decentralized(*args, **kwargs)
_decentralized_legacy_r.__name__ = "run_decentralized_legacy_r"


def _decentralized_vote(*args, **kwargs):
    """Kim et al. aggregator opt-in: majority vote over final-round answers
    (no LLM synth call). Useful for benchmarks where short factoid agreement
    makes vote viable (e.g., BrowseComp). For multi-component lists
    (FanOutQA, QAMPARI) consider _independent_concat-style union instead."""
    kwargs["consensus_vote"] = True
    return run_decentralized(*args, **kwargs)
_decentralized_vote.__name__ = "run_decentralized_vote"


def _independent_concat(*args, **kwargs):
    """Kim et al. aggregator opt-in: pure concatenation, no LLM synth call.
    Best for multi-component recall benchmarks (FanOutQA, QAMPARI) where
    union of agents' answers maximizes recall against gold list components.
    Worse for single-answer benchmarks where the grader must pick one."""
    kwargs["concat_synthesis"] = True
    return run_independent(*args, **kwargs)
_independent_concat.__name__ = "run_independent_concat"


TOPOLOGY_RUNNERS = {
    "sas": run_sas,
    "adaptive": run_adaptive,               # C1: cheap gate + always-run escalate
    # Justified defaults — full Kim et al. alignment, modulo serial execution
    # (intentional, for clean NVML energy attribution) and fixed R (intentional,
    # for clean R-axis experimental control).
    "independent": run_independent,         # concat aggregator
    "centralized": run_centralized,         # M per-worker LLM coord + team_context
    "decentralized": run_decentralized,     # answer-only comm + vote + Kim R semantic
    "hybrid": run_hybrid,
    # Side experiments (latent-comm ablations etc.) — unrelated to Kim alignment
    "independent_share": run_independent_share,
    "independent_share_minimal": _independent_share_minimal,
    # LEGACY — only to reproduce pre-fix data, NOT for paper-reported numbers
    "centralized_legacy_broadcast": _centralized_legacy_broadcast,  # broken broadcast protocol
    "decentralized_legacy_r": _decentralized_legacy_r,              # Du R semantic + early-stop active
    # Optional Kim-aligned aggregator variants (justifiable for some benchmarks)
    "decentralized_vote": _decentralized_vote,         # majority vote (BrowseComp etc.)
    "independent_concat": _independent_concat,         # concat aggregator (FanOutQA, QAMPARI)
    # ─── Phase A: receiver-side counterfactual ablation ───
    # Modify what the receiver sees about peers. Speaker still decodes fully.
    # Tests whether the receiver actually uses the dropped peer text content.
    # decentralized_full: canonical "decentralized" is answer_only, so this
    # full-channel variant (tool summaries + final answers) is needed to
    # establish the full→answer_only link ON-CONFIG in the channel-muting
    # rerun (2026-07-13). Old-era evidence for that link carries Du-R-semantics
    # + era-efficiency confounds.
    "decentralized_full":         _decent_variant(comm_mode="full"),
    "decentralized_answer_only":  _decent_variant(comm_mode="answer_only"),
    "decentralized_truncate100":  _decent_variant(comm_mode="truncate100"),
    "decentralized_truncate300":  _decent_variant(comm_mode="truncate300"),
    "decentralized_empty":        _decent_variant(comm_mode="empty"),
    # Strict control for 'empty': remove all peer-related framing entirely.
    # Tests whether 'empty' mode's accuracy is driven by the placeholder
    # acting as a meaningful signal vs the extra refinement compute alone.
    "decentralized_empty_silent": _decent_variant(comm_mode="empty_silent"),
    # ─── Phase B-2: speaker-side decode reduction ───
    # Instructs the receiver agent (which becomes a speaker on its turn) to
    # output only its final answer — no reasoning narrative. Directly attacks
    # decode cost (the dominant energy term) given Phase A's finding that the
    # verbose explanation isn't used downstream.
    "decentralized_terse":              _decent_variant(comm_mode="full",         terse_speaker=True),
    "decentralized_terse_answer_only":  _decent_variant(comm_mode="answer_only",  terse_speaker=True),
    # ─── Minimum-output: suppress ALL agent text during the trajectory ───
    # Stricter than `terse`: terse only restricts the FINAL response. Minimal
    # also suppresses the reasoning narration BETWEEN tool calls. Tests
    # whether the agent's intermediate ReAct narration is decode-waste or
    # load-bearing for the agent's own reasoning.
    "decentralized_minimal":            _decent_variant(comm_mode="full",         minimal_output=True),
    "decentralized_minimal_empty":      _decent_variant(comm_mode="empty",        minimal_output=True),
    # ─── Specialist tool-partition (disjoint capabilities) ───
    # Each agent gets a disjoint slice of the tool list. Tests whether
    # channel-muting holds when agents have GENUINELY disjoint capabilities
    # (not just overlapping access to a shared tool environment).
    # Round-robin: clean ablation, mixes tools across domains per agent.
    "decentralized_specialist":       _decent_variant(comm_mode="full",  tool_partition="round_robin"),
    "decentralized_specialist_empty": _decent_variant(comm_mode="empty", tool_partition="round_robin"),
    # Categorical WorkBench partition: deployment-faithful specialist roles
    # (Scheduler / Communicator / CRM-Analyst). Maps directly to production
    # CrewAI/MetaGPT/Agentforce patterns where agents own disjoint domains.
    "decentralized_specialist_categorical":       _decent_variant(comm_mode="full",  tool_partition="categorical_workbench"),
    "decentralized_specialist_categorical_empty": _decent_variant(comm_mode="empty", tool_partition="categorical_workbench"),
    # ─── Mid-stream communication (no premature commitment) ───
    # Agents run a single continuous react loop. Peer progress is injected
    # mid-stream as a user message at evenly-spaced checkpoints, without
    # asking for a final answer. Tests whether Du et al.'s commit-then-revise
    # protocol is the reason R doesn't help.
    "decentralized_midstream": run_decentralized_midstream,
}
