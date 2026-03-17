"""5 MAS topologies (Kim et al. 2025) with agentic tool calling.

Each topology function takes:
    client, model, task_question, tools, execute_tool, energy_monitor,
    extra_body=None, **kwargs
and returns a dict with: answer, call_records, total_usage.

All LLM calls are serial for clean NVML energy attribution.
Tool calling is handled by react_loop() in llm.py.
"""

from llm import chat, react_loop
from prompts import (
    SAS_PROMPT, INDEPENDENT_AGENT_PROMPT, WORKER_PROMPT,
    ORCHESTRATOR_PROMPT, DEBATE_AGENT_PROMPT,
    SYNTHESIZER_PROMPT, DEBATE_SYNTHESIZER_PROMPT,
    HYBRID_WORKER_PROMPT,
    format_decompose_prompt, format_synthesis_prompt,
    format_review_prompt, format_centralized_synthesis_prompt,
    format_debate_prompt, format_debate_synthesis,
    format_peer_debate_prompt,
    parse_subtasks, _summarize_trajectory,
)
from config import (
    N_AGENTS, N_ROUNDS, N_PEER_ROUNDS, MAX_REACT_STEPS,
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
    """
    max_steps = kwargs.get("max_react_steps", MAX_REACT_STEPS)
    agent_results = []
    all_records = []

    for i in range(N_AGENTS):
        messages = [
            {"role": "system", "content": INDEPENDENT_AGENT_PROMPT},
            {"role": "user", "content": task_question},
        ]
        result = react_loop(
            client=client, model=model, messages=messages,
            tools=tools, execute_tool=execute_tool,
            energy_monitor=energy_monitor,
            max_steps=max_steps,
            temperature=INDEPENDENT_TEMP,
            seed=BASE_SEED + i,
            agent_id=f"independent_{i}",
            extra_body=extra_body,
        )
        agent_results.append(result)
        all_records.extend(result["call_records"])

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
    }


# ─────────────────────────────────────────────────────────
# Topology 3: Centralized (orchestrator + M workers × R rounds)
# ─────────────────────────────────────────────────────────

def run_centralized(client, model, task_question, tools, execute_tool,
                    energy_monitor, extra_body=None, **kwargs):
    """Orchestrator decomposes task; workers execute with tools; orchestrator
    reviews and synthesizes.

    LLM calls: 1 (decompose) + R * M * S_avg + (R-1) reviews + 1 (synthesis)
    = O(rnk) + O(r).  Memory: O(r*n*k).

    Per Kim et al.: C = {(a_orch, ai) : ∀i} — workers communicate ONLY
    through the orchestrator, not with each other. Workers are logically
    parallel within each round (parallelization factor = n, sequential
    depth = r). They share the backend environment state but receive no
    explicit shared_history from peer workers.
    """
    max_steps = kwargs.get("max_react_steps", MAX_REACT_STEPS)
    all_records = []
    all_usages = []

    # Phase 1: Orchestrator decomposes
    energy_monitor.start()
    decompose_text, decompose_usage = chat(
        client, model,
        messages=[
            {"role": "system", "content": ORCHESTRATOR_PROMPT},
            {"role": "user", "content": format_decompose_prompt(
                task_question, tools, n_workers=N_AGENTS
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

    # Phase 2: R rounds of worker execution + orchestrator review
    worker_results = [None] * N_AGENTS
    feedback = None

    for round_idx in range(N_ROUNDS):
        for i in range(N_AGENTS):
            worker_messages = [
                {"role": "system", "content": WORKER_PROMPT},
            ]

            if round_idx > 0 and feedback and worker_results[i]:
                # Subsequent rounds: include prior result + feedback
                worker_messages.append(
                    {"role": "user", "content": f"Your subtask: {subtasks[i]}"}
                )
                worker_messages.append(
                    {"role": "assistant",
                     "content": worker_results[i].get("final_response", "")}
                )
                worker_messages.append(
                    {"role": "user",
                     "content": f"Feedback from coordinator: {feedback}\n\n"
                                f"Please revise your approach."}
                )
            else:
                worker_messages.append(
                    {"role": "user",
                     "content": f"Your assigned subtask: {subtasks[i]}\n\n"
                                f"Complete this using the available tools."}
                )

            result = react_loop(
                client=client, model=model, messages=worker_messages,
                tools=tools, execute_tool=execute_tool,
                energy_monitor=energy_monitor,
                max_steps=max_steps,
                temperature=DEBATE_TEMP, seed=BASE_SEED + i,
                agent_id=f"worker_{i}_r{round_idx}",
                extra_body=extra_body,
            )
            worker_results[i] = result
            all_records.extend(result["call_records"])
            all_usages.append(result["total_usage"])

        # Orchestrator review (except last round)
        if round_idx < N_ROUNDS - 1:
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

    # Phase 3: Orchestrator synthesis
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
    }


# ─────────────────────────────────────────────────────────
# Topology 4: Decentralized (M agents × R debate rounds)
# ─────────────────────────────────────────────────────────

def run_decentralized(client, model, task_question, tools, execute_tool,
                      energy_monitor, extra_body=None, **kwargs):
    """M agents each run ReAct, then debate for R rounds by seeing
    each other's trajectories and revising.

    LLM calls: M * S_initial + R * M * S_debate + 1 (synthesis).
    """
    max_steps = kwargs.get("max_react_steps", MAX_REACT_STEPS)
    all_records = []
    all_usages = []

    agent_histories = [None] * N_AGENTS
    agent_trajectories = [None] * N_AGENTS

    # Phase 1: Initial independent ReAct
    for i in range(N_AGENTS):
        messages = [
            {"role": "system", "content": DEBATE_AGENT_PROMPT},
            {"role": "user", "content": task_question},
        ]
        result = react_loop(
            client=client, model=model, messages=messages,
            tools=tools, execute_tool=execute_tool,
            energy_monitor=energy_monitor,
            max_steps=max_steps,
            temperature=DEBATE_TEMP,
            seed=BASE_SEED + i,
            agent_id=f"debater_{i}_init",
            extra_body=extra_body,
        )
        agent_histories[i] = result["messages"]
        agent_trajectories[i] = {
            "final_response": result["final_response"],
            "steps": result["steps"],
            "tool_summary": _summarize_trajectory(result["messages"]),
        }
        all_records.extend(result["call_records"])
        all_usages.append(result["total_usage"])

    # Phase 2: Debate rounds
    for round_idx in range(N_ROUNDS):
        new_trajectories = [None] * N_AGENTS
        for i in range(N_AGENTS):
            debate_msg = {
                "role": "user",
                "content": format_debate_prompt(agent_trajectories, i),
            }
            messages = agent_histories[i] + [debate_msg]

            result = react_loop(
                client=client, model=model, messages=messages,
                tools=tools, execute_tool=execute_tool,
                energy_monitor=energy_monitor,
                max_steps=min(max_steps, 10),
                temperature=DEBATE_TEMP,
                seed=BASE_SEED + i + (round_idx + 1) * N_AGENTS,
                agent_id=f"debater_{i}_r{round_idx}",
                extra_body=extra_body,
            )
            agent_histories[i] = result["messages"]
            new_trajectories[i] = {
                "final_response": result["final_response"],
                "steps": result["steps"],
                "tool_summary": _summarize_trajectory(result["messages"]),
            }
            all_records.extend(result["call_records"])
            all_usages.append(result["total_usage"])

        agent_trajectories = new_trajectories

    # Phase 3: Final synthesis
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
    }


# ─────────────────────────────────────────────────────────
# Topology 5: Hybrid (Centralized + peer communication)
# ─────────────────────────────────────────────────────────

def run_hybrid(client, model, task_question, tools, execute_tool,
               energy_monitor, extra_body=None, **kwargs):
    """Centralized orchestration with limited peer communication.

    Extends Centralized by inserting P text-only peer exchange rounds
    between worker execution and orchestrator review in each round.

    LLM calls: 1 (decompose)
             + R * [M*S (execution) + P*M (peer exchange) + 1 (review)]
             + 1 (synthesis)
    = O(rnk) + O(r) + O(pn)  — matches Kim et al. Table 2.

    Kim et al.: 515% token overhead, 13.6 success/1K tokens.
    Workers execute against shared backend with tool access.
    Peer exchange is text-only (chat, no tools) — "limited peer
    communication" per Kim et al.
    """
    max_steps = kwargs.get("max_react_steps", MAX_REACT_STEPS)
    all_records = []
    all_usages = []

    # ── Phase 1: Orchestrator decomposes ──
    energy_monitor.start()
    decompose_text, decompose_usage = chat(
        client, model,
        messages=[
            {"role": "system", "content": ORCHESTRATOR_PROMPT},
            {"role": "user", "content": format_decompose_prompt(
                task_question, tools, n_workers=N_AGENTS
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

    # ── Phase 2: R rounds of (execution → peer debate → orchestrator review) ──
    worker_results = [None] * N_AGENTS
    worker_histories = [None] * N_AGENTS  # preserved across peer rounds
    feedback = None

    for round_idx in range(N_ROUNDS):

        # ── 2a. Worker execution (same as Centralized) ──
        for i in range(N_AGENTS):
            worker_messages = [
                {"role": "system", "content": HYBRID_WORKER_PROMPT},
            ]

            if round_idx > 0 and feedback and worker_results[i]:
                worker_messages.append(
                    {"role": "user", "content": f"Your subtask: {subtasks[i]}"}
                )
                worker_messages.append(
                    {"role": "assistant",
                     "content": worker_results[i].get("final_response", "")}
                )
                worker_messages.append(
                    {"role": "user",
                     "content": f"Feedback from coordinator: {feedback}\n\n"
                                f"Please revise your approach."}
                )
            else:
                worker_messages.append(
                    {"role": "user",
                     "content": f"Your assigned subtask: {subtasks[i]}\n\n"
                                f"Complete this using the available tools."}
                )

            result = react_loop(
                client=client, model=model, messages=worker_messages,
                tools=tools, execute_tool=execute_tool,
                energy_monitor=energy_monitor,
                max_steps=max_steps,
                temperature=DEBATE_TEMP, seed=BASE_SEED + i,
                agent_id=f"hybrid_worker_{i}_r{round_idx}",
                extra_body=extra_body,
            )
            worker_results[i] = result
            worker_histories[i] = result["messages"]
            all_records.extend(result["call_records"])
            all_usages.append(result["total_usage"])

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
        if round_idx < N_ROUNDS - 1:
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
    }


TOPOLOGY_RUNNERS = {
    "sas": run_sas,
    "independent": run_independent,
    "centralized": run_centralized,
    "decentralized": run_decentralized,
    "hybrid": run_hybrid,
}
