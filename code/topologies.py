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

from llm import chat, react_loop
from prompts import (
    SAS_PROMPT, INDEPENDENT_AGENT_PROMPT, INDEPENDENT_AGENT_PROMPT_MINIMAL,
    WORKER_PROMPT,
    ORCHESTRATOR_PROMPT, DEBATE_AGENT_PROMPT, DEBATE_AGENT_PROMPT_MINIMAL,
    SYNTHESIZER_PROMPT, DEBATE_SYNTHESIZER_PROMPT,
    HYBRID_WORKER_PROMPT,
    format_decompose_prompt, format_synthesis_prompt,
    format_review_prompt, format_centralized_synthesis_prompt,
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


def _agents_converged(results):
    """True if all agents finished in 1 step (no tool calls -- just a text response).

    This means every agent already has its answer and additional rounds
    would only re-process the growing context for no benefit.
    """
    return all(r.get("steps", 0) == 1 for r in results)


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
        for i in range(N_AGENTS):
            if round_idx == 0:
                worker_messages = [
                    {"role": "system", "content": WORKER_PROMPT},
                    {"role": "user",
                     "content": (
                         f"Full task context:\n{task_question}\n\n"
                         f"Your assigned subtask: {subtasks[i]}"
                     )},
                ]
            else:
                # Continue from prior history with feedback appended
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
                agent_id=f"worker_{i}_r{round_idx}",
                extra_body=extra_body,
            )
            worker_results[i] = result
            worker_histories[i] = result["messages"]
            all_records.extend(result["call_records"])
            all_usages.append(result["total_usage"])

        # Early stop: all workers converged (no tool calls this round)
        if round_idx > 0 and _agents_converged(worker_results):
            break

        # Orchestrator review (except last round)
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
        "rounds_used": rounds_used,
    }


# ─────────────────────────────────────────────────────────
# Topology 4: Decentralized (M agents × R debate rounds)
# ─────────────────────────────────────────────────────────

def run_decentralized(client, model, task_question, tools, execute_tool,
                      energy_monitor, extra_body=None, **kwargs):
    """M agents each run ReAct, then debate for R rounds by seeing
    each other's trajectories and revising.

    Kim et al.: "3 agents through 3 debate rounds with 3 iterations
    per round." The initial independent phase counts as round 1 (no
    prior trajectories to debate), then DECENTRALIZED_ROUNDS debate
    rounds follow. Total = 1 + DECENTRALIZED_ROUNDS = 3 rounds.

    LLM calls: M*k (initial) + R*M*k (debate) + 1 (synthesis).
    """
    debate_steps = kwargs.get("max_react_steps", DECENTRALIZED_DEBATE_STEPS)
    n_debate_rounds = kwargs.get("n_rounds", DECENTRALIZED_ROUNDS)
    comm_mode = kwargs.get("comm_mode", "full")
    terse_speaker = kwargs.get("terse_speaker", False)
    # minimal_output: suppress text alongside tool calls AND in final answer
    # via a stricter system prompt. Tests whether the agent's intermediate
    # ReAct narration (between tool calls) is decode-waste or load-bearing.
    minimal_output = kwargs.get("minimal_output", False)
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
    init_results = []
    for i in range(N_AGENTS):
        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": task_question},
        ]
        result = react_loop(
            client=client, model=model, messages=messages,
            tools=_agent_tools(i), execute_tool=execute_tool,
            energy_monitor=energy_monitor,
            max_steps=debate_steps,
            temperature=DEBATE_TEMP,
            seed=BASE_SEED + i,
            agent_id=f"debater_{i}_init",
            extra_body=extra_body,
        )
        init_results.append(result)
        agent_histories[i] = result["messages"]
        agent_trajectories[i] = {
            "final_response": result["final_response"],
            "steps": result["steps"],
            "tool_summary": _summarize_trajectory(result["messages"]),
        }
        all_records.extend(result["call_records"])
        all_usages.append(result["total_usage"])

    # Phase 2: Debate rounds (early stop when all debaters converge)
    debate_rounds_used = 0
    for round_idx in range(n_debate_rounds):
        debate_rounds_used = round_idx + 1
        new_trajectories = [None] * N_AGENTS
        round_results = []
        for i in range(N_AGENTS):
            debate_msg = {
                "role": "user",
                "content": format_debate_prompt(agent_trajectories, i,
                                                comm_mode=comm_mode,
                                                terse_speaker=terse_speaker),
            }
            messages = agent_histories[i] + [debate_msg]

            result = react_loop(
                client=client, model=model, messages=messages,
                tools=_agent_tools(i), execute_tool=execute_tool,
                energy_monitor=energy_monitor,
                max_steps=debate_steps,
                temperature=DEBATE_TEMP,
                seed=BASE_SEED + i + (round_idx + 1) * N_AGENTS,
                agent_id=f"debater_{i}_r{round_idx}",
                extra_body=extra_body,
            )
            round_results.append(result)
            agent_histories[i] = result["messages"]
            new_trajectories[i] = {
                "final_response": result["final_response"],
                "steps": result["steps"],
                "tool_summary": _summarize_trajectory(result["messages"]),
            }
            all_records.extend(result["call_records"])
            all_usages.append(result["total_usage"])

        agent_trajectories = new_trajectories

        # Early stop: all debaters converged (no tool calls this round)
        if _agents_converged(round_results):
            break

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


TOPOLOGY_RUNNERS = {
    "sas": run_sas,
    "independent": run_independent,
    "independent_share": run_independent_share,
    "independent_share_minimal": _independent_share_minimal,
    "centralized": run_centralized,
    "decentralized": run_decentralized,
    "hybrid": run_hybrid,
    # ─── Phase A: receiver-side counterfactual ablation ───
    # Modify what the receiver sees about peers. Speaker still decodes fully.
    # Tests whether the receiver actually uses the dropped peer text content.
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
