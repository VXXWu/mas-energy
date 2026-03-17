"""System prompts and formatting functions for MAS topologies.

All prompts are designed for agentic tool-calling tasks.
Agents interact with benchmark-specific tools via the ReAct loop.
"""

# --- Agent system prompts ---

SAS_PROMPT = "You are a helpful assistant with access to tools. Use the provided tools to complete the user's task. When you have gathered enough information or completed the required actions, provide your final answer directly."

INDEPENDENT_AGENT_PROMPT = "You are a helpful assistant with access to tools. Use the provided tools to complete the user's task. When you have gathered enough information or completed the required actions, provide your final answer directly."

WORKER_PROMPT = "You are a tool-use assistant. Complete your assigned subtask using the available tools. Make function calls as needed. When done, provide a summary of what you accomplished and your result."

ORCHESTRATOR_PROMPT = "You are a coordination agent managing a team of workers. You do not call tools yourself. Instead, you decompose tasks, review worker outputs, provide feedback, and synthesize final answers."

DEBATE_AGENT_PROMPT = "You are a helpful assistant with access to tools, participating in a collaborative problem-solving process. Use tools to investigate the task thoroughly. Provide your reasoning and final answer clearly."

SYNTHESIZER_PROMPT = "You are a synthesis agent. Given multiple agents' approaches to the same task, synthesize the best final answer based on their findings. Do not analytically compare or cross-validate the responses against each other."

DEBATE_SYNTHESIZER_PROMPT = "You are a synthesis agent. Given multiple agents' responses after debate, synthesize the best final answer based on all agents' work."

HYBRID_WORKER_PROMPT = "You are a tool-use assistant working as part of a coordinated team. Complete your assigned subtask using the available tools. You may also see results from peer workers — use their findings to inform your approach. When done, provide a summary of what you accomplished and your result."


# --- Formatting functions ---

def format_decompose_prompt(question, tools, n_workers=3):
    """Format the orchestrator's task decomposition prompt."""
    tool_names = [t["function"]["name"] for t in tools]
    return (
        f"Task: {question}\n\n"
        f"Available tools: {', '.join(tool_names)}\n\n"
        f"Decompose this task into {n_workers} subtasks, one per worker. "
        f"Each worker can make function calls using the available tools.\n"
        f"Format each subtask on a separate line starting with 'SUBTASK:'"
    )


def format_synthesis_prompt(question, agent_results):
    """Format the synthesis prompt with agent responses and tool trajectories."""
    parts = [f"Task: {question}\n"]
    for i, result in enumerate(agent_results):
        trajectory = _summarize_trajectory(result.get("messages", []))
        parts.append(
            f"[Agent {i}]:\n"
            f"  Tool trajectory: {trajectory}\n"
            f"  Final answer: {result.get('final_response', 'No response')}\n"
        )
    parts.append(
        "\nBased on these agents' findings, provide the best final answer."
    )
    return "\n".join(parts)


def format_review_prompt(question, subtasks, worker_results):
    """Format the orchestrator's review prompt."""
    parts = [f"Original task: {question}\n\nWorker reports:\n"]
    for i, (st, wr) in enumerate(zip(subtasks, worker_results)):
        parts.append(
            f"[Worker {i}] Subtask: {st}\n"
            f"  Steps taken: {wr.get('steps', '?')}\n"
            f"  Result: {wr.get('final_response', 'No response')}\n"
        )
    parts.append("\nEvaluate quality. Provide feedback for revision if needed.")
    return "\n".join(parts)


def format_centralized_synthesis_prompt(question, subtasks, worker_results):
    """Format the orchestrator's final synthesis prompt."""
    parts = [f"Original task: {question}\n\nFinal worker reports:\n"]
    for i, (st, wr) in enumerate(zip(subtasks, worker_results)):
        parts.append(
            f"[Worker {i}] Subtask: {st}\n"
            f"  Result: {wr.get('final_response', 'No response')}\n"
        )
    parts.append("\nSynthesize these into a single final answer.")
    return "\n".join(parts)


def format_debate_prompt(agent_trajectories, exclude_idx):
    """Format the debate prompt showing other agents' work."""
    parts = []
    for i, traj in enumerate(agent_trajectories):
        if i == exclude_idx or traj is None:
            continue
        parts.append(
            f"[Agent {i}]:\n"
            f"  Tool calls: {traj.get('tool_summary', 'None')}\n"
            f"  Final answer: {traj.get('final_response', 'No response')}\n"
        )
    other_text = "\n".join(parts)
    return (
        f"These are the approaches and results from other agents:\n\n"
        f"{other_text}\n\n"
        f"Review their tool-calling strategies and results. "
        f"If you believe your approach was correct, reaffirm it. "
        f"If you see a better approach or errors in your work, "
        f"make additional tool calls to correct or verify. "
        f"Provide your updated final answer."
    )


def format_debate_synthesis(question, agent_trajectories):
    """Format the final synthesis after debate rounds."""
    parts = [f"Task: {question}\n\nFinal agent responses after debate:\n"]
    for i, traj in enumerate(agent_trajectories):
        parts.append(
            f"[Agent {i}]:\n"
            f"  Tool calls: {traj.get('tool_summary', 'None')}\n"
            f"  Answer: {traj.get('final_response', 'No response')}\n"
        )
    parts.append("\nSynthesize the best final answer based on all agents' work.")
    return "\n".join(parts)


def format_peer_debate_prompt(subtasks, worker_trajectories, exclude_idx):
    """Format peer debate prompt showing other workers' subtask results.

    Unlike format_debate_prompt (decentralized), this includes subtask
    assignments so workers understand the division of labor.
    """
    parts = []
    for i, traj in enumerate(worker_trajectories):
        if i == exclude_idx or traj is None:
            continue
        parts.append(
            f"[Worker {i}] Subtask: {subtasks[i]}\n"
            f"  Tool calls: {traj.get('tool_summary', 'None')}\n"
            f"  Result: {traj.get('final_response', 'No response')}\n"
        )
    other_text = "\n".join(parts)
    return (
        f"Here are the results from your peer workers:\n\n"
        f"{other_text}\n\n"
        f"Review their work in relation to your subtask. "
        f"If their findings change your understanding, revise your answer. "
        f"Provide your updated final answer."
    )


def format_shared_history(shared_history):
    """Format prior workers' tool calls for the next worker's context."""
    parts = ["Previous workers have taken the following actions:\n"]
    for msg in shared_history:
        role = msg.get("role", "")
        if role == "assistant":
            content = msg.get("content", "")
            tool_calls = msg.get("tool_calls", [])
            if tool_calls:
                for tc in tool_calls:
                    fn = tc.get("function", {})
                    parts.append(f"  Called: {fn.get('name', '?')}({fn.get('arguments', '')[:80]})")
            elif content:
                parts.append(f"  Response: {content[:200]}")
        elif role == "tool":
            content = msg.get("content", "")
            parts.append(f"  Result: {content[:200]}")
    return "\n".join(parts)


def parse_subtasks(orchestrator_text, n_agents, fallback_task):
    """Parse orchestrator decomposition into subtasks."""
    lines = orchestrator_text.strip().split("\n")
    subtasks = []
    for line in lines:
        line = line.strip()
        if line.upper().startswith("SUBTASK:"):
            subtasks.append(line[len("SUBTASK:"):].strip())
    while len(subtasks) < n_agents:
        subtasks.append(fallback_task)
    return subtasks[:n_agents]


def _summarize_trajectory(messages):
    """Extract a compact summary of tool calls from message history."""
    calls = []
    for m in messages:
        if isinstance(m, dict) and m.get("role") == "assistant" and m.get("tool_calls"):
            for tc in m["tool_calls"]:
                fn = tc.get("function", {})
                name = fn.get("name", "?")
                args_str = fn.get("arguments", "")[:60]
                calls.append(f"{name}({args_str})")
    return " -> ".join(calls) if calls else "No tool calls"
