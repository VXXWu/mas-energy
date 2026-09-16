"""System prompts and formatting functions for MAS topologies.

All prompts are designed for agentic tool-calling tasks.
Agents interact with benchmark-specific tools via the ReAct loop.
"""

import logging
import re

log = logging.getLogger(__name__)

# --- Agent system prompts ---

SAS_PROMPT = (
    "You are a helpful assistant with access to tools. "
    "IMPORTANT: You MUST use the provided tools to investigate the codebase, read files, and make changes. "
    "Do NOT describe a solution in prose without first calling tools — natural-language reasoning about the "
    "task is not enough. Always start by using tools (e.g. bash to explore, edit to modify) before proposing "
    "any answer. After you have gathered enough information or completed the required actions via tool calls, "
    "provide your final answer directly."
)

INDEPENDENT_AGENT_PROMPT = "You are a helpful assistant with access to tools. Use the provided tools to complete the user's task. When you have gathered enough information or completed the required actions, provide your final answer directly."

WORKER_PROMPT = "You are a helpful assistant with access to tools. Complete your assigned subtask using the provided tools. When you have gathered enough information or completed the required actions, provide your final answer directly."

ORCHESTRATOR_PROMPT = "You are a coordination agent managing a team of workers. You do not call tools yourself. Instead, you decompose tasks, review worker outputs, provide feedback, and synthesize final answers."

DEBATE_AGENT_PROMPT = "You are a helpful assistant with access to tools, participating in a collaborative problem-solving process. Use tools to investigate the task thoroughly. Provide your reasoning and final answer clearly."

# Minimum-output variant: suppresses ALL agent text during the trajectory.
# Differs from `terse` (which only restricts the final response):
#  - terse:    agent narrates between tool calls, terse FINAL answer
#  - minimal:  agent emits NO text alongside tool calls, only outputs text
#              at the very end as the final answer
# Tests whether the agent's intermediate ReAct narration is doing useful
# self-reasoning work vs being decode-waste.
DEBATE_AGENT_PROMPT_MINIMAL = (
    "You are a helpful assistant with access to tools. Investigate the task using the tools provided. "
    "When making tool calls, output ONLY the tool calls — do not write any explanation, reasoning, or commentary alongside them. "
    "When you have enough information, output ONLY your final answer — no preamble, no acknowledgements, no reasoning trace. Just the answer."
)

# Independent-agent variant of the minimal-output system prompt. Same content
# constraint as DEBATE_AGENT_PROMPT_MINIMAL, framed for the independent
# (no-debate) solo-work case so independent_share + minimal can share the
# same speaker-side intervention without the debate framing.
INDEPENDENT_AGENT_PROMPT_MINIMAL = (
    "You are a helpful assistant with access to tools. Use the tools provided to complete the task. "
    "When making tool calls, output ONLY the tool calls — do not write any explanation, reasoning, or commentary alongside them. "
    "When you have enough information, output ONLY your final answer — no preamble, no acknowledgements, no reasoning trace. Just the answer."
)

SYNTHESIZER_PROMPT = "You are a synthesis agent. Given multiple agents' approaches to the same task, synthesize the best final answer based on their findings. Do not analytically compare or cross-validate the responses against each other."

DEBATE_SYNTHESIZER_PROMPT = "You are a synthesis agent. Given multiple agents' responses after debate, synthesize the best final answer based on all agents' work."

HYBRID_WORKER_PROMPT = "You are a helpful assistant with access to tools, working as part of a coordinated team. Complete your assigned subtask using the provided tools. You may also see results from peer workers — use their findings to inform your approach. When you have gathered enough information or completed the required actions, provide your final answer directly."


# --- Formatting functions ---

def format_decompose_prompt(question, tools, n_workers=3):
    """Format the orchestrator's task decomposition prompt.

    Workers are 0-indexed (Worker 0, Worker 1, ..., Worker {n_workers-1}).
    Subtask text must NOT embed worker numbers — the runtime assigns workers
    by position in the SUBTASK list. Including 'Worker N, ...' inside a
    subtask creates an off-by-one collision with broadcast feedback addressing.
    """
    tool_names = [t["function"]["name"] for t in tools]
    worker_ids = ", ".join(f"Worker {i}" for i in range(n_workers))
    return (
        f"Task: {question}\n\n"
        f"Available tools: {', '.join(tool_names)}\n\n"
        f"Decompose this task into {n_workers} subtasks. Workers are 0-indexed "
        f"({worker_ids}). The first SUBTASK line is assigned to Worker 0, the "
        f"second to Worker 1, and so on. Each worker can make function calls "
        f"using the available tools. Assign each worker a DIFFERENT, "
        f"NON-OVERLAPPING portion of the search space (e.g., different time "
        f"periods, categories, or regions) so that together they cover the "
        f"full scope of the task without redundant work.\n"
        f"Format each subtask on a separate line starting with 'SUBTASK:'.\n"
        f"DO NOT embed 'Worker N' or 'Agent N' references inside the subtask "
        f"text — the runtime assigns workers by position."
    )


def format_synthesis_prompt(question, agent_results):
    """Format the synthesis prompt with agent responses and tool trajectories."""
    parts = [f"Task: {question}\n"]
    for i, result in enumerate(agent_results):
        trajectory = _summarize_trajectory(result.get("messages", []))
        answer = result.get('final_response') or 'No response'
        parts.append(
            f"[Agent {i}]:\n"
            f"  Tool trajectory: {trajectory}\n"
            f"  Final answer: {answer}\n"
        )
    parts.append(
        "\nBased on these agents' findings, provide the best final answer."
    )
    return "\n".join(parts)


def format_review_prompt(question, subtasks, worker_results):
    """Format the orchestrator's review prompt.

    Constrains the orchestrator to feedback-only output: must NOT produce a
    final synthesized answer (that's the synthesizer's job). Forbidding
    pre-commitment in the review prevents the broadcast-feedback context
    pollution where workers see the orchestrator's draft answer and may
    defer to it rather than refine independently.
    """
    parts = [f"Original task: {question}\n\nWorker reports:\n"]
    for i, (st, wr) in enumerate(zip(subtasks, worker_results)):
        result = wr.get('final_response') or 'No response'
        parts.append(
            f"[Worker {i}] Subtask: {st}\n"
            f"  Steps taken: {wr.get('steps', '?')}\n"
            f"  Result: {result}\n"
        )
    parts.append(
        "\nEvaluate quality. Provide feedback for revision if needed.\n\n"
        "CONSTRAINTS:\n"
        "- Address workers by 0-indexed ID: 'Worker 0', 'Worker 1', etc.\n"
        "- DO NOT produce a final synthesized answer in this feedback. The "
        "synthesizer will produce the final answer separately.\n"
        "- DO NOT include phrases like 'Final Answer:', 'Final Answer "
        "Formulation:', 'Revised Plan:', or any draft of the final answer.\n"
        "- Provide ONLY concrete revision guidance for the workers."
    )
    return "\n".join(parts)


def format_review_prompt_routed(question, subtasks, worker_results):
    """Orchestrator review prompt that instructs PER-WORKER feedback.

    Unlike format_review_prompt (one global string broadcast to all workers),
    this asks the orchestrator to emit one feedback block per worker, parsed
    and routed individually so each worker only sees feedback addressed to it.
    Avoids context pollution where all workers receive feedback about other
    workers' subtasks they cannot act on.
    """
    n = len(subtasks)
    parts = [f"Original task: {question}\n\nWorker reports:\n"]
    for i, (st, wr) in enumerate(zip(subtasks, worker_results)):
        result = wr.get('final_response') or 'No response'
        parts.append(
            f"[Worker {i}] Subtask: {st}\n"
            f"  Steps taken: {wr.get('steps', '?')}\n"
            f"  Result: {result}\n"
        )
    example_blocks = "\n".join(
        f"[Worker {i}]: <specific feedback for Worker {i}, or 'OK' if no revision needed>"
        for i in range(n)
    )
    parts.append(
        "\nEvaluate each worker's quality SEPARATELY. Provide PER-WORKER feedback, "
        "one block per worker, in EXACTLY this format:\n\n"
        f"{example_blocks}\n\n"
        "CONSTRAINTS:\n"
        "- Each worker will only see their own block — do not reference other "
        "workers' work inside a block, since the addressee cannot see it.\n"
        "- Workers are 0-indexed (Worker 0, Worker 1, ...).\n"
        "- DO NOT produce a final synthesized answer in any block. The "
        "synthesizer will produce the final answer separately.\n"
        "- DO NOT include phrases like 'Final Answer:', 'Final Answer "
        "Formulation:', or any draft of the final answer.\n"
        "- Provide ONLY concrete revision guidance for that specific worker."
    )
    return "\n".join(parts)


def parse_routed_feedback(text, n_agents):
    """Parse orchestrator output into per-worker feedback strings.

    Expected format: blocks starting with [Worker N]: or Worker N:.
    Multi-line continuations are appended to the current worker's block.
    Returns list[str] of length n_agents.

    Fallbacks:
      - If NO worker tags parse: returns [text] * n_agents (effective broadcast,
        same behaviour as the non-routed variant — preserves accuracy floor
        when orchestrator ignores the format instruction).
      - If SOME workers tagged but others missing: missing workers get an
        explicit OK message ("Your work looks good — continue with your subtask").
    """
    lines = text.split("\n")
    feedback = [""] * n_agents
    current_worker = None

    for line in lines:
        m = re.match(r'^\s*\[?Worker\s+(\d+)\]?\s*[:\-]\s*(.*)$', line, re.IGNORECASE)
        if m:
            idx = int(m.group(1))
            if 0 <= idx < n_agents:
                current_worker = idx
                feedback[current_worker] = m.group(2).strip()
            else:
                current_worker = None
        elif current_worker is not None:
            extra = line.strip()
            if extra:
                feedback[current_worker] = (feedback[current_worker] + " " + extra).strip()

    if not any(f for f in feedback):
        # No worker tags parsed — broadcast the entire text (degrade gracefully)
        return [text] * n_agents

    for i in range(n_agents):
        if not feedback[i]:
            feedback[i] = "Your work looks good — continue with your subtask."

    return feedback


def format_centralized_synthesis_prompt(question, subtasks, worker_results):
    """Format the orchestrator's final synthesis prompt."""
    parts = [f"Original task: {question}\n\nFinal worker reports:\n"]
    for i, (st, wr) in enumerate(zip(subtasks, worker_results)):
        result = wr.get('final_response') or 'No response'
        parts.append(
            f"[Worker {i}] Subtask: {st}\n"
            f"  Result: {result}\n"
        )
    parts.append("\nSynthesize these into a single final answer.")
    return "\n".join(parts)


def format_centralized_code_synthesis_prompt(question, subtasks, worker_results,
                                             worker_patches):
    """Orchestrator's final curation prompt for artifact-producing tasks (SWE).

    The orchestrator (lead) writes the final artifact itself using tools, in its
    OWN clean workspace, curating the team's work rather than shipping the raw
    union of their edits. Supports two team layouts:
      - shared repo (len(worker_patches) == 1): the team worked in one repository;
        the single combined diff is their accumulated edits.
      - isolated (len(worker_patches) == N): one candidate diff per worker.
    Worker rationales come from worker_results regardless of layout.
    """
    parts = [
        f"Task: {question}\n",
        "You are the lead engineer. Your team attempted this task; their combined "
        "edits have ALREADY BEEN APPLIED to your workspace as a starting point. Your "
        "job is to CURATE them into the single correct final fix using the tools. Do "
        "NOT start from scratch and do NOT just re-read the code.\n",
        "Guidance:\n"
        "- Run `git diff` to see the applied team edits, then use edit_file to remove "
        "wrong/spurious changes, fix bugs, and add anything missing.\n"
        "- The team edits are proposals, not final: your workspace already contains "
        "them, so you must actively correct them with edit_file, not just explore.\n"
        "- Verify with run_tests where possible, then call submit_patch to finalize.\n",
    ]
    parts.append("\n----- Worker reports -----")
    for i, st in enumerate(subtasks):
        rationale = (worker_results[i] or {}).get("final_response") or "No response"
        parts.append(f"[Worker {i}] Subtask: {st}\n  Summary: {rationale}")

    nonempty = [(j, (p or "").strip()) for j, p in enumerate(worker_patches)
                if (p or "").strip()]
    if len(worker_patches) == 1:
        combined = (worker_patches[0] or "").strip() or "(no edits produced)"
        parts.append(
            "\n----- Team's combined edits (shared repository) -----\n"
            f"{combined}\n"
        )
    else:
        parts.append("\n----- Candidate patches (one per worker) -----")
        if not nonempty:
            parts.append("(no worker produced a patch)")
        for j, patch in nonempty:
            parts.append(f"\n[Worker {j} candidate]\n{patch}\n")
    return "\n".join(parts)


def format_debate_prompt(agent_trajectories, exclude_idx,
                         comm_mode="full", terse_speaker=False):
    """Format the debate prompt showing other agents' work.

    Two orthogonal controls for the latent comm pilot:

    comm_mode — what the *receiver sees* about peers (Phase A):
      'full'          : original — tool calls + final answer
      'answer_only'   : final answer only, no tool calls
      'truncate100'   : first 100 chars of final answer
      'truncate300'   : first 300 chars of final answer
      'empty'         : peer slot present but content empty (control)
      'empty_silent'  : NO peer slot at all, NO mention of other agents — pure
                        self-refinement instruction. Strict-ablation control
                        for `empty`: rules out placeholder-interpretation
                        confound where '(output not shared)' might trigger
                        meaningful behavior change.

    terse_speaker — instructs the *receiver agent* (about to decode its own
    response) to keep its output minimal. This is Phase B-2: directly attack
    speaker-side decode, since Phase A established the verbose explanation
    isn't used downstream anyway. Combined with answer_only, the system is
    explicitly producing+consuming only answer text.
    """
    # Strict-ablation control: no peer block, no peer-related framing.
    # Equivalent compute (still goes through ReAct loop with tool access),
    # but the receiver has no signal that other agents exist. Tests whether
    # the 'empty' mode's accuracy is real or driven by the placeholder string
    # acting as an instruction.
    if comm_mode == "empty_silent":
        if terse_speaker:
            return (
                "Continue refining your answer to the task above. You may make "
                "additional tool calls if needed. Then output ONLY your final "
                "answer — no reasoning, no explanation, no commentary."
            )
        return (
            "Continue refining your answer to the task above. You may make "
            "additional tool calls to verify or improve your answer. "
            "Provide your updated final answer."
        )
    parts = []
    for i, traj in enumerate(agent_trajectories):
        if i == exclude_idx or traj is None:
            continue
        tool_summary = traj.get('tool_summary', 'None')
        final_resp = traj.get('final_response', 'No response')
        if comm_mode == "answer_only":
            parts.append(f"[Agent {i}] Final answer: {final_resp}\n")
        elif comm_mode == "truncate100":
            parts.append(f"[Agent {i}] Final answer: {final_resp[:100]}\n")
        elif comm_mode == "truncate300":
            parts.append(f"[Agent {i}] Final answer: {final_resp[:300]}\n")
        elif comm_mode == "empty":
            parts.append(f"[Agent {i}] (output not shared)\n")
        else:  # 'full'
            parts.append(
                f"[Agent {i}]:\n"
                f"  Tool calls: {tool_summary}\n"
                f"  Final answer: {final_resp}\n"
            )
    other_text = "\n".join(parts)
    instruction_default = (
        "Review their tool-calling strategies and results. "
        "If you believe your approach was correct, reaffirm it. "
        "If you see a better approach or errors in your work, "
        "make additional tool calls to correct or verify. "
        "Provide your updated final answer."
    )
    instruction_terse = (
        "Use any tool calls you need to verify. "
        "Then output ONLY your final answer — no reasoning, no explanation, "
        "no acknowledgement of other agents, no commentary. "
        "Just the answer."
    )
    instruction = instruction_terse if terse_speaker else instruction_default
    return (
        f"These are the approaches and results from other agents:\n\n"
        f"{other_text}\n\n"
        f"{instruction}"
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
    """Parse orchestrator decomposition into subtasks.

    Accepts multiple formats:
      SUBTASK: ...
      1. ...  /  1) ...
      - ...  /  * ...
      Worker 1: ...  /  Agent 1: ...
    Falls back to giving each worker the full task if parsing fails.
    """
    lines = orchestrator_text.strip().split("\n")
    subtasks = []

    for line in lines:
        line = line.strip()
        if not line:
            continue
        # SUBTASK: prefix
        if line.upper().startswith("SUBTASK:"):
            subtasks.append(line[len("SUBTASK:"):].strip())
        # Numbered: "1. ..." or "1) ..."
        elif re.match(r'^\d+[\.\)]\s+', line):
            subtasks.append(re.sub(r'^\d+[\.\)]\s+', '', line).strip())
        # Bullet: "- ..." or "* ..."
        elif re.match(r'^[-*•]\s+', line):
            subtasks.append(re.sub(r'^[-*•]\s+', '', line).strip())
        # "Worker N:" or "Agent N:"
        elif re.match(r'^(Worker|Agent|Sub-?agent)\s*\d*\s*:', line, re.IGNORECASE):
            subtasks.append(re.sub(r'^(Worker|Agent|Sub-?agent)\s*\d*\s*:\s*', '', line, flags=re.IGNORECASE).strip())

    # Filter out empty entries
    subtasks = [s for s in subtasks if len(s) > 3]

    if not subtasks:
        log.warning("Failed to parse subtasks from orchestrator output, using fallback")

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
