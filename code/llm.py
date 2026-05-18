"""SGLang client wrapper with ReAct loop for agentic tool calling.

All LLM calls go through SGLang's OpenAI-compatible HTTP API.
Tool calling is client-managed: SGLang returns tool_calls, we execute
tools locally, append results, and send the next request.
"""

import json
from openai import OpenAI, BadRequestError
from config import (
    SGLANG_URL, SGLANG_API_KEY, MAX_TOKENS, MAX_REACT_STEPS, BASE_SEED,
    SGLANG_CONTEXT_LENGTH, SAVE_TRANSCRIPTS,
)


def _serialize_messages(messages):
    """Compact JSON-serializable view of a messages list, suitable for embedding
    in a call_record. Truncates very long content to avoid runaway file sizes."""
    out = []
    for m in messages:
        item = {"role": m.get("role", "?")}
        content = m.get("content")
        if isinstance(content, str):
            item["content"] = content[:8000]
        elif content is not None:
            item["content"] = str(content)[:8000]
        if "tool_calls" in m and m["tool_calls"]:
            item["tool_calls"] = [
                {
                    "id": tc.get("id"),
                    "name": tc.get("function", {}).get("name"),
                    "arguments": (tc.get("function", {}).get("arguments") or "")[:2000],
                }
                for tc in m["tool_calls"]
            ]
        if "tool_call_id" in m:
            item["tool_call_id"] = m["tool_call_id"]
        out.append(item)
    return out

CHARS_PER_TOKEN = 3  # conservative (real ~3.5); overestimate tokens to prevent overflow


def make_client(base_url=None, timeout=120):
    return OpenAI(
        base_url=base_url or SGLANG_URL,
        api_key=SGLANG_API_KEY,
        timeout=timeout,
    )


def _estimate_tokens(messages):
    """Rough token count: ~3 chars per token (conservative), plus overhead per message."""
    total = 0
    for m in messages:
        content = m.get("content") or ""
        total += len(content) // CHARS_PER_TOKEN + 4
        for tc in m.get("tool_calls", []):
            fn = tc.get("function", {})
            total += len(fn.get("arguments", "")) // CHARS_PER_TOKEN + 10
    return total


def _truncate_messages(messages, max_context_tokens=None):
    """Drop oldest tool-calling rounds to fit within context window.

    Preserves: system message, initial user message, most recent messages.
    Drops oldest (assistant w/ tool_calls + tool results) groups from the
    middle first. Keeps conversation structure valid (tool results always
    have a matching tool_call in a preceding assistant message).
    """
    # Reserve extra 5% safety margin beyond MAX_TOKENS for estimation error
    limit = max_context_tokens or SGLANG_CONTEXT_LENGTH
    available = int(limit * 0.95) - MAX_TOKENS
    if _estimate_tokens(messages) <= available:
        return messages

    # Parse messages into segments:
    # - "tool_round": assistant(tool_calls) + all following tool messages
    # - "other": any other single message (system, user, assistant text)
    segments = []
    i = 0
    while i < len(messages):
        msg = messages[i]
        if msg.get("role") == "assistant" and msg.get("tool_calls"):
            group = [msg]
            j = i + 1
            while j < len(messages) and messages[j].get("role") == "tool":
                group.append(messages[j])
                j += 1
            segments.append(("tool_round", group))
            i = j
        else:
            segments.append(("other", [msg]))
            i += 1

    # Keep first 2 segments (system + user prompt), drop oldest tool_rounds
    prefix_count = min(2, len(segments))
    prefix_segs = segments[:prefix_count]
    rest_segs = list(segments[prefix_count:])

    while rest_segs:
        flat = [m for _, seg_msgs in prefix_segs + rest_segs for m in seg_msgs]
        if _estimate_tokens(flat) <= available:
            break
        # Drop the oldest tool_round from rest
        dropped = False
        for idx, (seg_type, _) in enumerate(rest_segs):
            if seg_type == "tool_round":
                rest_segs.pop(idx)
                dropped = True
                break
        if not dropped:
            # No more tool_rounds to drop; drop oldest "other" (except feedback)
            rest_segs.pop(0)

    result = [m for _, seg_msgs in prefix_segs + rest_segs for m in seg_msgs]
    if len(result) < len(messages):
        print(f"  Context truncation: {len(messages)}→{len(result)} messages "
              f"(~{_estimate_tokens(messages)}→{_estimate_tokens(result)} tokens)")
    return result


def chat(client, model, messages, temperature=0.0, seed=None,
         max_tokens=None, extra_body=None):
    """Single LLM call (no tool calling). For synthesis, orchestrator, etc.

    Returns (response_text, usage_dict). On context overflow, returns
    ("", empty_usage) after logging the error.

    When SAVE_TRANSCRIPTS=True, the returned usage dict additionally contains
    a `_transcript` key with the request messages and response text. The
    caller (in topologies.py) is responsible for forwarding this into the
    energy_monitor.stop() metadata so it lands in the call_record.
    """
    messages = _truncate_messages(messages)

    kwargs = dict(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens or MAX_TOKENS,
        seed=seed if seed is not None else BASE_SEED,
    )
    if extra_body:
        kwargs["extra_body"] = extra_body

    try:
        resp = client.chat.completions.create(**kwargs)
    except BadRequestError as e:
        print(f"  chat() context overflow ({_estimate_tokens(messages)} est. tokens): {e}")
        return "", {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    text = resp.choices[0].message.content or ""
    usage = {
        "prompt_tokens": resp.usage.prompt_tokens,
        "completion_tokens": resp.usage.completion_tokens,
        "total_tokens": resp.usage.total_tokens,
    }
    if SAVE_TRANSCRIPTS:
        usage["_transcript"] = {
            "request_messages": _serialize_messages(messages),
            "response_text": text[:8000],
        }
    return text, usage


def react_loop(client, model, messages, tools, execute_tool, energy_monitor,
               max_steps=None, temperature=0.0, seed=None, agent_id="agent_0",
               extra_body=None, mid_stream_injections=None):
    """Client-managed ReAct loop for SGLang tool calling.

    Each LLM inference call and each tool execution is individually
    wrapped with energy measurement for clean attribution.

    Args:
        client: OpenAI client pointing at SGLang
        model: model path string
        messages: initial message list (system + user)
        tools: list of tool schemas (OpenAI function calling format)
        execute_tool: callable(name, args) -> result string/dict
        energy_monitor: EnergyMonitor instance
        max_steps: max tool-calling rounds (default MAX_REACT_STEPS)
        temperature: sampling temperature
        seed: random seed
        agent_id: identifier for energy record attribution
        extra_body: model-specific params (e.g. enable_thinking)

    Returns dict:
        messages: full conversation history
        final_response: model's final text (or None if overflow)
        steps: number of tool-calling steps taken
        call_records: list of per-step energy records
        total_usage: aggregated token counts
    """
    max_steps = max_steps or MAX_REACT_STEPS
    call_records = []
    total_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    valid_tool_names = {t["function"]["name"] for t in tools} if tools else set()

    # mid_stream_injections: dict mapping step_number -> message_content_string
    # At the specified step, a user message is inserted into the conversation
    # BEFORE the LLM call, simulating mid-stream peer communication.
    _injections = mid_stream_injections or {}

    for step in range(max_steps):
        # --- Mid-stream injection (e.g. peer info during debate) ---
        if step in _injections:
            messages.append({
                "role": "user",
                "content": _injections[step],
            })

        # --- Truncate if context has grown too large ---
        messages = _truncate_messages(messages)

        # --- LLM inference (energy-measured) ---
        energy_monitor.start()
        kwargs = dict(
            model=model,
            messages=messages,
            tools=tools,
            tool_choice="auto",
            temperature=temperature,
            max_tokens=MAX_TOKENS,
            seed=seed if seed is not None else BASE_SEED,
        )
        if extra_body:
            kwargs["extra_body"] = extra_body

        try:
            resp = client.chat.completions.create(**kwargs)
        except BadRequestError as e:
            # Context overflow despite truncation — return partial results
            energy_monitor.stop(metadata={
                "agent_id": agent_id,
                "call_type": f"react_step_{step}_overflow",
            })
            print(f"  Context overflow at step {step} ({agent_id}): {e}")
            return {
                "messages": messages,
                "final_response": None,
                "steps": step,
                "call_records": call_records,
                "total_usage": total_usage,
                "overflow": True,
            }

        msg_for_record = resp.choices[0].message
        step_metadata = {
            "agent_id": agent_id,
            "call_type": f"react_step_{step}",
            "prompt_tokens": resp.usage.prompt_tokens,
            "completion_tokens": resp.usage.completion_tokens,
            "total_tokens": resp.usage.total_tokens,
        }
        if SAVE_TRANSCRIPTS:
            step_metadata["request_messages"] = _serialize_messages(messages)
            tc_list = msg_for_record.tool_calls or []
            step_metadata["response"] = {
                "content": (msg_for_record.content or "")[:8000],
                "tool_calls": [
                    {
                        "name": tc.function.name,
                        "arguments": (tc.function.arguments or "")[:2000],
                    }
                    for tc in tc_list
                ],
            }
        record = energy_monitor.stop(metadata=step_metadata)
        call_records.append(record)
        total_usage["prompt_tokens"] += resp.usage.prompt_tokens
        total_usage["completion_tokens"] += resp.usage.completion_tokens
        total_usage["total_tokens"] += resp.usage.total_tokens

        msg = resp.choices[0].message

        # No tool calls → model produced final text response
        if not msg.tool_calls:
            messages.append({"role": "assistant", "content": msg.content or ""})
            return {
                "messages": messages,
                "final_response": msg.content or "",
                "steps": step + 1,
                "call_records": call_records,
                "total_usage": total_usage,
            }

        # Check for hallucinated tool names (e.g. "finish") — treat as termination
        if all(tc.function.name not in valid_tool_names for tc in msg.tool_calls):
            final_text = msg.content or ""
            print(f"  {agent_id} step {step}: hallucinated tools "
                  f"{[tc.function.name for tc in msg.tool_calls]}, treating as final response")
            messages.append({"role": "assistant", "content": final_text})
            return {
                "messages": messages,
                "final_response": final_text,
                "steps": step + 1,
                "call_records": call_records,
                "total_usage": total_usage,
            }

        # --- Append assistant's tool call(s) to history ---
        # Filter out any hallucinated tool calls from the list
        real_tool_calls = [tc for tc in msg.tool_calls if tc.function.name in valid_tool_names]
        messages.append({
            "role": "assistant",
            "content": msg.content,
            "tool_calls": [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in real_tool_calls
            ],
        })

        # --- Execute each tool call (energy-measured separately) ---
        for tc in real_tool_calls:
            try:
                args = json.loads(tc.function.arguments)
            except json.JSONDecodeError:
                args = {"raw": tc.function.arguments}

            energy_monitor.start()
            try:
                result = execute_tool(tc.function.name, args)
            except Exception as e:
                result = f"Error executing {tc.function.name}: {e}"
            tool_metadata = {
                "agent_id": agent_id,
                "call_type": "tool_execution",
                "tool_name": tc.function.name,
            }
            if SAVE_TRANSCRIPTS:
                result_preview = result if isinstance(result, str) else json.dumps(result)
                tool_metadata["tool_call"] = {
                    "name": tc.function.name,
                    "arguments": json.dumps(args)[:2000],
                    "result": result_preview[:4000],
                }
            tool_record = energy_monitor.stop(metadata=tool_metadata)
            call_records.append(tool_record)

            result_str = json.dumps(result) if not isinstance(result, str) else result
            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": result_str,
            })

    # Exceeded max_steps — force a final text response (no tools)
    messages = _truncate_messages(messages)
    energy_monitor.start()
    kwargs_final = dict(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=MAX_TOKENS,
        seed=seed if seed is not None else BASE_SEED,
    )
    if extra_body:
        kwargs_final["extra_body"] = extra_body

    try:
        resp = client.chat.completions.create(**kwargs_final)
        final_text = resp.choices[0].message.content or ""
        record = energy_monitor.stop(metadata={
            "agent_id": agent_id,
            "call_type": "react_wrapup",
            "prompt_tokens": resp.usage.prompt_tokens,
            "completion_tokens": resp.usage.completion_tokens,
            "total_tokens": resp.usage.total_tokens,
        })
        call_records.append(record)
        total_usage["prompt_tokens"] += resp.usage.prompt_tokens
        total_usage["completion_tokens"] += resp.usage.completion_tokens
        total_usage["total_tokens"] += resp.usage.total_tokens
        messages.append({"role": "assistant", "content": final_text})
    except BadRequestError:
        energy_monitor.stop(metadata={
            "agent_id": agent_id,
            "call_type": "react_wrapup_overflow",
        })
        final_text = None

    return {
        "messages": messages,
        "final_response": final_text,
        "steps": max_steps,
        "call_records": call_records,
        "total_usage": total_usage,
    }


def warmup(client, model, n=5, extra_body=None):
    """Send warmup calls to stabilize GPU clocks."""
    kwargs = dict(
        model=model,
        messages=[{"role": "user", "content": "Hello"}],
        max_tokens=10,
    )
    if extra_body:
        kwargs["extra_body"] = extra_body
    for _ in range(n):
        client.chat.completions.create(**kwargs)
