"""SGLang client wrapper with ReAct loop for agentic tool calling.

All LLM calls go through SGLang's OpenAI-compatible HTTP API.
Tool calling is client-managed: SGLang returns tool_calls, we execute
tools locally, append results, and send the next request.
"""

import json
from openai import OpenAI, BadRequestError
from config import (
    SGLANG_URL, SGLANG_API_KEY, MAX_TOKENS, MAX_REACT_STEPS, BASE_SEED,
)


def make_client(base_url=None, timeout=120):
    return OpenAI(
        base_url=base_url or SGLANG_URL,
        api_key=SGLANG_API_KEY,
        timeout=timeout,
    )


def chat(client, model, messages, temperature=0.0, seed=None,
         max_tokens=None, extra_body=None):
    """Single LLM call (no tool calling). For synthesis, orchestrator, etc.

    Returns (response_text, usage_dict).
    """
    kwargs = dict(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens or MAX_TOKENS,
        seed=seed if seed is not None else BASE_SEED,
    )
    if extra_body:
        kwargs["extra_body"] = extra_body

    resp = client.chat.completions.create(**kwargs)
    text = resp.choices[0].message.content or ""
    usage = {
        "prompt_tokens": resp.usage.prompt_tokens,
        "completion_tokens": resp.usage.completion_tokens,
        "total_tokens": resp.usage.total_tokens,
    }
    return text, usage


def react_loop(client, model, messages, tools, execute_tool, energy_monitor,
               max_steps=None, temperature=0.0, seed=None, agent_id="agent_0",
               extra_body=None):
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
        final_response: model's final text (or None if max_steps hit)
        steps: number of tool-calling steps taken
        call_records: list of per-step energy records
        total_usage: aggregated token counts
    """
    max_steps = max_steps or MAX_REACT_STEPS
    call_records = []
    total_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    for step in range(max_steps):
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
            # Context overflow — stop the energy measurement and return partial results
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

        record = energy_monitor.stop(metadata={
            "agent_id": agent_id,
            "call_type": f"react_step_{step}",
            "prompt_tokens": resp.usage.prompt_tokens,
            "completion_tokens": resp.usage.completion_tokens,
            "total_tokens": resp.usage.total_tokens,
        })
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

        # --- Append assistant's tool call(s) to history ---
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
                for tc in msg.tool_calls
            ],
        })

        # --- Execute each tool call (energy-measured separately) ---
        for tc in msg.tool_calls:
            try:
                args = json.loads(tc.function.arguments)
            except json.JSONDecodeError:
                args = {"raw": tc.function.arguments}

            energy_monitor.start()
            try:
                result = execute_tool(tc.function.name, args)
            except Exception as e:
                result = f"Error executing {tc.function.name}: {e}"
            tool_record = energy_monitor.stop(metadata={
                "agent_id": agent_id,
                "call_type": "tool_execution",
                "tool_name": tc.function.name,
            })
            call_records.append(tool_record)

            result_str = json.dumps(result) if not isinstance(result, str) else result
            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": result_str,
            })

    # Exceeded max_steps — extract whatever the model last said
    return {
        "messages": messages,
        "final_response": None,
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
