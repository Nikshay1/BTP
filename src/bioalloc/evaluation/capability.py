"""A tiny live check of the provider before the pilot (docs/pilot.md, step 2).

Up to four requests, a few hundred tokens in all: a JSON answer, the same request again with the
same seed, and a tool round trip with the calculator. It reports what the adapters assume but
could not verify offline: that usage is reported (and whether reasoning tokens are itemized),
the finish reasons, whether the seed reproduces the reply, and that tool calls and their read-back
work.
"""

from __future__ import annotations

import json
from typing import Any, Literal

from bioalloc.llm.base import LLMRequest, Message, Transport
from bioalloc.tasks.common import CALCULATOR

MODEL = "openai/gpt-oss-120b"


async def capability_check(
    transport: Transport, model: str = MODEL, reasoning_effort: Literal["low"] | None = "low"
) -> dict[str, Any]:
    ask = LLMRequest(
        model=model,
        messages=(
            Message(
                role="system", content='Reply with a single JSON object: {"answer": <number>}.'
            ),
            Message(role="user", content="What is 17 + 25?"),
        ),
        max_completion_tokens=256,
        temperature=0.6,
        seed=12345,
        reasoning_effort=reasoning_effort,
    )
    first = await transport.send(ask)
    second = await transport.send(ask.model_copy(update={"sample_id": "repeat"}))
    tool_ask = LLMRequest(
        model=model,
        messages=(
            Message(role="system", content="Use the calculate tool, then reply with the result."),
            Message(role="user", content="What is 123 * 47?"),
        ),
        max_completion_tokens=256,
        temperature=0.6,
        seed=12345,
        reasoning_effort=reasoning_effort,
        tools=(CALCULATOR.spec.wire(),),
    )
    tool_reply = await transport.send(tool_ask)
    readback = None
    if tool_reply.tool_calls:
        tc = tool_reply.tool_calls[0]
        value = CALCULATOR.run(json.loads(tc.arguments or "{}"))
        readback = await transport.send(
            tool_ask.model_copy(
                update={
                    "messages": (
                        *tool_ask.messages,
                        Message(
                            role="assistant",
                            content=tool_reply.content,
                            tool_calls=tool_reply.tool_calls,
                        ),
                        Message(role="tool", content=json.dumps(value), tool_call_id=tc.id),
                    ),
                    "sample_id": "readback",
                }
            )
        )
    replies = [r for r in (first, second, tool_reply, readback) if r is not None]
    return {
        "model_returned": first.model,
        "json_answer": first.content,
        "finish_reasons": [r.finish_reason for r in replies],
        "usage": [r.usage.model_dump() for r in replies],
        "reasoning_tokens_itemized": all(r.usage.reasoning_tokens is not None for r in replies),
        "same_seed_same_reply": first.content == second.content,
        "tool_called": bool(tool_reply.tool_calls),
        "tool_readback": readback.content if readback else None,
        "total_tokens": sum(r.usage.prompt_tokens + r.usage.completion_tokens for r in replies),
    }
