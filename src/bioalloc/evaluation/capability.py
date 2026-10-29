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

from bioalloc.budget import reserve_prompt_tokens
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
    readback_ask = tool_ask
    if tool_reply.tool_calls:
        # Every call gets its result, or the provider rejects the unanswered ones.
        results = tuple(
            Message(
                role="tool",
                content=json.dumps(CALCULATOR.run(json.loads(tc.arguments or "{}"))),
                tool_call_id=tc.id,
            )
            for tc in tool_reply.tool_calls
        )
        readback_ask = tool_ask.model_copy(
            update={
                "messages": (
                    *tool_ask.messages,
                    Message(
                        role="assistant",
                        content=tool_reply.content,
                        tool_calls=tool_reply.tool_calls,
                    ),
                    *results,
                ),
                "sample_id": "readback",
            }
        )
        readback = await transport.send(readback_ask)
    sent = [first, second, tool_reply, readback]
    replies = [r for r in sent if r is not None]
    requests = [ask, ask, tool_ask, readback_ask]
    problems = []
    if not tool_reply.tool_calls:
        problems.append("the model did not call the tool")
    if readback is not None and "5781" not in readback.content.replace(",", ""):
        problems.append("the read-back reply does not contain 123 * 47 = 5781")
    if any(r.finish_reason == "length" for r in replies):
        problems.append("a reply was cut off at the completion limit")
    return {
        "ok": not problems,
        "problems": problems,
        # The ledger's reservation must not fall below real prompt usage (it is charged in full
        # when usage is unknown).
        "prompt_estimate_vs_actual": [
            [reserve_prompt_tokens(q), r.usage.prompt_tokens]
            for q, r in zip(requests, sent, strict=True)
            if r is not None
        ],
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
