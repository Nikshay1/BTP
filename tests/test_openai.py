"""OpenAI Responses transport (D32) against a mocked HTTP server. No network, no key.

The reply fixtures follow the shapes seen from the live API on 14 November 2026.
"""

import asyncio
import json
from typing import Any

import httpx
import pytest

from bioalloc.budget import Account, MeteredClient
from bioalloc.llm.base import (
    LLMError,
    LLMRequest,
    Message,
    ProviderThrottled,
    RateLimited,
    ToolCall,
)
from bioalloc.llm.openai import MODEL, OpenAITransport, is_openai_model
from bioalloc.schemas import Budget
from bioalloc.tasks.common import CALCULATOR


def reply(output: list[dict[str, Any]], **over: Any) -> dict[str, Any]:
    body: dict[str, Any] = {
        "id": "resp_1",
        "model": MODEL,
        "status": "completed",
        "output": output,
        "usage": {
            "input_tokens": 63,
            "input_tokens_details": {"cached_tokens": 0},
            "output_tokens": 220,
            "output_tokens_details": {"reasoning_tokens": 198},
        },
    }
    body.update(over)
    return body


TEXT = {
    "type": "message",
    "role": "assistant",
    "content": [{"type": "output_text", "text": '{"answer": 5}'}],
}
CALL = {
    "type": "function_call",
    "id": "fc_1",
    "call_id": "call_1",
    "name": "calculate",
    "arguments": '{"expression": "123 * 47"}',
}


def transport(handler) -> tuple[OpenAITransport, list[httpx.Request]]:  # type: ignore[no-untyped-def]
    seen: list[httpx.Request] = []

    def wrapped(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return handler(request)  # type: ignore[no-any-return]

    client = httpx.AsyncClient(transport=httpx.MockTransport(wrapped))
    return OpenAITransport("test-key", client=client), seen


def req(**over: Any) -> LLMRequest:
    base: dict[str, Any] = dict(
        model=MODEL,
        messages=(
            Message(role="system", content="sys"),
            Message(role="user", content="hi", source="w0"),
        ),
        max_completion_tokens=256,
        temperature=1.0,
        seed=7,
        reasoning_effort="low",
    )
    base.update(over)
    return LLMRequest(**base)


def test_body_maps_a_tool_conversation_to_input_items() -> None:
    t, seen = transport(lambda r: httpx.Response(200, json=reply([TEXT])))
    call = ToolCall(id="call_1", name="calculate", arguments='{"expression": "1+1"}')
    messages = (
        Message(role="system", content="sys"),
        Message(role="user", content="hi"),
        Message(role="assistant", content="", tool_calls=(call,)),
        Message(role="tool", content='{"result": 2}', tool_call_id="call_1"),
    )
    asyncio.run(t.send(req(messages=messages, tools=(CALCULATOR.spec.wire(),))))
    sent = json.loads(seen[0].content)
    assert seen[0].url == "https://api.openai.com/v1/responses"
    assert seen[0].headers["authorization"] == "Bearer test-key"
    assert sent["input"] == [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "hi"},
        {
            "type": "function_call",
            "call_id": "call_1",
            "name": "calculate",
            "arguments": '{"expression": "1+1"}',
        },
        {"type": "function_call_output", "call_id": "call_1", "output": '{"result": 2}'},
    ]
    assert sent["tools"][0]["name"] == "calculate" and "function" not in sent["tools"][0]
    assert sent["reasoning"] == {"effort": "low"} and sent["max_output_tokens"] == 256
    assert sent["store"] is False and sent["tool_choice"] == "auto"
    assert "seed" not in sent and "include_reasoning" not in sent


def test_reply_text_usage_and_reasoning_counted_once() -> None:
    t, _ = transport(lambda r: httpx.Response(200, json=reply([{"type": "reasoning"}, TEXT])))
    out = asyncio.run(t.send(req()))
    assert out.content == '{"answer": 5}' and out.finish_reason == "stop"
    assert (out.usage.prompt_tokens, out.usage.completion_tokens) == (63, 220)
    assert out.usage.reasoning_tokens == 198 and out.request_id == "resp_1"


def test_function_calls_become_tool_calls() -> None:
    t, _ = transport(lambda r: httpx.Response(200, json=reply([{"type": "reasoning"}, CALL])))
    out = asyncio.run(t.send(req()))
    assert out.finish_reason == "tool_calls"
    assert out.tool_calls == (
        ToolCall(id="call_1", name="calculate", arguments='{"expression": "123 * 47"}'),
    )


def test_cut_off_reply_is_length() -> None:
    body = reply([], status="incomplete", incomplete_details={"reason": "max_output_tokens"})
    t, _ = transport(lambda r: httpx.Response(200, json=body))
    assert asyncio.run(t.send(req())).finish_reason == "length"


@pytest.mark.parametrize(
    "body",
    [reply([TEXT], usage=None), reply([TEXT], status="failed"), {"output": []}],
)
def test_malformed_or_failed_replies_raise(body: dict[str, Any]) -> None:
    t, _ = transport(lambda r: httpx.Response(200, json=body))
    with pytest.raises(LLMError):
        asyncio.run(t.send(req()))


def test_spent_credit_is_a_quota_and_a_plain_429_is_not() -> None:
    def err(code: str) -> httpx.Response:
        return httpx.Response(429, json={"error": {"code": code, "message": "x"}})

    t, _ = transport(lambda r: err("insufficient_quota"))
    with pytest.raises(RateLimited) as quota:
        asyncio.run(t.send(req()))
    assert quota.value.quota
    t, _ = transport(lambda r: err("rate_limit_exceeded"))
    with pytest.raises(RateLimited) as rate:
        asyncio.run(t.send(req()))
    assert not rate.value.quota


def test_spent_credit_ends_the_run_without_retrying() -> None:
    t, seen = transport(
        lambda r: httpx.Response(429, json={"error": {"code": "insufficient_quota"}})
    )
    account = Account(Budget(max_tokens=10_000, max_model_calls=5))
    slept: list[float] = []

    async def sleep(s: float) -> None:
        slept.append(s)

    client = MeteredClient(t, account, "worker", "w", sleep=sleep)
    with pytest.raises(ProviderThrottled):
        asyncio.run(client.complete(req()))
    assert len(seen) == 1 and slept == [] and account.root.throttled


@pytest.mark.parametrize(
    ("model", "expected"),
    [
        ("gpt-6-luna", True),
        ("gpt-6.1-sol", True),
        ("openai/gpt-oss-120b", False),
        ("global.anthropic.claude-sonnet-4-6", False),
    ],
)
def test_model_routing(model: str, expected: bool) -> None:
    assert is_openai_model(model) is expected


def test_key_is_required() -> None:
    with pytest.raises(ValueError):
        OpenAITransport("")
