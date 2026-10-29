"""Groq transport against a mocked HTTP server. No network, no key.

The reply fixtures are hand-written from Groq's documented OpenAI-compatible schema, not recordings
of the live API.
"""

import asyncio
import json
from typing import Any

import httpx
import pytest

from bioalloc.budget import Account, MeteredClient
from bioalloc.llm.base import (
    ConnectionFailed,
    LLMError,
    LLMRequest,
    Message,
    ProviderThrottled,
    RateLimited,
    RequestRejected,
    ToolCall,
    TransportTimeout,
)
from bioalloc.llm.groq import MODEL, GroqTransport, wire_request
from bioalloc.schemas import Budget


def ok_body(**over: Any) -> dict[str, Any]:
    body: dict[str, Any] = {
        "id": "chatcmpl-1",
        "model": MODEL,
        "choices": [
            {
                "index": 0,
                "finish_reason": "stop",
                "message": {"role": "assistant", "content": '{"answer": 5}'},
            }
        ],
        "usage": {
            "prompt_tokens": 120,
            "completion_tokens": 40,
            "completion_tokens_details": {"reasoning_tokens": 25},
            "prompt_tokens_details": {"cached_tokens": 0},
        },
    }
    body.update(over)
    return body


def transport(handler) -> tuple[GroqTransport, list[httpx.Request]]:  # type: ignore[no-untyped-def]
    seen: list[httpx.Request] = []

    def wrapped(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return handler(request)  # type: ignore[no-any-return]

    client = httpx.AsyncClient(transport=httpx.MockTransport(wrapped))
    return GroqTransport("test-key", client=client), seen


def req(**over: Any) -> LLMRequest:
    base: dict[str, Any] = dict(
        model=MODEL,
        messages=(
            Message(role="system", content="sys"),
            Message(role="user", content="hi", source="w0"),
        ),
        max_completion_tokens=256,
        temperature=0.6,
        seed=42,
        sample_id="direct:a:0/1",
        reasoning_effort="low",
    )
    base.update(over)
    return LLMRequest(**base)


def test_request_body_maps_fields_and_drops_internal_metadata() -> None:
    t, seen = transport(lambda r: httpx.Response(200, json=ok_body()))
    asyncio.run(t.send(req()))
    sent = json.loads(seen[0].content)
    assert seen[0].url == "https://api.groq.com/openai/v1/chat/completions"
    assert seen[0].headers["authorization"] == "Bearer test-key"
    assert sent == {
        "model": MODEL,
        "messages": [{"role": "system", "content": "sys"}, {"role": "user", "content": "hi"}],
        "max_completion_tokens": 256,
        "temperature": 0.6,
        "stream": False,
        "n": 1,
        "include_reasoning": False,
        "seed": 42,
        "reasoning_effort": "low",
    }


def test_reply_usage_counts_reasoning_once() -> None:
    t, _ = transport(lambda r: httpx.Response(200, json=ok_body(), headers={"x-request-id": "rq"}))
    reply = asyncio.run(t.send(req()))
    assert reply.content == '{"answer": 5}' and reply.finish_reason == "stop"
    assert (reply.usage.prompt_tokens, reply.usage.completion_tokens) == (120, 40)
    assert reply.usage.reasoning_tokens == 25 and reply.request_id == "rq"


def test_missing_reasoning_details_are_unknown_not_zero() -> None:
    body = ok_body(usage={"prompt_tokens": 5, "completion_tokens": 3})
    t, _ = transport(lambda r: httpx.Response(200, json=body))
    assert asyncio.run(t.send(req())).usage.reasoning_tokens is None


@pytest.mark.parametrize(
    "body",
    [
        ok_body(usage=None),
        {"choices": []},
        ok_body(usage={"prompt_tokens": 5}),
        ok_body(
            usage={
                "prompt_tokens": 5,
                "completion_tokens": 3,
                "completion_tokens_details": {"reasoning_tokens": 9},
            }
        ),
    ],
)
def test_malformed_replies_raise_instead_of_inventing_usage(body: dict[str, Any]) -> None:
    t, _ = transport(lambda r: httpx.Response(200, json=body))
    with pytest.raises(LLMError):
        asyncio.run(t.send(req()))


def test_tool_calls_round_trip() -> None:
    body = ok_body(
        choices=[
            {
                "index": 0,
                "finish_reason": "tool_calls",
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_9",
                            "type": "function",
                            "function": {"name": "add", "arguments": '{"a":1,"b":2}'},
                        }
                    ],
                },
            }
        ]
    )
    t, seen = transport(lambda r: httpx.Response(200, json=body))
    reply = asyncio.run(t.send(req(tools=({"type": "function", "function": {"name": "add"}},))))
    assert reply.content == "" and reply.finish_reason == "tool_calls"
    assert reply.tool_calls == (ToolCall(id="call_9", name="add", arguments='{"a":1,"b":2}'),)
    assert json.loads(seen[0].content)["tool_choice"] == "auto"
    readback = req(
        messages=(
            Message(role="user", content="q"),
            Message(role="assistant", content="", tool_calls=reply.tool_calls),
            Message(role="tool", content="3", tool_call_id="call_9"),
        )
    )
    msgs = wire_request(readback)["messages"]
    assert msgs[1]["tool_calls"][0]["function"]["arguments"] == '{"a":1,"b":2}'
    assert msgs[2] == {"role": "tool", "content": "3", "tool_call_id": "call_9"}


def test_unknown_finish_reason_is_other() -> None:
    body = ok_body(
        choices=[
            {"index": 0, "finish_reason": "weird", "message": {"role": "assistant", "content": "x"}}
        ]
    )
    t, _ = transport(lambda r: httpx.Response(200, json=body))
    assert asyncio.run(t.send(req())).finish_reason == "other"


@pytest.mark.parametrize(
    ("status", "error"),
    [
        (400, RequestRejected),
        (401, RequestRejected),
        (403, RequestRejected),
        (500, LLMError),
        (503, LLMError),
    ],
)
def test_http_errors_map_to_ledger_classes(status: int, error: type[Exception]) -> None:
    t, _ = transport(lambda r: httpx.Response(status, text="nope"))
    with pytest.raises(error) as info:
        asyncio.run(t.send(req()))
    if status >= 500:
        assert not isinstance(info.value, RequestRejected)


@pytest.mark.parametrize(
    ("header", "expected"), [("7", 7.0), ("0.5", 0.5), ("soon", None), (None, None)]
)
def test_rate_limit_carries_retry_after(header: str | None, expected: float | None) -> None:
    headers = {"retry-after": header} if header is not None else {}
    t, _ = transport(lambda r: httpx.Response(429, headers=headers))
    with pytest.raises(RateLimited) as info:
        asyncio.run(t.send(req()))
    assert info.value.retry_after == expected


def test_timeouts_and_connection_failures() -> None:
    def timeout(r: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("slow", request=r)

    def refused(r: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=r)

    with pytest.raises(TransportTimeout):
        asyncio.run(transport(timeout)[0].send(req()))
    with pytest.raises(ConnectionFailed):
        asyncio.run(transport(refused)[0].send(req()))


def metered(handler, budget: Budget | None = None):  # type: ignore[no-untyped-def]
    t, seen = transport(handler)
    sleeps: list[float] = []
    events: list[tuple[str, dict]] = []  # type: ignore[type-arg]

    async def fake_sleep(s: float) -> None:
        sleeps.append(s)

    account = Account(budget or Budget(max_tokens=10_000, max_model_calls=5))
    client = MeteredClient(
        t, account, "worker", "w", sleep=fake_sleep, on_event=lambda k, d: events.append((k, d))
    )
    return client, account, seen, sleeps, events


def test_rate_limit_is_retried_without_charge_then_succeeds() -> None:
    replies = iter(
        [
            httpx.Response(429, headers={"retry-after": "2"}),
            httpx.Response(429),
            httpx.Response(200, json=ok_body()),
        ]
    )
    client, account, seen, sleeps, events = metered(lambda r: next(replies))
    reply = asyncio.run(client.complete(req()))
    assert reply.usage.completion_tokens == 40
    assert len(seen) == 3 and sleeps == [2.0, 2.0]  # Retry-After, then backoff 2**1
    cost = account.snapshot()
    assert cost.model_calls == 1 and cost.usage_complete and account.reserved_tokens == 0
    assert [k for k, _ in events].count("rate_limited") == 2


def test_rate_limit_retries_are_bounded() -> None:
    client, account, seen, sleeps, _ = metered(lambda r: httpx.Response(429))
    # Past every retry the provider is unavailable, not the task failed (R18).
    with pytest.raises(ProviderThrottled):
        asyncio.run(client.complete(req()))
    assert len(seen) == 4 and sleeps == [1.0, 2.0, 4.0]
    assert account.snapshot().model_calls == 0


def test_rate_limit_wait_respects_the_deadline() -> None:
    client, account, seen, sleeps, _ = metered(
        lambda r: httpx.Response(429, headers={"retry-after": "60"}),
        Budget(max_tokens=10_000, max_model_calls=5, max_elapsed_s=10),
    )
    with pytest.raises(ProviderThrottled):
        asyncio.run(client.complete(req()))
    assert sleeps == [] and account.snapshot().model_calls == 0


def test_server_error_is_charged_as_unknown_usage_and_not_resent() -> None:
    client, account, seen, sleeps, _ = metered(lambda r: httpx.Response(502))
    with pytest.raises(LLMError):
        asyncio.run(client.complete(req()))
    assert len(seen) == 1 and sleeps == []
    cost = account.snapshot()
    assert cost.model_calls == 1 and not cost.usage_complete


def test_api_key_is_never_written_to_the_trace() -> None:
    client, _, _, _, events = metered(lambda r: httpx.Response(200, json=ok_body()))
    asyncio.run(client.complete(req()))
    assert "test-key" not in json.dumps([d for _, d in events])
