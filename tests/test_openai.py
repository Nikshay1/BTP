"""OpenAI transport (D32) against a mocked HTTP server. No network, no key."""

import asyncio
import json

import httpx
import pytest

from bioalloc.budget import Account, MeteredClient
from bioalloc.llm.base import LLMRequest, Message, ProviderThrottled, RateLimited
from bioalloc.llm.openai import MODEL, OpenAITransport, is_openai_model
from bioalloc.schemas import Budget

OK = {
    "id": "chatcmpl-1",
    "model": MODEL,
    "choices": [
        {"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": "{}"}}
    ],
    "usage": {
        "prompt_tokens": 29,
        "completion_tokens": 27,
        "completion_tokens_details": {"reasoning_tokens": 13},
        "prompt_tokens_details": {"cached_tokens": 0},
    },
}


def transport(handler) -> tuple[OpenAITransport, list[httpx.Request]]:  # type: ignore[no-untyped-def]
    seen: list[httpx.Request] = []

    def wrapped(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return handler(request)  # type: ignore[no-any-return]

    client = httpx.AsyncClient(transport=httpx.MockTransport(wrapped))
    return OpenAITransport("test-key", client=client), seen


def req() -> LLMRequest:
    return LLMRequest(
        model=MODEL,
        messages=(Message(role="user", content="hi"),),
        max_completion_tokens=256,
        temperature=1.0,
        seed=7,
        reasoning_effort="low",
    )


def test_body_goes_to_openai_without_groq_only_fields() -> None:
    t, seen = transport(lambda r: httpx.Response(200, json=OK))
    reply = asyncio.run(t.send(req()))
    sent = json.loads(seen[0].content)
    assert seen[0].url == "https://api.openai.com/v1/chat/completions"
    assert seen[0].headers["authorization"] == "Bearer test-key"
    assert "include_reasoning" not in sent
    assert sent["model"] == MODEL and sent["reasoning_effort"] == "low" and sent["seed"] == 7
    assert reply.usage.completion_tokens == 27 and reply.usage.reasoning_tokens == 13


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
