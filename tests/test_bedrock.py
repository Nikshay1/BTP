"""Bedrock transport against a stand-in boto3 client. No network, no credentials.

The reply fixtures are hand-written from the documented Anthropic Messages schema, not recordings
of the live API.
"""

import asyncio
import io
import json
import math
from typing import Any

import pytest
from botocore.exceptions import ClientError, EndpointConnectionError, ReadTimeoutError

from bioalloc.budget import (
    TOOL_OVERHEAD_TOKENS,
    Account,
    MeteredClient,
    Pricing,
    reserve_prompt_tokens,
)
from bioalloc.cli import is_bedrock_model
from bioalloc.evaluation.capability import capability_check
from bioalloc.llm.base import (
    ConnectionFailed,
    LLMError,
    LLMRequest,
    Message,
    RateLimited,
    RequestRejected,
    ToolCall,
    TransportTimeout,
)
from bioalloc.llm.bedrock import MODEL, BedrockTransport, wire_request
from bioalloc.schemas import Budget
from bioalloc.tasks.common import CALCULATOR


def ok_body(**over: Any) -> dict[str, Any]:
    body: dict[str, Any] = {
        "id": "msg_1",
        "model": "claude-sonnet-4-6",
        "type": "message",
        "role": "assistant",
        "content": [{"type": "text", "text": '{"answer": 5}'}],
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 120, "output_tokens": 40},
    }
    body.update(over)
    return body


def client_error(code: str, status: int) -> ClientError:
    return ClientError(
        {
            "Error": {"Code": code, "Message": "nope"},
            "ResponseMetadata": {"HTTPStatusCode": status},
        },
        "InvokeModel",
    )


class FakeBedrock:
    """Records each invoke_model call; ``handler`` gets the parsed body and returns a reply body
    or raises."""

    def __init__(self, handler: Any) -> None:
        self.handler = handler
        self.seen: list[dict[str, Any]] = []

    def invoke_model(self, **kwargs: Any) -> dict[str, Any]:
        body = json.loads(kwargs["body"])
        self.seen.append({"modelId": kwargs["modelId"], **body})
        out = self.handler(body)
        return {
            "body": io.BytesIO(json.dumps(out).encode()),
            "ResponseMetadata": {"RequestId": "req-1"},
        }


def transport(handler: Any) -> tuple[BedrockTransport, FakeBedrock]:
    fake = FakeBedrock(handler)
    return BedrockTransport(client=fake), fake


def req(**over: Any) -> LLMRequest:
    base: dict[str, Any] = {
        "model": MODEL,
        "messages": (
            Message(role="system", content="Be brief."),
            Message(role="user", content="2+3?", source=None),
        ),
        "max_completion_tokens": 256,
        "temperature": 0.6,
        "seed": 7,
        "reasoning_effort": "low",
    }
    base.update(over)
    return LLMRequest(**base)


def test_request_body_lifts_system_and_sends_effort_but_no_seed() -> None:
    body = wire_request(req(tools=(CALCULATOR.spec.wire(),)))
    assert body["anthropic_version"] == "bedrock-2023-05-31"
    assert body["system"] == "Be brief."
    assert body["messages"] == [{"role": "user", "content": [{"type": "text", "text": "2+3?"}]}]
    assert body["output_config"] == {"effort": "low"}
    assert body["max_tokens"] == 256 and body["temperature"] == 0.6
    assert "seed" not in body and "thinking" not in body
    assert body["tools"][0]["name"] == "calculate"
    assert body["tools"][0]["input_schema"] == CALCULATOR.spec.parameters
    assert body["tool_choice"] == {"type": "auto"}


def test_no_effort_field_when_unset() -> None:
    assert "output_config" not in wire_request(req(reasoning_effort=None))


def test_tool_results_of_one_turn_go_back_in_one_user_message() -> None:
    calls = (
        ToolCall(id="t1", name="calculate", arguments='{"expression": "1+1"}'),
        ToolCall(id="t2", name="calculate", arguments='{"expression": "2+2"}'),
    )
    messages = (
        *req().messages,
        Message(role="assistant", content="", tool_calls=calls),
        Message(role="tool", content="2", tool_call_id="t1"),
        Message(role="tool", content="4", tool_call_id="t2"),
        Message(role="user", content="Now add them."),
    )
    body = wire_request(req(messages=messages))
    assert [m["role"] for m in body["messages"]] == ["user", "assistant", "user"]
    assistant = body["messages"][1]["content"]
    assert [b["type"] for b in assistant] == ["tool_use", "tool_use"]
    assert assistant[0]["input"] == {"expression": "1+1"}
    results = body["messages"][2]["content"]
    assert [b["type"] for b in results] == ["tool_result", "tool_result", "text"]
    assert [b.get("tool_use_id") for b in results[:2]] == ["t1", "t2"]


def test_unsupported_requests_are_rejected_before_sending() -> None:
    t, fake = transport(lambda b: ok_body())
    late_system = (*req().messages, Message(role="system", content="late"))
    for bad in (req(temperature=1.5), req(messages=late_system)):
        with pytest.raises(RequestRejected):
            asyncio.run(t.send(bad))
    assert fake.seen == []


def test_reply_with_text_and_tool_use_is_parsed() -> None:
    body = ok_body(
        content=[
            {"type": "text", "text": "Let me compute."},
            {
                "type": "tool_use",
                "id": "toolu_1",
                "name": "calculate",
                "input": {"expression": "6*7"},
            },
        ],
        stop_reason="tool_use",
        usage={"input_tokens": 100, "output_tokens": 30, "cache_read_input_tokens": 20},
    )
    t, fake = transport(lambda b: body)
    reply = asyncio.run(t.send(req()))
    assert fake.seen[0]["modelId"] == MODEL
    assert reply.content == "Let me compute."
    assert reply.tool_calls == (
        ToolCall(id="toolu_1", name="calculate", arguments='{"expression": "6*7"}'),
    )
    assert reply.finish_reason == "tool_calls"
    # Cache reads are billed prompt tokens and are counted once.
    assert reply.usage.prompt_tokens == 120 and reply.usage.cached_prompt_tokens == 20
    assert reply.usage.completion_tokens == 30 and reply.usage.reasoning_tokens is None
    assert reply.request_id == "req-1"


@pytest.mark.parametrize(
    ("stop", "finish"),
    [("end_turn", "stop"), ("max_tokens", "length"), ("refusal", "content_filter"), ("x", "other")],
)
def test_finish_reasons(stop: str, finish: str) -> None:
    t, _ = transport(lambda b: ok_body(stop_reason=stop))
    assert asyncio.run(t.send(req())).finish_reason == finish


def test_missing_usage_is_an_error_not_zero() -> None:
    body = ok_body()
    del body["usage"]
    t, _ = transport(lambda b: body)
    with pytest.raises(LLMError) as info:
        asyncio.run(t.send(req()))
    assert not isinstance(info.value, RequestRejected)


def raising(exc: Exception) -> Any:
    def handler(body: dict[str, Any]) -> Any:
        raise exc

    return handler


@pytest.mark.parametrize(
    ("exc", "expected", "rejected"),
    [
        (client_error("ThrottlingException", 429), RateLimited, True),
        (client_error("ServiceQuotaExceededException", 400), RateLimited, True),
        (client_error("ValidationException", 400), RequestRejected, True),
        (client_error("AccessDeniedException", 403), RequestRejected, True),
        (client_error("ModelTimeoutException", 408), LLMError, False),
        (client_error("InternalServerException", 500), LLMError, False),
        (EndpointConnectionError(endpoint_url="https://x"), ConnectionFailed, True),
        (ReadTimeoutError(endpoint_url="https://x"), TransportTimeout, False),
    ],
)
def test_errors_are_mapped_for_the_ledger(exc: Exception, expected: type, rejected: bool) -> None:
    t, _ = transport(raising(exc))
    with pytest.raises(expected) as info:
        asyncio.run(t.send(req()))
    assert isinstance(info.value, RequestRejected) is rejected


def test_throttling_is_retried_by_the_metered_client_without_charge() -> None:
    replies = iter([client_error("ThrottlingException", 429), None])

    def handler(body: dict[str, Any]) -> Any:
        exc = next(replies)
        if exc is not None:
            raise exc
        return ok_body()

    t, fake = transport(handler)
    sleeps: list[float] = []

    async def fake_sleep(s: float) -> None:
        sleeps.append(s)

    account = Account(Budget(max_tokens=10_000, max_model_calls=5))
    client = MeteredClient(t, account, "worker", "w", sleep=fake_sleep)
    reply = asyncio.run(client.complete(req()))
    assert reply.usage.completion_tokens == 40
    assert len(fake.seen) == 2 and sleeps == [1.0]  # no retry-after on Bedrock: backoff from 1 s
    cost = account.snapshot()
    assert cost.model_calls == 1 and cost.usage_complete and account.reserved_tokens == 0


def test_capability_check_against_a_stand_in_bedrock() -> None:
    def handler(body: dict[str, Any]) -> Any:
        last = body["messages"][-1]["content"][-1]
        if body.get("tools") and last["type"] != "tool_result":
            return ok_body(
                content=[
                    {
                        "type": "tool_use",
                        "id": "toolu_1",
                        "name": "calculate",
                        "input": {"expression": "123*47"},
                    },
                ],
                stop_reason="tool_use",
            )
        if body.get("tools"):
            return ok_body(content=[{"type": "text", "text": "5781"}])
        return ok_body(content=[{"type": "text", "text": '{"answer": 42}'}])

    t, fake = transport(handler)
    report = asyncio.run(capability_check(t, model=MODEL))
    assert len(fake.seen) == 4
    assert report["tool_called"] and report["tool_readback"] == "5781"
    assert report["finish_reasons"] == ["stop", "stop", "tool_calls", "stop"]
    readback = fake.seen[3]["messages"]
    assert readback[-1]["content"][0]["type"] == "tool_result"
    assert readback[-1]["content"][0]["tool_use_id"] == "toolu_1"


def test_bedrock_models_are_recognised() -> None:
    assert is_bedrock_model(MODEL)
    assert not is_bedrock_model("openai/gpt-oss-120b")


SONNET = Pricing(input_per_m=3.0, output_per_m=15.0)


def metered(handler: Any) -> tuple[MeteredClient, Account, FakeBedrock, list[float]]:
    t, fake = transport(handler)
    sleeps: list[float] = []

    async def fake_sleep(s: float) -> None:
        sleeps.append(s)

    account = Account(Budget(max_tokens=10_000, max_model_calls=5))
    client = MeteredClient(t, account, "worker", "w", pricing=SONNET, sleep=fake_sleep)
    return client, account, fake, sleeps


def sequence(*steps: Any) -> Any:
    it = iter(steps)

    def handler(body: dict[str, Any]) -> Any:
        step = next(it)
        if isinstance(step, Exception):
            raise step
        return step

    return handler


def test_success_after_throttling_charges_exactly_the_reported_usage() -> None:
    client, account, fake, sleeps = metered(
        sequence(
            client_error("ThrottlingException", 429),
            client_error("ModelNotReadyException", 429),
            ok_body(),
        )
    )
    asyncio.run(client.complete(req()))
    cost = account.snapshot()
    assert len(fake.seen) == 3 and sleeps == [1.0, 2.0]
    assert cost.model_calls == 1 and cost.usage_complete and account.reserved_tokens == 0
    assert cost.usd == pytest.approx((120 * 3.0 + 40 * 15.0) / 1e6)


@pytest.mark.parametrize(
    "exc",
    [
        client_error("ModelErrorException", 424),
        client_error("ModelTimeoutException", 408),
        client_error("InternalServerException", 500),
        ReadTimeoutError(endpoint_url="https://x"),
    ],
)
def test_possibly_billed_failures_charge_the_full_reservation(exc: Exception) -> None:
    client, account, fake, sleeps = metered(sequence(exc))
    request = req(tools=(CALCULATOR.spec.wire(),))
    with pytest.raises(LLMError):
        asyncio.run(client.complete(request))
    cost = account.snapshot()
    assert len(fake.seen) == 1 and sleeps == []  # never resent
    assert not cost.usage_complete and account.reserved_tokens == 0
    reserved = reserve_prompt_tokens(request)
    assert cost.usd == pytest.approx((reserved * 3.0 + 256 * 15.0) / 1e6)


def test_reservation_covers_the_hidden_tool_prompt() -> None:
    with_tools = req(tools=(CALCULATOR.spec.wire(),))
    assert TOOL_OVERHEAD_TOKENS >= 350  # Anthropic's documented hidden tool prompt
    spec_tokens = math.ceil(len(json.dumps(CALCULATOR.spec.wire())) / 3)
    assert reserve_prompt_tokens(with_tools) == (
        reserve_prompt_tokens(req()) + spec_tokens + TOOL_OVERHEAD_TOKENS
    )


def test_capability_check_answers_every_parallel_tool_call() -> None:
    def handler(body: dict[str, Any]) -> Any:
        last = body["messages"][-1]["content"]
        if body.get("tools") and last[-1]["type"] != "tool_result":
            return ok_body(
                content=[
                    {
                        "type": "tool_use",
                        "id": "a",
                        "name": "calculate",
                        "input": {"expression": "123*47"},
                    },
                    {
                        "type": "tool_use",
                        "id": "b",
                        "name": "calculate",
                        "input": {"expression": "47*123"},
                    },
                ],
                stop_reason="tool_use",
            )
        if body.get("tools"):
            return ok_body(content=[{"type": "text", "text": "5,781"}])
        return ok_body(content=[{"type": "text", "text": '{"answer": 42}'}])

    t, fake = transport(handler)
    report = asyncio.run(capability_check(t, model=MODEL))
    results = fake.seen[3]["messages"][-1]
    assert results["role"] == "user"
    assert [b["tool_use_id"] for b in results["content"]] == ["a", "b"]
    assert report["ok"], report["problems"]
    assert len(report["prompt_estimate_vs_actual"]) == 4


def test_capability_check_fails_without_a_tool_call() -> None:
    t, _ = transport(lambda b: ok_body(content=[{"type": "text", "text": "5781"}]))
    report = asyncio.run(capability_check(t, model=MODEL))
    assert not report["ok"] and "did not call the tool" in report["problems"][0]
