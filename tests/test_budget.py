import asyncio

import pytest

from bioalloc.budget import Account, BudgetExceeded, MeteredClient, Pricing, communication_tokens
from bioalloc.llm.base import LLMError, LLMRequest, Message, RequestRejected, TransportTimeout
from bioalloc.llm.fake import FakeTransport
from bioalloc.schemas import Budget


def req(text: str = "hello", max_completion: int = 20, source: str | None = None) -> LLMRequest:
    return LLMRequest(
        model="m",
        messages=(Message(role="user", content=text, source=source),),
        max_completion_tokens=max_completion,
    )


def client(fake: FakeTransport, tokens: int = 1000, calls: int = 10, **kw: object) -> MeteredClient:
    account = Account(Budget(max_tokens=tokens, max_model_calls=calls))
    return MeteredClient(fake, account, actor="worker", agent="w1", **kw)  # type: ignore[arg-type]


def test_call_is_charged_from_reported_usage() -> None:
    c = client(FakeTransport().always("ok"), pricing=Pricing(1.0, 2.0))
    asyncio.run(c.complete(req("x" * 40)))
    cost = c.account.snapshot()
    assert cost.worker.prompt == 10 and cost.worker.completion == 1
    assert cost.coordinator.total == 0
    assert cost.model_calls == 1
    assert cost.usd == pytest.approx((10 * 1.0 + 1 * 2.0) / 1e6)
    assert c.account.reserved_tokens == 0


def test_oversized_call_is_never_sent() -> None:
    fake = FakeTransport().always("ok")
    c = client(fake, tokens=50)
    with pytest.raises(BudgetExceeded):
        asyncio.run(c.complete(req("x" * 300)))
    assert fake.requests == []
    assert c.account.snapshot().model_calls == 0


def test_call_limit() -> None:
    fake = FakeTransport().always("a", "b")
    c = client(fake, calls=1)
    asyncio.run(c.complete(req()))
    with pytest.raises(BudgetExceeded, match="call limit"):
        asyncio.run(c.complete(req()))
    assert len(fake.requests) == 1


def test_child_spending_counts_against_parent_and_is_capped_by_it() -> None:
    parent = client(FakeTransport().always("ok", "ok"), tokens=200)
    child = parent.child(Budget(max_tokens=10_000, max_model_calls=10))
    assert child.account.remaining().max_tokens == 200
    asyncio.run(child.complete(req()))
    assert parent.account.snapshot().tokens == child.account.snapshot().tokens
    assert child.account.snapshot().tokens.total > 0
    assert child.account.remaining().max_tokens == parent.account.remaining().max_tokens


def test_concurrent_reservations_cannot_overspend_parent() -> None:
    parent = Account(Budget(max_tokens=100, max_model_calls=10))
    a = parent.child(Budget(max_tokens=100, max_model_calls=10))
    b = parent.child(Budget(max_tokens=100, max_model_calls=10))
    a.reserve(30, 30)
    with pytest.raises(BudgetExceeded):
        b.reserve(30, 30)
    # It would fit once the call in flight settles, so the client waits rather than fails.
    assert b.would_fit_after_settling(60)


def test_failed_attempts_consume_budget_and_timeouts_charge_the_reservation() -> None:
    fake = FakeTransport().always(TransportTimeout("slow"), "ok")
    c = client(fake)
    with pytest.raises(TransportTimeout):
        asyncio.run(c.complete(req("hi", max_completion=20)))
    cost = c.account.snapshot()
    assert cost.model_calls == 1 and cost.tokens.total > 20
    assert cost.usage_complete is False
    asyncio.run(c.complete(req()))
    assert c.account.snapshot().model_calls == 2


def test_rejected_request_releases_reservation_without_charge() -> None:
    c = client(FakeTransport().always(RequestRejected("bad request")))
    with pytest.raises(RequestRejected):
        asyncio.run(c.complete(req()))
    assert c.account.reserved_tokens == 0 and c.account.snapshot().model_calls == 0


def test_other_errors_are_charged_as_unknown_usage() -> None:
    c = client(FakeTransport().always(LLMError("connection reset")))
    with pytest.raises(LLMError):
        asyncio.run(c.complete(req()))
    cost = c.account.snapshot()
    assert c.account.reserved_tokens == 0
    assert cost.model_calls == 1 and cost.usage_complete is False


def test_time_limit() -> None:
    now = [0.0]
    account = Account(
        Budget(max_tokens=1000, max_model_calls=10, max_elapsed_s=5), clock=lambda: now[0]
    )
    account.reserve(5, 5)
    now[0] = 6.0
    assert account.exhausted()
    with pytest.raises(BudgetExceeded, match="time"):
        account.reserve(5, 5)


def test_overrun_blocks_further_calls() -> None:
    account = Account(Budget(max_tokens=100, max_model_calls=10))
    rid = account.reserve(80, 10)
    account.settle(rid, "worker", prompt=95, completion=10, communication=0, reasoning=0, usd=0)
    assert account.overrun and account.exhausted()
    assert account.snapshot().overrun
    with pytest.raises(BudgetExceeded, match="overran"):
        account.reserve(1, 0)


def test_communication_share_follows_provenance() -> None:
    msgs = (
        Message(role="system", content="a" * 50),
        Message(role="user", content="b" * 50, source="w2"),
        Message(role="assistant", content="c" * 100, source="w1"),
    )
    r = LLMRequest(model="m", messages=msgs, max_completion_tokens=1)
    assert communication_tokens(r, prompt_tokens=100, agent="w1") == 25
    assert communication_tokens(r, prompt_tokens=100, agent="w2") == 50


def test_events_are_emitted() -> None:
    events: list[str] = []
    c = client(FakeTransport().always("ok"), on_event=lambda k, d: events.append(k))
    asyncio.run(c.complete(req(source="coordinator")))
    assert events == ["llm_request", "llm_reply"]
