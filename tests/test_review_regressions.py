"""Regression tests for the findings of the M1 code review (2026-10-26, GPT-6.1 Sol)."""

import asyncio
import json
from pathlib import Path

import pytest
from conftest import SlowTransport

from bioalloc.budget import Account, CachedReplyInMeasurement, MeteredClient, Pricing
from bioalloc.coordinator import Coordinator
from bioalloc.evaluation.runner import run_toy
from bioalloc.evaluation.toy_data import make_instance, optimal_legs
from bioalloc.executors.base import ExecutionContext, ExecutorOutcome, subtask_messages
from bioalloc.executors.direct import Direct
from bioalloc.graph import TaskGraph
from bioalloc.llm.base import LLMReply, LLMRequest, Message, TransportTimeout, Usage
from bioalloc.llm.fake import FakeTransport
from bioalloc.policies import Constant, Uniform
from bioalloc.schemas import Budget, Subtask
from bioalloc.trace import Tracer

INSTANCE = make_instance(seed=0)
LEG1, LEG2 = optimal_legs(INSTANCE)


def req(text: str = "q", max_completion: int = 50) -> LLMRequest:
    return LLMRequest(
        model="m",
        messages=(Message(role="user", content=text),),
        max_completion_tokens=max_completion,
    )


def is_leg(n: int):  # type: ignore[no-untyped-def]
    marker = "Find the cheapest route from start" if n == 1 else "Continue from the node"
    return lambda r: marker in r.messages[1].content


class Echo:
    """Trivial family: every output passes; the answer is all outputs by id."""

    name = "echo"

    def checks(self, subtask: Subtask):  # type: ignore[no-untyped-def]
        return []

    def aggregate(self, graph: TaskGraph, outputs):  # type: ignore[no-untyped-def]
        return dict(outputs)


# Finding 1: siblings contend for reservations, not for budget.
def test_siblings_wait_for_reservations_instead_of_failing(tmp_path: Path) -> None:
    graph = TaskGraph(
        id="siblings",
        nodes=(Subtask(id="a", instruction="a"), Subtask(id="b", instruction="b")),
        final_nodes=("a", "b"),
    )
    tiny = LLMReply(content="{}", usage=Usage(prompt_tokens=1, completion_tokens=1), model="m")
    fake = FakeTransport().always(tiny, tiny)
    llm = MeteredClient(
        SlowTransport(fake),
        Account(Budget(max_tokens=400, max_model_calls=10)),
        "coordinator",
        "coordinator",
    )
    # Each Direct call reserves roughly 300 tokens, so only one fits at a time.
    coord = Coordinator(
        {"direct": Direct(model="m", max_completion_tokens=200)}, Constant("direct")
    )
    tracer = Tracer(tmp_path, "r")
    result = asyncio.run(coord.run(graph, Echo(), llm, tracer))
    assert result.status == "completed", result.attempts
    assert result.cost.model_calls == 2 and result.cost.tokens.total == 4


# Finding 2: wall-time limits bound the wait for a reply.
def test_deadline_cuts_off_a_slow_reply() -> None:
    fake = FakeTransport().always("late")
    account = Account(Budget(max_tokens=10_000, max_model_calls=5, max_elapsed_s=0.02))
    c = MeteredClient(SlowTransport(fake, delay=1.0), account, "worker", "w")
    with pytest.raises(TransportTimeout):
        asyncio.run(c.complete(req()))
    cost = account.snapshot()
    assert cost.model_calls == 1 and not cost.usage_complete and account.reserved_tokens == 0


# Finding 3: cancellation settles the reservation and leaves a trace event.
def test_cancellation_settles_reservation() -> None:
    events: list[str] = []
    account = Account(Budget(max_tokens=10_000, max_model_calls=5))
    c = MeteredClient(
        SlowTransport(FakeTransport().always("x"), delay=1.0),
        account,
        "worker",
        "w",
        on_event=lambda k, d: events.append(k),
    )

    async def go() -> None:
        task = asyncio.create_task(c.complete(req()))
        await asyncio.sleep(0.01)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(go())
    assert account.reserved_tokens == 0 and account.reserved_calls == 0
    assert account.snapshot().model_calls == 1
    assert events == ["llm_request", "llm_error"]


# Finding 4: malformed output is rejected and retried, never crashes.
@pytest.mark.parametrize("bad", ['{"route": []}', '{"route": [1, 2]}', "[]", "null", '"A"'])
def test_malformed_route_is_rejected_and_retried(tmp_path: Path, bad: str) -> None:
    fake = FakeTransport().on(is_leg(1), bad, json.dumps({"route": LEG1}))
    fake.on(is_leg(2), json.dumps({"route": LEG2}))
    out = run_toy(0, "good", tmp_path, transport=fake)
    assert out.result.status == "completed" and out.grade.correct
    assert [a.accepted for a in out.result.attempts] == [False, True, True]


# Finding 5: an overrun is visible in the result.
def test_overrun_is_reported(tmp_path: Path) -> None:
    huge = LLMReply(
        content=json.dumps({"route": LEG1}),
        usage=Usage(prompt_tokens=5_000, completion_tokens=10),
        model="m",
    )
    fake = FakeTransport().on(is_leg(1), huge).on(is_leg(2), json.dumps({"route": LEG2}))
    out = run_toy(
        0, "good", tmp_path, transport=fake, budget=Budget(max_tokens=3_000, max_model_calls=10)
    )
    assert out.result.cost.overrun and not out.result.within_budget
    assert out.result.status == "budget_exhausted"


# Finding 6: a run never appends to another run's trace.
def test_runs_get_distinct_directories_and_traces_are_exclusive(tmp_path: Path) -> None:
    a = run_toy(0, "good", tmp_path)
    b = run_toy(0, "good", tmp_path)
    assert a.run_dir != b.run_dir
    with pytest.raises(FileExistsError):
        Tracer(a.run_dir, "again")


# Finding 7: a ready descendant does not wait for an unrelated slow sibling.
def test_descendant_starts_before_slow_sibling_finishes(tmp_path: Path) -> None:
    graph = TaskGraph(
        id="waves",
        nodes=(
            Subtask(id="a", instruction="fast a"),
            Subtask(id="b", instruction="after a", depends_on=("a",)),
            Subtask(id="c", instruction="slow c"),
        ),
        final_nodes=("b", "c"),
    )
    order: list[str] = []
    fake = FakeTransport().always("{}", "{}", "{}")

    class Recording(SlowTransport):
        async def send(self, request: LLMRequest) -> LLMReply:
            reply = await super().send(request)
            order.append(request.messages[1].content.removeprefix("Subtask: "))
            return reply

    slow = Recording(fake, delay=0.01, delays=[(lambda r: "slow c" in r.messages[1].content, 0.2)])
    llm = MeteredClient(
        slow, Account(Budget(max_tokens=50_000, max_model_calls=10)), "coordinator", "coordinator"
    )
    coord = Coordinator({"direct": Direct(model="m")}, Constant("direct"))
    asyncio.run(coord.run(graph, Echo(), llm, Tracer(tmp_path, "r")))
    assert order == ["fast a", "after a", "slow c"]


# Finding 8: measurement mode refuses cached replies at the client.
def test_measurement_mode_rejects_cached_replies() -> None:
    cached = LLMReply(
        content="x", usage=Usage(prompt_tokens=1, completion_tokens=1), model="m", from_cache=True
    )
    account = Account(Budget(max_tokens=1000, max_model_calls=5))
    c = MeteredClient(FakeTransport().always(cached), account, "worker", "w", mode="measurement")
    with pytest.raises(CachedReplyInMeasurement):
        asyncio.run(c.complete(req()))
    assert account.snapshot().model_calls == 0


# Finding 12: unknown usage is charged as prompt estimate plus maximum completion.
def test_timeout_charges_completion_at_output_rate() -> None:
    account = Account(Budget(max_tokens=10_000, max_model_calls=5))
    c = MeteredClient(
        FakeTransport().always(TransportTimeout("t")),
        account,
        "worker",
        "w",
        pricing=Pricing(1.0, 10.0),
    )
    with pytest.raises(TransportTimeout):
        asyncio.run(c.complete(req("x" * 30, max_completion=50)))
    cost = account.snapshot()
    assert cost.worker.completion == 50
    assert cost.usd == pytest.approx((cost.worker.prompt * 1.0 + 50 * 10.0) / 1e6)


# Finding 10 (decision D8): a retry is told why the previous answer was rejected.
def test_retry_receives_rejected_output_and_reasons(tmp_path: Path) -> None:
    bad = json.dumps({"route": [LEG1[0], "Z", LEG1[-1]]})
    fake = FakeTransport().on(is_leg(1), bad, json.dumps({"route": LEG1}))
    fake.on(is_leg(2), json.dumps({"route": LEG2}))
    run_toy(0, "good", tmp_path, transport=fake)
    first, retry = fake.requests[0], fake.requests[1]
    assert len(retry.messages) == len(first.messages) + 2
    assert retry.messages[-2].source == "direct:leg1:0" and bad in retry.messages[-2].content
    assert "missing edge" in retry.messages[-1].content and retry.messages[-1].source is None


# Finding 11: model seeds and random draws come from stable identities.
def test_seeds_are_stable_and_order_independent(tmp_path: Path) -> None:
    def seeds() -> list[int | None]:
        fake = FakeTransport().on(is_leg(1), json.dumps({"route": LEG1}))
        fake.on(is_leg(2), json.dumps({"route": LEG2}))
        run_toy(0, "good", tmp_path, transport=fake)
        return [r.seed for r in fake.requests]

    first = seeds()
    assert None not in first and first == seeds() and len(set(first)) == 2
    policy = Uniform(["x", "y", "z"], seed=1)
    s = Subtask(id="a", instruction="a")
    assert [policy.select(s, []) for _ in range(5)] == [policy.select(s, [])] * 5


# Finding 13: raw task inputs are never counted as communication.
def test_inputs_are_not_communication_even_with_model_written_instruction() -> None:
    sub = Subtask(id="a", instruction="short", inputs={"big": "x" * 1000}, written_by="decomposer")
    ctx = ExecutionContext(llm=None, span_id="s")  # type: ignore[arg-type]
    msgs = subtask_messages("sys", sub, ctx)
    sources = {m.content[:9]: m.source for m in msgs}
    assert sources["Subtask: "] == "decomposer"
    assert all(m.source is None for m in msgs if "big" in m.content)


def test_executor_outcome_default_status() -> None:
    assert ExecutorOutcome(output=1).status == "completed"
