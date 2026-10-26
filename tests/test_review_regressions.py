"""Regression tests for the findings of the M1 code review (2026-10-26, GPT-6.1 Sol)."""

import asyncio
import json
from pathlib import Path

import pytest
from conftest import SlowTransport

from bioalloc.budget import Account, CachedReplyInMeasurement, MeteredClient, Pricing
from bioalloc.coordinator import Coordinator, CoordinatorConfig
from bioalloc.evaluation.runner import run_toy
from bioalloc.evaluation.toy_data import make_instance, optimal_legs
from bioalloc.executors.base import ExecutionContext, ExecutorOutcome, subtask_messages
from bioalloc.executors.direct import Direct
from bioalloc.graph import TaskGraph
from bioalloc.llm.base import LLMReply, LLMRequest, Message, TransportTimeout, Usage
from bioalloc.llm.fake import FakeTransport
from bioalloc.policies import Constant, Uniform
from bioalloc.schemas import Budget, Subtask
from bioalloc.tasks.toy import ToyRoutes
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
    assert len(retry.messages) == len(first.messages) + 3
    wrapper, output, reasons = retry.messages[-3:]
    assert wrapper.source is None
    assert output.source == "direct:leg1:0" and output.content == bad
    assert "missing edge" in reasons.content and reasons.source is None


# Finding 11: model seeds and random draws come from stable identities, not completion order.
def test_seeds_do_not_depend_on_completion_order(tmp_path: Path) -> None:
    graph = TaskGraph(
        id="order",
        nodes=(Subtask(id="a", instruction="a"), Subtask(id="b", instruction="b")),
        final_nodes=("a", "b"),
    )

    def seeds(slow_one: str) -> dict[str, int | None]:
        fake = FakeTransport().always("{}", "{}")
        slow = SlowTransport(
            fake,
            delay=0.001,
            delays=[(lambda r: r.messages[1].content == f"Subtask: {slow_one}", 0.05)],
        )
        llm = MeteredClient(
            slow,
            Account(Budget(max_tokens=50_000, max_model_calls=10)),
            "coordinator",
            "coordinator",
        )
        coord = Coordinator({"direct": Direct(model="m")}, Constant("direct"), seed=7)
        asyncio.run(coord.run(graph, Echo(), llm, Tracer(tmp_path / slow_one, "r")))
        return {r.messages[1].content: r.seed for r in fake.requests}

    a_slow, b_slow = seeds("a"), seeds("b")
    assert a_slow == b_slow and None not in a_slow.values() and len(set(a_slow.values())) == 2
    policy = Uniform(["x", "y", "z"], seed=1)
    s = Subtask(id="a", instruction="a")
    assert [policy.select("t", s, []) for _ in range(5)] == [policy.select("t", s, [])] * 5
    draws = {policy.select(f"task{i}", s, []) for i in range(30)}
    assert len(draws) > 1  # the same subtask id in different tasks gets independent draws


# Finding 13: raw task inputs are never counted as communication.
def test_inputs_are_not_communication_even_with_model_written_instruction() -> None:
    sub = Subtask(id="a", instruction="short", inputs={"big": "x" * 1000}, written_by="decomposer")
    ctx = ExecutionContext(llm=None, span_id="s")  # type: ignore[arg-type]
    msgs = subtask_messages("sys", sub, ctx)
    sources = {m.content[:9]: m.source for m in msgs}
    assert sources["Subtask: "] == "decomposer"
    assert all(m.source is None for m in msgs if "big" in m.content)


# Finding 9: the policy receives the structured attempt record, not a collapsed reward.
def test_policy_receives_structured_attempt_records(tmp_path: Path) -> None:
    seen = []

    class Spy(Constant):
        def update(self, subtask, attempt):  # type: ignore[no-untyped-def]
            seen.append(attempt)

    bad = json.dumps({"route": [LEG1[0], "Z", LEG1[-1]]})
    fake = FakeTransport().on(is_leg(1), bad, json.dumps({"route": LEG1}))
    fake.on(is_leg(2), json.dumps({"route": LEG2}))
    llm = MeteredClient(
        fake, Account(Budget(max_tokens=50_000, max_model_calls=10)), "coordinator", "coordinator"
    )
    coord = Coordinator({"direct": Direct(model="m")}, Spy("direct"))
    asyncio.run(coord.run(INSTANCE.graph, ToyRoutes(), llm, Tracer(tmp_path, "r")))
    first = seen[0]
    assert (first.subtask_id, first.attempt, first.accepted, first.verdict) == (
        "leg1",
        0,
        False,
        "fail",
    )
    assert {c.check for c in first.checks} == {"schema", "route"} and first.tokens > 0
    assert [a.accepted for a in seen] == [False, True, True]


# Re-review: cancelling a run stops its workers and settles their reservations.
def test_cancelling_a_run_stops_workers(tmp_path: Path) -> None:
    graph = TaskGraph(id="c", nodes=(Subtask(id="a", instruction="a"),), final_nodes=("a",))
    account = Account(Budget(max_tokens=50_000, max_model_calls=10))
    llm = MeteredClient(
        SlowTransport(FakeTransport().always("{}"), delay=1.0),
        account,
        "coordinator",
        "coordinator",
    )
    coord = Coordinator({"direct": Direct(model="m")}, Constant("direct"))

    async def go() -> None:
        run = asyncio.create_task(coord.run(graph, Echo(), llm, Tracer(tmp_path, "r")))
        await asyncio.sleep(0.02)
        loop = asyncio.get_running_loop()
        start = loop.time()
        run.cancel()
        with pytest.raises(asyncio.CancelledError):
            await run
        # Workers are cancelled, not waited for (the transport would take a second).
        assert loop.time() - start < 0.5
        others = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
        assert others == []

    asyncio.run(go())
    assert account.reserved_tokens == 0 and account.reserved_calls == 0


# Re-review: an executor bug in one branch is raised, and every sibling is stopped and observed.
def test_executor_exceptions_propagate_and_siblings_are_stopped(tmp_path: Path) -> None:
    graph = TaskGraph(
        id="e",
        nodes=(Subtask(id="a", instruction="a"), Subtask(id="b", instruction="b")),
        final_nodes=("a", "b"),
    )

    class Broken:
        name = "broken"

        async def execute(self, subtask, ctx):  # type: ignore[no-untyped-def]
            raise RuntimeError(f"bug in {subtask.id}")

    llm = MeteredClient(
        FakeTransport(),
        Account(Budget(max_tokens=1000, max_model_calls=5)),
        "coordinator",
        "coordinator",
    )
    coord = Coordinator({"broken": Broken()}, Constant("broken"))
    with pytest.raises(RuntimeError, match="bug in"):
        asyncio.run(coord.run(graph, Echo(), llm, Tracer(tmp_path, "r")))


# Re-review: a call cut off by the deadline before reaching the transport is not charged.
def test_unsent_call_is_not_charged() -> None:
    now = [0.0]

    class Stall:
        async def send(self, request):  # type: ignore[no-untyped-def]
            raise AssertionError("must not be sent")

    account = Account(
        Budget(max_tokens=10_000, max_model_calls=5, max_elapsed_s=1.0), clock=lambda: now[0]
    )
    events: list[dict] = []  # type: ignore[type-arg]

    def hook(kind, data):  # type: ignore[no-untyped-def]
        if kind == "llm_request":
            now[0] = 2.0  # the deadline passes after admission, before sending
        events.append(data)

    c = MeteredClient(Stall(), account, "worker", "w", on_event=hook)
    with pytest.raises(TransportTimeout):
        asyncio.run(c.complete(req()))
    assert account.snapshot().model_calls == 0 and account.reserved_tokens == 0
    assert events[-1]["usage"] == "none"


# Re-review: a dispatch cap's deadline bounds the executor; its overrun shows in the run.
def test_dispatch_limits_are_enforced_and_reported(tmp_path: Path) -> None:
    graph = TaskGraph(id="d", nodes=(Subtask(id="a", instruction="a"),), final_nodes=("a",))

    class Dawdler:
        """Spends its time outside model calls, so only the dispatch deadline can stop it."""

        name = "dawdler"

        async def execute(self, subtask, ctx):  # type: ignore[no-untyped-def]
            await asyncio.sleep(1.0)
            return ExecutorOutcome(output={})

    llm = MeteredClient(
        FakeTransport(),
        Account(Budget(max_tokens=50_000, max_model_calls=10)),
        "coordinator",
        "coordinator",
    )
    cfg = CoordinatorConfig(
        max_attempts=1,
        dispatch_budget=Budget(max_tokens=5_000, max_model_calls=2, max_elapsed_s=0.02),
    )
    result = asyncio.run(
        Coordinator({"dawdler": Dawdler()}, Constant("dawdler"), cfg).run(
            graph, Echo(), llm, Tracer(tmp_path / "t", "r")
        )
    )
    assert result.attempts[0].result.status == "timeout"

    big = LLMReply(content="{}", usage=Usage(prompt_tokens=2_000, completion_tokens=1), model="m")
    llm = MeteredClient(
        FakeTransport().always(big),
        Account(Budget(max_tokens=50_000, max_model_calls=10)),
        "coordinator",
        "coordinator",
    )
    cfg = CoordinatorConfig(
        max_attempts=1, dispatch_budget=Budget(max_tokens=1_500, max_model_calls=2)
    )
    result = asyncio.run(
        Coordinator(
            {"direct": Direct(model="m", max_completion_tokens=100)}, Constant("direct"), cfg
        ).run(graph, Echo(), llm, Tracer(tmp_path / "o", "r"))
    )
    assert result.cost.overrun and not result.within_budget


# Re-review: a rejected "unknown" result passes its reason to the retry.
def test_rejected_unknown_keeps_its_reason(tmp_path: Path) -> None:
    from bioalloc.schemas import CheckResult

    class Unsure:
        name = "unsure"

        def evaluate(self, subtask, output, upstream):  # type: ignore[no-untyped-def]
            return CheckResult(check=self.name, verdict="unknown", reason="cannot validate")

    class UnsureFamily(Echo):
        def checks(self, subtask):  # type: ignore[no-untyped-def]
            return [Unsure()]

    graph = TaskGraph(id="u", nodes=(Subtask(id="a", instruction="a"),), final_nodes=("a",))
    fake = FakeTransport().always("{}", "{}")
    llm = MeteredClient(
        fake, Account(Budget(max_tokens=50_000, max_model_calls=10)), "coordinator", "coordinator"
    )
    cfg = CoordinatorConfig(accept_unknown=False)
    asyncio.run(
        Coordinator({"direct": Direct(model="m")}, Constant("direct"), cfg).run(
            graph, UnsureFamily(), llm, Tracer(tmp_path, "r")
        )
    )
    assert "unsure: cannot validate" in fake.requests[1].messages[-1].content


# Re-review: only agent-written content counts as communication, not the template wrapper.
def test_feedback_wrapper_is_not_communication() -> None:
    from bioalloc.budget import communication_tokens
    from bioalloc.executors.base import Feedback

    sub = Subtask(id="a", instruction="a")
    ctx = ExecutionContext(
        llm=None,
        span_id="s",  # type: ignore[arg-type]
        feedback=(Feedback(agent="w0", output="", reasons=("r",)),),
    )
    msgs = tuple(subtask_messages("sys", sub, ctx))
    foreign = [m for m in msgs if m.source == "w0"]
    assert [m.content for m in foreign] == ['""']
    r = LLMRequest(model="m", messages=msgs, max_completion_tokens=1)
    total_chars = sum(len(m.content) for m in msgs)
    assert communication_tokens(r, 1000, "w1") == round(1000 * 2 / total_chars)


# Re-review: admission is first come first served.
def test_admission_is_fifo() -> None:
    account = Account(Budget(max_tokens=100, max_model_calls=10))
    order: list[str] = []
    gate = asyncio.Event()

    class Gated:
        async def send(self, request):  # type: ignore[no-untyped-def]
            order.append(request.messages[0].content)
            if request.messages[0].content == "holder":
                await gate.wait()
            return LLMReply(
                content="ok", usage=Usage(prompt_tokens=1, completion_tokens=1), model="m"
            )

    def call(name: str, completion: int):  # type: ignore[no-untyped-def]
        c = MeteredClient(
            Gated(), account.child(Budget(max_tokens=100, max_model_calls=10)), "worker", name
        )
        return c.complete(
            LLMRequest(
                model="m",
                messages=(Message(role="user", content=name),),
                max_completion_tokens=completion,
            )
        )

    async def go() -> None:
        holder = asyncio.create_task(call("holder", 50))  # reserves about 56
        await asyncio.sleep(0)
        big = asyncio.create_task(call("big", 50))  # does not fit until holder settles
        await asyncio.sleep(0)
        small = asyncio.create_task(call("small", 10))  # would fit now, but arrived later
        await asyncio.sleep(0.01)
        assert order == ["holder"]
        gate.set()
        await asyncio.gather(holder, big, small)

    asyncio.run(go())
    assert order == ["holder", "big", "small"]


# Third review: a batch with a cancelled worker and a failing one leaves no exception unobserved.
def test_mixed_cancelled_and_failed_batch_is_fully_observed(tmp_path: Path) -> None:
    import gc

    graph = TaskGraph(
        id="m",
        nodes=(Subtask(id="a", instruction="a"), Subtask(id="b", instruction="b")),
        final_nodes=("a", "b"),
    )

    class Mixed:
        name = "mixed"

        async def execute(self, subtask, ctx):  # type: ignore[no-untyped-def]
            if subtask.id == "a":
                raise asyncio.CancelledError()
            raise RuntimeError("bug in b")

    unobserved: list[str] = []
    llm = MeteredClient(
        FakeTransport(),
        Account(Budget(max_tokens=1000, max_model_calls=5)),
        "coordinator",
        "coordinator",
    )
    coord = Coordinator({"mixed": Mixed()}, Constant("mixed"))

    async def go() -> None:
        asyncio.get_running_loop().set_exception_handler(
            lambda loop, ctx: unobserved.append(ctx["message"])
        )
        with pytest.raises(RuntimeError, match="bug in b"):
            await coord.run(graph, Echo(), llm, Tracer(tmp_path, "r"))
        gc.collect()

    asyncio.run(go())
    gc.collect()
    assert unobserved == []


def test_fixed_policy_maps_contracts_to_executors() -> None:
    from bioalloc.policies import FixedByContract

    p = FixedByContract({"pddl_plan": "ant", "gsm8k_answer": "bee"}, default="direct")
    assert p.select("t", Subtask(id="a", instruction="x", contract="pddl_plan"), []) == "ant"
    assert p.select("t", Subtask(id="a", instruction="x", contract="gsm8k_answer"), []) == "bee"
    assert p.select("t", Subtask(id="a", instruction="x"), []) == "direct"
