import asyncio
import json
from pathlib import Path

from bioalloc.budget import Account, MeteredClient
from bioalloc.coordinator import Coordinator, CoordinatorConfig
from bioalloc.evaluation.toy_data import make_instance, optimal_legs
from bioalloc.executors.base import ExecutionContext
from bioalloc.executors.octopus import Octopus
from bioalloc.llm.fake import FakeTransport
from bioalloc.policies import Constant
from bioalloc.schemas import Budget, CheckResult, Subtask
from bioalloc.tasks.toy import ToyRoutes
from bioalloc.trace import Tracer, read_trace

INSTANCE = make_instance(0)
LEG1, LEG2 = optimal_legs(INSTANCE)
BAD = json.dumps({"route": [LEG1[0], "Z", LEG1[-1]]})


def is_leg(n: int):  # type: ignore[no-untyped-def]
    marker = "Find the cheapest route from start" if n == 1 else "Continue from the node"
    return lambda r: r.messages[1].source != "x" and marker in r.messages[1].content


def run(fake: FakeTransport, tmp: Path, executor: Octopus):  # type: ignore[no-untyped-def]
    llm = MeteredClient(
        fake, Account(Budget(max_tokens=200_000, max_model_calls=30)), "coordinator", "coordinator"
    )
    coord = Coordinator(
        {executor.name: executor}, Constant(executor.name), CoordinatorConfig(max_attempts=1)
    )
    tracer = Tracer(tmp, "r")
    result = asyncio.run(coord.run(INSTANCE.graph, ToyRoutes(), llm, tracer))
    tracer.close()
    return result


def test_octopus_corrects_itself_from_check_reasons_inside_one_dispatch(tmp_path: Path) -> None:
    fake = FakeTransport().on(is_leg(1), BAD, json.dumps({"route": LEG1}))
    fake.on(is_leg(2), json.dumps({"route": LEG2}))
    result = run(fake, tmp_path, Octopus(model="m"))
    assert result.status == "completed"
    # One dispatch per subtask: the correction happened inside the executor, not by a retry.
    assert [a.result.subtask_id for a in result.attempts] == ["leg1", "leg2"]
    second_leg1_call = fake.requests[1]
    assert "missing edge" in second_leg1_call.messages[-1].content
    assert second_leg1_call.messages[-2].content == BAD
    stops = [e.data["reason"] for e in read_trace(tmp_path) if e.kind == "executor_stop"]
    assert stops == ["unverified_answer", "unverified_answer"]  # toy has no goal check


def test_step_limit_without_answer_is_rejected(tmp_path: Path) -> None:
    fake = FakeTransport().on(is_leg(1), *[BAD] * 3)
    result = run(fake, tmp_path, Octopus(model="m", max_steps=3))
    first = result.attempts[0]
    assert not first.accepted and first.checks[-1].check == "generation"
    assert len(fake.requests) == 3 and result.status == "failed"


def test_goal_verification_stops_the_loop() -> None:
    class Goal:
        name = "goal"
        evidence = "goal"

        def evaluate(self, subtask, output, upstream):  # type: ignore[no-untyped-def]
            ok = output == {"x": 1}
            return CheckResult(
                check="goal",
                verdict="pass" if ok else "fail",
                reason="" if ok else "wrong",
                evidence="goal",
            )

    fake = FakeTransport().always('{"x": 0}', '{"x": 1}', '{"x": 2}')
    account = Account(Budget(max_tokens=50_000, max_model_calls=10))
    events: list[tuple[str, dict]] = []  # type: ignore[type-arg]
    ctx = ExecutionContext(
        llm=MeteredClient(fake, account, "worker", "w"),
        span_id="s",
        checks=(Goal(),),
        emit=lambda k, d: events.append((k, d)),
    )
    out = asyncio.run(Octopus(model="m").execute(Subtask(id="a", instruction="x"), ctx))
    assert out.output == {"x": 1} and len(fake.requests) == 2
    assert ("executor_stop", {"reason": "goal_verified"}) in events


def test_reporting_counterpart_charges_supervisor_as_coordinator(tmp_path: Path) -> None:
    supervise = lambda r: "supervise" in r.messages[0].content  # noqa: E731
    fake = FakeTransport()
    fake.on(
        supervise, json.dumps({"decision": "continue", "instruction": "Use only listed edges."})
    )
    fake.on(is_leg(1), BAD, json.dumps({"route": LEG1}))
    fake.on(is_leg(2), json.dumps({"route": LEG2}))
    result = run(
        fake, tmp_path, Octopus(model="m", report_every_step=True, name="octopus_reporting")
    )
    assert result.status == "completed"
    sup = [r for r in fake.requests if supervise(r)]
    assert len(sup) == 1  # only after the rejected step; accepted answers return at once
    assert result.cost.coordinator.total > 0
    retry = [r for r in fake.requests if is_leg(1)(r)][1]
    instr = [m for m in retry.messages if m.content == "Use only listed edges."]
    assert instr and instr[0].source and instr[0].source.startswith("supervisor:")


def test_supervisor_stop_without_answer_is_rejected(tmp_path: Path) -> None:
    supervise = lambda r: "supervise" in r.messages[0].content  # noqa: E731
    fake = FakeTransport().on(supervise, json.dumps({"decision": "stop"}))
    fake.on(is_leg(1), BAD)
    result = run(
        fake, tmp_path, Octopus(model="m", report_every_step=True, name="octopus_reporting")
    )
    assert not result.attempts[0].accepted and result.status == "failed"
