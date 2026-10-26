import asyncio
import json
from pathlib import Path

from conftest import SlowTransport

from bioalloc.budget import Account, MeteredClient
from bioalloc.coordinator import Coordinator, CoordinatorConfig
from bioalloc.evaluation.toy_data import grade, make_instance, optimal_legs
from bioalloc.executors.conventional import PlanThenExecute, SingleAgent, act_observe
from bioalloc.executors.octopus import Octopus
from bioalloc.graph import TaskGraph
from bioalloc.llm.base import LLMRequest
from bioalloc.llm.fake import FakeTransport
from bioalloc.policies import Constant
from bioalloc.schemas import Budget, Subtask
from bioalloc.tasks.toy import ToyRoutes
from bioalloc.trace import Tracer

INST = make_instance(0)
LEG1, LEG2 = optimal_legs(INST)


def run(executor, fake, tmp: Path, config: CoordinatorConfig | None = None):  # type: ignore[no-untyped-def]
    llm = MeteredClient(
        fake, Account(Budget(max_tokens=10**6, max_model_calls=50)), "coordinator", "coordinator"
    )
    coord = Coordinator({executor.name: executor}, Constant(executor.name), config)
    return asyncio.run(coord.run(INST.graph, ToyRoutes(), llm, Tracer(tmp, "r")))


def test_act_observe_is_the_octopus_loop() -> None:
    w = act_observe("m")
    assert isinstance(w, Octopus) and w.name == "act_observe" and not w.report_every_step


def test_plan_then_execute_plans_once_then_answers(tmp_path: Path) -> None:
    def respond(req: LLMRequest) -> str:
        if "numbered plan" in req.messages[0].content:
            return "1. read edges 2. pick cheapest"
        leg2 = any("Continue from the node" in m.content for m in req.messages)
        return json.dumps({"route": LEG2 if leg2 else LEG1})

    fake = FakeTransport().respond(lambda r: True, respond)
    result = run(PlanThenExecute(model="m"), fake, tmp_path)
    assert result.status == "completed" and grade(INST, result.answer).correct
    assert len(fake.requests) == 4
    execute = fake.requests[1]
    plan_msg = [m for m in execute.messages if m.content.startswith("1. read edges")]
    assert plan_msg and plan_msg[0].source == "plan_then_execute:leg1:0"


def test_single_agent_runs_serially_remembers_and_has_one_identity(tmp_path: Path) -> None:
    def respond(req: LLMRequest) -> str:
        leg2 = (
            "Continue from the node" in req.messages[-3].content
            or "Continue from the node" in req.messages[-1].content
        )
        return json.dumps({"route": LEG2 if leg2 else LEG1})

    fake = FakeTransport().respond(lambda r: True, respond)
    result = run(SingleAgent(model="m"), fake, tmp_path, CoordinatorConfig(serial=True))
    assert result.status == "completed" and grade(INST, result.answer).correct
    first, second = fake.requests
    # The second subtask's prompt contains the first subtask's whole exchange.
    contents = [m.content for m in second.messages]
    assert any("Find the cheapest route from start" in c for c in contents)
    assert json.dumps({"route": LEG1}) in contents
    # Its own earlier output is not communication.
    assert result.cost.communication == 0
    assert {a.result.span_id for a in result.attempts}  # traced
    assert first.sample_id.startswith("single_agent/") and second.sample_id.startswith(
        "single_agent/"
    )


def test_serial_mode_runs_one_subtask_at_a_time(tmp_path: Path) -> None:
    graph = TaskGraph(
        id="s",
        nodes=(Subtask(id="a", instruction="a"), Subtask(id="b", instruction="b")),
        final_nodes=("a", "b"),
    )
    active = 0
    peak = 0

    class Probe(SlowTransport):
        async def send(self, request):  # type: ignore[no-untyped-def]
            nonlocal active, peak
            active += 1
            peak = max(peak, active)
            try:
                return await super().send(request)
            finally:
                active -= 1

    class Echo:
        name = "echo"

        def checks(self, s):  # type: ignore[no-untyped-def]
            return []

        def aggregate(self, g, outputs):  # type: ignore[no-untyped-def]
            return dict(outputs)

    for serial, expected in ((False, 2), (True, 1)):
        peak = 0
        fake = FakeTransport().always("{}", "{}")
        llm = MeteredClient(
            Probe(fake, delay=0.02),
            Account(Budget(max_tokens=10**6, max_model_calls=10)),
            "coordinator",
            "coordinator",
        )
        from bioalloc.executors.direct import Direct

        coord = Coordinator(
            {"direct": Direct(model="m")}, Constant("direct"), CoordinatorConfig(serial=serial)
        )
        asyncio.run(coord.run(graph, Echo(), llm, Tracer(tmp_path / str(serial), "r")))
        assert peak == expected
