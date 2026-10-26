import asyncio
import json
from pathlib import Path

from bioalloc.budget import Account, MeteredClient
from bioalloc.coordinator import Coordinator
from bioalloc.evaluation.toy_data import grade, make_instance, optimal_legs
from bioalloc.executors.specialist import Specialist
from bioalloc.llm.base import LLMRequest
from bioalloc.llm.fake import FakeTransport
from bioalloc.policies import Constant
from bioalloc.schemas import Budget
from bioalloc.tasks.toy import ToyRoutes
from bioalloc.trace import Tracer, read_trace

INST = make_instance(0)
LEG1, LEG2 = optimal_legs(INST)
DETOUR = ["A", "E", "G"]  # valid but costlier than LEG1 for seed 0


def make_fake(critic_rejects_first: bool) -> FakeTransport:
    reviews = iter([False, True] if critic_rejects_first else [True, True, True])

    def respond(req: LLMRequest) -> str:
        system = req.messages[0].content
        if "coordinate workers" in system:
            return json.dumps(
                {"advice": {"leg1": "Compare total costs.", "leg2": "Use cheap edges."}}
            )
        if "review a worker" in system:
            ok = next(reviews)
            return json.dumps({"accept": ok, "critique": "" if ok else "A-D-G is cheaper."})
        leg2 = any("Continue from the node" in m.content for m in req.messages)
        revised = any(m.source == "critic" for m in req.messages)
        if leg2:
            return json.dumps({"route": LEG2})
        return json.dumps({"route": LEG1 if revised or not critic_rejects_first else DETOUR})

    return FakeTransport().respond(lambda r: True, respond)


def run(fake: FakeTransport, tmp: Path):  # type: ignore[no-untyped-def]
    llm = MeteredClient(
        fake, Account(Budget(max_tokens=10**6, max_model_calls=50)), "coordinator", "coordinator"
    )
    tracer = Tracer(tmp, "r")
    result = asyncio.run(
        Coordinator({"specialist": Specialist(model="m")}, Constant("specialist")).run(
            INST.graph, ToyRoutes(), llm, tracer
        )
    )
    tracer.close()
    return result


def test_planner_once_critic_triggers_one_revision(tmp_path: Path) -> None:
    fake = make_fake(critic_rejects_first=True)
    result = run(fake, tmp_path)
    assert result.status == "completed" and grade(INST, result.answer).correct
    planner_calls = [r for r in fake.requests if "coordinate workers" in r.messages[0].content]
    assert len(planner_calls) == 1  # once per task, even though leg1 and leg2 both need advice
    assert result.cost.coordinator.total > 0  # the planner is a coordinator role
    revision = [r for r in fake.requests if any(m.source == "critic" for m in r.messages)]
    assert len(revision) == 1
    advice = [m for m in fake.requests[1].messages if m.content.startswith("Advice:")]
    assert advice and advice[0].source == "planner"
    # Within one dispatch per subtask: the revision is not an outer retry.
    assert [a.result.subtask_id for a in result.attempts] == ["leg1", "leg2"]
    reviews = [e.data for e in read_trace(tmp_path) if e.kind == "specialist_review"]
    assert [r["accept"] for r in reviews] == [False, True]


def test_accepted_result_is_returned_without_revision(tmp_path: Path) -> None:
    fake = make_fake(critic_rejects_first=False)
    result = run(fake, tmp_path)
    assert result.status == "completed"
    assert not any(any(m.source == "critic" for m in r.messages) for r in fake.requests)
