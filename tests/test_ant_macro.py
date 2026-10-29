"""Macro ant with milestone rewards (D30)."""

import asyncio
import json
import re
from pathlib import Path

from bioalloc.budget import Account, MeteredClient
from bioalloc.coordinator import Coordinator, CoordinatorConfig
from bioalloc.evaluation import logistics_gen as gen
from bioalloc.evaluation.registry import TaskLoader
from bioalloc.executors.ant_macro import MacroAnt, MacroParams, feature, macro_controls
from bioalloc.llm.base import LLMRequest
from bioalloc.llm.fake import FakeTransport
from bioalloc.policies import Constant
from bioalloc.schemas import Budget
from bioalloc.trace import Tracer, read_trace

TASK = "logisticsgen-v1-c3-p5-00-0"
WITNESS = list(gen.instance(TASK).witness)


def plan_so_far(req: LLMRequest) -> list[str]:
    m = re.search(r"Plan so far: (\[.*?\])\.", req.messages[-1].content, re.S)
    assert m
    return json.loads(m.group(1))


def run(tmp: Path, ant: MacroAnt, proposer) -> tuple:  # type: ignore[no-untyped-def, type-arg]
    task = TaskLoader().load(TASK)
    assert task.public is not None
    fake = FakeTransport().always().respond(lambda r: True, proposer)
    llm = MeteredClient(
        fake, Account(Budget(max_tokens=10**7, max_model_calls=50)), "coordinator", "coordinator"
    )
    coord = Coordinator({ant.name: ant}, Constant(ant.name), CoordinatorConfig(max_attempts=1))
    tracer = Tracer(tmp, "r")
    result = asyncio.run(coord.run(task.public.as_single_node(), task.task_family, llm, tracer))
    tracer.close()
    events = list(read_trace(tmp))
    return task, result, fake, events


def witness_segments(req: LLMRequest) -> str:
    """Best: the next 12 witness actions; second: an illegal segment."""
    done = len(plan_so_far(req))
    return json.dumps({"segments": [WITNESS[done : done + 12], ["(fly-airplane a9 x y)"]]})


def test_witness_segments_reach_the_goal_with_a_certificate(tmp_path: Path) -> None:
    params = MacroParams(max_decisions=len(WITNESS) // 12 + 1)
    task, result, fake, events = run(tmp_path, MacroAnt(model="m", params=params), witness_segments)
    assert task.grade(result.answer).correct
    stop = next(e.data for e in events if e.kind == "executor_stop")
    assert stop["reason"] == "certificate" and stop["plan_length"] == len(WITNESS)
    choices = [e.data for e in events if e.kind == "ant_macro_choices"]
    assert all(c["rejected"] for c in choices)  # the illegal second segment is always dropped
    assert any(c["milestone"] for c in choices)


def test_an_illegal_only_proposal_is_fed_back(tmp_path: Path) -> None:
    calls: list[LLMRequest] = []

    def bad(req: LLMRequest) -> str:
        calls.append(req)
        return json.dumps({"segments": [["(fly-airplane a9 x y)"]]})

    run(tmp_path, MacroAnt(model="m", params=MacroParams(max_decisions=2)), bad)
    assert "could not be applied" in calls[1].messages[-1].content


def test_a_milestone_deposit_changes_a_later_choice() -> None:
    # Hand-computed: two legal candidates of different shapes; after a deposit on the second
    # shape its weight overtakes the better-ranked first one.
    ant = MacroAnt(model="m")
    tau = {"a": 1.0, "b": 1.0}
    ant._evaporate(tau)
    tau["b"] += ant.params.delta  # milestone on shape b
    w_first = tau["a"] ** 1.0 * (1 / 1) ** 2.0
    w_second = tau["b"] ** 1.0 * (1 / 2) ** 2.0
    assert w_first == 0.9 and abs(w_second - 0.475) < 1e-9
    tau["b"] += 3 * ant.params.delta  # three more milestones on b
    assert tau["b"] * 0.25 > tau["a"]


def test_controls_change_exactly_one_thing() -> None:
    c = macro_controls("m")
    assert (
        c["ant_macro_no_deposit"].params.delta == 0 and c["ant_macro_no_deposit"].params.rho == 0.1
    )
    assert c["ant_macro_no_evaporation"].params.rho == 0
    assert (c["ant_macro_uniform"].params.alpha, c["ant_macro_uniform"].params.beta) == (0, 0)
    assert c["macro_greedy"].params.greedy and not c["ant_macro"].params.greedy


def test_feature_is_the_order_of_first_use() -> None:
    steps = [("drive-truck", "t0"), ("load-truck", "p0"), ("drive-truck", "t0")]
    assert feature(steps) == "drive-truck>load-truck"
