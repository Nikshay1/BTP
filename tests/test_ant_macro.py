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


def choices(events: list) -> list[dict]:  # type: ignore[type-arg]
    return [e.data for e in events if e.kind == "ant_macro_choices"]


def test_a_completing_segment_is_taken_even_by_the_greedy_control(tmp_path: Path) -> None:
    # R17 (Sol): the sampler could leave a segment that reaches the goal untaken.
    def propose(req: LLMRequest) -> str:
        done = len(plan_so_far(req))
        rest = WITNESS[done:]
        if len(rest) <= 12:
            return json.dumps({"segments": [rest[:1], rest]})  # rank 2 completes the plan
        return json.dumps({"segments": [rest[:12]]})

    greedy = macro_controls("m")["macro_greedy"]
    task, result, _, events = run(tmp_path, greedy, propose)
    assert task.grade(result.answer).correct
    last = choices(events)[-1]
    assert last["completing"] and last["chosen"] == 2
    assert [c["p"] for c in last["candidates"]] == [0.0, 1.0]


def test_a_package_counts_as_a_milestone_once(tmp_path: Path) -> None:
    # R17: delivering, picking up again and redelivering the same package was rewarded twice.
    from bioalloc.tasks.strips import parse_problem

    goals = set(parse_problem(gen.instance(TASK).problem).goal)
    first = next(
        i
        for i, step in enumerate(WITNESS)
        if step.startswith("(unload-truck")
        and ("at", step.strip("()").split()[1], step.strip("()").split()[3]) in goals
    )
    _, pkg, truck, loc = WITNESS[first].strip("()").split()
    script = [
        WITNESS[: first + 1],
        [f"(load-truck {pkg} {truck} {loc})"],
        [f"(unload-truck {pkg} {truck} {loc})"],
    ]
    calls = iter(script)
    _, _, _, events = run(
        tmp_path,
        MacroAnt(model="m", params=MacroParams(max_decisions=3)),
        lambda r: json.dumps({"segments": [next(calls)]}),
    )
    milestones = [c["milestone"] for c in choices(events)]
    assert milestones[0] >= 1 and milestones[1:] == [0, 0]  # was [.., 0, 1]


def test_a_new_shape_has_no_edge_over_a_decayed_unrewarded_one(tmp_path: Path) -> None:
    # R17: unseen shapes started at tau0 while old ones had evaporated (0.7826/0.2174, not 0.8/0.2).
    init = gen.instance(TASK).problem
    truck_at = re.search(r"\(at t0 (l0-\d)\)", init).group(1)  # type: ignore[union-attr]
    other = next(f"l0-{k}" for k in range(3) if f"l0-{k}" != truck_at)
    drive = [f"(drive-truck t0 {truck_at} {other} c0)", f"(drive-truck t0 {other} {truck_at} c0)"]
    plane_at = re.search(r"\(at a0 (l\d-0)\)", init).group(1)  # type: ignore[union-attr]
    elsewhere = next(f"l{c}-0" for c in range(3) if f"l{c}-0" != plane_at)
    fly = [f"(fly-airplane a0 {plane_at} {elsewhere})", f"(fly-airplane a0 {elsewhere} {plane_at})"]
    script = iter([[drive], [drive], [drive], [drive, fly]])
    _, _, _, events = run(
        tmp_path,
        MacroAnt(model="m", params=MacroParams(max_decisions=4)),
        lambda r: json.dumps({"segments": next(script)}),
    )
    last = choices(events)[-1]
    assert [round(c["p"], 6) for c in last["candidates"]] == [0.8, 0.2]
    assert last["candidates"][0]["weight"] == last["candidates"][1]["weight"] * 4


def test_rejections_are_fed_back_even_when_another_segment_was_legal(tmp_path: Path) -> None:
    params = MacroParams(max_decisions=2)
    _, _, fake, _ = run(tmp_path, MacroAnt(model="m", params=params), witness_segments)
    assert "could not be applied" in fake.requests[1].messages[-1].content


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
