import asyncio
import json
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest

from bioalloc.actions import FINISH, Action
from bioalloc.budget import Account, MeteredClient
from bioalloc.coordinator import Coordinator, CoordinatorConfig
from bioalloc.evaluation.toy_data import make_instance, optimal_legs
from bioalloc.executors.ant import Ant, AntParams, ant_controls
from bioalloc.graph import TaskGraph
from bioalloc.llm.base import LLMRequest
from bioalloc.llm.fake import FakeTransport
from bioalloc.policies import Constant
from bioalloc.schemas import Budget, CheckResult, Subtask
from bioalloc.tasks.toy import ToyRoutes
from bioalloc.trace import Tracer, read_trace

PROPOSE = lambda r: "Propose up to five" in r.messages[0].content  # noqa: E731


def test_update_rule_hand_computed() -> None:
    ant = Ant(model="m", params=AntParams(rho=0.1, delta=1.0, tau_min=0.5))
    tau = {"MOVE": 1.0, "FINISH": 1.0, "WAIT": 0.52}
    ant._update(tau, {"MOVE", "FINISH"}, reward=1.0)
    assert tau == pytest.approx({"MOVE": 1.9, "FINISH": 1.9, "WAIT": 0.5})  # floor applies
    ant._update(tau, {"MOVE"}, reward=0.0)
    assert tau == pytest.approx({"MOVE": 1.71, "FINISH": 1.71, "WAIT": 0.5})


def test_controls_change_exactly_one_thing() -> None:
    c = ant_controls("m")
    assert c["ant_no_deposit"].params.delta == 0 and c["ant_no_deposit"].params.rho == 0.1
    assert c["ant_no_evaporation"].params.rho == 0 and c["ant_no_evaporation"].params.delta == 1
    u = c["ant_uniform"]
    assert (u.params.alpha, u.params.beta) == (0, 0)
    tau = {"MOVE": 1.0, "FINISH": 1.0}
    c["ant_no_evaporation"]._update(tau, set(), reward=0.0)
    assert tau == {"MOVE": 1.0, "FINISH": 1.0}


def test_choice_weights_hand_computed() -> None:
    a, b = Action(kind="A"), Action(kind="B")
    ant = Ant(model="m", params=AntParams(alpha=1, beta=2))
    assert ant.choice_weights([a, b], {"A": 1.0, "B": 3.0}) == pytest.approx([1.0, 0.75])
    uniform = ant_controls("m")["ant_uniform"]
    assert uniform.choice_weights([a, b], {"A": 1.0, "B": 3.0}) == [1.0, 1.0]


def route_proposer(best: dict[str, list[str]]):  # type: ignore[no-untyped-def]
    """Fake proposer for the toy: ranks the next node on the cheapest route first."""

    def respond(req: LLMRequest) -> str:
        text = req.messages[-1].content
        route = json.loads(re.search(r"Route so far: (\[.*?\])", text).group(1))  # type: ignore[union-attr]
        goal = re.search(r"Goal: (\w)", text).group(1)  # type: ignore[union-attr]
        edges = json.loads(re.search(r"Edges from \w: (\[.*\])\.", text).group(1))  # type: ignore[union-attr]
        if route[-1] == goal:
            return json.dumps({"candidates": [{"kind": FINISH, "args": {}}]})
        path = best[goal]
        nxt = path[path.index(route[-1]) + 1] if route[-1] in path else None
        others = [v for v, _ in edges if v != nxt]
        ranked = ([nxt] if nxt else []) + others
        return json.dumps({"candidates": [{"kind": "MOVE", "args": {"to": v}} for v in ranked]})

    return respond


def run_toy(fake: FakeTransport, tmp: Path, ant: Ant, budget: Budget | None = None):  # type: ignore[no-untyped-def]
    inst = make_instance(0)
    llm = MeteredClient(
        fake,
        Account(budget or Budget(max_tokens=10**7, max_model_calls=500)),
        "coordinator",
        "coordinator",
    )
    coord = Coordinator({ant.name: ant}, Constant(ant.name), CoordinatorConfig(max_attempts=1))
    tracer = Tracer(tmp, "r")
    result = asyncio.run(coord.run(inst.graph, ToyRoutes(), llm, tracer))
    tracer.close()
    return inst, result


def test_ant_solves_the_toy_end_to_end(tmp_path: Path) -> None:
    inst = make_instance(0)
    leg1, leg2 = optimal_legs(inst)
    fake = FakeTransport().respond(PROPOSE, route_proposer({leg1[-1]: leg1, leg2[-1]: leg2}))
    inst, result = run_toy(fake, tmp_path, Ant(model="m"))
    assert result.status == "completed"
    paths = [e for e in read_trace(tmp_path) if e.kind == "ant_path"]
    # No goal check in the toy: reward stays 0. Sampled detours can wander past the choice limit
    # without finishing, so a leg stops by stagnation or at the path limit.
    assert paths and all(e.data["reward"] == 0 for e in paths)
    stops = [e.data for e in read_trace(tmp_path) if e.kind == "executor_stop"]
    assert len(stops) == 2 and {s["reason"] for s in stops} <= {"stagnation", "path_limit"}
    assert all(e.data["path"] < 8 for e in paths)
    assert result.answer["route"][0] == "A" and result.answer["route"][-1] == "H"  # type: ignore[index]


def test_illegal_and_unknown_proposals_are_dropped(tmp_path: Path) -> None:
    junk = json.dumps(
        {
            "candidates": [
                {"kind": "TELEPORT", "args": {}},
                {"kind": "MOVE", "args": {"to": "Z"}},
                {"kind": FINISH, "args": {}},
                "garbage",
            ]
        }
    )
    fake = FakeTransport().respond(PROPOSE, lambda r: junk)
    _, result = run_toy(fake, tmp_path, Ant(model="m", params=AntParams(max_paths=2)))
    assert result.status == "failed"
    assert result.attempts[0].checks[-1].reason == "no path reached FINISH"


class Counter:
    """Test family with a real goal check: reach exactly 3 from 0 with INC/DEC, then FINISH."""

    name = "counter"

    class Space:
        kinds: tuple[str, ...] = ("INC", "DEC", FINISH)

        def start(self, subtask: Subtask, upstream: Mapping[str, Any]) -> int:
            return 0

        def legal(self, state: int, action: Action) -> str | None:
            return None

        def apply(self, state: int, action: Action) -> tuple[int, str]:
            n = state + (1 if action.kind == "INC" else -1)
            return n, f"value {n}"

        def finish(self, state: int, action: Action) -> Any:
            return {"value": state}

        def describe(self, state: int) -> str:
            return f"value={state}"

    class Goal:
        name = "goal"
        evidence = "goal"

        def evaluate(self, subtask, output, upstream):  # type: ignore[no-untyped-def]
            ok = output == {"value": 3}
            return CheckResult(check="goal", verdict="pass" if ok else "fail", evidence="goal")

    def checks(self, s):  # type: ignore[no-untyped-def]
        return [self.Goal()]

    def aggregate(self, g, outputs):  # type: ignore[no-untyped-def]
        return outputs["c"]

    def action_space(self, s):  # type: ignore[no-untyped-def]
        return self.Space()


def counter_proposer(req: LLMRequest) -> str:
    value = int(re.search(r"value=(-?\d+)", req.messages[-1].content).group(1))  # type: ignore[union-attr]
    first = "FINISH" if value == 3 else "INC"
    rest = [k for k in ("INC", "DEC", "FINISH") if k != first]
    return json.dumps({"candidates": [{"kind": k, "args": {}} for k in [first, *rest]]})


def run_counter(tmp: Path, ant: Ant) -> tuple[Any, list[Any]]:
    graph = TaskGraph(id="cnt", nodes=(Subtask(id="c", instruction="reach 3"),), final_nodes=("c",))
    fake = FakeTransport().respond(PROPOSE, counter_proposer)
    llm = MeteredClient(
        fake, Account(Budget(max_tokens=10**7, max_model_calls=500)), "coordinator", "coordinator"
    )
    coord = Coordinator(
        {ant.name: ant}, Constant(ant.name), CoordinatorConfig(max_attempts=1), seed=3
    )
    tracer = Tracer(tmp, "r")
    result = asyncio.run(coord.run(graph, Counter(), llm, tracer))
    tracer.close()
    return result, [e.data for e in read_trace(tmp) if e.kind == "ant_path"]


def test_verified_paths_are_reinforced_and_state_never_leaks_between_runs(tmp_path: Path) -> None:
    result, paths = run_counter(tmp_path / "1", Ant(model="m"))
    assert result.answer == {"value": 3}
    verified = [p for p in paths if p["reward"] == 1]
    assert verified, paths
    assert verified[0]["tau"]["INC"] > 1.0 and verified[0]["tau"]["FINISH"] > 1.0
    # A second run starts from the frozen snapshot again, not from the first run's weights.
    _, again = run_counter(tmp_path / "2", Ant(model="m"))
    assert again[0]["tau"] == paths[0]["tau"]


def test_no_deposit_control_never_reinforces(tmp_path: Path) -> None:
    _, paths = run_counter(tmp_path, ant_controls("m")["ant_no_deposit"])
    assert all(v <= 1.0 for p in paths for v in p["tau"].values())


def test_budget_before_any_finished_path_is_budget_exhaustion(tmp_path: Path) -> None:
    inst = make_instance(0)
    leg1, leg2 = optimal_legs(inst)
    fake = FakeTransport().respond(PROPOSE, route_proposer({leg1[-1]: leg1, leg2[-1]: leg2}))
    _, result = run_toy(fake, tmp_path, Ant(model="m"), Budget(max_tokens=10**7, max_model_calls=1))
    assert result.status == "budget_exhausted"


def test_ant_sees_inputs_upstream_and_retry_feedback_with_provenance() -> None:
    from bioalloc.executors.base import ExecutionContext, Feedback, Upstream

    seen: list[LLMRequest] = []

    def respond(req: LLMRequest) -> str:
        seen.append(req)
        return json.dumps({"candidates": [{"kind": "FINISH", "args": {}}]})

    fake = FakeTransport().respond(PROPOSE, respond)
    sub = Subtask(
        id="c",
        instruction="reach 3",
        inputs={"question": "What is 1+2?"},
        depends_on=("u",),
        written_by="decomposer",
    )
    ctx = ExecutionContext(
        llm=MeteredClient(
            fake, Account(Budget(max_tokens=10**6, max_model_calls=50)), "worker", "ant:c:1"
        ),
        span_id="s",
        actions=Counter.Space(),
        checks=(Counter.Goal(),),
        upstream={"u": Upstream(subtask_id="u", agent="direct:u:0", output={"n": 1})},
        feedback=(Feedback(agent="ant:c:0", output={"value": 0}, reasons=("goal: wrong",)),),
    )
    asyncio.run(Ant(model="m", params=AntParams(max_paths=1)).execute(sub, ctx))
    msgs = seen[0].messages
    text = "\n".join(m.content for m in msgs)
    assert "What is 1+2?" in text and "goal: wrong" in text and "value=0" in text
    sources = {m.source for m in msgs}
    assert {"decomposer", "direct:u:0", "ant:c:0"} <= sources
