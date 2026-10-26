import asyncio
import json
from pathlib import Path

import numpy as np
import pytest
from test_planbench import DOMAIN, PROB

from bioalloc.budget import Account, MeteredClient
from bioalloc.coordinator import Coordinator
from bioalloc.evaluation.planbench_data import public_task as plan_task
from bioalloc.evaluation.toolsuite_data import make_instance as tool_instance
from bioalloc.executors.direct import Direct
from bioalloc.features import NAMES, family_of, subtask_features, vector
from bioalloc.llm.fake import FakeTransport
from bioalloc.policies import Attempt, LinUCB, Rule, RuleTable, provisional_reward
from bioalloc.schemas import Budget, CheckResult, Subtask
from bioalloc.tasks.planbench import PlanBench
from bioalloc.tasks.toolsuite import ToolSuite
from bioalloc.trace import Tracer, read_trace


def test_features_are_observable_and_fixed_order() -> None:
    plan = plan_task("p", DOMAIN, PROB).as_single_node().nodes[0]
    f = subtask_features(plan, PlanBench(), ready_siblings=1, attempt=0)
    assert list(f) == list(NAMES)
    assert f["family_planning"] == 1 and f["goal_checkable"] == 1 and f["has_tools"] == 0
    tool = tool_instance("compare", 1).graph.node("difference")
    g = subtask_features(tool, ToolSuite(), ready_siblings=2, attempt=1)
    assert g["family_tools"] == 1 and g["goal_checkable"] == 0 and g["has_tools"] == 1
    assert g["dependencies"] == 0.5 and g["ready_siblings"] == 0.5 and g["attempt"] == 1
    assert family_of(Subtask(id="x", instruction="y")) == "other"


def x_of(**on: float) -> Subtask:
    f = {n: 0.0 for n in NAMES} | {"bias": 1.0} | on
    return Subtask(id="s", instruction="i", features=f)


def test_linucb_matches_hand_computation_and_learns_per_context() -> None:
    p = LinUCB(arms=["a", "b"], alpha=0.0)
    planning, tools = x_of(family_planning=1.0), x_of(family_tools=1.0)
    for _ in range(5):
        p.learn(planning, "b", 1.0)
        p.learn(tools, "a", 1.0)
        p.learn(planning, "a", 0.0)
        p.learn(tools, "b", 0.0)
    assert p.select("t", planning, []) == "b" and p.select("t", tools, []) == "a"
    # Hand computation for arm b's estimate on the planning context.
    x_p, x_t = np.array(vector(planning.features)), np.array(vector(tools.features))
    A = np.eye(len(NAMES)) + 5 * np.outer(x_p, x_p) + 5 * np.outer(x_t, x_t)
    b = 5 * x_p
    assert p.scores(planning)["b"] == pytest.approx(float(np.linalg.solve(A, b) @ x_p))


def test_exploration_bonus_and_ties() -> None:
    p = LinUCB(arms=["a", "b"], alpha=1.0)
    s = x_of()
    assert p.select("t", s, []) == "a"  # untrained: equal scores, earlier arm
    p.learn(s, "a", 0.2)
    assert p.select("t", s, []) == "b"  # b is less explored, so its bonus is larger


def test_frozen_policy_does_not_learn_and_round_trips() -> None:
    p = LinUCB(arms=["a", "b"])
    p.learn(x_of(), "a", 1.0)
    frozen = LinUCB.from_state(json.loads(json.dumps(p.state())), frozen=True)
    before = frozen.scores(x_of())
    frozen.learn(x_of(), "b", 1.0)
    assert frozen.scores(x_of()) == before == p.scores(x_of())


def test_provisional_reward_separates_strong_and_weak_acceptance() -> None:
    def att(accepted: bool, evidence: str) -> Attempt:
        return Attempt(
            subtask_id="s",
            attempt=0,
            executor="a",
            status="completed",
            verdict="pass" if accepted else "fail",
            accepted=accepted,
            tokens=0,
            checks=(
                CheckResult(check="c", verdict="pass" if accepted else "fail", evidence=evidence),
            ),
        )  # type: ignore[arg-type]

    assert provisional_reward(att(True, "goal")) == 1.0
    assert provisional_reward(att(True, "format")) == 0.5
    assert provisional_reward(att(False, "goal")) == 0.0


def test_rule_table_first_match_wins() -> None:
    table = RuleTable(
        [
            Rule(when={"goal_checkable": (">=", 1)}, executor="ant"),
            Rule(when={"family_reasoning": ("==", 1)}, executor="bee"),
        ],
        default="direct",
    )
    assert table.select("t", x_of(goal_checkable=1.0, family_reasoning=1.0), []) == "ant"
    assert table.select("t", x_of(family_reasoning=1.0), []) == "bee"
    assert table.select("t", x_of(), []) == "direct"
    assert Rule.model_validate_json(table.rules[0].model_dump_json()) == table.rules[0]


def test_coordinator_attaches_features_and_traces_them(tmp_path: Path) -> None:
    inst = tool_instance("compare", 1)
    fake = FakeTransport().respond(
        lambda r: True, lambda r: json.dumps({"items": [1], "answer": 1})
    )
    llm = MeteredClient(
        fake, Account(Budget(max_tokens=10**6, max_model_calls=50)), "coordinator", "coordinator"
    )
    policy = LinUCB(arms=["direct"])
    asyncio.run(
        Coordinator({"direct": Direct(model="m")}, policy).run(
            inst.graph, ToolSuite(), llm, Tracer(tmp_path, "r")
        )
    )
    dispatches = [e.data for e in read_trace(tmp_path) if e.kind == "dispatch"]
    first_two = [d["features"]["ready_siblings"] for d in dispatches[:2]]
    assert first_two == [0.5, 0.5]  # a-customers and b-customers are ready together
    assert all(set(d["features"]) == set(NAMES) for d in dispatches)
