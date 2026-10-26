import asyncio
import json
import re
from pathlib import Path

import networkx as nx

from bioalloc.budget import Account, MeteredClient
from bioalloc.coordinator import Coordinator
from bioalloc.evaluation.synthetic_data import Factors, grade, make_instance, one_factor_designs
from bioalloc.executors.direct import Direct
from bioalloc.llm.base import LLMRequest
from bioalloc.llm.fake import FakeTransport
from bioalloc.policies import Constant
from bioalloc.schemas import Budget
from bioalloc.tasks.synthetic import SyntheticRoutes
from bioalloc.tasks.toy import edge_costs
from bioalloc.trace import Tracer


def test_factors_shape_the_graph() -> None:
    inst = make_instance(Factors(alternatives=4, depth=3, width=2, hops=3), seed=1)
    ids = [n.id for n in inst.graph.nodes]
    assert ids == ["c0-leg0", "c0-leg1", "c0-leg2", "c1-leg0", "c1-leg1", "c1-leg2"]
    assert inst.graph.node("c0-leg2").depends_on == ("c0-leg1",)
    assert inst.graph.node("c1-leg0").depends_on == ()
    leg = inst.graph.node("c0-leg0")
    assert len(leg.inputs["edges"]) == 4 + 4 * 4 + 4  # type: ignore[arg-type]
    assert "cost_bound" not in leg.inputs
    obs = make_instance(Factors(observable=True), seed=1)
    assert obs.graph.nodes[0].inputs["cost_bound"] == obs.reference.optimal_cost


def test_generation_is_reproducible_and_seed_dependent() -> None:
    f = Factors(alternatives=8, depth=2)
    assert make_instance(f, 3).graph == make_instance(f, 3).graph
    assert make_instance(f, 3).graph != make_instance(f, 4).graph
    assert (
        make_instance(f, 3).graph.id != make_instance(f.model_copy(update={"width": 2}), 3).graph.id
    )


def best_route(leg_inputs: dict, start: str) -> list[str]:  # type: ignore[type-arg]
    g = nx.DiGraph()
    for (u, v), w in edge_costs(leg_inputs["edges"]).items():
        g.add_edge(u, v, weight=w)
    return nx.shortest_path(g, start, leg_inputs["goal"], weight="weight")  # type: ignore[no-any-return]


def test_end_to_end_with_optimal_and_suboptimal_answers(tmp_path: Path) -> None:
    inst = make_instance(Factors(alternatives=4, depth=2, width=2, observable=True), seed=5)
    starts: dict[str, str] = {}

    by_goal = {str(n.inputs["goal"]): n for n in inst.graph.nodes}

    def respond(req: LLMRequest) -> str:
        goal = re.search(r'"goal": "(N\d+)"', req.messages[2].content).group(1)  # type: ignore[union-attr]
        sub = by_goal[goal]
        start = sub.inputs.get("start") or starts[str(sub.inputs["start_after"])]
        route = best_route(sub.inputs, str(start))
        starts[sub.id] = route[-1]
        return json.dumps({"route": route})

    fake = FakeTransport().respond(lambda r: True, respond)
    llm = MeteredClient(
        fake, Account(Budget(max_tokens=10**6, max_model_calls=50)), "coordinator", "coordinator"
    )
    result = asyncio.run(
        Coordinator({"direct": Direct(model="m")}, Constant("direct")).run(
            inst.graph, SyntheticRoutes(), llm, Tracer(tmp_path, "r")
        )
    )
    assert result.status == "completed" and grade(inst, result.answer).correct
    goal_checks = [c for a in result.attempts for c in a.checks if c.check == "cost_bound"]
    assert len(goal_checks) == 4 and all(
        c.verdict == "pass" and c.evidence == "goal" for c in goal_checks
    )
    # A valid but costlier chain is graded wrong.
    worse = json.loads(json.dumps(result.answer))
    leg = inst.graph.node("c0-leg0")
    alt = [
        r
        for r in nx.all_simple_paths(
            nx.DiGraph([(u, v) for u, v, _ in leg.inputs["edges"]]),  # type: ignore[misc]
            leg.inputs["start"],
            leg.inputs["goal"],
        )
    ]
    costs = edge_costs(leg.inputs["edges"])
    best = min(alt, key=lambda r: sum(costs[e] for e in zip(r, r[1:], strict=False)))
    other = max(alt, key=lambda r: sum(costs[e] for e in zip(r, r[1:], strict=False)))
    if other != best:
        chain = worse["routes"]["c0"]
        worse["routes"]["c0"] = other + chain[len(best) :]
        assert not grade(inst, worse).correct


def test_one_factor_designs_cover_each_level_both_ways() -> None:
    designs = one_factor_designs()
    assert Factors(alternatives=8, observable=True) in [
        d.model_copy(update={"depth": 1, "width": 1}) for d in designs
    ]
    assert len(designs) == len({d.model_dump_json() for d in designs}) == 14
