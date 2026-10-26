"""Generator, references and grader for the controlled synthetic route tasks.

Instances are generated from (factors, seed) and are fully reproducible; the generator version is
part of every task id, so a change to the generator can never be mixed with old instances.
"""

from __future__ import annotations

import itertools
import random
from dataclasses import dataclass

import networkx as nx
from pydantic import JsonValue

from bioalloc.evaluation.toy_data import Grade
from bioalloc.graph import TaskGraph
from bioalloc.schemas import Subtask, Wire
from bioalloc.tasks.synthetic import CONTRACT
from bioalloc.tasks.toy import ROUTE_SCHEMA, edge_costs, route_cost, route_of

GENERATOR_VERSION = 1
LEG_INSTRUCTION = (
    "Find the cheapest route from start to goal along the directed weighted edges "
    "[from, to, cost]. Return the nodes visited in order, including both ends."
)
NEXT_LEG_INSTRUCTION = (
    "Continue from the node where the route of subtask {prev} ended, and find the cheapest route "
    "from there to goal along the directed weighted edges [from, to, cost]. Return the nodes "
    "visited in order, including both ends."
)


class Factors(Wire):
    alternatives: int = 2
    depth: int = 1
    width: int = 1
    hops: int = 3
    observable: bool = False


class SyntheticReference(Wire):
    optimal_cost: float
    leg_optima: dict[str, float]


@dataclass(frozen=True)
class SyntheticInstance:
    graph: TaskGraph
    factors: Factors
    reference: SyntheticReference
    group: str


def _leg_network(
    rng: random.Random,
    names: itertools.count[int],
    start: str,
    goal: str,
    alternatives: int,
    hops: int,
) -> tuple[list[JsonValue], float]:
    layers = [[start]]
    for _ in range(hops - 1):
        layers.append([f"N{next(names)}" for _ in range(alternatives)])
    layers.append([goal])
    g = nx.DiGraph()
    edges: list[JsonValue] = []
    for a, b in itertools.pairwise(layers):
        for u in a:
            for v in b:
                w = rng.randint(1, 9)
                g.add_edge(u, v, weight=w)
                edges.append([u, v, w])
    rng.shuffle(edges)
    return edges, float(nx.shortest_path_length(g, start, goal, weight="weight"))


def make_instance(factors: Factors, seed: int) -> SyntheticInstance:
    rng = random.Random(f"synthetic:v{GENERATOR_VERSION}:{factors.model_dump_json()}:{seed}")
    names = itertools.count()
    nodes: list[Subtask] = []
    optima: dict[str, float] = {}
    for c in range(factors.width):
        here = f"N{next(names)}"
        for leg in range(factors.depth):
            goal = f"N{next(names)}"
            edges, best = _leg_network(rng, names, here, goal, factors.alternatives, factors.hops)
            leg_id = f"c{c}-leg{leg}"
            optima[leg_id] = best
            inputs: dict[str, JsonValue] = {"edges": edges, "goal": goal}
            if leg == 0:
                inputs["start"] = here
                instruction = LEG_INSTRUCTION
                deps: tuple[str, ...] = ()
            else:
                prev = f"c{c}-leg{leg - 1}"
                inputs["start_after"] = prev
                instruction = NEXT_LEG_INSTRUCTION.format(prev=prev)
                deps = (prev,)
            if factors.observable:
                inputs["cost_bound"] = best
            nodes.append(
                Subtask(
                    id=leg_id,
                    instruction=instruction,
                    inputs=inputs,
                    depends_on=deps,
                    output_schema=ROUTE_SCHEMA,
                    contract=CONTRACT,
                )
            )
            here = goal
    key = (
        f"a{factors.alternatives}-d{factors.depth}-w{factors.width}-h{factors.hops}"
        f"-{'obs' if factors.observable else 'hid'}"
    )
    task_id = f"synth-v{GENERATOR_VERSION}-{key}-s{seed}"
    graph = TaskGraph(id=task_id, nodes=tuple(nodes), final_nodes=tuple(n.id for n in nodes))
    reference = SyntheticReference(optimal_cost=sum(optima.values()), leg_optima=optima)
    # Every generated instance is its own template group; instances never share structure by design.
    return SyntheticInstance(graph, factors, reference, group=f"synthetic:{task_id}")


def grade(instance: SyntheticInstance, answer: JsonValue) -> Grade:
    routes = answer.get("routes") if isinstance(answer, dict) else None
    if not isinstance(routes, dict):
        return Grade(correct=False, detail="no routes")
    total = 0.0
    for c in range(instance.factors.width):
        legs = [n for n in instance.graph.nodes if n.id.startswith(f"c{c}-")]
        route = route_of({"route": routes.get(f"c{c}")})
        first, last = legs[0], legs[-1]
        if not route or route[0] != first.inputs["start"] or route[-1] != last.inputs["goal"]:
            return Grade(correct=False, detail=f"chain c{c}: wrong endpoints")
        costs: dict[tuple[str, str], float] = {}
        for leg in legs:
            costs |= edge_costs(leg.inputs["edges"])
        cost = route_cost(route, costs)
        if cost is None:
            return Grade(correct=False, detail=f"chain c{c}: invalid edge")
        total += cost
    if total > instance.reference.optimal_cost:
        return Grade(
            correct=False,
            detail=f"valid but not cheapest ({total} > {instance.reference.optimal_cost})",
        )
    return Grade(correct=True)


FACTOR_LEVELS = {"alternatives": (2, 4, 8), "depth": (1, 3, 6), "width": (1, 2, 4)}
BASELINE = Factors(alternatives=4, depth=1, width=1)


def one_factor_designs() -> list[Factors]:
    """Vary one factor at a time around the baseline, each with and without an observable goal."""
    designs: list[Factors] = []
    for factor, levels in FACTOR_LEVELS.items():
        for level in levels:
            for observable in (False, True):
                f = BASELINE.model_copy(update={factor: level, "observable": observable})
                if f not in designs:
                    designs.append(f)
    return designs
