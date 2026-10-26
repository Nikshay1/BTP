"""Generator, reference and grader for the toy route-planning family."""

from __future__ import annotations

import random
from dataclasses import dataclass

import networkx as nx
from pydantic import JsonValue

from bioalloc.graph import TaskGraph
from bioalloc.schemas import Subtask, Wire
from bioalloc.tasks.toy import ROUTE_SCHEMA, edge_costs, route_cost, route_of


class Grade(Wire):
    correct: bool
    detail: str = ""


class RouteReference(Wire):
    start: str
    waypoint: str
    goal: str
    optimal_cost: float


@dataclass(frozen=True)
class ToyInstance:
    graph: TaskGraph
    reference: RouteReference


def make_instance(seed: int, n_nodes: int = 8, p_edge: float = 0.3) -> ToyInstance:
    """A random weighted digraph on nodes A, B, ... with a route start -> waypoint -> goal."""
    rng = random.Random(seed)
    names = [chr(ord("A") + i) for i in range(n_nodes)]
    while True:
        g = nx.DiGraph()
        g.add_nodes_from(names)
        for u in names:
            for v in names:
                if u != v and rng.random() < p_edge:
                    g.add_edge(u, v, weight=rng.randint(1, 9))
        start, goal = names[0], names[-1]
        waypoint = rng.choice(names[1:-1])
        if nx.has_path(g, start, waypoint) and nx.has_path(g, waypoint, goal):
            break
    optimal = nx.shortest_path_length(
        g, start, waypoint, weight="weight"
    ) + nx.shortest_path_length(g, waypoint, goal, weight="weight")
    edges: list[JsonValue] = [[u, v, d["weight"]] for u, v, d in sorted(g.edges(data=True))]
    leg1 = Subtask(
        id="leg1",
        instruction="Find the cheapest route from start to goal along the directed weighted edges "
        "[from, to, cost]. Return the nodes visited in order, including both ends.",
        inputs={"edges": edges, "start": start, "goal": waypoint},
        output_schema=ROUTE_SCHEMA,
    )
    leg2 = Subtask(
        id="leg2",
        instruction="Continue from the node where the route of subtask leg1 ended, and find the "
        "cheapest route from there to goal along the directed weighted edges [from, to, cost]. "
        "Return the nodes visited in order, including both ends.",
        inputs={"edges": edges, "start_after": "leg1", "goal": goal},
        depends_on=("leg1",),
        output_schema=ROUTE_SCHEMA,
    )
    graph = TaskGraph(id=f"toy-{seed}", nodes=(leg1, leg2), final_nodes=("leg1", "leg2"))
    reference = RouteReference(start=start, waypoint=waypoint, goal=goal, optimal_cost=optimal)
    return ToyInstance(graph, reference)


def grade(instance: ToyInstance, answer: JsonValue) -> Grade:
    ref = instance.reference
    route = route_of(answer)
    if not route:
        return Grade(correct=False, detail="no route")
    if route[0] != ref.start or route[-1] != ref.goal or ref.waypoint not in route:
        return Grade(correct=False, detail="wrong endpoints or waypoint skipped")
    cost = route_cost(route, edge_costs(instance.graph.nodes[0].inputs["edges"]))
    if cost is None:
        return Grade(correct=False, detail="invalid edge")
    if cost > ref.optimal_cost:
        return Grade(correct=False, detail=f"valid but not cheapest ({cost} > {ref.optimal_cost})")
    return Grade(correct=True)


def optimal_legs(instance: ToyInstance) -> tuple[list[str], list[str]]:
    """Cheapest routes for both legs; used to script the fake model in demos and tests."""
    g = nx.DiGraph()
    for (u, v), w in edge_costs(instance.graph.nodes[0].inputs["edges"]).items():
        g.add_edge(u, v, weight=w)
    ref = instance.reference
    leg1: list[str] = nx.shortest_path(g, ref.start, ref.waypoint, weight="weight")
    leg2: list[str] = nx.shortest_path(g, ref.waypoint, ref.goal, weight="weight")
    return leg1, leg2
