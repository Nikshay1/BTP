"""Toy family for M1: two-leg route planning on a small weighted directed graph.

Leg 1 asks for the cheapest route from the start to a waypoint. Leg 2 continues from wherever leg 1
ended to the goal, so it depends on leg 1's accepted output. The final answer joins the legs and
adds up the edge costs. The public route check verifies endpoints and edges but not optimality;
whether the route is cheapest is known only to the grader. This family tests the plumbing and is
excluded from every reported result (decision D4).
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any

from pydantic import JsonValue

from bioalloc.actions import FINISH, Action
from bioalloc.checks import RuntimeCheck, SchemaCheck
from bioalloc.graph import TaskGraph
from bioalloc.schemas import CheckResult, Evidence, Subtask

ROUTE_SCHEMA: dict[str, JsonValue] = {
    "type": "object",
    "properties": {"route": {"type": "array", "items": {"type": "string"}, "minItems": 1}},
    "required": ["route"],
}


def edge_costs(edges: Any) -> dict[tuple[str, str], float]:
    return {(str(u), str(v)): float(w) for u, v, w in edges}


def route_of(output: JsonValue) -> list[str] | None:
    route = output.get("route") if isinstance(output, dict) else None
    if isinstance(route, list) and route and all(isinstance(n, str) for n in route):
        return [str(n) for n in route]
    return None


def route_cost(route: Sequence[str], costs: Mapping[tuple[str, str], float]) -> float | None:
    """Total cost of a route, or None if it uses an edge that does not exist."""
    total = 0.0
    for edge in zip(route, route[1:], strict=False):
        if edge not in costs:
            return None
        total += costs[edge]
    return total


def leg_start(subtask: Subtask, upstream: Mapping[str, JsonValue]) -> str | None:
    if "start" in subtask.inputs:
        return str(subtask.inputs["start"])
    after = subtask.inputs.get("start_after")
    route = route_of(upstream.get(str(after))) if after is not None else None
    return route[-1] if route else None


class RouteCheck:
    """Public constraint check: the route starts and ends where asked and uses only real edges.

    Operational evidence only: it cannot tell whether the route is the cheapest, which is the goal.
    """

    name = "route"
    evidence: Evidence = "operational"

    def evaluate(
        self, subtask: Subtask, output: JsonValue, upstream: Mapping[str, JsonValue]
    ) -> CheckResult:
        route = route_of(output)
        if route is None:
            return CheckResult(
                check=self.name, evidence=self.evidence, verdict="fail", reason="no route in output"
            )
        start = leg_start(subtask, upstream)
        goal = str(subtask.inputs["goal"])
        if start is None:
            return CheckResult(
                check=self.name, evidence=self.evidence, verdict="unknown", reason="start not known"
            )
        if route[0] != start or route[-1] != goal:
            return CheckResult(
                check=self.name,
                verdict="fail",
                reason=f"route must go from {start} to {goal}, got {route[0]} to {route[-1]}",
            )
        if route_cost(route, edge_costs(subtask.inputs["edges"])) is None:
            return CheckResult(
                check=self.name,
                evidence=self.evidence,
                verdict="fail",
                reason="route uses a missing edge",
            )
        return CheckResult(check=self.name, evidence=self.evidence, verdict="pass")


class RouteActions:
    """MOVE(to) along an existing edge from the current node; FINISH once at the goal."""

    kinds: tuple[str, ...] = ("MOVE", FINISH)

    def __init__(self, subtask: Subtask) -> None:
        self.subtask = subtask
        self.costs = edge_costs(subtask.inputs["edges"])
        self.goal = str(subtask.inputs["goal"])

    def start(self, subtask: Subtask, upstream: Mapping[str, JsonValue]) -> list[str]:
        start = leg_start(subtask, upstream)
        return [start] if start is not None else []

    def legal(self, state: list[str], action: Action) -> str | None:
        if not state:
            return "start node unknown"
        if action.kind == FINISH:
            return None if state[-1] == self.goal else f"not at the goal {self.goal} yet"
        to = action.args.get("to")
        if action.kind != "MOVE" or not isinstance(to, str):
            return "expected MOVE with a string 'to'"
        if (state[-1], to) not in self.costs:
            return f"no edge {state[-1]} -> {to}"
        return None

    def apply(self, state: list[str], action: Action) -> tuple[list[str], str]:
        to = str(action.args["to"])
        return [*state, to], f"moved to {to} at cost {self.costs[(state[-1], to)]}"

    def finish(self, state: list[str], action: Action) -> JsonValue:
        return {"route": list(state)}

    def tool_call(self, state: list[str], action: Action, call_id: str) -> None:
        return None

    def observe(self, state: list[str], action: Action, result: object) -> tuple[list[str], str]:
        raise ValueError("route actions are local")

    def describe_actions(self) -> str:
        return (
            '{"kind": "MOVE", "args": {"to": "<next node along an existing edge>"}}\n'
            '{"kind": "FINISH", "args": {}} (only at the goal)'
        )

    def describe(self, state: list[str]) -> str:
        here = state[-1] if state else "?"
        out = sorted((v, w) for (u, v), w in self.costs.items() if u == here)
        return (
            f"Route so far: {json.dumps(state)}. Current node: {here}. Goal: {self.goal}. "
            f"Edges from {here}: {json.dumps([[v, w] for v, w in out])}."
        )


class ToyRoutes:
    name = "toy_routes"

    def action_space(self, subtask: Subtask) -> RouteActions:
        return RouteActions(subtask)

    def checks(self, subtask: Subtask) -> Sequence[RuntimeCheck]:
        return [SchemaCheck(), RouteCheck()]

    def aggregate(self, graph: TaskGraph, outputs: Mapping[str, JsonValue]) -> JsonValue:
        full: list[str] = []
        for node_id in graph.final_nodes:
            leg = route_of(outputs[node_id]) or []
            full += leg if not full else leg[1:]
        costs = edge_costs(graph.nodes[0].inputs["edges"])
        cost = route_cost(full, costs)
        return {"route": list(full), "cost": cost}
