"""Controlled synthetic route tasks, public side (decision D16).

A task has ``width`` independent chains; each chain has ``depth`` legs, and each leg is a small
layered network from its start to its goal with ``alternatives`` nodes per intermediate layer
(fully connected between layers, random positive costs), so a leg has alternatives^(hops-1)
candidate routes. Leg i+1 of a chain starts where leg i ended, so it depends on it. The factors
are set one at a time to test the hypotheses of RQ1: more alternatives favour search, more width
gives parallel siblings, more depth lengthens dependency chains.

When ``observable`` is true each leg carries a public ``cost_bound`` equal to its optimal cost, so
a public check can verify the goal (goal evidence, D15); otherwise only route validity is visible.
This makes "is the goal checkable" a controlled factor too.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence

from pydantic import JsonValue

from bioalloc.checks import RuntimeCheck, SchemaCheck
from bioalloc.graph import TaskGraph
from bioalloc.schemas import CheckResult, Evidence, Subtask
from bioalloc.tasks.toy import RouteActions, RouteCheck, edge_costs, route_cost, route_of

CONTRACT = "synthetic_leg"
_LEG_ID = re.compile(r"^c(\d+)-leg(\d+)$")


class CostBoundCheck:
    """Goal check for observable tasks: a valid route whose cost is within the public bound."""

    name = "cost_bound"
    evidence: Evidence = "goal"

    def evaluate(
        self, subtask: Subtask, output: JsonValue, upstream: Mapping[str, JsonValue]
    ) -> CheckResult:
        bound = subtask.inputs.get("cost_bound")
        route = route_of(output)
        if not isinstance(bound, int | float) or route is None:
            return CheckResult(
                check=self.name,
                verdict="unknown",
                evidence=self.evidence,
                reason="no bound or no route",
            )
        cost = route_cost(route, edge_costs(subtask.inputs["edges"]))
        if cost is None:
            return CheckResult(
                check=self.name,
                verdict="fail",
                evidence="operational",
                reason="route uses a missing edge",
            )
        if cost > bound:
            return CheckResult(
                check=self.name,
                verdict="fail",
                evidence=self.evidence,
                reason=f"route cost {cost} exceeds the bound {bound}",
            )
        return CheckResult(check=self.name, verdict="pass", evidence=self.evidence)


class SyntheticRoutes:
    name = "synthetic_routes"

    def checks(self, subtask: Subtask) -> Sequence[RuntimeCheck]:
        if subtask.contract != CONTRACT:
            return []
        checks: list[RuntimeCheck] = [SchemaCheck(), RouteCheck()]
        if "cost_bound" in subtask.inputs:
            checks.append(CostBoundCheck())
        return checks

    def aggregate(self, graph: TaskGraph, outputs: Mapping[str, JsonValue]) -> JsonValue:
        chains: dict[str, list[str]] = {}
        total = 0.0
        valid = True
        for node in graph.nodes:
            m = _LEG_ID.match(node.id)
            if m is None:
                continue
            route = route_of(outputs.get(node.id)) or []
            chain = chains.setdefault(f"c{m.group(1)}", [])
            chain += route if not chain else route[1:]
            cost = route_cost(route, edge_costs(node.inputs["edges"]))
            if cost is None:
                valid = False
            else:
                total += cost
        routes: dict[str, JsonValue] = {k: list[JsonValue](v) for k, v in sorted(chains.items())}
        return {"routes": routes, "cost": total if valid else None}

    def action_space(self, subtask: Subtask) -> RouteActions:
        return RouteActions(subtask)

    def normalize(self, subtask: Subtask, output: JsonValue) -> str | None:
        route = route_of(output)
        return "-".join(route) if route else None
