"""PlanBench planning tasks (Valmeekam et al., 2023), public side: Blocksworld and Logistics.

A task is one subtask whose inputs are the PDDL domain and problem; the output is
{"plan": ["(action arg ...)", ...]}. Because the domain model is part of the input, a plan
validator is a legitimate runtime check (research plan, Section 4.4): illegal steps fail with
operational evidence, and a legal plan that reaches the goal passes with goal evidence (D15).
Ant's action kinds are the domain's operators plus FINISH.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from functools import lru_cache

from pydantic import JsonValue

from bioalloc.actions import FINISH, Action
from bioalloc.checks import RuntimeCheck, SchemaCheck
from bioalloc.graph import TaskGraph
from bioalloc.schemas import CheckResult, Evidence, Subtask
from bioalloc.tasks.strips import (
    Atom,
    Domain,
    PDDLError,
    Problem,
    applicable,
    apply,
    parse_domain,
    parse_problem,
    parse_step,
    simulate,
)

CONTRACT = "pddl_plan"
PLAN_SCHEMA: dict[str, JsonValue] = {
    "type": "object",
    "properties": {"plan": {"type": "array", "items": {"type": "string"}}},
    "required": ["plan"],
}
INSTRUCTION = (
    "Find a plan that solves the planning problem in the input 'problem' for the domain in the "
    "input 'domain' (both in PDDL). Reply with the plan as a list of ground actions, one string "
    'per step, e.g. {"plan": ["(pick-up a)", "(stack a b)"]}.'
)


@lru_cache(maxsize=64)
def _models(domain_text: str, problem_text: str) -> tuple[Domain, Problem]:
    return parse_domain(domain_text), parse_problem(problem_text)


def models(subtask: Subtask) -> tuple[Domain, Problem]:
    return _models(str(subtask.inputs["domain"]), str(subtask.inputs["problem"]))


def plan_of(output: JsonValue) -> list[str] | None:
    plan = output.get("plan") if isinstance(output, dict) else None
    if isinstance(plan, list) and all(isinstance(s, str) for s in plan):
        return [str(s) for s in plan]
    return None


class PlanCheck:
    """Simulates the plan on the public domain and problem."""

    name = "plan"
    evidence: Evidence = "goal"

    def evaluate(
        self, subtask: Subtask, output: JsonValue, upstream: Mapping[str, JsonValue]
    ) -> CheckResult:
        plan = plan_of(output)
        if plan is None:
            return CheckResult(
                check=self.name, verdict="fail", evidence="format", reason="no list of plan steps"
            )
        try:
            domain, problem = models(subtask)
        except PDDLError as exc:
            return CheckResult(
                check=self.name,
                verdict="unknown",
                evidence="operational",
                reason=f"cannot read the PDDL: {exc}",
            )
        sim = simulate(domain, problem, plan)
        if not sim.valid:
            return CheckResult(
                check=self.name, verdict="fail", evidence="operational", reason=str(sim.error)
            )
        if not sim.goal_reached:
            return CheckResult(
                check=self.name, verdict="fail", evidence="goal", reason=str(sim.error)
            )
        return CheckResult(check=self.name, verdict="pass", evidence="goal")


@dataclass
class _PlanState:
    facts: frozenset[Atom]
    plan: list[str] = field(default_factory=list)


class OperatorActions:
    """One action kind per domain operator (upper case) plus FINISH; args {"objects": [...]}."""

    def __init__(self, domain: Domain, problem: Problem) -> None:
        self.domain = domain
        self.problem = problem
        self.kinds: tuple[str, ...] = (*(op.upper() for op in sorted(domain.operators)), FINISH)

    def _step(self, action: Action) -> Atom:
        objects = action.args.get("objects", [])
        if not isinstance(objects, list):
            objects = []
        return (action.kind.lower(), *(str(o).lower() for o in objects))

    def start(self, subtask: Subtask, upstream: Mapping[str, JsonValue]) -> _PlanState:
        return _PlanState(self.problem.init)

    def legal(self, state: _PlanState, action: Action) -> str | None:
        if action.kind == FINISH:
            return None
        return applicable(self.domain, self.problem, state.facts, self._step(action))

    def apply(self, state: _PlanState, action: Action) -> tuple[_PlanState, str]:
        step = self._step(action)
        text = "(" + " ".join(step) + ")"
        return _PlanState(apply(self.domain, state.facts, step), [*state.plan, text]), f"did {text}"

    def finish(self, state: _PlanState, action: Action) -> JsonValue:
        return {"plan": list(state.plan)}

    def tool_call(self, state: _PlanState, action: Action, call_id: str) -> None:
        return None

    def observe(self, state: _PlanState, action: Action, result: object) -> tuple[_PlanState, str]:
        raise ValueError("planning actions are local")

    def describe_actions(self) -> str:
        lines = []
        for name in sorted(self.domain.operators):
            op = self.domain.operators[name]
            params = ", ".join(p.lstrip("?") for p in op.params)
            lines.append(f'{name.upper()} {{"objects": [{params}]}} (objects in this order)')
        lines.append("FINISH {} (when the goal holds)")
        return "\n".join(lines)

    def describe(self, state: _PlanState) -> str:
        facts = sorted(" ".join(a) for a in state.facts)
        goal = [" ".join(g) for g in self.problem.goal]
        return (
            f"True facts: {json.dumps(facts)}. Goal: {json.dumps(goal)}. "
            f"Plan so far: {json.dumps(state.plan)}."
        )


def normalize_plan(output: JsonValue) -> str | None:
    plan = plan_of(output)
    if plan is None:
        return None
    try:
        return json.dumps([" ".join(parse_step(s)) for s in plan])
    except PDDLError:
        return None


class PlanBench:
    name = "planbench"

    def checks(self, subtask: Subtask) -> Sequence[RuntimeCheck]:
        if subtask.contract == CONTRACT:
            return [SchemaCheck(), PlanCheck()]
        return []

    def aggregate(self, graph: TaskGraph, outputs: Mapping[str, JsonValue]) -> JsonValue:
        return outputs[graph.final_nodes[-1]]

    def normalize(self, subtask: Subtask, output: JsonValue) -> str | None:
        # Distinct valid plans stay distinct options: there is no public equivalence relation.
        return normalize_plan(output)

    def action_space(self, subtask: Subtask) -> OperatorActions | None:
        if "domain" not in subtask.inputs or "problem" not in subtask.inputs:
            return None
        domain, problem = models(subtask)
        return OperatorActions(domain, problem)
