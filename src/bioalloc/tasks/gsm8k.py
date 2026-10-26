"""GSM8K: grade-school math word problems (Cobbe et al., 2021), public side.

Each task is one subtask: the question, an output schema {"answer": number}, and a calculator. There
is no goal check: a system cannot verify the answer, only its format, so acceptance here is weak
evidence (D15). Bee normalizes answers numerically. Ant's action kinds are NOTE (record a quantity
from the text), CALCULATE (evaluate an expression into a named quantity) and FINISH; the same
calculator is offered to every other executor as a tool.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from pydantic import JsonValue

from bioalloc.actions import FINISH, Action
from bioalloc.checks import RuntimeCheck, SchemaCheck
from bioalloc.graph import TaskGraph
from bioalloc.llm.base import ToolCall
from bioalloc.schemas import CheckResult, Evidence, Subtask
from bioalloc.tasks.common import CALCULATOR, normalize_number
from bioalloc.tools import Tool, ToolResult

CONTRACT = "gsm8k_answer"
ANSWER_SCHEMA: dict[str, JsonValue] = {
    "type": "object",
    "properties": {"answer": {"type": ["number", "string"]}},
    "required": ["answer"],
}
INSTRUCTION = (
    "Solve the math word problem in the input 'question'. Reply with the final numeric answer "
    'as {"answer": <number>}.'
)


class NumericAnswerCheck:
    """The answer must be a single number. Format evidence only."""

    name = "numeric"
    evidence: Evidence = "format"

    def evaluate(
        self, subtask: Subtask, output: JsonValue, upstream: Mapping[str, JsonValue]
    ) -> CheckResult:
        value = output.get("answer") if isinstance(output, dict) else None
        if normalize_number(value) is None:
            return CheckResult(
                check=self.name,
                verdict="fail",
                evidence=self.evidence,
                reason="the answer is not a single number",
            )
        return CheckResult(check=self.name, verdict="pass", evidence=self.evidence)


@dataclass
class _State:
    quantities: dict[str, float | int] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)


class ArithmeticActions:
    """NOTE records a quantity read from the problem; CALCULATE evaluates an expression over
    numbers and known quantity names with the calculator tool; FINISH gives the answer."""

    kinds: tuple[str, ...] = ("NOTE", "CALCULATE", FINISH)

    def start(self, subtask: Subtask, upstream: Mapping[str, JsonValue]) -> _State:
        return _State()

    def describe_actions(self) -> str:
        return (
            'NOTE {"name": "<new quantity name>", "value": <number from the problem>}\n'
            'CALCULATE {"name": "<new quantity name>", "expression": "<arithmetic over numbers '
            'and known quantity names>"} (runs the calculator)\n'
            'FINISH {"answer": <number, or the name of a known quantity>}'
        )

    def legal(self, state: _State, action: Action) -> str | None:
        a = action.args
        if action.kind == "NOTE":
            ok = isinstance(a.get("name"), str) and isinstance(a.get("value"), int | float)
            return None if ok else "NOTE needs a string 'name' and a numeric 'value'"
        if action.kind == "CALCULATE":
            if not isinstance(a.get("name"), str) or not isinstance(a.get("expression"), str):
                return "CALCULATE needs a string 'name' and a string 'expression'"
            return None
        if action.kind == FINISH:
            target = a.get("answer")
            if isinstance(target, str) and target in state.quantities:
                return None
            return None if normalize_number(target) is not None else "FINISH needs an 'answer'"
        return f"unknown action {action.kind}"

    def _substitute(self, state: _State, expression: str) -> str:
        names = sorted(state.quantities, key=len, reverse=True)
        if not names:
            return expression
        pattern = re.compile(r"\b(" + "|".join(re.escape(n) for n in names) + r")\b")
        return pattern.sub(lambda m: f"({state.quantities[m.group(1)]})", expression)

    def apply(self, state: _State, action: Action) -> tuple[_State, str]:
        name = str(action.args["name"])
        value = action.args["value"]
        assert isinstance(value, int | float)
        return _State({**state.quantities, name: value}, [*state.notes, name]), f"{name} = {value}"

    def tool_call(self, state: _State, action: Action, call_id: str) -> ToolCall | None:
        if action.kind != "CALCULATE":
            return None
        expression = self._substitute(state, str(action.args["expression"]))
        return ToolCall(
            id=call_id, name="calculate", arguments=json.dumps({"expression": expression})
        )

    def observe(self, state: _State, action: Action, result: ToolResult) -> tuple[_State, str]:
        name = str(action.args["name"])
        if not result.ok or not isinstance(result.output, int | float):
            return state, f"calculation failed: {result.error}"
        return (
            _State({**state.quantities, name: result.output}, [*state.notes, name]),
            f"{name} = {result.output}",
        )

    def finish(self, state: _State, action: Action) -> JsonValue:
        target = action.args.get("answer")
        if isinstance(target, str) and target in state.quantities:
            return {"answer": state.quantities[target]}
        return {"answer": target}

    def describe(self, state: _State) -> str:
        return f"Known quantities: {json.dumps(state.quantities)}"


class GSM8K:
    name = "gsm8k"

    def checks(self, subtask: Subtask) -> Sequence[RuntimeCheck]:
        if subtask.contract == CONTRACT:
            return [SchemaCheck(), NumericAnswerCheck()]
        return []

    def aggregate(self, graph: TaskGraph, outputs: Mapping[str, JsonValue]) -> JsonValue:
        return outputs[graph.final_nodes[-1]]

    def normalize(self, subtask: Subtask, output: JsonValue) -> str | None:
        if subtask.contract != CONTRACT:
            return None
        return normalize_number(output.get("answer") if isinstance(output, dict) else None)

    def action_space(self, subtask: Subtask) -> ArithmeticActions:
        return ArithmeticActions()

    def tools(self) -> dict[str, Tool]:
        return {"calculate": CALCULATOR}
