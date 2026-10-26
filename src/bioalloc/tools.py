"""Deterministic tools and the restricted tool service executors use.

Tools are local, deterministic functions with a JSON Schema for their arguments. The service only
runs tools the subtask allows, validates arguments before running, bounds run time and output size,
counts every attempted execution (failures included) against the budget, and traces each call.
Tool output is not written by an agent, so reading it is not communication (D6).
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

import jsonschema
from pydantic import JsonValue

from bioalloc.budget import Account
from bioalloc.llm.base import ToolCall
from bioalloc.schemas import Wire


class ToolSpec(Wire):
    name: str
    description: str
    parameters: dict[str, JsonValue]

    def wire(self) -> dict[str, JsonValue]:
        """OpenAI-compatible function tool definition."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


@dataclass(frozen=True)
class Tool:
    spec: ToolSpec
    run: Callable[[dict[str, Any]], JsonValue]


class ToolResult(Wire):
    call_id: str
    name: str
    ok: bool
    output: JsonValue = None
    error: str | None = None

    def message_content(self) -> str:
        if self.ok:
            return json.dumps(self.output)
        return json.dumps({"error": self.error})


@dataclass
class ToolService:
    registry: Mapping[str, Tool]
    allowed: tuple[str, ...]
    account: Account
    emit: Callable[[str, dict[str, Any]], None] = lambda kind, data: None
    timeout_s: float = 5.0
    max_output_chars: int = 4000
    _count: int = field(default=0, init=False)

    def specs(self) -> tuple[ToolSpec, ...]:
        return tuple(self.registry[n].spec for n in self.allowed if n in self.registry)

    def wire(self) -> tuple[dict[str, JsonValue], ...]:
        return tuple(s.wire() for s in self.specs())

    async def call(self, tc: ToolCall) -> ToolResult:
        self.account.record_tool_call()  # raises BudgetExceeded at the limit; nothing runs
        result = await self._run(tc)
        self.emit(
            "tool_call",
            {
                "call_id": tc.id,
                "name": tc.name,
                "arguments": tc.arguments,
                "ok": result.ok,
                "error": result.error,
                "output": result.output,
            },
        )
        return result

    async def _run(self, tc: ToolCall) -> ToolResult:
        def fail(error: str) -> ToolResult:
            return ToolResult(call_id=tc.id, name=tc.name, ok=False, error=error)

        if tc.name not in self.allowed or tc.name not in self.registry:
            return fail(f"tool {tc.name!r} is not available for this subtask")
        tool = self.registry[tc.name]
        try:
            args = json.loads(tc.arguments or "{}")
        except json.JSONDecodeError as exc:
            return fail(f"arguments are not valid JSON: {exc.msg}")
        if not isinstance(args, dict):
            return fail("arguments must be a JSON object")
        try:
            jsonschema.validate(args, tool.spec.parameters)
        except jsonschema.ValidationError as exc:
            return fail(f"invalid arguments: {exc.message}")
        try:
            output = await asyncio.wait_for(asyncio.to_thread(tool.run, args), self.timeout_s)
        except TimeoutError:
            return fail(f"tool timed out after {self.timeout_s} s")
        except Exception as exc:  # a tool error is an observation for the model, not a crash
            return fail(f"{type(exc).__name__}: {exc}")
        text = json.dumps(output)
        if len(text) > self.max_output_chars:
            return fail(f"output too large ({len(text)} characters)")
        return ToolResult(call_id=tc.id, name=tc.name, ok=True, output=output)
