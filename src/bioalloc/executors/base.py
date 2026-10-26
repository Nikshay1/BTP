"""The common executor interface.

An executor receives a subtask and an execution context. The context carries the outputs of the
subtasks it depends on, a metered model client whose account is the budget for this dispatch, a
seeded random generator, and an event hook into the run trace. Executors differ in control flow,
communication rules, memory and stopping conditions; prompts for shared steps are common.
"""

from __future__ import annotations

import json
import random
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol

from pydantic import JsonValue

from bioalloc.budget import MeteredClient
from bioalloc.llm.base import Message
from bioalloc.schemas import ExecutorStatus, Subtask, Wire


class Upstream(Wire):
    """Accepted output of a subtask this one depends on."""

    subtask_id: str
    agent: str
    output: JsonValue


@dataclass
class ExecutionContext:
    llm: MeteredClient
    span_id: str
    upstream: dict[str, Upstream] = field(default_factory=dict)
    rng: random.Random = field(default_factory=lambda: random.Random(0))
    emit: Callable[[str, dict[str, Any]], None] = lambda kind, data: None

    @property
    def agent(self) -> str:
        return self.llm.agent


class Executor(Protocol):
    name: str

    async def execute(self, subtask: Subtask, ctx: ExecutionContext) -> ExecutorOutcome: ...


@dataclass(frozen=True)
class ExecutorOutcome:
    """What an executor returns; the coordinator adds cost and span to build an ExecutorResult."""

    output: JsonValue
    status: ExecutorStatus = "completed"
    confidence: float | None = None


def subtask_messages(system: str, subtask: Subtask, ctx: ExecutionContext) -> list[Message]:
    """Standard messages for a subtask: template, task text, then each upstream result."""
    body = [f"Subtask: {subtask.instruction}"]
    if subtask.inputs:
        body.append("Inputs:\n" + json.dumps(subtask.inputs, indent=1))
    if subtask.output_schema is not None:
        body.append("Output schema:\n" + json.dumps(subtask.output_schema))
    messages = [
        Message(role="system", content=system),
        Message(role="user", content="\n\n".join(body), source=subtask.written_by),
    ]
    for dep in subtask.depends_on:
        up = ctx.upstream.get(dep)
        if up is not None:
            messages.append(
                Message(
                    role="user",
                    content=f"Result of subtask {dep}:\n{json.dumps(up.output)}",
                    source=up.agent,
                )
            )
    return messages


_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


def parse_json(text: str) -> JsonValue:
    """Parse a JSON reply, tolerating code fences and surrounding prose. Returns the raw text if
    no JSON can be found, so that the runtime checks (not the parser) decide what fails."""
    candidates = [text.strip()]
    candidates += [m.strip() for m in _FENCE.findall(text)]
    start, end = text.find("{"), text.rfind("}")
    if 0 <= start < end:
        candidates.append(text[start : end + 1])
    for c in candidates:
        try:
            value: JsonValue = json.loads(c)
            return value
        except json.JSONDecodeError:
            continue
    return text
