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
from bioalloc.checks import RuntimeCheck, combine
from bioalloc.llm.base import LLMReply, LLMRequest, Message
from bioalloc.schemas import CheckResult, ExecutorStatus, Subtask, Wire
from bioalloc.tools import ToolService


class Upstream(Wire):
    """Accepted output of a subtask this one depends on."""

    subtask_id: str
    agent: str
    output: JsonValue


class Feedback(Wire):
    """A rejected earlier attempt at the same subtask and the public reasons it was rejected."""

    agent: str
    output: JsonValue
    reasons: tuple[str, ...]


@dataclass
class ExecutionContext:
    llm: MeteredClient
    span_id: str
    upstream: dict[str, Upstream] = field(default_factory=dict)
    # Earlier rejected attempts at this subtask, oldest first (decision D8).
    feedback: tuple[Feedback, ...] = ()
    # Derived from the run seed, task, subtask and attempt; seeds the RNG and model sampling.
    seed: int = 0
    rng: random.Random = field(default_factory=lambda: random.Random(0))
    emit: Callable[[str, dict[str, Any]], None] = lambda kind, data: None
    # The subtask's public runtime checks, the same ones the coordinator applies afterwards.
    checks: tuple[RuntimeCheck, ...] = ()
    # Tools the subtask allows, or None if it allows none.
    tools: ToolService | None = None

    @property
    def agent(self) -> str:
        return self.llm.agent

    def check(self, subtask: Subtask, output: JsonValue) -> tuple[str, tuple[CheckResult, ...]]:
        """Run the public runtime checks on a candidate output (e.g. to score an ant path or to
        test an octopus goal). Costs no tokens; every call is traced."""
        plain = {k: v.output for k, v in self.upstream.items()}
        results = tuple(c.evaluate(subtask, output, plain) for c in self.checks)
        verdict = combine(list(results))
        self.emit(
            "executor_check",
            {"verdict": verdict, "checks": [r.model_dump(mode="json") for r in results]},
        )
        return verdict, results


class Executor(Protocol):
    name: str

    async def execute(self, subtask: Subtask, ctx: ExecutionContext) -> ExecutorOutcome: ...


@dataclass(frozen=True)
class ExecutorOutcome:
    """What an executor returns; the coordinator adds cost and span to build an ExecutorResult."""

    output: JsonValue
    status: ExecutorStatus = "completed"
    confidence: float | None = None
    # Set when the output comes from an unfinished generation (cut off at the token limit, or a
    # tool episode that ran out of completions). The coordinator then rejects it whatever it parses
    # to, so that a JSON fragment cannot pass as an answer.
    incomplete: str | None = None


def incompleteness(episode: Episode) -> str | None:
    if not episode.finished:
        return "the tool episode ended before a final answer"
    if episode.reply.finish_reason == "length":
        return "the reply was cut off at the completion limit"
    if episode.reply.finish_reason == "content_filter":
        return "the reply was filtered by the provider"
    return None


def subtask_messages(system: str, subtask: Subtask, ctx: ExecutionContext) -> list[Message]:
    """Standard messages for a subtask, one per source so that communication is counted only for
    content another agent wrote (D6): template, instruction, task inputs, upstream results, and
    the rejected earlier attempts with the public reasons for rejection."""
    messages = [
        Message(role="system", content=system),
        Message(role="user", content=f"Subtask: {subtask.instruction}", source=subtask.written_by),
    ]
    data = []
    if subtask.inputs:
        data.append("Inputs:\n" + json.dumps(subtask.inputs, indent=1))
    if subtask.output_schema is not None:
        data.append("Output schema:\n" + json.dumps(subtask.output_schema))
    if data:
        messages.append(Message(role="user", content="\n\n".join(data)))
    for dep in subtask.depends_on:
        up = ctx.upstream.get(dep)
        if up is not None:
            messages.append(Message(role="user", content=f"Result of subtask {dep}:"))
            messages.append(Message(role="user", content=json.dumps(up.output), source=up.agent))
    for fb in ctx.feedback:
        messages.append(Message(role="user", content="An earlier attempt answered:"))
        messages.append(Message(role="user", content=json.dumps(fb.output), source=fb.agent))
        reasons = "; ".join(fb.reasons) or "no reason given"
        messages.append(
            Message(role="user", content=f"That answer was rejected by the checks: {reasons}")
        )
    return messages


@dataclass(frozen=True)
class Episode:
    reply: LLMReply
    messages: tuple[Message, ...]
    completions: int
    finished: bool  # False if the completion limit was reached while the model still wanted tools


async def tool_episode(ctx: ExecutionContext, base: LLMRequest, max_completions: int) -> Episode:
    """One uninterrupted tool-use episode (D11): complete, run any requested tools, read the
    results back, and repeat until the model answers without tools or ``max_completions`` is hit.
    Without tools in the context this is exactly one completion."""
    messages = list(base.messages)
    tools = ctx.tools.wire() if ctx.tools is not None else ()
    for n in range(1, max_completions + 1):
        request = base.model_copy(
            update={
                "messages": tuple(messages),
                "tools": tools,
                "sample_id": f"{base.sample_id}/{n}",
            }
        )
        reply = await ctx.llm.complete(request)
        if not reply.tool_calls or ctx.tools is None:
            return Episode(reply, tuple(messages), n, True)
        if n == max_completions:
            break  # results could never be read back, so the tools are not run
        messages.append(
            Message(
                role="assistant",
                content=reply.content,
                source=ctx.agent,
                tool_calls=reply.tool_calls,
            )
        )
        for tc in reply.tool_calls:
            result = await ctx.tools.call(tc)
            messages.append(
                Message(role="tool", content=result.message_content(), tool_call_id=tc.id)
            )
    return Episode(reply, tuple(messages), max_completions, False)


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
