"""The common executor interface.

An executor receives a subtask and an execution context. The context carries the outputs of the
subtasks it depends on, a metered model client whose account is the budget for this dispatch, a
seeded random generator, and an event hook into the run trace. Executors differ in control flow,
communication rules, memory and stopping conditions; prompts for shared steps are common.
"""

from __future__ import annotations

import json
import random
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol

from pydantic import JsonValue

from bioalloc.actions import ActionSpace
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
    # The family's typed action space for this subtask, if it has one (D10).
    actions: ActionSpace | None = None
    # State private to one task run, shared by attempts within it and dropped afterwards
    # (e.g. ant weights, D10). Never shared across tasks, systems or repetitions.
    task_state: dict[str, Any] = field(default_factory=dict)
    # Maps a candidate output to the option it supports, or None if it is not a usable answer.
    # Families give a target-independent normalizer; the default is canonical JSON (D14).
    normalize: Callable[[JsonValue], str | None] = field(default_factory=lambda: canonical_option)

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


def canonical_option(output: JsonValue) -> str | None:
    """Default normalizer: canonical JSON; raw unparsed text and empty outputs are not options."""
    if output is None or isinstance(output, str):
        return None
    return json.dumps(output, sort_keys=True, separators=(",", ":"))


_OPEN = {"{": "}", "[": "]"}
_LOOKS_LIKE = {"{": '"}', "[": '{["'}  # what may follow the opener (after spaces) in JSON


def _looks_like_json(text: str, i: int) -> bool:
    """Whether the bracket at ``i`` opens something that looks like JSON: an object starting
    with a key (or empty), or an array of objects, arrays or strings. Prose such as
    "\frac{1}{2}", "{n}", "[1]" or "[note]" does not."""
    j = i + 1
    while j < len(text) and text[j] in " \t\r\n":
        j += 1
    return j < len(text) and text[j] in _LOOKS_LIKE[text[i]]


def _end_of(text: str, i: int) -> tuple[int, bool]:
    """(index after the container opened at ``i``, whether its brackets nest correctly). If they
    do not, the end is where the outer bracket's own kind balances again, ignoring the other kind,
    so recovery never resumes inside it; -1 if it never closes."""
    opener = text[i]
    stack, depth, nested_ok = [_OPEN[opener]], 1, True
    in_string = escaped = False
    for j in range(i + 1, len(text)):
        c = text[j]
        if in_string:
            if escaped:
                escaped = False
            elif c == "\\":
                escaped = True
            elif c == '"':
                in_string = False
            continue
        if c == '"':
            in_string = True
        elif c in _OPEN:
            depth += c == opener
            if nested_ok:
                stack.append(_OPEN[c])
        elif c in "}]":
            depth -= c == _OPEN[opener]
            if nested_ok:
                if c == stack[-1]:
                    stack.pop()
                else:
                    nested_ok = False
            if nested_ok and not stack:
                return j + 1, True
            if not nested_ok and depth == 0:
                return j + 1, False
    return -1, False


def _scan_top_level(text: str) -> list[JsonValue | None]:
    """The JSON-looking containers that start outside any other one, in order: each its value,
    or None if it is malformed (bad nesting, never closed, or not valid JSON). A malformed
    container is skipped whole, nothing inside it is ever taken, and one that never closes ends
    the scan."""
    found: list[JsonValue | None] = []
    i = 0
    while i < len(text):
        if text[i] not in _OPEN or not _looks_like_json(text, i):
            i += 1
            continue
        end, ok = _end_of(text, i)
        if end == -1:
            found.append(None)
            break
        value: JsonValue | None = None
        if ok:
            try:
                value = json.loads(text[i:end])
            except ValueError:  # invalid JSON, or a number beyond Python's conversion limit
                value = None
        found.append(value)
        i = end
    return found


def parse_json(text: str, envelope: str | None = None) -> JsonValue:
    """Parse a JSON reply, tolerating code fences and surrounding prose. Returns the raw text if
    no answer can be read, so that the runtime checks (not the parser) decide what fails.

    The answer is the last top-level JSON value in the reply: a model that answers, notices a
    mistake and answers again means its last object (the prompts ask it to write the corrected
    object last and no example objects). If the last JSON-looking container is malformed, there is
    no answer: an earlier value or a fragment of the broken one is never taken instead. Brackets
    in prose that do not look like JSON are ignored. With ``envelope``, the answer is the last
    top-level object holding that protocol key; well-formed objects without it after it (a quoted
    item, say) are passed over, but a malformed container after it means no answer. The rule is
    the same for every system and never looks at whether a value would pass a check."""
    try:
        whole: JsonValue = json.loads(text.strip())
    except ValueError:
        pass
    else:
        if envelope is None or (isinstance(whole, dict) and envelope in whole):
            return whole
        return text
    found = _scan_top_level(text)
    for value in reversed(found):
        if value is None:
            return text
        if envelope is None or (isinstance(value, dict) and envelope in value):
            return value
    return text
