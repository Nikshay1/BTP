"""Octopus: bounded local execution with a private scratchpad, reporting only at the end.

The worker receives the subtask and a local budget and runs its own act-observe loop. Each step is
one completion plus the tools it calls. When the model answers without tools, the answer is run
through the public checks: a goal-verified answer ends the loop (D15); a rejected answer goes back
to the worker with the public reasons; an answer that passes without goal evidence is returned as
unverified. The transcript is private and dropped after the report. Limits: ``max_steps`` model
steps (tool calls are bounded by the task's tool budget). Any answer no check rejects is returned at
once, so the loop only continues after a rejection; if it ends without an answer (step limit,
supervisor stop) the outcome is marked incomplete and the coordinator rejects it. Running out of
budget mid-loop is reported as budget exhaustion.

``report_every_step=True`` is the plain counterpart in Table 3 of the research plan: the same
worker, but after every step that does not end the dispatch (each tool step and each rejected
answer) it reports to a coordinator model call (actor ``coordinator``) that
answers continue or stop and may add one instruction. The report and the coordinator's reply are
both charged, and the instruction counts as communication. The same class with
``report_every_step=False`` and no octopus-specific behaviour is the ordinary act-observe worker of
the conventional baselines (D18); the two are close computationally, which the paper states.
"""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass

from pydantic import JsonValue

from bioalloc import prompts
from bioalloc.checks import goal_verified
from bioalloc.executors.base import (
    ExecutionContext,
    ExecutorOutcome,
    parse_json,
    subtask_messages,
)
from bioalloc.llm.base import LLMRequest, Message
from bioalloc.schemas import Subtask

REPORT_CHARS = 1500


@dataclass
class Octopus:
    model: str
    max_steps: int = 12
    max_completion_tokens: int = 2048
    temperature: float = 0.6
    reasoning_effort: str | None = None
    report_every_step: bool = False
    name: str = "octopus"

    def _request(self, messages: list[Message], ctx: ExecutionContext, step: int) -> LLMRequest:
        return LLMRequest(
            model=self.model,
            messages=tuple(messages),
            max_completion_tokens=self.max_completion_tokens,
            temperature=self.temperature,
            reasoning_effort=self.reasoning_effort,  # type: ignore[arg-type]
            seed=ctx.seed + step,
            sample_id=f"{ctx.agent}/step{step}",
        )

    async def execute(self, subtask: Subtask, ctx: ExecutionContext) -> ExecutorOutcome:
        return await self.execute_with_history(subtask, ctx, None)

    async def execute_with_history(
        self, subtask: Subtask, ctx: ExecutionContext, history: list[Message] | None
    ) -> ExecutorOutcome:
        """Run the loop. With ``history`` (the single agent's), earlier subtasks' messages come
        first and this subtask's messages are appended to it, so the worker remembers."""
        messages = subtask_messages(prompts.load("solve_subtask"), subtask, ctx)
        if history is not None:
            messages = [messages[0], *history, *messages[1:]]

        def stop(reason: str, output: JsonValue = None) -> ExecutorOutcome:
            ctx.emit("executor_stop", {"reason": reason})
            if history is not None:
                history[:] = messages[1:]  # everything but the system prompt
            if output is None:
                return ExecutorOutcome(output=None, incomplete=f"no answer ({reason})")
            return ExecutorOutcome(output=output)

        for step in range(self.max_steps):
            # One step = one completion, plus the tools it calls (an act-observe step).
            request = self._request(messages, ctx, step)
            if ctx.tools is not None:
                request = request.model_copy(update={"tools": ctx.tools.wire()})
            reply = await ctx.llm.complete(request)
            last = step == self.max_steps - 1
            if reply.tool_calls and ctx.tools is not None and last:
                break  # no later step could read the tool results, so they are not run
            if reply.tool_calls and ctx.tools is not None:
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
                if self.report_every_step:
                    decision, instruction = await self._supervise(messages, ctx, step)
                    if decision == "stop":
                        return stop("supervisor_stop")
                    if instruction:
                        messages.append(
                            Message(
                                role="user", content=instruction, source=f"supervisor:{ctx.agent}"
                            )
                        )
                continue
            messages.append(Message(role="assistant", content=reply.content, source=ctx.agent))
            problem = {
                "length": "the reply was cut off at the completion limit",
                "content_filter": "the reply was filtered by the provider",
            }.get(reply.finish_reason)
            if problem is not None:
                messages.append(Message(role="user", content=f"Your last step failed: {problem}."))
            else:
                candidate = parse_json(reply.content)
                verdict, results = ctx.check(subtask, candidate)
                if goal_verified(results):
                    return stop("goal_verified", candidate)
                if verdict != "fail":
                    return stop("unverified_answer", candidate)
                reasons = "; ".join(
                    f"{r.check}: {r.reason}" for r in results if r.verdict == "fail"
                )
                messages.append(
                    Message(role="user", content=f"The checks rejected that answer: {reasons}")
                )
            if self.report_every_step:
                decision, instruction = await self._supervise(messages, ctx, step)
                if decision == "stop":
                    return stop("supervisor_stop")
                if instruction:
                    messages.append(
                        Message(role="user", content=instruction, source=f"supervisor:{ctx.agent}")
                    )
        return stop("step_limit")

    async def _supervise(
        self, messages: list[Message], ctx: ExecutionContext, step: int
    ) -> tuple[str, str]:
        """The per-step report and the coordinator's reply (plain counterpart only)."""
        report = json.dumps([m.content for m in messages[-3:]])[-REPORT_CHARS:]
        supervisor = dataclasses.replace(
            ctx.llm, actor="coordinator", agent=f"supervisor:{ctx.agent}"
        )
        request = LLMRequest(
            model=self.model,
            messages=(
                Message(role="system", content=prompts.load("supervise_step")),
                Message(role="user", content=report, source=ctx.agent),
            ),
            max_completion_tokens=256,
            temperature=self.temperature,
            reasoning_effort=self.reasoning_effort,  # type: ignore[arg-type]
            seed=ctx.seed + 10_000 + step,
            sample_id=f"supervisor:{ctx.agent}/step{step}",
        )
        reply = await supervisor.complete(request)
        parsed = parse_json(reply.content, envelope="decision")
        if isinstance(parsed, dict):
            decision = "stop" if parsed.get("decision") == "stop" else "continue"
            instruction = str(parsed.get("instruction") or "")[:500]
            return decision, instruction
        return "continue", ""
