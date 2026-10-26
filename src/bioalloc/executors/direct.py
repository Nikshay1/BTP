"""Direct: one model call, no discussion. The reference point for every other executor."""

from __future__ import annotations

from dataclasses import dataclass

from bioalloc import prompts
from bioalloc.executors.base import ExecutionContext, ExecutorOutcome, parse_json, subtask_messages
from bioalloc.llm.base import LLMRequest
from bioalloc.schemas import Subtask


@dataclass
class Direct:
    model: str
    max_completion_tokens: int = 1024
    temperature: float = 1.0
    reasoning_effort: str | None = None
    name: str = "direct"

    async def execute(self, subtask: Subtask, ctx: ExecutionContext) -> ExecutorOutcome:
        request = LLMRequest(
            model=self.model,
            messages=tuple(subtask_messages(prompts.load("solve_subtask"), subtask, ctx)),
            max_completion_tokens=self.max_completion_tokens,
            temperature=self.temperature,
            reasoning_effort=self.reasoning_effort,  # type: ignore[arg-type]
            sample_id=ctx.agent,
        )
        reply = await ctx.llm.complete(request)
        return ExecutorOutcome(output=parse_json(reply.content))
