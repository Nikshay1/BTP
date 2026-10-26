"""Direct: one model call, no discussion. The reference point for every other executor.

Without tools it makes exactly one completion. With tools it runs one bounded tool episode (at most
``max_completions`` completions, reading tool results back) and nothing else: no resampling,
critique or check-driven revision (decision D11). The coordinator's outer retries still apply.
"""

from __future__ import annotations

from dataclasses import dataclass

from bioalloc import prompts
from bioalloc.executors.base import (
    ExecutionContext,
    ExecutorOutcome,
    incompleteness,
    parse_json,
    subtask_messages,
    tool_episode,
)
from bioalloc.llm.base import LLMRequest
from bioalloc.schemas import Subtask


@dataclass
class Direct:
    model: str
    max_completion_tokens: int = 1024
    temperature: float = 1.0
    reasoning_effort: str | None = None
    max_completions: int = 4
    name: str = "direct"

    async def execute(self, subtask: Subtask, ctx: ExecutionContext) -> ExecutorOutcome:
        request = LLMRequest(
            model=self.model,
            messages=tuple(subtask_messages(prompts.load("solve_subtask"), subtask, ctx)),
            max_completion_tokens=self.max_completion_tokens,
            temperature=self.temperature,
            reasoning_effort=self.reasoning_effort,  # type: ignore[arg-type]
            seed=ctx.seed,
            sample_id=ctx.agent,
        )
        episode = await tool_episode(ctx, request, self.max_completions)
        return ExecutorOutcome(
            output=parse_json(episode.reply.content), incomplete=incompleteness(episode)
        )
