"""Ordinary workers for the conventional baselines (decision D18).

* ``act_observe``: the same bounded act-observe loop as octopus (D18: shared code), so the
  conventional pool is not artificially weak. Octopus and this worker coincide computationally;
  the paper says so, and RQ3 rests on the components that do differ (ant vs plan-then-execute,
  bee vs majority vote) and on the component controls.
* ``plan_then_execute``: one metered planning call, then Direct's bounded episode with that plan in
  context; no pheromone table, quorum or inhibition.
* ``majority``: ``MajorityVote`` from the bee module.
* ``single_agent``: one persistent worker that solves every subtask of a task in turn, keeping its
  own history across subtasks; the coordinator runs it serially in a fixed order and gives it one
  identity, so reading its own earlier results is not communication.
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
from bioalloc.executors.octopus import Octopus
from bioalloc.llm.base import LLMRequest, Message
from bioalloc.schemas import Subtask


def act_observe(model: str, **kw: object) -> Octopus:
    return Octopus(model=model, name="act_observe", **kw)  # type: ignore[arg-type]


@dataclass
class PlanThenExecute:
    model: str
    max_completion_tokens: int = 2048
    temperature: float = 0.6
    reasoning_effort: str | None = None
    max_completions: int = 4
    name: str = "plan_then_execute"

    def _request(
        self, messages: list[Message], ctx: ExecutionContext, tag: str, limit: int
    ) -> LLMRequest:
        return LLMRequest(
            model=self.model,
            messages=tuple(messages),
            max_completion_tokens=limit,
            temperature=self.temperature,
            reasoning_effort=self.reasoning_effort,  # type: ignore[arg-type]
            seed=ctx.seed,
            sample_id=f"{ctx.agent}/{tag}",
        )

    async def execute(self, subtask: Subtask, ctx: ExecutionContext) -> ExecutorOutcome:
        base = subtask_messages(prompts.load("solve_subtask"), subtask, ctx)
        planning = [Message(role="system", content=prompts.load("plan_steps")), *base[1:]]
        plan = await ctx.llm.complete(self._request(planning, ctx, "plan", 512))
        messages = [
            *base,
            Message(role="assistant", content=plan.content, source=ctx.agent),
            Message(role="user", content="Now carry out the plan and give the answer."),
        ]
        episode = await tool_episode(
            ctx,
            self._request(messages, ctx, "execute", self.max_completion_tokens),
            self.max_completions,
        )
        return ExecutorOutcome(
            output=parse_json(episode.reply.content), incomplete=incompleteness(episode)
        )


HISTORY_KEY = "single_agent:history"


@dataclass
class SingleAgent:
    """One persistent act-observe worker over the whole task graph (D11)."""

    model: str
    max_steps: int = 12
    max_completion_tokens: int = 2048
    temperature: float = 0.6
    reasoning_effort: str | None = None
    name: str = "single_agent"
    # The coordinator gives this executor one agent identity for the whole task and runs subtasks
    # one at a time in declaration (topological) order.
    persistent: bool = True

    async def execute(self, subtask: Subtask, ctx: ExecutionContext) -> ExecutorOutcome:
        history: list[Message] = ctx.task_state.setdefault(HISTORY_KEY, [])
        worker = Octopus(
            model=self.model,
            max_steps=self.max_steps,
            max_completion_tokens=self.max_completion_tokens,
            temperature=self.temperature,
            reasoning_effort=self.reasoning_effort,
            name=self.name,
        )
        before = len(history)
        outcome = await worker.execute_with_history(subtask, ctx, history)
        ctx.emit(
            "single_agent_history", {"messages_before": before, "messages_after": len(history)}
        )
        return outcome
