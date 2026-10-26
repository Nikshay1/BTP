"""The conventional coordinator with specialist agents (research plan, Section 4.4 (v); D18).

* planner (actor ``coordinator``): once per task, reads the stored graph (subtask ids,
  instructions, dependencies) and gives bounded advice per subtask; it cannot change the graph;
* executor: the ordinary act-observe worker, with the planner's advice for its subtask in context;
* critic (actor ``worker``): reviews each result against the public context and may ask for one
  revision, which the executor then makes with the critique in context.

The coordinator's public checks still decide acceptance, and its outer retries are the same as
for every system. The revision is part of one dispatch, not an extra retry. All three roles'
calls are metered against the task budget; advice and critique count as communication.
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
from dataclasses import dataclass

from bioalloc import prompts
from bioalloc.executors.base import ExecutionContext, ExecutorOutcome, parse_json, subtask_messages
from bioalloc.executors.octopus import Octopus
from bioalloc.llm.base import LLMRequest, Message
from bioalloc.schemas import Subtask

ADVICE_KEY = "specialist:advice"
LOCK_KEY = "specialist:lock"
GRAPH_KEY = "specialist:graph"  # set by the coordinator: [{id, instruction, depends_on}, ...]


@dataclass
class Specialist:
    model: str
    max_steps: int = 12
    max_completion_tokens: int = 2048
    temperature: float = 0.6
    reasoning_effort: str | None = None
    name: str = "specialist"

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

    async def _advice(self, ctx: ExecutionContext) -> dict[str, str]:
        lock: asyncio.Lock = ctx.task_state.setdefault(LOCK_KEY, asyncio.Lock())
        async with lock:
            if ADVICE_KEY in ctx.task_state:
                return ctx.task_state[ADVICE_KEY]  # type: ignore[no-any-return]
            graph = ctx.task_state.get(GRAPH_KEY, [])
            planner = dataclasses.replace(ctx.llm, actor="coordinator", agent="planner")
            messages = [
                Message(role="system", content=prompts.load("plan_graph")),
                Message(role="user", content=json.dumps(graph, indent=1)),
            ]
            reply = await planner.complete(self._request(messages, ctx, "planner", 1024))
            parsed = parse_json(reply.content)
            raw = parsed.get("advice") if isinstance(parsed, dict) else None
            advice = {str(k): str(v)[:500] for k, v in raw.items()} if isinstance(raw, dict) else {}
            ctx.task_state[ADVICE_KEY] = advice
            ctx.emit("specialist_plan", {"advice": advice})
            return advice

    def _worker(self) -> Octopus:
        return Octopus(
            model=self.model,
            max_steps=self.max_steps,
            max_completion_tokens=self.max_completion_tokens,
            temperature=self.temperature,
            reasoning_effort=self.reasoning_effort,
            name=self.name,
        )

    async def execute(self, subtask: Subtask, ctx: ExecutionContext) -> ExecutorOutcome:
        advice = (await self._advice(ctx)).get(subtask.id)
        context = (
            [Message(role="user", content=f"Advice: {advice}", source="planner")] if advice else []
        )
        outcome = await self._worker().execute_with_history(subtask, ctx, list(context))
        if outcome.incomplete is not None:
            return outcome
        critic = dataclasses.replace(ctx.llm, agent="critic")
        review = subtask_messages(prompts.load("review_result"), subtask, ctx)
        review += [
            Message(role="user", content="Worker's answer:"),
            Message(role="user", content=json.dumps(outcome.output), source=ctx.agent),
        ]
        verdict = parse_json(
            (await critic.complete(self._request(review, ctx, "critic", 512))).content
        )
        accept = isinstance(verdict, dict) and verdict.get("accept") is True
        critique = str(verdict.get("critique", ""))[:800] if isinstance(verdict, dict) else ""
        ctx.emit("specialist_review", {"accept": accept, "critique": critique})
        if accept or not critique:
            return outcome
        revision_context = [
            *context,
            Message(role="user", content="Your previous answer:"),
            Message(role="user", content=json.dumps(outcome.output), source=ctx.agent),
            Message(role="user", content=f"Reviewer: {critique}", source="critic"),
        ]
        return await self._worker().execute_with_history(subtask, ctx, revision_context)
