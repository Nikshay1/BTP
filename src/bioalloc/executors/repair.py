"""Scouts that repair their own rejected answers (decision D28; Sol's design in R14).

A schedule names which scout takes each solve episode, one episode at a time (per-task model
concurrency 1). A scout's first episode answers independently with the shared solve prompt and
Direct's bounded episode. Each later episode of the same scout revises with short memory: the base
subtask messages, its own latest answer and the latest feedback on it, worded as octopus words it
("The checks rejected that answer: ..." or "Your last step failed: ..."), never older attempts and
never another scout's. Only the final reply of an episode is remembered; tool calls inside it are
not (D28 limits this to tasks where that loses nothing, Logistics first).

After every episode a goal-verified answer (D15) ends the search at once (D27). An answer that
passes the public checks without a certificate commits its scout, which takes no further episodes.

Consensus decides what happens with committed answers:

* ``"bee"``: the module answers as soon as one option holds a quorum of ceil(theta * k) committed
  scouts (k is the registered population, launched or not). If the schedule ends without a quorum
  while answers are committed, D14's rounds follow: inhibition with probability ``sigma``, then
  recruitment of uncommitted scouts by one metered assessment each, then the quorum test; the
  answer is the option with most support, then most summed confidence, then canonical order.
* ``"plurality"``: the matched ordinary control. The same schedule, memory, feedback, limits and
  certificate stop, but no quorum stop, inhibition or recruitment: the full schedule runs and the
  plurality of committed answers is returned (canonical order breaks ties).

On Logistics every accepted plan is a certificate, so the two consensus modes behave identically
there (R14); they differ where several accepted but unverified answers can coexist.
"""

from __future__ import annotations

import dataclasses
from collections import Counter
from dataclasses import dataclass, field

from pydantic import JsonValue

from bioalloc import prompts
from bioalloc.budget import BudgetExceeded
from bioalloc.checks import goal_verified
from bioalloc.executors.base import (
    ExecutionContext,
    ExecutorOutcome,
    incompleteness,
    parse_json,
    subtask_messages,
    tool_episode,
)
from bioalloc.executors.bee import Bee, Scout
from bioalloc.llm.base import LLMError, LLMRequest, Message
from bioalloc.schemas import Subtask

# Launch orders registered in D28 (scout index per solve episode).
SHORT_MEMORY = (0, 0, 0, 0, 0, 0)
STAGED_TWO = (0, 0, 1, 0, 1, 0)
ALL_START_THREE = (0, 1, 2, 0, 1, 2)


@dataclass
class _Memory:
    last_reply: str | None = None
    feedback: str | None = None
    episodes: int = 0


@dataclass
class ScoutRepair(Bee):
    """Bee's scouts with self-repair under a fixed launch schedule (D28)."""

    schedule: tuple[int, ...] = STAGED_TWO
    consensus: str = "bee"  # "bee" or "plurality"
    theta: float = 0.6
    sigma: float = 0.2
    name: str = "bee_staged"
    _memory: dict[int, _Memory] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        super().__post_init__()
        self.k = max(self.schedule) + 1
        if self.consensus not in ("bee", "plurality"):
            raise ValueError(f"unknown consensus {self.consensus!r}")

    async def execute(self, subtask: Subtask, ctx: ExecutionContext) -> ExecutorOutcome:
        scouts = [Scout(i, f"{ctx.agent}/scout{i}") for i in range(self.k)]
        memory = {i: _Memory() for i in range(self.k)}
        budget_hit = False
        for slot, index in enumerate(self.schedule):
            scout = scouts[index]
            if scout.option is not None:
                continue  # committed scouts take no further episodes
            try:
                await self._episode(subtask, ctx, scout, memory[index], slot)
            except BudgetExceeded:
                budget_hit = True
                break
            except LLMError:
                continue  # a failed call is no answer; the slot is spent
            certified = self._certified(scouts)
            if certified is not None:
                return self._finish_certified(certified, scouts, ctx)
            support = self._support(scouts)
            if self.consensus == "bee" and support and max(support.values()) >= self.quorum:
                return self._finish(scouts, ctx, "quorum")
        ctx.emit(
            "repair_schedule_end",
            {
                "episodes": {str(i): m.episodes for i, m in memory.items()},
                "support": dict(self._support(scouts)),
                "budget": budget_hit,
            },
        )
        if self.consensus == "plurality":
            return self._plurality(scouts, ctx, budget_hit)
        return await self._consensus(subtask, ctx, scouts, budget_hit)

    async def _episode(
        self, subtask: Subtask, ctx: ExecutionContext, scout: Scout, mem: _Memory, slot: int
    ) -> None:
        messages = subtask_messages(prompts.load("solve_subtask"), subtask, ctx)
        if mem.last_reply is not None and mem.feedback is not None:
            messages += [
                Message(role="assistant", content=mem.last_reply, source=scout.agent),
                Message(role="user", content=mem.feedback),
            ]
        client = dataclasses.replace(ctx.llm, agent=scout.agent)
        request = LLMRequest(
            model=self.model,
            messages=tuple(messages),
            max_completion_tokens=self.max_completion_tokens,
            temperature=self.temperature,
            reasoning_effort=self.reasoning_effort,  # type: ignore[arg-type]
            seed=ctx.seed + 1000 * slot + scout.index,
            sample_id=f"{scout.agent}/e{mem.episodes}",
        )
        mem.episodes += 1
        episode = await tool_episode(dataclasses.replace(ctx, llm=client), request, 4)
        mem.last_reply = episode.reply.content
        problem = incompleteness(episode)
        if problem is not None:
            mem.feedback = f"Your last step failed: {problem}."
            ctx.emit("repair_episode", {"scout": scout.index, "slot": slot, "outcome": "cut off"})
            return
        output = parse_json(episode.reply.content)
        verdict, results = ctx.check(subtask, output)
        if verdict == "fail":
            reasons = "; ".join(f"{r.check}: {r.reason}" for r in results if r.verdict == "fail")
            mem.feedback = f"The checks rejected that answer: {reasons}"
            ctx.emit("repair_episode", {"scout": scout.index, "slot": slot, "outcome": "rejected"})
            return
        scout.option, scout.output = ctx.normalize(output), output
        scout.verified = goal_verified(results)
        outcome = "certificate" if scout.verified else "committed"
        ctx.emit("repair_episode", {"scout": scout.index, "slot": slot, "outcome": outcome})

    def _plurality(
        self, scouts: list[Scout], ctx: ExecutionContext, budget_hit: bool
    ) -> ExecutorOutcome:
        tally = Counter(s.option for s in scouts if s.option is not None)
        ctx.emit(
            "executor_stop", {"reason": "budget" if budget_hit else "votes", "tally": dict(tally)}
        )
        if not tally:
            if budget_hit:
                raise BudgetExceeded("budget ran out before any scout had a usable answer")
            return ExecutorOutcome(output=None, incomplete="no scout produced a usable answer")
        winner = min(tally, key=lambda o: (-tally[o], o))
        return ExecutorOutcome(output=next(s.output for s in scouts if s.option == winner))

    async def _consensus(
        self, subtask: Subtask, ctx: ExecutionContext, scouts: list[Scout], budget_hit: bool
    ) -> ExecutorOutcome:
        """D14's rounds over the committed answers, if any and the budget allows."""
        reason = "budget" if budget_hit else "round_limit"
        if not budget_hit and self._support(scouts):
            for rnd in range(1, self.max_rounds + 1):
                self._inhibit(scouts, ctx)
                support = self._support(scouts)
                if not support:
                    break
                uncommitted = [s for s in scouts if s.option is None]
                if uncommitted and await self._recruit(
                    subtask, ctx, scouts, uncommitted, support, rnd
                ):
                    reason = "budget"
                    break
                support = self._support(scouts)
                ctx.emit("bee_round", {"round": rnd, "support": dict(support)})
                if support and max(support.values()) >= self.quorum:
                    reason = "quorum"
                    break
        return self._finish(scouts, ctx, reason)


def repair_executors(model: str, **kw: JsonValue) -> dict[str, ScoutRepair]:
    """The D28 candidates and their matched plurality controls."""
    specs = {
        "scout_short_memory": (SHORT_MEMORY, "plurality"),
        "bee_staged": (STAGED_TWO, "bee"),
        "majority_staged": (STAGED_TWO, "plurality"),
        "bee_feedback3": (ALL_START_THREE, "bee"),
        "majority_feedback3": (ALL_START_THREE, "plurality"),
    }
    return {
        name: ScoutRepair(model=model, schedule=sched, consensus=cons, name=name, **kw)  # type: ignore[arg-type]
        for name, (sched, cons) in specs.items()
    }
