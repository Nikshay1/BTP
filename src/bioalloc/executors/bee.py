"""Bee: evidence accumulation with cross inhibition and a quorum (Section 4.2; decision D14).

``k`` scouts answer the subtask independently, each with Direct's bounded episode and the shared
solve prompt, under distinct seeds. A scout whose answer is usable (it parses, finished, and no
public check rejects it) commits to the option its normalized answer represents; otherwise it stays
uncommitted. The module answers as soon as one option holds a quorum of ceil(theta * k) scouts.

Each later round works on snapshots and applies its decisions together:

1. inhibition: every committed scout whose option has a rival receives one stop signal and returns
   to the uncommitted state with probability ``sigma``;
2. recruitment: each uncommitted scout is shown the answer of a supporter of an option drawn in
   proportion to the remaining support, and decides with one metered assessment call whether to
   commit to it; if no option has supporters left, uncommitted scouts answer independently again;
3. the quorum is tested.

With ``sigma = 0`` and every scout committed, nothing can change: the module stops at once and
returns the plurality option (the degenerate case is traced). Otherwise it stops at quorum, after
``max_rounds`` rounds, or when the budget runs out, returning the option with most support, then
highest summed acceptance confidence, then the first in canonical order.

``MajorityVote`` is the plain counterpart: independent answers and a plurality vote, no
recruitment, inhibition or quorum. With ``max_calls`` it keeps sampling until a given call
allowance is spent (the call-yoked control); it is also the ordinary majority-vote worker of the
conventional baselines (D18).
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
import math
from collections import Counter
from dataclasses import dataclass, field

from pydantic import JsonValue

from bioalloc import prompts
from bioalloc.budget import BudgetExceeded, MeteredClient
from bioalloc.checks import goal_verified
from bioalloc.executors.base import (
    ExecutionContext,
    ExecutorOutcome,
    incompleteness,
    parse_json,
    subtask_messages,
    tool_episode,
)
from bioalloc.llm.base import LLMError, LLMRequest, Message
from bioalloc.schemas import Budget, Subtask


@dataclass
class Scout:
    index: int
    agent: str
    option: str | None = None
    output: JsonValue = None
    confidence: float = 0.0
    # The last option this scout committed to, kept when it is released, so that a run cut short
    # right after inhibition still returns an answer it had.
    last_option: str | None = None
    last_output: JsonValue = None
    # The scout's own answer passed a goal-level public check (D15): a certificate.
    verified: bool = False


@dataclass
class _Answerer:
    model: str
    max_completion_tokens: int
    temperature: float
    reasoning_effort: str | None

    async def answer(
        self,
        subtask: Subtask,
        ctx: ExecutionContext,
        agent: str,
        seed: int,
        tag: str,
        client: MeteredClient | None = None,
    ) -> tuple[str | None, JsonValue, bool]:
        """One independent answer through Direct's bounded episode; (option, output, whether a
        goal-level check verified it)."""
        client = dataclasses.replace(client or ctx.llm, agent=agent)
        sub_ctx = dataclasses.replace(ctx, llm=client)
        request = LLMRequest(
            model=self.model,
            messages=tuple(subtask_messages(prompts.load("solve_subtask"), subtask, ctx)),
            max_completion_tokens=self.max_completion_tokens,
            temperature=self.temperature,
            reasoning_effort=self.reasoning_effort,  # type: ignore[arg-type]
            seed=seed,
            sample_id=f"{agent}/{tag}",
        )
        episode = await tool_episode(sub_ctx, request, 4)
        if incompleteness(episode) is not None:
            return None, None, False
        output = parse_json(episode.reply.content)
        verdict, results = ctx.check(subtask, output)
        if verdict == "fail":
            return None, output, False
        return ctx.normalize(output), output, goal_verified(results)


async def _gather_budgeted(coros: list) -> tuple[list, bool]:  # type: ignore[type-arg]
    """Run concurrently; a call refused for budget or failed counts as no answer."""
    results = await asyncio.gather(*coros, return_exceptions=True)
    out: list = []  # type: ignore[type-arg]
    budget_hit = False
    for r in results:
        if isinstance(r, BudgetExceeded):
            budget_hit = True
            out.append(None)
        elif isinstance(r, LLMError):
            out.append(None)
        elif isinstance(r, BaseException):
            raise r
        else:
            out.append(r)
    return out, budget_hit


@dataclass
class Bee:
    model: str
    k: int = 5
    theta: float = 0.8
    sigma: float = 0.2
    max_rounds: int = 5
    max_completion_tokens: int = 2048
    temperature: float = 0.6
    reasoning_effort: str | None = None
    name: str = "bee"
    # D27: return a goal-verified scout answer at once instead of waiting for a quorum.
    stop_on_certificate: bool = True
    _answerer: _Answerer = field(init=False)

    def __post_init__(self) -> None:
        self._answerer = _Answerer(
            self.model, self.max_completion_tokens, self.temperature, self.reasoning_effort
        )

    @property
    def quorum(self) -> int:
        return math.ceil(self.theta * self.k)

    async def execute(self, subtask: Subtask, ctx: ExecutionContext) -> ExecutorOutcome:
        scouts = [Scout(i, f"{ctx.agent}/scout{i}") for i in range(self.k)]
        budget_hit = await self._independent(subtask, ctx, scouts, 0)
        reason = "round_limit"
        for rnd in range(1, self.max_rounds + 1):
            support = self._support(scouts)
            ctx.emit("bee_round", {"round": rnd - 1, "support": dict(support)})
            certified = self._certified(scouts)
            if certified is not None:
                return self._finish_certified(certified, scouts, ctx)
            if support and max(support.values()) >= self.quorum:
                reason = "quorum"
                break
            if budget_hit:
                reason = "budget"
                break
            uncommitted = [s for s in scouts if s.option is None]
            if self.sigma == 0 and not uncommitted:
                reason = "absorbing"
                break
            self._inhibit(scouts, ctx)
            support = self._support(scouts)
            uncommitted = [s for s in scouts if s.option is None]
            if not support:
                budget_hit = await self._independent(subtask, ctx, uncommitted, rnd)
            else:
                budget_hit = await self._recruit(subtask, ctx, scouts, uncommitted, support, rnd)
        else:
            support = self._support(scouts)
            ctx.emit("bee_round", {"round": self.max_rounds, "support": dict(support)})
            certified = self._certified(scouts)
            if certified is not None:
                return self._finish_certified(certified, scouts, ctx)
            if support and max(support.values()) >= self.quorum:
                reason = "quorum"
        return self._finish(scouts, ctx, reason)

    def _certified(self, scouts: list[Scout]) -> Scout | None:
        """The first scout (by index) holding a goal-verified answer, if stopping on one is on."""
        if not self.stop_on_certificate:
            return None
        return next((s for s in scouts if s.verified and s.option is not None), None)

    def _finish_certified(
        self, scout: Scout, scouts: list[Scout], ctx: ExecutionContext
    ) -> ExecutorOutcome:
        support = self._support(scouts)
        ctx.emit(
            "executor_stop",
            {
                "reason": "certificate",
                "support": dict(support),
                "quorum_reached": bool(support) and max(support.values()) >= self.quorum,
                "from_released": False,
                "scout": scout.index,
            },
        )
        return ExecutorOutcome(output=scout.output)

    def _support(self, scouts: list[Scout]) -> Counter[str]:
        return Counter(s.option for s in scouts if s.option is not None)

    async def _independent(
        self, subtask: Subtask, ctx: ExecutionContext, who: list[Scout], rnd: int
    ) -> bool:
        coros = [
            self._answerer.answer(subtask, ctx, s.agent, ctx.seed + 1000 * rnd + s.index, f"r{rnd}")
            for s in who
        ]
        results, budget_hit = await _gather_budgeted(coros)
        for s, r in zip(who, results, strict=True):
            if r is not None:
                s.option, s.output, s.verified = r
        return budget_hit

    def _inhibit(self, scouts: list[Scout], ctx: ExecutionContext) -> None:
        support = self._support(scouts)
        released = [
            s
            for s in scouts
            if s.option is not None
            and any(o != s.option for o in support)
            and ctx.rng.random() < self.sigma
        ]
        for s in released:
            s.last_option, s.last_output = s.option, s.output
            s.option, s.verified = None, False
        ctx.emit("bee_inhibition", {"released": [s.index for s in released]})

    async def _recruit(
        self,
        subtask: Subtask,
        ctx: ExecutionContext,
        scouts: list[Scout],
        uncommitted: list[Scout],
        support: Counter[str],
        rnd: int,
    ) -> bool:
        options = sorted(support)
        draws = [
            ctx.rng.choices(options, weights=[support[o] for o in options])[0] for _ in uncommitted
        ]
        coros = [
            self._assess(subtask, ctx, s, self._representative(scouts, o), rnd)
            for s, o in zip(uncommitted, draws, strict=True)
        ]
        results, budget_hit = await _gather_budgeted(coros)
        decisions = []
        for s, o, r in zip(uncommitted, draws, results, strict=True):
            accepted, confidence = r if r is not None else (False, 0.0)
            decisions.append({"scout": s.index, "accept": accepted})
            if accepted:
                rep = self._representative(scouts, o)
                s.option, s.output, s.confidence = o, rep.output, confidence
                s.verified = rep.verified
        ctx.emit("bee_recruitment", {"round": rnd, "decisions": decisions})
        return budget_hit

    def _representative(self, scouts: list[Scout], option: str) -> Scout:
        return min(
            (s for s in scouts if s.option == option), key=lambda s: (-s.confidence, s.index)
        )

    async def _assess(
        self, subtask: Subtask, ctx: ExecutionContext, scout: Scout, rep: Scout, rnd: int
    ) -> tuple[bool, float]:
        client = dataclasses.replace(ctx.llm, agent=scout.agent)
        messages = subtask_messages(prompts.load("assess_candidate"), subtask, ctx)
        messages += [
            Message(role="user", content="Proposed answer:"),
            Message(role="user", content=json.dumps(rep.output), source=rep.agent),
        ]
        request = LLMRequest(
            model=self.model,
            messages=tuple(messages),
            max_completion_tokens=512,
            temperature=self.temperature,
            reasoning_effort=self.reasoning_effort,  # type: ignore[arg-type]
            seed=ctx.seed + 10_000 * rnd + scout.index,
            sample_id=f"{scout.agent}/assess{rnd}",
        )
        reply = await client.complete(request)
        parsed = parse_json(reply.content, envelope="accept")
        if not isinstance(parsed, dict):
            return False, 0.0
        confidence = parsed.get("confidence")
        conf = float(confidence) if isinstance(confidence, int | float) else 0.0
        return parsed.get("accept") is True, max(0.0, min(1.0, conf))

    def _finish(self, scouts: list[Scout], ctx: ExecutionContext, reason: str) -> ExecutorOutcome:
        support = self._support(scouts)
        from_released = False
        if not support:
            # Everyone was released and the run ended before anyone recommitted: fall back to the
            # options scouts held before release.
            for s in scouts:
                if s.last_option is not None:
                    s.option, s.output = s.last_option, s.last_output
            support = self._support(scouts)
            from_released = bool(support)
        ctx.emit(
            "executor_stop",
            {
                "reason": reason,
                "support": dict(support),
                "quorum_reached": reason == "quorum",
                "from_released": from_released,
            },
        )
        if not support:
            if reason == "budget":
                raise BudgetExceeded("budget ran out before any scout had a usable answer")
            return ExecutorOutcome(output=None, incomplete="no scout produced a usable answer")
        conf = {o: sum(s.confidence for s in scouts if s.option == o) for o in support}
        winner = min(support, key=lambda o: (-support[o], -conf[o], o))
        return ExecutorOutcome(output=self._representative(scouts, winner).output)


@dataclass
class MajorityVote:
    """Independent answers and a plurality vote (Table 3 control; D18 ordinary worker)."""

    model: str
    k: int = 5
    # Call-yoked control: keep sampling until this many model calls are spent (overrides k).
    max_calls: int | None = None
    max_completion_tokens: int = 2048
    temperature: float = 0.6
    reasoning_effort: str | None = None
    name: str = "majority"
    # D27: a goal-verified vote is returned at once (the same certificate access as bee).
    stop_on_certificate: bool = True

    async def execute(self, subtask: Subtask, ctx: ExecutionContext) -> ExecutorOutcome:
        answerer = _Answerer(
            self.model, self.max_completion_tokens, self.temperature, self.reasoning_effort
        )
        votes: list[tuple[str, JsonValue, bool]] = []
        budget_hit = False
        if self.max_calls is None:
            coros = [
                answerer.answer(subtask, ctx, f"{ctx.agent}/voter{i}", ctx.seed + 100 * i, "vote")
                for i in range(self.k)
            ]
            results, budget_hit = await _gather_budgeted(coros)
            votes = [r for r in results if r is not None and r[0] is not None]
        else:
            # Call-yoked: each voter runs under a sub-account capped by the calls still allowed, so
            # a multi-call tool episode cannot overshoot; attempts are bounded too, so rejected
            # requests (which spend nothing) cannot loop forever.
            allowance = self.max_calls
            spent_at_start = ctx.llm.account.snapshot().model_calls
            for i in range(self.max_calls):
                left = allowance - (ctx.llm.account.snapshot().model_calls - spent_at_start)
                if left <= 0:
                    break
                capped = ctx.llm.child(
                    Budget(max_tokens=ctx.llm.account.limits.max_tokens, max_model_calls=left)
                )
                results, refused = await _gather_budgeted(
                    [
                        answerer.answer(
                            subtask,
                            ctx,
                            f"{ctx.agent}/voter{i}",
                            ctx.seed + 100 * i,
                            "vote",
                            client=capped,
                        )
                    ]
                )
                if results[0] is not None and results[0][0] is not None:
                    votes.append(results[0])
                    if self.stop_on_certificate and results[0][2]:
                        break
                if refused and ctx.llm.account.exhausted():
                    budget_hit = True
                    break
        tally = Counter(o for o, _, _ in votes)
        certified = next((v for v in votes if v[2]), None) if self.stop_on_certificate else None
        if certified is not None:
            ctx.emit("executor_stop", {"reason": "certificate", "tally": dict(tally)})
            return ExecutorOutcome(output=certified[1])
        ctx.emit(
            "executor_stop", {"reason": "budget" if budget_hit else "votes", "tally": dict(tally)}
        )
        if not tally:
            if budget_hit:
                raise BudgetExceeded("budget ran out before any usable vote")
            return ExecutorOutcome(output=None, incomplete="no usable vote")
        winner = min(tally, key=lambda o: (-tally[o], o))
        return ExecutorOutcome(output=next(out for o, out, _ in votes if o == winner))
