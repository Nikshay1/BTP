"""Budget accounts and the metered model client.

Every model call is reserved against a budget before it is sent and settled from the provider's
reported usage afterwards. Accounts form a tree: a dispatch gets a child account whose spending
also counts against every ancestor, so an executor can never be admitted beyond what the task has
left.

Admission. A call whose reservation does not fit is not sent. If it would fit once calls already
in flight have settled (their reservations are worst cases and usually larger than what they end
up costing), the call waits for them instead of failing; it fails only when the budget itself is
spent. Waiting calls are admitted strictly in arrival order, so a large call cannot be overtaken
indefinitely by smaller ones. Siblings therefore draw from one shared pool, first come first
served, and the same rule applies to every system compared.

Reservations are estimates. The prompt size of a call is not known exactly before sending (it
depends on the provider's tokenizer), so a reservation uses three characters per token over the
whole request plus its maximum completion. If a provider still reports more and the total crosses
a limit, the overrun is recorded in the cost of that account and of every ancestor
(``Cost.overrun``), no further calls are admitted under the account that overran, and the run is
reported as over budget.

Deadlines. With a wall-time limit, waiting for admission and waiting for a reply are both bounded
by the time left on the tightest ancestor; a call cut off by the deadline raises
``TransportTimeout``.

Unknown usage. A call that reached the transport but produced no reply (timeout, cancellation,
transport error) may still have been billed. It is charged its full reservation, prompt estimate
as prompt tokens and maximum completion as completion tokens, and the cost is marked
``usage_complete=False``. A ``RequestRejected`` error, or a call cut off before it reached the
transport, releases the reservation without charge.
"""

from __future__ import annotations

import asyncio
import json
import math
import time
from collections import deque
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from itertools import count
from typing import Any

from bioalloc.llm.base import (
    LLMReply,
    LLMRequest,
    RateLimited,
    RequestRejected,
    RunMode,
    Transport,
    TransportTimeout,
)
from bioalloc.schemas import Actor, Budget, Cost, Tokens

Clock = Callable[[], float]
EventHook = Callable[[str, dict[str, Any]], None]


class BudgetExceeded(Exception):
    """A call does not fit in what is left of the budget. Nothing was sent."""


class CachedReplyInMeasurement(RuntimeError):
    """A measured run received a cached reply; measured runs must hit the provider."""


def _chars_to_tokens(n: int) -> int:
    return math.ceil(n / 3)


def reserve_prompt_tokens(request: LLMRequest) -> int:
    """Conservative estimate of a request's prompt tokens, covering every part that is sent."""
    tokens = sum(_chars_to_tokens(len(m.content)) + 4 for m in request.messages)
    extras: list[object] = [*request.tools]
    extras += [tc.model_dump() for m in request.messages for tc in m.tool_calls]
    if request.response_format is not None:
        extras.append(request.response_format)
    tokens += sum(_chars_to_tokens(len(json.dumps(x))) for x in extras)
    return tokens


@dataclass(frozen=True)
class Pricing:
    """US dollars per million tokens."""

    input_per_m: float = 0.0
    output_per_m: float = 0.0

    def usd(self, prompt: int, completion: int) -> float:
        return (prompt * self.input_per_m + completion * self.output_per_m) / 1e6


@dataclass
class _Spent:
    coordinator_prompt: int = 0
    coordinator_completion: int = 0
    worker_prompt: int = 0
    worker_completion: int = 0
    communication: int = 0
    reasoning: int = 0
    model_calls: int = 0
    tool_calls: int = 0
    usd: float = 0.0
    usage_complete: bool = True
    # Some account in this subtree (this one or a dispatch below it) went over its token limit.
    limit_exceeded: bool = False

    @property
    def tokens(self) -> int:
        return (
            self.coordinator_prompt
            + self.coordinator_completion
            + self.worker_prompt
            + self.worker_completion
        )


@dataclass(frozen=True)
class _Reservation:
    prompt: int
    completion: int

    @property
    def tokens(self) -> int:
        return self.prompt + self.completion


@dataclass
class Account:
    limits: Budget
    parent: Account | None = None
    clock: Clock = time.monotonic
    started: float = field(init=False)
    spent: _Spent = field(default_factory=_Spent, init=False)
    reserved_tokens: int = field(default=0, init=False)
    reserved_calls: int = field(default=0, init=False)
    overrun: bool = field(default=False, init=False)
    _reservations: dict[int, _Reservation] = field(default_factory=dict, init=False)
    _ids: count[int] = field(default_factory=count, init=False)
    _changed: asyncio.Event | None = field(default=None, init=False)
    # Calls waiting for admission, in arrival order (used at the root only).
    _queue: deque[object] = field(default_factory=deque, init=False)

    def __post_init__(self) -> None:
        self.started = self.clock()

    def child(self, limits: Budget) -> Account:
        """A sub-account; its limits are further capped by what every ancestor has left."""
        return Account(limits=limits, parent=self, clock=self.clock)

    def _chain(self) -> list[Account]:
        chain: list[Account] = []
        acc: Account | None = self
        while acc is not None:
            chain.append(acc)
            acc = acc.parent
        return chain

    @property
    def root(self) -> Account:
        return self._chain()[-1]

    def elapsed(self) -> float:
        return self.clock() - self.started

    def time_left(self) -> float | None:
        left = [
            a.limits.max_elapsed_s - a.elapsed()
            for a in self._chain()
            if a.limits.max_elapsed_s is not None
        ]
        return max(min(left), 0.0) if left else None

    def remaining(self, *, include_reservations: bool = True) -> Budget:
        """What can still be reserved here, taking every ancestor into account."""
        tokens: list[int] = []
        calls: list[int] = []
        for acc in self._chain():
            held_t = acc.reserved_tokens if include_reservations else 0
            held_c = acc.reserved_calls if include_reservations else 0
            tokens.append(acc.limits.max_tokens - acc.spent.tokens - held_t)
            calls.append(acc.limits.max_model_calls - acc.spent.model_calls - held_c)
        tool_left = [
            a.limits.max_tool_calls - a.spent.tool_calls
            for a in self._chain()
            if a.limits.max_tool_calls is not None
        ]
        return Budget(
            max_tokens=max(min(tokens), 0),
            max_model_calls=max(min(calls), 0),
            max_elapsed_s=self.time_left(),
            max_tool_calls=max(min(tool_left), 0) if tool_left else None,
        )

    def any_overrun(self) -> bool:
        return any(a.overrun for a in self._chain())

    def exhausted(self) -> bool:
        """True when no further call can ever be admitted (ignores calls in flight)."""
        r = self.remaining(include_reservations=False)
        out_of_time = r.max_elapsed_s is not None and r.max_elapsed_s <= 0
        return self.any_overrun() or r.max_model_calls == 0 or r.max_tokens == 0 or out_of_time

    def _refusal(self, tokens: int, *, include_reservations: bool) -> str | None:
        if self.any_overrun():
            return "an earlier call overran the budget"
        r = self.remaining(include_reservations=include_reservations)
        if r.max_elapsed_s is not None and r.max_elapsed_s <= 0:
            return "time limit reached"
        if r.max_model_calls < 1:
            return "model call limit reached"
        if tokens > r.max_tokens:
            return f"call needs up to {tokens} tokens, {r.max_tokens} left"
        return None

    def reserve(self, prompt: int, completion: int) -> int:
        """Reserve one model call. Raises ``BudgetExceeded`` if it does not fit right now."""
        reason = self._refusal(prompt + completion, include_reservations=True)
        if reason is not None:
            raise BudgetExceeded(reason)
        rid = next(self._ids)
        self._reservations[rid] = _Reservation(prompt, completion)
        for acc in self._chain():
            acc.reserved_tokens += prompt + completion
            acc.reserved_calls += 1
        return rid

    def would_fit_after_settling(self, tokens: int) -> bool:
        """Whether a refused call would fit once the calls in flight have settled."""
        held = any(a.reserved_calls for a in self._chain())
        return held and self._refusal(tokens, include_reservations=False) is None

    async def wait_for_change(self, timeout: float | None) -> None:
        root = self.root
        if root._changed is None:
            root._changed = asyncio.Event()
        await asyncio.wait_for(root._changed.wait(), timeout)

    def _notify(self) -> None:
        root = self.root
        if root._changed is not None:
            root._changed.set()
            root._changed = None

    def _release(self, rid: int) -> _Reservation:
        res = self._reservations.pop(rid)
        for acc in self._chain():
            acc.reserved_tokens -= res.tokens
            acc.reserved_calls -= 1
        return res

    def settle(
        self,
        rid: int,
        actor: Actor,
        prompt: int,
        completion: int,
        communication: int,
        reasoning: int,
        usd: float,
    ) -> None:
        self._release(rid)
        for acc in self._chain():
            s = acc.spent
            if actor == "coordinator":
                s.coordinator_prompt += prompt
                s.coordinator_completion += completion
            else:
                s.worker_prompt += prompt
                s.worker_completion += completion
            s.communication += communication
            s.reasoning += reasoning
            s.model_calls += 1
            s.usd += usd
            if s.tokens > acc.limits.max_tokens:
                acc.overrun = True
        if any(a.overrun for a in self._chain()):
            for acc in self._chain():
                acc.spent.limit_exceeded = True
        self._notify()

    def settle_unknown(self, rid: int, actor: Actor, pricing: Pricing) -> _Reservation:
        """Charge the full reservation for a call that was sent but whose usage is unknown."""
        res = self._reservations[rid]
        self.settle(
            rid, actor, res.prompt, res.completion, 0, 0, pricing.usd(res.prompt, res.completion)
        )
        for acc in self._chain():
            acc.spent.usage_complete = False
        return res

    def cancel(self, rid: int) -> None:
        """Release a reservation for a call that was never processed."""
        self._release(rid)
        self._notify()

    def record_tool_call(self) -> None:
        """Count one tool execution attempt. Raises if a tool-call limit is already reached."""
        for acc in self._chain():
            limit = acc.limits.max_tool_calls
            if limit is not None and acc.spent.tool_calls >= limit:
                raise BudgetExceeded("tool call limit reached")
        for acc in self._chain():
            acc.spent.tool_calls += 1

    def snapshot(self) -> Cost:
        s = self.spent
        return Cost(
            coordinator=Tokens(prompt=s.coordinator_prompt, completion=s.coordinator_completion),
            worker=Tokens(prompt=s.worker_prompt, completion=s.worker_completion),
            communication=s.communication,
            reasoning=s.reasoning,
            model_calls=s.model_calls,
            tool_calls=s.tool_calls,
            elapsed_s=self.elapsed(),
            usd=s.usd,
            usage_complete=s.usage_complete,
            overrun=s.limit_exceeded,
        )


def communication_tokens(request: LLMRequest, prompt_tokens: int, agent: str) -> int:
    """Share of a call's prompt tokens spent reading content written by another agent (D6)."""
    total = sum(len(m.content) for m in request.messages)
    if total == 0:
        return 0
    foreign = sum(len(m.content) for m in request.messages if m.source not in (None, agent))
    return round(prompt_tokens * foreign / total)


@dataclass
class MeteredClient:
    """The only way executors and the coordinator reach a model."""

    transport: Transport
    account: Account
    actor: Actor
    agent: str
    pricing: Pricing = field(default_factory=Pricing)
    mode: RunMode = "debug"
    on_event: EventHook | None = None
    # Rate-limit handling (D19): wait at least Retry-After, else exponential backoff from 1 s capped
    # at 30 s; give up after this many retries.
    max_rate_limit_retries: int = 3
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep

    def child(
        self, limits: Budget, *, actor: Actor | None = None, agent: str | None = None
    ) -> MeteredClient:
        return MeteredClient(
            transport=self.transport,
            account=self.account.child(limits),
            actor=actor or self.actor,
            agent=agent or self.agent,
            pricing=self.pricing,
            mode=self.mode,
            on_event=self.on_event,
            max_rate_limit_retries=self.max_rate_limit_retries,
            sleep=self.sleep,
        )

    def _emit(self, kind: str, data: dict[str, Any]) -> None:
        if self.on_event is not None:
            self.on_event(kind, data)

    async def _admit(self, prompt: int, completion: int) -> int:
        """Reserve a call, waiting in arrival order behind earlier calls of the same task."""
        root = self.account.root
        ticket = object()
        root._queue.append(ticket)
        try:
            while True:
                if root._queue[0] is ticket:
                    try:
                        return self.account.reserve(prompt, completion)
                    except BudgetExceeded:
                        if not self.account.would_fit_after_settling(prompt + completion):
                            raise
                try:
                    await self.account.wait_for_change(self.account.time_left())
                except TimeoutError:
                    raise BudgetExceeded("time limit reached while waiting for budget") from None
        finally:
            root._queue.remove(ticket)
            self.account._notify()

    async def complete(self, request: LLMRequest) -> LLMReply:
        """One model call. A rate-limited attempt releases its reservation, waits, and goes through
        admission again; it is an HTTP retry, not a model call, and is traced as such."""
        for retry in range(self.max_rate_limit_retries + 1):
            try:
                return await self._complete_once(request)
            except RateLimited as exc:
                if retry == self.max_rate_limit_retries:
                    raise
                delay = exc.retry_after if exc.retry_after is not None else min(30.0, 2.0**retry)
                left = self.account.time_left()
                if left is not None and delay >= left:
                    raise TransportTimeout("deadline reached while rate limited") from exc
                self._emit(
                    "rate_limited", {"agent": self.agent, "retry": retry + 1, "delay_s": delay}
                )
                await self.sleep(delay)
        raise AssertionError("unreachable")

    async def _complete_once(self, request: LLMRequest) -> LLMReply:
        prompt_est = reserve_prompt_tokens(request)
        rid = await self._admit(prompt_est, request.max_completion_tokens)
        reserved = prompt_est + request.max_completion_tokens
        self._emit(
            "llm_request",
            {
                "agent": self.agent,
                "actor": self.actor,
                "reserved_tokens": reserved,
                "request": request.model_dump(mode="json"),
            },
        )
        sent = False

        async def send() -> LLMReply:
            nonlocal sent
            sent = True
            return await self.transport.send(request)

        try:
            reply = await asyncio.wait_for(send(), self.account.time_left())
        except BaseException as exc:
            if isinstance(exc, RequestRejected) or not sent:
                # Never processed: rejected by the provider, or cut off before it was sent.
                self.account.cancel(rid)
                self._emit("llm_error", {"agent": self.agent, "error": repr(exc), "usage": "none"})
                if isinstance(exc, TimeoutError):
                    raise TransportTimeout("deadline reached before the request was sent") from exc
                raise
            # Sent, but no reply: timeout, deadline, cancellation or transport failure.
            res = self.account.settle_unknown(rid, self.actor, self.pricing)
            self._emit(
                "llm_error",
                {
                    "agent": self.agent,
                    "actor": self.actor,
                    "error": repr(exc),
                    "usage": "unknown",
                    "charged_prompt_tokens": res.prompt,
                    "charged_completion_tokens": res.completion,
                },
            )
            if isinstance(exc, TimeoutError) and not isinstance(exc, TransportTimeout):
                raise TransportTimeout("deadline reached while waiting for the reply") from exc
            raise
        if reply.from_cache and self.mode == "measurement":
            self.account.cancel(rid)
            self._emit("llm_error", {"agent": self.agent, "error": "cached reply", "usage": "none"})
            raise CachedReplyInMeasurement("measured runs must not use cached replies")
        u = reply.usage
        comm = communication_tokens(request, u.prompt_tokens, self.agent)
        self.account.settle(
            rid,
            self.actor,
            u.prompt_tokens,
            u.completion_tokens,
            comm,
            u.reasoning_tokens or 0,
            self.pricing.usd(u.prompt_tokens, u.completion_tokens),
        )
        self._emit(
            "llm_reply",
            {
                "agent": self.agent,
                "actor": self.actor,
                "communication_tokens": comm,
                "reply": reply.model_dump(mode="json"),
            },
        )
        return reply
