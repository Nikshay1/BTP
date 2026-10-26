"""Budget accounts and the metered model client.

Every model call is reserved against a budget before it is sent and settled from the provider's
reported usage afterwards. A call whose reservation does not fit is never sent. Accounts form a
tree: a dispatch gets a child account whose spending also counts against every ancestor, so an
executor can never spend more than the task has left.

The prompt size of a call is not known exactly before sending (it depends on the provider's
tokenizer), so reservations use a conservative estimate of three characters per token. If a
provider still reports more than the estimate and the total crosses a limit, the account records
an overrun; no further calls are allowed and the overrun is visible in the cost record.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from itertools import count
from typing import Any

from bioalloc.llm.base import LLMReply, LLMRequest, Message, Transport, TransportTimeout
from bioalloc.schemas import Actor, Budget, Cost, Tokens

Clock = Callable[[], float]
EventHook = Callable[[str, dict[str, Any]], None]


class BudgetExceeded(Exception):
    """A call or dispatch would not fit in the remaining budget. Nothing was sent."""


def reserve_prompt_tokens(messages: tuple[Message, ...]) -> int:
    return sum(math.ceil(len(m.content) / 3) + 4 for m in messages)


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

    @property
    def tokens(self) -> int:
        return (
            self.coordinator_prompt
            + self.coordinator_completion
            + self.worker_prompt
            + self.worker_completion
        )


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
    _reservations: dict[int, tuple[int, int]] = field(default_factory=dict, init=False)
    _ids: count[int] = field(default_factory=count, init=False)

    def __post_init__(self) -> None:
        self.started = self.clock()

    def child(self, limits: Budget) -> Account:
        """A sub-account; its limits are further capped by what this account has left."""
        return Account(limits=limits, parent=self, clock=self.clock)

    def _chain(self) -> list[Account]:
        chain: list[Account] = []
        acc: Account | None = self
        while acc is not None:
            chain.append(acc)
            acc = acc.parent
        return chain

    def elapsed(self) -> float:
        return self.clock() - self.started

    def remaining(self) -> Budget:
        """What can still be reserved here, taking every ancestor into account."""
        tokens = calls = None
        time_left: float | None = None
        for acc in self._chain():
            t = acc.limits.max_tokens - acc.spent.tokens - acc.reserved_tokens
            c = acc.limits.max_model_calls - acc.spent.model_calls - acc.reserved_calls
            tokens = t if tokens is None else min(tokens, t)
            calls = c if calls is None else min(calls, c)
            if acc.limits.max_elapsed_s is not None:
                left = acc.limits.max_elapsed_s - acc.elapsed()
                time_left = left if time_left is None else min(time_left, left)
        assert tokens is not None and calls is not None
        return Budget(
            max_tokens=max(tokens, 0),
            max_model_calls=max(calls, 0),
            max_elapsed_s=None if time_left is None else max(time_left, 0.0),
        )

    def exhausted(self) -> bool:
        r = self.remaining()
        out_of_time = r.max_elapsed_s is not None and r.max_elapsed_s <= 0
        return any(a.overrun for a in self._chain()) or r.max_model_calls == 0 or out_of_time

    def reserve(self, tokens: int) -> int:
        """Reserve one model call of up to ``tokens`` tokens. Raises if it does not fit."""
        if any(a.overrun for a in self._chain()):
            raise BudgetExceeded("an earlier call overran the budget")
        r = self.remaining()
        if r.max_elapsed_s is not None and r.max_elapsed_s <= 0:
            raise BudgetExceeded("time limit reached")
        if r.max_model_calls < 1:
            raise BudgetExceeded("model call limit reached")
        if tokens > r.max_tokens:
            raise BudgetExceeded(f"call needs up to {tokens} tokens, {r.max_tokens} left")
        rid = next(self._ids)
        self._reservations[rid] = (tokens, 1)
        for acc in self._chain():
            acc.reserved_tokens += tokens
            acc.reserved_calls += 1
        return rid

    def _release(self, rid: int) -> int:
        tokens, calls = self._reservations.pop(rid)
        for acc in self._chain():
            acc.reserved_tokens -= tokens
            acc.reserved_calls -= calls
        return tokens

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

    def settle_unknown(self, rid: int, actor: Actor, pricing: Pricing) -> None:
        """Charge the full reservation for a call whose usage is unknown (e.g. a timeout)."""
        tokens = self._reservations[rid][0]
        self.settle(rid, actor, tokens, 0, 0, 0, pricing.usd(tokens, 0))
        for acc in self._chain():
            acc.spent.usage_complete = False

    def cancel(self, rid: int) -> None:
        """Release a reservation for a call that was never sent."""
        self._release(rid)

    def record_tool_call(self) -> None:
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
        )


def communication_tokens(messages: tuple[Message, ...], prompt_tokens: int, agent: str) -> int:
    """Share of a call's prompt tokens spent reading content written by another agent (D6)."""
    total = sum(len(m.content) for m in messages)
    if total == 0:
        return 0
    foreign = sum(len(m.content) for m in messages if m.source not in (None, agent))
    return round(prompt_tokens * foreign / total)


@dataclass
class MeteredClient:
    """The only way executors and the coordinator reach a model."""

    transport: Transport
    account: Account
    actor: Actor
    agent: str
    pricing: Pricing = field(default_factory=Pricing)
    on_event: EventHook | None = None

    def child(
        self, limits: Budget, *, actor: Actor | None = None, agent: str | None = None
    ) -> MeteredClient:
        return MeteredClient(
            transport=self.transport,
            account=self.account.child(limits),
            actor=actor or self.actor,
            agent=agent or self.agent,
            pricing=self.pricing,
            on_event=self.on_event,
        )

    def _emit(self, kind: str, data: dict[str, Any]) -> None:
        if self.on_event is not None:
            self.on_event(kind, data)

    async def complete(self, request: LLMRequest) -> LLMReply:
        reservation = reserve_prompt_tokens(request.messages) + request.max_completion_tokens
        rid = self.account.reserve(reservation)
        self._emit(
            "llm_request",
            {
                "agent": self.agent,
                "actor": self.actor,
                "reserved_tokens": reservation,
                "request": request.model_dump(mode="json"),
            },
        )
        try:
            reply = await self.transport.send(request)
        except TransportTimeout as exc:
            self.account.settle_unknown(rid, self.actor, self.pricing)
            self._emit(
                "llm_error",
                {
                    "agent": self.agent,
                    "error": repr(exc),
                    "usage": "unknown",
                    "charged_tokens": reservation,
                },
            )
            raise
        except Exception as exc:
            # No reply and no evidence of consumption: release the reservation.
            self.account.cancel(rid)
            self._emit("llm_error", {"agent": self.agent, "error": repr(exc), "usage": "none"})
            raise
        u = reply.usage
        comm = communication_tokens(request.messages, u.prompt_tokens, self.agent)
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
