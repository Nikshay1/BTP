"""Deterministic fake model for tests and offline demos.

Replies are scripted with rules. Each rule has a predicate over the request and a list of
replies consumed in order; the first rule whose predicate matches and still has replies answers.
A scripted reply can be text, a full ``LLMReply``, or an exception to raise. Token usage of text
replies is a deterministic estimate (four characters per token), so budgets behave as with a
real model but nothing here is a measurement.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field

from bioalloc.llm.base import LLMError, LLMReply, LLMRequest, Usage

Script = str | LLMReply | LLMError
Predicate = Callable[[LLMRequest], bool]


def estimate_tokens(text: str) -> int:
    return max(1, math.ceil(len(text) / 4))


def always(_: LLMRequest) -> bool:
    return True


def contains(text: str) -> Predicate:
    """Matches requests whose last message contains ``text``."""
    return lambda req: text in req.messages[-1].content


@dataclass
class _Rule:
    predicate: Predicate
    replies: list[Script]
    # If set, answers every matching request (replies is ignored).
    responder: Callable[[LLMRequest], Script] | None = None


@dataclass
class FakeTransport:
    model: str = "fake-model"
    rules: list[_Rule] = field(default_factory=list)
    requests: list[LLMRequest] = field(default_factory=list)

    def on(self, predicate: Predicate, *replies: Script) -> FakeTransport:
        self.rules.append(_Rule(predicate, list(replies)))
        return self

    def always(self, *replies: Script) -> FakeTransport:
        return self.on(always, *replies)

    def respond(
        self, predicate: Predicate, responder: Callable[[LLMRequest], Script]
    ) -> FakeTransport:
        """Answer every matching request by calling ``responder`` on it."""
        self.rules.append(_Rule(predicate, [], responder))
        return self

    async def send(self, request: LLMRequest) -> LLMReply:
        self.requests.append(request)
        for rule in self.rules:
            if rule.responder is not None and rule.predicate(request):
                script = rule.responder(request)
                break
            if rule.replies and rule.predicate(request):
                script = rule.replies.pop(0)
                break
        else:
            raise AssertionError(f"FakeTransport has no scripted reply for: {request.messages[-1]}")
        if isinstance(script, LLMError):
            raise script
        if isinstance(script, LLMReply):
            return script
        completion = min(estimate_tokens(script), request.max_completion_tokens)
        return LLMReply(
            content=script,
            usage=Usage(
                prompt_tokens=sum(estimate_tokens(m.content) for m in request.messages),
                completion_tokens=completion,
            ),
            model=self.model,
            request_id=f"fake-{len(self.requests)}",
        )

    def remaining(self) -> Iterable[Script]:
        for rule in self.rules:
            yield from rule.replies
