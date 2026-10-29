"""Model request and reply contracts, and the transport protocol.

A transport sends one request to a provider (or a fake) and returns the reply with the provider's
reported usage. Executors never hold a transport: they get a metered client (``bioalloc.budget``)
that charges every call to a budget before it is sent.
"""

from __future__ import annotations

import hashlib
import json
from typing import Literal, Protocol

from pydantic import Field, JsonValue, NonNegativeFloat, NonNegativeInt, model_validator

from bioalloc.schemas import Wire

RunMode = Literal["debug", "measurement"]


class ToolCall(Wire):
    """A tool call requested by the model. ``arguments`` is the raw JSON text it produced."""

    id: str
    name: str
    arguments: str


class Message(Wire):
    role: Literal["system", "user", "assistant", "tool"]
    content: str
    # Agent that wrote this content, or None for task text, fixed templates and tool output. Used to
    # count communication tokens (decision D6).
    source: str | None = None
    # Assistant messages: the tool calls the model made. Tool messages: the call answered.
    tool_calls: tuple[ToolCall, ...] = ()
    tool_call_id: str | None = None


class LLMRequest(Wire):
    model: str
    messages: tuple[Message, ...] = Field(min_length=1)
    max_completion_tokens: int = Field(gt=0)
    temperature: float = Field(default=1.0, ge=0.0)
    seed: int | None = None
    # Distinguishes deliberately independent samples of an otherwise identical request, so that a
    # cache or a fake never collapses k scouts into one answer.
    sample_id: str = "0"
    reasoning_effort: Literal["low", "medium", "high"] | None = None
    response_format: dict[str, JsonValue] | None = None
    tools: tuple[dict[str, JsonValue], ...] = ()

    def key(self) -> str:
        """Stable hash of the full request, used for caching and fake scripts."""
        canonical = json.dumps(self.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode()).hexdigest()


class Usage(Wire):
    prompt_tokens: NonNegativeInt
    completion_tokens: NonNegativeInt
    reasoning_tokens: NonNegativeInt | None = None  # subset of completion_tokens
    cached_prompt_tokens: NonNegativeInt | None = None  # subset of prompt_tokens

    @model_validator(mode="after")
    def _subsets(self) -> Usage:
        if self.reasoning_tokens is not None and self.reasoning_tokens > self.completion_tokens:
            raise ValueError("reasoning_tokens must be included in completion_tokens")
        if self.cached_prompt_tokens is not None and self.cached_prompt_tokens > self.prompt_tokens:
            raise ValueError("cached_prompt_tokens must be included in prompt_tokens")
        return self


FinishReason = Literal["stop", "length", "tool_calls", "content_filter", "other"]


class LLMReply(Wire):
    content: str
    tool_calls: tuple[ToolCall, ...] = ()
    usage: Usage
    # "length" means the reply was cut off at max_completion_tokens.
    finish_reason: FinishReason = "stop"
    model: str  # model identifier as returned by the provider
    request_id: str | None = None
    latency_s: NonNegativeFloat = 0.0
    from_cache: bool = False


class LLMError(Exception):
    """A request that produced no usable reply. Unless it is a ``RequestRejected``, the provider
    may have consumed tokens, so the metered client charges the full reservation."""


class TransportTimeout(LLMError):
    """The request timed out; the provider may still have consumed tokens."""


class RequestRejected(LLMError):
    """The provider refused the request before processing it (e.g. invalid request, rate limit).
    Nothing was consumed, so the reservation is released."""


class RateLimited(RequestRejected):
    """HTTP 429. ``retry_after`` is the provider's requested wait in seconds, if it gave one."""

    def __init__(self, message: str, retry_after: float | None = None) -> None:
        super().__init__(message)
        self.retry_after = retry_after


class ProviderThrottled(Exception):  # noqa: N818 (it is a condition of the provider, not a bug)
    """The provider keeps refusing for rate or quota reasons (a daily token quota, or 429s past
    every retry). Deliberately not an ``LLMError``: no executor may treat it as "no answer", because
    then a run that never reached the model would be recorded as a task failure (R18). It ends the
    whole run, which an experiment leaves unrecorded so that a later resume redoes it."""


class ConnectionFailed(RequestRejected):
    """The connection could not be opened, so the request never reached the provider."""


class Transport(Protocol):
    async def send(self, request: LLMRequest) -> LLMReply: ...
