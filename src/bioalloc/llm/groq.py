"""Transport for GPT-OSS on Groq's OpenAI-compatible chat completions API (decisions D3, D19).

One request per call, no streaming, no client-side retries: rate-limit retries happen in the
metered client so that each attempt goes through budget admission. Errors are mapped so that the
ledger charges correctly:

* 429 -> ``RateLimited`` (rejected, nothing charged), with the ``retry-after`` delay;
* other 4xx -> ``RequestRejected`` (nothing charged);
* connection failures before sending -> ``ConnectionFailed`` (nothing charged);
* timeouts -> ``TransportTimeout`` and 5xx or malformed replies -> ``LLMError``; both may have been
  billed, so the metered client charges the full reservation as unknown usage.

Reasoning tokens are read from ``usage.completion_tokens_details.reasoning_tokens``; they are part
of ``completion_tokens`` and are never added again. A reply without usage is treated as an error
(unknown usage), never as zero tokens. Not yet verified against the live API: seed behaviour,
whether reasoning details are always present, and the exact name of the parameter that hides
reasoning text.
"""

from __future__ import annotations

import json
import time
from typing import Any

import httpx

from bioalloc.llm.base import (
    ConnectionFailed,
    FinishReason,
    LLMError,
    LLMReply,
    LLMRequest,
    Message,
    RateLimited,
    RequestRejected,
    ToolCall,
    TransportTimeout,
    Usage,
)

BASE_URL = "https://api.groq.com/openai/v1"
MODEL = "openai/gpt-oss-120b"
_FINISH: dict[str, FinishReason] = {
    "stop": "stop",
    "length": "length",
    "tool_calls": "tool_calls",
    "content_filter": "content_filter",
}


def wire_message(m: Message) -> dict[str, Any]:
    out: dict[str, Any] = {"role": m.role, "content": m.content}
    if m.tool_calls:
        out["tool_calls"] = [
            {
                "id": tc.id,
                "type": "function",
                "function": {"name": tc.name, "arguments": tc.arguments},
            }
            for tc in m.tool_calls
        ]
    if m.tool_call_id is not None:
        out["tool_call_id"] = m.tool_call_id
    return out


def wire_request(request: LLMRequest, *, include_reasoning: bool = False) -> dict[str, Any]:
    """The JSON body sent to Groq. Internal fields (message sources, sample ids) are not sent."""
    body: dict[str, Any] = {
        "model": request.model,
        "messages": [wire_message(m) for m in request.messages],
        "max_completion_tokens": request.max_completion_tokens,
        "temperature": request.temperature,
        "stream": False,
        "n": 1,
        "include_reasoning": include_reasoning,
    }
    if request.seed is not None:
        body["seed"] = request.seed
    if request.reasoning_effort is not None:
        body["reasoning_effort"] = request.reasoning_effort
    if request.tools:
        body["tools"] = list(request.tools)
        body["tool_choice"] = "auto"
    if request.response_format is not None:
        body["response_format"] = request.response_format
    return body


def parse_retry_after(value: str | None) -> float | None:
    if value is None:
        return None
    try:
        seconds = float(value)
    except ValueError:
        return None
    return seconds if seconds >= 0 else None


def parse_reply(data: Any, latency_s: float, request_id: str | None) -> LLMReply:
    try:
        choice = data["choices"][0]
        message = choice["message"]
        usage = data["usage"]
        prompt = int(usage["prompt_tokens"])
        completion = int(usage["completion_tokens"])
    except (KeyError, IndexError, TypeError, ValueError) as exc:
        raise LLMError(f"malformed reply or missing usage: {exc!r}") from exc
    reasoning = (usage.get("completion_tokens_details") or {}).get("reasoning_tokens")
    cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens")
    try:
        calls = tuple(
            ToolCall(
                id=tc["id"],
                name=tc["function"]["name"],
                arguments=tc["function"].get("arguments") or "",
            )
            for tc in message.get("tool_calls") or ()
        )
        return LLMReply(
            content=message.get("content") or "",
            tool_calls=calls,
            usage=Usage(
                prompt_tokens=prompt,
                completion_tokens=completion,
                reasoning_tokens=None if reasoning is None else int(reasoning),
                cached_prompt_tokens=None if cached is None else int(cached),
            ),
            finish_reason=_FINISH.get(str(choice.get("finish_reason")), "other"),
            model=str(data.get("model", "")),
            request_id=request_id or data.get("id"),
            latency_s=latency_s,
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise LLMError(f"malformed reply: {exc!r}") from exc


class GroqTransport:
    def __init__(
        self,
        api_key: str,
        *,
        client: httpx.AsyncClient | None = None,
        base_url: str = BASE_URL,
        timeout_s: float = 120.0,
        include_reasoning: bool = False,
    ) -> None:
        if not api_key:
            raise ValueError("a Groq API key is required")
        self._client = client or httpx.AsyncClient(timeout=timeout_s)
        self._url = f"{base_url}/chat/completions"
        self._headers = {"Authorization": f"Bearer {api_key}"}
        self.include_reasoning = include_reasoning

    async def send(self, request: LLMRequest) -> LLMReply:
        body = wire_request(request, include_reasoning=self.include_reasoning)
        start = time.monotonic()
        try:
            response = await self._client.post(self._url, json=body, headers=self._headers)
        except (httpx.ConnectError, httpx.ConnectTimeout) as exc:
            raise ConnectionFailed(f"could not connect: {exc!r}") from exc
        except httpx.TimeoutException as exc:
            raise TransportTimeout(f"request timed out: {exc!r}") from exc
        except httpx.HTTPError as exc:
            raise LLMError(f"transport error: {exc!r}") from exc
        latency = time.monotonic() - start
        request_id = response.headers.get("x-request-id")
        if response.status_code == 429:
            raise RateLimited(
                f"rate limited (request {request_id})",
                retry_after=parse_retry_after(response.headers.get("retry-after")),
            )
        if 400 <= response.status_code < 500:
            raise RequestRejected(f"HTTP {response.status_code}: {response.text[:500]}")
        if response.status_code >= 500 or response.status_code != 200:
            raise LLMError(f"HTTP {response.status_code}: {response.text[:500]}")
        try:
            data = response.json()
        except json.JSONDecodeError as exc:
            raise LLMError("reply is not JSON") from exc
        return parse_reply(data, latency, request_id)

    async def aclose(self) -> None:
        await self._client.aclose()
