"""Transport for Claude Sonnet 4.6 on Amazon Bedrock (decision D23).

Requests go through boto3's ``invoke_model`` with the Anthropic Messages body and
``anthropic_version: bedrock-2023-05-31``. The Anthropic SDK is not used: its current releases
send an API version that Bedrock's ARN-versioned models (Sonnet 4.6 among them) reject.

One request per call, no client-side retries (botocore's are switched off), so that every attempt
goes through budget admission in the metered client. Errors are mapped as in the Groq adapter:

* throttling and quota errors -> ``RateLimited`` (nothing charged; Bedrock sends no retry-after);
* other 4xx and missing credentials -> ``RequestRejected`` (nothing charged);
* connection failures before sending -> ``ConnectionFailed`` (nothing charged);
* read timeouts -> ``TransportTimeout`` and 5xx or malformed replies -> ``LLMError``; both may have
  been billed, so the metered client charges the full reservation as unknown usage.

The internal message format is OpenAI-shaped; it is converted here. Leading system messages
become the top-level ``system`` field, assistant tool calls become ``tool_use`` blocks, and
consecutive tool results are sent together in one user turn. Thinking is not enabled;
``reasoning_effort`` is sent as ``output_config.effort``. Sonnet 4.6 has no seed, so the seed is
not sent. Prompt caching is not used, so prompt tokens are all billed at the input price.
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any

import boto3
from botocore.config import Config
from botocore.exceptions import (
    BotoCoreError,
    ClientError,
    ConnectTimeoutError,
    EndpointConnectionError,
    NoCredentialsError,
    ReadTimeoutError,
)

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

MODEL = "global.anthropic.claude-sonnet-4-6"
ANTHROPIC_VERSION = "bedrock-2023-05-31"
_FINISH: dict[str, FinishReason] = {
    "end_turn": "stop",
    "stop_sequence": "stop",
    "max_tokens": "length",
    "tool_use": "tool_calls",
    "refusal": "content_filter",
}
_THROTTLED = {"ThrottlingException", "ServiceQuotaExceededException", "ModelNotReadyException"}


def wire_tool(spec: dict[str, Any]) -> dict[str, Any]:
    """An OpenAI-style function tool as an Anthropic tool definition."""
    fn = spec["function"]
    return {
        "name": fn["name"],
        "description": fn.get("description", ""),
        "input_schema": fn.get("parameters") or {"type": "object", "properties": {}},
    }


def _tool_input(tc: ToolCall) -> Any:
    try:
        return json.loads(tc.arguments or "{}")
    except json.JSONDecodeError as exc:
        raise RequestRejected(f"tool call {tc.id} has arguments that are not JSON") from exc


def wire_messages(messages: tuple[Message, ...]) -> tuple[str, list[dict[str, Any]]]:
    """The system text and the Anthropic message list for our message sequence."""
    i = 0
    system: list[str] = []
    while i < len(messages) and messages[i].role == "system":
        system.append(messages[i].content)
        i += 1
    turns: list[dict[str, Any]] = []

    def add(role: str, blocks: list[dict[str, Any]]) -> None:
        # Consecutive same-role turns are merged: all tool results of one assistant turn must
        # come back in a single user message.
        if turns and turns[-1]["role"] == role:
            turns[-1]["content"].extend(blocks)
        else:
            turns.append({"role": role, "content": blocks})

    for m in messages[i:]:
        if m.role == "system":
            raise RequestRejected("system messages are only supported at the start")
        if m.role == "user":
            add("user", [{"type": "text", "text": m.content or "(empty)"}])
        elif m.role == "tool":
            if m.tool_call_id is None:
                raise RequestRejected("tool message without tool_call_id")
            add(
                "user",
                [{"type": "tool_result", "tool_use_id": m.tool_call_id, "content": m.content}],
            )
        else:
            blocks: list[dict[str, Any]] = []
            if m.content:
                blocks.append({"type": "text", "text": m.content})
            blocks.extend(
                {"type": "tool_use", "id": tc.id, "name": tc.name, "input": _tool_input(tc)}
                for tc in m.tool_calls
            )
            add("assistant", blocks or [{"type": "text", "text": "(empty)"}])
    return "\n\n".join(system), turns


def wire_request(request: LLMRequest) -> dict[str, Any]:
    """The JSON body sent to Bedrock. Internal fields (sources, sample ids, seed) are not sent."""
    if request.temperature > 1.0:
        raise RequestRejected("Claude accepts temperatures from 0 to 1")
    if request.response_format is not None:
        raise RequestRejected("response_format is not supported by the Bedrock adapter")
    system, messages = wire_messages(request.messages)
    body: dict[str, Any] = {
        "anthropic_version": ANTHROPIC_VERSION,
        "max_tokens": request.max_completion_tokens,
        "temperature": request.temperature,
        "messages": messages,
    }
    if system:
        body["system"] = system
    if request.reasoning_effort is not None:
        body["output_config"] = {"effort": request.reasoning_effort}
    if request.tools:
        body["tools"] = [wire_tool(t) for t in request.tools]
        body["tool_choice"] = {"type": "auto"}
    return body


def parse_reply(data: Any, latency_s: float, request_id: str | None) -> LLMReply:
    try:
        usage = data["usage"]
        # Without caching these are zero or absent; with caching they are billed prompt tokens.
        prompt = (
            int(usage["input_tokens"])
            + int(usage.get("cache_read_input_tokens") or 0)
            + int(usage.get("cache_creation_input_tokens") or 0)
        )
        completion = int(usage["output_tokens"])
        blocks = data["content"]
        texts = [b["text"] for b in blocks if b.get("type") == "text"]
        calls = tuple(
            ToolCall(id=b["id"], name=b["name"], arguments=json.dumps(b.get("input", {})))
            for b in blocks
            if b.get("type") == "tool_use"
        )
        cached = usage.get("cache_read_input_tokens")
        return LLMReply(
            content="".join(texts),
            tool_calls=calls,
            usage=Usage(
                prompt_tokens=prompt,
                completion_tokens=completion,
                cached_prompt_tokens=None if cached is None else int(cached),
            ),
            finish_reason=_FINISH.get(str(data.get("stop_reason")), "other"),
            model=str(data.get("model", "")),
            request_id=request_id or data.get("id"),
            latency_s=latency_s,
        )
    except (KeyError, IndexError, TypeError, ValueError) as exc:
        raise LLMError(f"malformed reply or missing usage: {exc!r}") from exc


def default_client(region: str, timeout_s: float) -> Any:
    config = Config(
        region_name=region,
        read_timeout=timeout_s,
        connect_timeout=10,
        retries={"total_max_attempts": 1, "mode": "standard"},
        max_pool_connections=32,
    )
    return boto3.client("bedrock-runtime", config=config)


class BedrockTransport:
    def __init__(
        self,
        *,
        region: str = "us-east-1",
        client: Any | None = None,
        timeout_s: float = 120.0,
    ) -> None:
        self._client = client if client is not None else default_client(region, timeout_s)

    def _invoke(self, model: str, body: dict[str, Any]) -> tuple[Any, str | None]:
        response = self._client.invoke_model(
            modelId=model,
            body=json.dumps(body),
            contentType="application/json",
            accept="application/json",
        )
        request_id = (response.get("ResponseMetadata") or {}).get("RequestId")
        return json.loads(response["body"].read()), request_id

    async def send(self, request: LLMRequest) -> LLMReply:
        body = wire_request(request)
        start = time.monotonic()
        try:
            data, request_id = await asyncio.to_thread(self._invoke, request.model, body)
        except ClientError as exc:
            error = exc.response.get("Error", {})
            code = str(error.get("Code", ""))
            status = int(exc.response.get("ResponseMetadata", {}).get("HTTPStatusCode", 0))
            message = f"{code} (HTTP {status}): {str(error.get('Message', ''))[:500]}"
            if code in _THROTTLED or status == 429:
                raise RateLimited(message) from exc
            if 400 <= status < 500 and code not in {"ModelTimeoutException", "ModelErrorException"}:
                raise RequestRejected(message) from exc
            raise LLMError(message) from exc
        except (EndpointConnectionError, ConnectTimeoutError) as exc:
            raise ConnectionFailed(f"could not connect: {exc!r}") from exc
        except NoCredentialsError as exc:
            raise RequestRejected("no AWS credentials found") from exc
        except ReadTimeoutError as exc:
            raise TransportTimeout(f"request timed out: {exc!r}") from exc
        except BotoCoreError as exc:
            raise LLMError(f"transport error: {exc!r}") from exc
        except json.JSONDecodeError as exc:
            raise LLMError("reply is not JSON") from exc
        return parse_reply(data, time.monotonic() - start, request_id)

    async def aclose(self) -> None:
        return None
