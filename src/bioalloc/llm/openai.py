"""Transport for OpenAI's Responses API, used for GPT-6 Luna (decision D32).

GPT-6 Luna takes function tools together with a reasoning effort other than ``none`` only on the
Responses API (``/v1/responses``), and at ``none`` it answers AIME without working (6/30 tuning
problems against 16/30 at ``low``), so this adapter speaks that API. It is stateless like the other
adapters: every call sends the whole conversation (``store`` is off) and the model's hidden
reasoning is not sent back between tool rounds, which the API accepts. HTTP status handling is the
Groq transport's (both are OpenAI-style), and a 429 whose ``error.code`` is ``insufficient_quota``
(the account's credit is spent) is marked as a quota that no retry clears, so the run ends
unrecorded instead of waiting out its deadline (R18).

Mapping. Chat messages become input items: system, user and assistant text keep their role,
an assistant's tool calls become ``function_call`` items and tool results ``function_call_output``
items, matched by ``call_id``. Tools are flattened from the chat form. ``max_completion_tokens``
becomes ``max_output_tokens`` (it covers reasoning tokens, as on chat completions). There is no
seed on this API, so repeated samples are not seed-reproducible (Sonnet on Bedrock had none either).
Usage ``output_tokens`` include reasoning tokens, billed as output; they are never added twice.

Checked against the live API on 14 November 2026: ``temperature`` must be 1; a tool round trip
replayed without reasoning items is accepted; usage itemizes reasoning and cached tokens.
"""

from __future__ import annotations

from typing import Any

import httpx

from bioalloc.llm.base import (
    FinishReason,
    LLMError,
    LLMReply,
    LLMRequest,
    Message,
    RequestRejected,
    ToolCall,
    Usage,
)
from bioalloc.llm.groq import GroqTransport

BASE_URL = "https://api.openai.com/v1"
MODEL = "gpt-6-luna"
_INCOMPLETE: dict[str, FinishReason] = {
    "max_output_tokens": "length",
    "content_filter": "content_filter",
}


def is_openai_model(model: str) -> bool:
    """OpenAI's own model names (``gpt-6-luna``); Groq's hosted ones carry a ``vendor/`` prefix."""
    return model.startswith("gpt-") and "/" not in model


def input_items(m: Message) -> list[dict[str, Any]]:
    if m.role == "tool":
        if m.tool_call_id is None:
            raise RequestRejected("tool message without tool_call_id")
        return [{"type": "function_call_output", "call_id": m.tool_call_id, "output": m.content}]
    items: list[dict[str, Any]] = []
    if m.content or not m.tool_calls:
        items.append({"role": m.role, "content": m.content})
    items += [
        {"type": "function_call", "call_id": tc.id, "name": tc.name, "arguments": tc.arguments}
        for tc in m.tool_calls
    ]
    return items


def wire_tool(tool: Any) -> dict[str, Any]:
    if not isinstance(tool, dict) or tool.get("type") != "function" or "function" not in tool:
        raise RequestRejected(f"unsupported tool spec: {tool!r}")
    fn = tool["function"]
    return {
        "type": "function",
        "name": fn["name"],
        "description": fn.get("description", ""),
        "parameters": fn.get("parameters", {"type": "object", "properties": {}}),
        "strict": False,
    }


def wire_format(response_format: Any) -> dict[str, Any]:
    kind = response_format.get("type") if isinstance(response_format, dict) else None
    if kind == "json_object":
        return {"type": "json_object"}
    if kind == "json_schema":
        schema = response_format["json_schema"]
        return {"type": "json_schema", **schema}
    raise RequestRejected(f"unsupported response_format: {response_format!r}")


def wire_request(request: LLMRequest) -> dict[str, Any]:
    """The JSON body sent to OpenAI. Internal fields (message sources, sample ids) are not sent."""
    body: dict[str, Any] = {
        "model": request.model,
        "input": [item for m in request.messages for item in input_items(m)],
        "max_output_tokens": request.max_completion_tokens,
        "temperature": request.temperature,
        "store": False,
    }
    if request.reasoning_effort is not None:
        body["reasoning"] = {"effort": request.reasoning_effort}
    if request.tools:
        body["tools"] = [wire_tool(t) for t in request.tools]
        body["tool_choice"] = "auto"
    if request.response_format is not None:
        body["text"] = {"format": wire_format(request.response_format)}
    return body


def parse_reply(data: Any, latency_s: float, request_id: str | None) -> LLMReply:
    try:
        status = data["status"]
        output = data["output"]
        usage = data["usage"]
        prompt = int(usage["input_tokens"])
        completion = int(usage["output_tokens"])
    except (KeyError, TypeError, ValueError) as exc:
        raise LLMError(f"malformed reply or missing usage: {exc!r}") from exc
    if status not in ("completed", "incomplete"):
        raise LLMError(f"response status {status!r}: {data.get('error')!r}")
    reasoning = (usage.get("output_tokens_details") or {}).get("reasoning_tokens")
    cached = (usage.get("input_tokens_details") or {}).get("cached_tokens")
    try:
        text = "".join(
            part["text"]
            for item in output
            if item["type"] == "message"
            for part in item["content"]
            if part["type"] == "output_text"
        )
        calls = tuple(
            ToolCall(id=item["call_id"], name=item["name"], arguments=item.get("arguments") or "")
            for item in output
            if item["type"] == "function_call"
        )
        finish: FinishReason
        if status == "incomplete":
            reason = (data.get("incomplete_details") or {}).get("reason")
            finish = _INCOMPLETE.get(str(reason), "other")
        else:
            finish = "tool_calls" if calls else "stop"
        return LLMReply(
            content=text,
            tool_calls=calls,
            usage=Usage(
                prompt_tokens=prompt,
                completion_tokens=completion,
                reasoning_tokens=None if reasoning is None else int(reasoning),
                cached_prompt_tokens=None if cached is None else int(cached),
            ),
            finish_reason=finish,
            model=str(data.get("model", "")),
            request_id=request_id or data.get("id"),
            latency_s=latency_s,
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise LLMError(f"malformed reply: {exc!r}") from exc


class OpenAITransport(GroqTransport):
    def __init__(
        self,
        api_key: str,
        *,
        client: httpx.AsyncClient | None = None,
        base_url: str = BASE_URL,
        timeout_s: float = 300.0,
    ) -> None:
        if not api_key:
            raise ValueError("an OpenAI API key is required")
        super().__init__(api_key, client=client, base_url=base_url, timeout_s=timeout_s)
        self._url = f"{base_url}/responses"

    def body(self, request: LLMRequest) -> dict[str, Any]:
        return wire_request(request)

    def parse(self, data: Any, latency_s: float, request_id: str | None) -> LLMReply:
        return parse_reply(data, latency_s, request_id)
