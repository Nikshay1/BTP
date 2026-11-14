"""Transport for OpenAI's chat completions API, used for GPT-6 Luna (decision D32).

The wire format is the OpenAI schema the Groq transport already speaks, so this reuses its request
and reply mapping and its error handling, with two differences: OpenAI rejects Groq's
``include_reasoning`` field, so it is not sent, and a 429 whose ``error.code`` is
``insufficient_quota`` (the account's credit is spent) is marked as a quota that no retry clears,
so the run ends unrecorded instead of waiting out its deadline (R18).

Checked against the live API on 14 November 2026 for ``gpt-6-luna``: ``seed``, ``n``,
``max_completion_tokens``, tools and ``reasoning_effort`` are accepted, but tools only with
``reasoning_effort`` ``none`` (other levels need the Responses API), so the Luna configs use
``none``, the counterpart of Sonnet's thinking off; ``temperature`` must be 1; usage reports
reasoning tokens inside ``completion_tokens``, which are billed as output.
"""

from __future__ import annotations

from typing import Any

import httpx

from bioalloc.llm.base import LLMRequest
from bioalloc.llm.groq import GroqTransport, wire_request

BASE_URL = "https://api.openai.com/v1"
MODEL = "gpt-6-luna"


def is_openai_model(model: str) -> bool:
    """OpenAI's own model names (``gpt-6-luna``); Groq's hosted ones carry a ``vendor/`` prefix."""
    return model.startswith("gpt-") and "/" not in model


class OpenAITransport(GroqTransport):
    def __init__(
        self,
        api_key: str,
        *,
        client: httpx.AsyncClient | None = None,
        base_url: str = BASE_URL,
        timeout_s: float = 120.0,
    ) -> None:
        if not api_key:
            raise ValueError("an OpenAI API key is required")
        super().__init__(api_key, client=client, base_url=base_url, timeout_s=timeout_s)

    def body(self, request: LLMRequest) -> dict[str, Any]:
        body = wire_request(request)
        del body["include_reasoning"]
        return body
