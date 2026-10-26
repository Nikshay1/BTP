import asyncio
import json

import httpx

from bioalloc.evaluation.capability import capability_check
from bioalloc.llm.groq import GroqTransport


def test_capability_check_against_mocked_groq() -> None:
    calls: list[dict] = []  # type: ignore[type-arg]

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        calls.append(body)
        usage = {
            "prompt_tokens": 40,
            "completion_tokens": 20,
            "completion_tokens_details": {"reasoning_tokens": 8},
        }
        if body.get("tools") and body["messages"][-1]["role"] != "tool":
            msg = {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "c1",
                        "type": "function",
                        "function": {"name": "calculate", "arguments": '{"expression": "123*47"}'},
                    }
                ],
            }
            return httpx.Response(
                200,
                json={
                    "model": "openai/gpt-oss-120b",
                    "usage": usage,
                    "choices": [{"finish_reason": "tool_calls", "message": msg}],
                },
            )
        content = "5781" if body.get("tools") else '{"answer": 42}'
        return httpx.Response(
            200,
            json={
                "model": "openai/gpt-oss-120b",
                "usage": usage,
                "choices": [
                    {"finish_reason": "stop", "message": {"role": "assistant", "content": content}}
                ],
            },
        )

    t = GroqTransport("k", client=httpx.AsyncClient(transport=httpx.MockTransport(handler)))
    report = asyncio.run(capability_check(t))
    assert (
        report["same_seed_same_reply"]
        and report["tool_called"]
        and report["tool_readback"] == "5781"
    )
    assert report["reasoning_tokens_itemized"] and report["finish_reasons"] == [
        "stop",
        "stop",
        "tool_calls",
        "stop",
    ]
    assert calls[-1]["messages"][-1] == {"role": "tool", "content": "5781", "tool_call_id": "c1"}
    assert report["total_tokens"] == 4 * 60
