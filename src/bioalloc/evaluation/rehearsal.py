"""A stand-in model for offline rehearsals of whole experiments (no network, no key).

It answers every prompt type with output of the right shape (a single-node decomposition, empty
advice, accepted reviews, FINISH proposals, a JSON object with the usual answer fields) and
reports a deterministic token estimate. Rehearsal results test the pipeline, never the methods,
and rehearsal runs are always in debug mode.
"""

from __future__ import annotations

import json
import math
from typing import Any

from bioalloc.llm.base import LLMReply, LLMRequest, Usage


def _tokens(text: str) -> int:
    return max(1, math.ceil(len(text) / 4))


class RehearsalTransport:
    def __init__(self) -> None:
        self.calls = 0

    async def send(self, request: LLMRequest) -> LLMReply:
        self.calls += 1
        system = request.messages[0].content
        body: dict[str, Any]
        if "plan how a task will be split" in system:
            body = {
                "nodes": [
                    {
                        "id": "s1",
                        "instruction": "Solve the whole task.",
                        "depends_on": [],
                        "inputs": [],
                    }
                ],
                "final": "s1",
            }
        elif "coordinate workers" in system:
            body = {"advice": {}}
        elif "review a worker" in system:
            body = {"accept": True, "critique": ""}
        elif "supervise a worker" in system:
            body = {"decision": "continue", "instruction": ""}
        elif "Propose up to five" in system:
            body = {
                "candidates": [{"kind": "FINISH", "args": {"answer": 0, "output": {"answer": 0}}}]
            }
        elif "has proposed an answer" in system:
            body = {"accept": True, "confidence": 0.5}
        elif "numbered plan" in system:
            return self._reply(request, "1. Read the task. 2. Answer it.")
        else:
            body = {"answer": 0, "plan": [], "items": [], "route": []}
        return self._reply(request, json.dumps(body))

    def _reply(self, request: LLMRequest, content: str) -> LLMReply:
        prompt = sum(_tokens(m.content) for m in request.messages)
        return LLMReply(
            content=content,
            usage=Usage(
                prompt_tokens=prompt,
                completion_tokens=min(_tokens(content), request.max_completion_tokens),
            ),
            model="rehearsal",
            request_id=f"rehearsal-{self.calls}",
        )
