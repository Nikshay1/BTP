import asyncio
from dataclasses import dataclass, field

import pytest

from bioalloc.llm.base import LLMReply, LLMRequest
from bioalloc.llm.fake import FakeTransport, Predicate


@dataclass
class SlowTransport:
    """Wraps a fake so that replies take real (event-loop) time, like a provider."""

    inner: FakeTransport
    delay: float = 0.01
    delays: list[tuple[Predicate, float]] = field(default_factory=list)

    async def send(self, request: LLMRequest) -> LLMReply:
        wait = next((d for p, d in self.delays if p(request)), self.delay)
        await asyncio.sleep(wait)
        return await self.inner.send(request)


@pytest.fixture
def slow() -> type[SlowTransport]:
    return SlowTransport
