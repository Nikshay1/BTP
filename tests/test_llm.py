import asyncio
from pathlib import Path

import pytest

from bioalloc.llm.base import LLMRequest, Message, TransportTimeout
from bioalloc.llm.cache import CacheNotAllowed, CachingTransport
from bioalloc.llm.fake import FakeTransport, contains


def req(text: str, sample_id: str = "0") -> LLMRequest:
    return LLMRequest(
        model="m",
        messages=(Message(role="user", content=text),),
        max_completion_tokens=64,
        sample_id=sample_id,
    )


def test_fake_follows_script_in_order() -> None:
    fake = FakeTransport().on(contains("leg 1"), "bad", "good").always("other")
    replies = [asyncio.run(fake.send(req(t))).content for t in ["leg 1", "leg 1", "leg 2"]]
    assert replies == ["bad", "good", "other"]
    assert len(fake.requests) == 3


def test_fake_is_deterministic() -> None:
    def run() -> list[tuple[str, int, int]]:
        fake = FakeTransport().always("alpha", "beta")
        out = [asyncio.run(fake.send(req("q"))) for _ in range(2)]
        return [(r.content, r.usage.prompt_tokens, r.usage.completion_tokens) for r in out]

    assert run() == run()


def test_fake_raises_scripted_errors_and_unscripted_requests() -> None:
    fake = FakeTransport().always(TransportTimeout("slow"))
    with pytest.raises(TransportTimeout):
        asyncio.run(fake.send(req("q")))
    with pytest.raises(AssertionError, match="no scripted reply"):
        asyncio.run(fake.send(req("q")))


def test_request_key_separates_samples() -> None:
    assert req("q").key() == req("q").key()
    assert req("q", sample_id="0").key() != req("q", sample_id="1").key()


def test_cache_replays_and_keeps_samples_distinct(tmp_path: Path) -> None:
    fake = FakeTransport().always("first", "second", "third")
    cache = CachingTransport(fake, tmp_path / "c.sqlite", mode="debug")
    a = asyncio.run(cache.send(req("q", "0")))
    b = asyncio.run(cache.send(req("q", "0")))
    c = asyncio.run(cache.send(req("q", "1")))
    assert (a.content, a.from_cache) == ("first", False)
    assert (b.content, b.from_cache) == ("first", True)
    assert (c.content, c.from_cache) == ("second", False)
    cache.close()


def test_cache_refused_in_measurement_mode(tmp_path: Path) -> None:
    with pytest.raises(CacheNotAllowed):
        CachingTransport(FakeTransport(), tmp_path / "c.sqlite", mode="measurement")
