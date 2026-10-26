import asyncio
import json
from pathlib import Path

import pytest

from bioalloc.budget import Account, MeteredClient
from bioalloc.llm.base import LLMRequest, Message, TransportTimeout
from bioalloc.llm.fake import FakeTransport
from bioalloc.schemas import Budget
from bioalloc.trace import (
    Tracer,
    finish_manifest,
    read_manifest,
    read_trace,
    span_tree,
    start_manifest,
    tokens_from_trace,
)


def req(text: str, source: str | None = None) -> LLMRequest:
    return LLMRequest(
        model="m",
        messages=(Message(role="user", content=text, source=source),),
        max_completion_tokens=16,
    )


def test_trace_reconstructs_ledger_totals(tmp_path: Path) -> None:
    tracer = Tracer(tmp_path, "run-1")
    root = tracer.span()
    child_span = tracer.span(parent=root)
    account = Account(Budget(max_tokens=10_000, max_model_calls=10))
    fake = FakeTransport().always("one", TransportTimeout("slow"), "three")
    coord = MeteredClient(fake, account, "coordinator", "coord", on_event=tracer.hook(root))
    worker = coord.child(Budget(max_tokens=5_000, max_model_calls=5), actor="worker", agent="w1")
    worker.on_event = tracer.hook(child_span)

    asyncio.run(coord.complete(req("plan the task")))
    with pytest.raises(TransportTimeout):
        asyncio.run(worker.complete(req("leg one", source="coord")))
    asyncio.run(worker.complete(req("leg one again", source="coord")))
    tracer.close()

    events = list(read_trace(tmp_path))
    assert [e.seq for e in events] == list(range(len(events)))
    totals = tokens_from_trace(events)
    ledger = account.snapshot()
    assert totals["model_calls"] == ledger.model_calls == 3
    assert totals["tokens"] == ledger.tokens.total
    assert totals["communication"] == ledger.communication > 0
    assert {e.span_id for e in span_tree(events, child_span)} == {child_span}
    assert len(span_tree(events, root)) == len(events)


def test_trace_survives_interruption(tmp_path: Path) -> None:
    tracer = Tracer(tmp_path, "run-2")
    s = tracer.span()
    tracer.emit(s, "start", {"x": 1})
    # No close(): every event is flushed when written.
    assert [e.kind for e in read_trace(tmp_path)] == ["start"]
    tracer.close()


def test_manifest_lifecycle(tmp_path: Path) -> None:
    m = start_manifest(
        tmp_path,
        "run-3",
        mode="debug",
        config={"budget": 100},
        model="fake",
        seeds={"sampling": 0},
        prompts={"direct": "Answer the question."},
    )
    assert read_manifest(tmp_path).status == "running"
    assert len(m.config_sha256) == 64 and "direct" in m.prompt_sha256
    assert "git_commit" in m.code and "uv_lock_sha256" in m.code
    finish_manifest(tmp_path, m, "completed", {"success": True})
    done = read_manifest(tmp_path)
    assert done.status == "completed" and done.finished is not None
    assert json.loads((tmp_path / "manifest.json").read_text())["summary"] == {"success": True}
