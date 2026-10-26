"""Run traces and manifests.

A run directory holds two files:

* ``trace.jsonl``: append-only events, one JSON object per line, flushed as they are written. This
  is the source of truth for every analysis; anything else (SQLite indexes, tables) is rebuilt from
  it.
* ``manifest.json``: what is needed to reproduce the run (code revision, lockfile hash, resolved
  configuration, prompt hashes, model, seeds, mode) and its status. A run that was interrupted is
  left with status ``running``.

Events are grouped in spans (the coordinator's run, one dispatch, one executor's work) so that an
executor's full trace can be read back from its root span.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from collections.abc import Iterator
from datetime import UTC, datetime
from itertools import count
from pathlib import Path
from typing import Any, TextIO

from pydantic import JsonValue

from bioalloc.budget import EventHook
from bioalloc.llm.base import RunMode
from bioalloc.schemas import Wire

SCHEMA_VERSION = 1
REPO_ROOT = Path(__file__).resolve().parents[2]


class TraceEvent(Wire):
    schema_version: int = SCHEMA_VERSION
    run_id: str
    seq: int
    time: str
    span_id: str
    parent_span_id: str | None
    kind: str
    data: dict[str, JsonValue]


def _now() -> str:
    return datetime.now(UTC).isoformat()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _git(*args: str) -> str | None:
    try:
        out = subprocess.run(
            ["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, check=True
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return out.stdout.strip()


def code_revision() -> dict[str, JsonValue]:
    status = _git("status", "--porcelain")
    lock = REPO_ROOT / "uv.lock"
    return {
        "git_commit": _git("rev-parse", "HEAD"),
        "git_dirty": bool(status) if status is not None else None,
        "uv_lock_sha256": hashlib.sha256(lock.read_bytes()).hexdigest() if lock.exists() else None,
    }


class Tracer:
    def __init__(self, run_dir: Path, run_id: str) -> None:
        self.run_dir = run_dir
        self.run_id = run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        # Exclusive create: a trace is never appended to by a second run.
        self._file: TextIO = (run_dir / "trace.jsonl").open("x", encoding="utf-8")
        self._seq = count()
        self._span_ids = count(1)
        self._parents: dict[str, str | None] = {}

    def span(self, parent: str | None = None) -> str:
        span_id = f"s{next(self._span_ids)}"
        self._parents[span_id] = parent
        return span_id

    def emit(self, span_id: str, kind: str, data: dict[str, Any] | None = None) -> TraceEvent:
        event = TraceEvent(
            run_id=self.run_id,
            seq=next(self._seq),
            time=_now(),
            span_id=span_id,
            parent_span_id=self._parents.get(span_id),
            kind=kind,
            data=data or {},
        )
        self._file.write(event.model_dump_json() + "\n")
        self._file.flush()
        return event

    def hook(self, span_id: str) -> EventHook:
        """An event hook for a ``MeteredClient`` that writes into ``span_id``."""

        def hook(kind: str, data: dict[str, Any]) -> None:
            self.emit(span_id, kind, data)

        return hook

    def close(self) -> None:
        self._file.close()


def read_trace(run_dir: Path) -> Iterator[TraceEvent]:
    with (run_dir / "trace.jsonl").open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield TraceEvent.model_validate_json(line)


def span_tree(events: list[TraceEvent], root: str) -> list[TraceEvent]:
    """All events in ``root`` and its descendant spans, in order."""
    members = {root}
    for e in events:  # parents are always created before children
        if e.parent_span_id in members:
            members.add(e.span_id)
    return [e for e in events if e.span_id in members]


def tokens_from_trace(events: list[TraceEvent]) -> dict[str, int]:
    """Recompute call and token totals from the trace alone, to cross-check the ledger."""
    totals = {"model_calls": 0, "tokens": 0, "communication": 0, "tool_calls": 0}
    for e in events:
        data: dict[str, Any] = e.data
        if e.kind == "llm_reply":
            usage = data["reply"]["usage"]
            totals["model_calls"] += 1
            totals["tokens"] += int(usage["prompt_tokens"]) + int(usage["completion_tokens"])
            totals["communication"] += int(data["communication_tokens"])
        elif e.kind == "tool_call":
            totals["tool_calls"] += 1
        elif e.kind == "llm_error" and data.get("usage") == "unknown":
            totals["model_calls"] += 1
            totals["tokens"] += int(data["charged_prompt_tokens"])
            totals["tokens"] += int(data["charged_completion_tokens"])
    return totals


class Manifest(Wire):
    schema_version: int = SCHEMA_VERSION
    run_id: str
    mode: RunMode
    status: str = "running"
    started: str
    finished: str | None = None
    code: dict[str, JsonValue]
    config: dict[str, JsonValue]
    config_sha256: str
    model: str
    seeds: dict[str, int]
    prompt_sha256: dict[str, str] = {}
    summary: dict[str, JsonValue] = {}


def start_manifest(
    run_dir: Path,
    run_id: str,
    *,
    mode: RunMode,
    config: dict[str, JsonValue],
    model: str,
    seeds: dict[str, int],
    prompts: dict[str, str] | None = None,
) -> Manifest:
    config_json = json.dumps(config, sort_keys=True, separators=(",", ":"))
    manifest = Manifest(
        run_id=run_id,
        mode=mode,
        started=_now(),
        code=code_revision(),
        config=config,
        config_sha256=sha256_text(config_json),
        model=model,
        seeds=seeds,
        prompt_sha256={name: sha256_text(text) for name, text in (prompts or {}).items()},
    )
    write_manifest(run_dir, manifest)
    return manifest


def finish_manifest(
    run_dir: Path, manifest: Manifest, status: str, summary: dict[str, JsonValue]
) -> Manifest:
    done = manifest.model_copy(update={"status": status, "finished": _now(), "summary": summary})
    write_manifest(run_dir, done)
    return done


def write_manifest(run_dir: Path, manifest: Manifest) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    tmp = run_dir / "manifest.json.tmp"
    tmp.write_text(manifest.model_dump_json(indent=2) + "\n", encoding="utf-8")
    tmp.replace(run_dir / "manifest.json")


def read_manifest(run_dir: Path) -> Manifest:
    return Manifest.model_validate_json((run_dir / "manifest.json").read_text(encoding="utf-8"))
