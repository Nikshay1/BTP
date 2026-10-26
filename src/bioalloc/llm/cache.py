"""Response cache for debugging and exact replay only.

Measured runs must hit the provider (research plan, Section 4.4, threats to validity), so this
cache refuses to be built in measurement mode. Replies served from it are marked ``from_cache``.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from bioalloc.llm.base import LLMReply, LLMRequest, RunMode, Transport


class CacheNotAllowed(RuntimeError):
    pass


class CachingTransport:
    def __init__(self, inner: Transport, path: Path, mode: RunMode) -> None:
        if mode == "measurement":
            raise CacheNotAllowed("the response cache cannot be used in measurement mode")
        self.inner = inner
        path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path)
        self.db.execute("CREATE TABLE IF NOT EXISTS replies (key TEXT PRIMARY KEY, reply TEXT)")

    async def send(self, request: LLMRequest) -> LLMReply:
        key = request.key()
        row = self.db.execute("SELECT reply FROM replies WHERE key = ?", (key,)).fetchone()
        if row is not None:
            return LLMReply.model_validate_json(row[0]).model_copy(update={"from_cache": True})
        reply = await self.inner.send(request)
        self.db.execute(
            "INSERT OR REPLACE INTO replies VALUES (?, ?)", (key, reply.model_dump_json())
        )
        self.db.commit()
        return reply

    def close(self) -> None:
        self.db.close()
