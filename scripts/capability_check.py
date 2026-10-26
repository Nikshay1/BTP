"""Run the live capability check against Groq (needs GROQ_API_KEY; costs well under a cent).

Usage: uv run python scripts/capability_check.py
"""

from __future__ import annotations

import asyncio
import json

from bioalloc.cli import read_env_key
from bioalloc.evaluation.capability import capability_check
from bioalloc.llm.groq import GroqTransport


def main() -> int:
    key = read_env_key("GROQ_API_KEY")
    if not key:
        print("GROQ_API_KEY is not set (environment or .env); nothing was sent.")
        return 2
    report = asyncio.run(capability_check(GroqTransport(key)))
    print(json.dumps(report, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
