"""Run the live capability check (costs well under a cent).

Usage: uv run python scripts/capability_check.py [bedrock|groq|openai]

bedrock (the default, D23) needs AWS credentials in .env or the usual AWS places; groq needs
GROQ_API_KEY; openai (D32, GPT-6 Luna, temperature 1 only) needs OPENAI_API_KEY.
"""

from __future__ import annotations

import asyncio
import json
import sys

from bioalloc.cli import bedrock_transport, read_env_key
from bioalloc.evaluation.capability import capability_check
from bioalloc.llm import bedrock, openai
from bioalloc.llm.groq import GroqTransport


def main() -> int:
    provider = sys.argv[1] if len(sys.argv) > 1 else "bedrock"
    if provider == "bedrock":
        report = asyncio.run(capability_check(bedrock_transport(), model=bedrock.MODEL))
    elif provider == "groq":
        key = read_env_key("GROQ_API_KEY")
        if not key:
            print("GROQ_API_KEY is not set (environment or .env); nothing was sent.")
            return 2
        report = asyncio.run(capability_check(GroqTransport(key)))
    elif provider == "openai":
        key = read_env_key("OPENAI_API_KEY")
        if not key:
            print("OPENAI_API_KEY is not set (environment or .env); nothing was sent.")
            return 2
        transport = openai.OpenAITransport(key)
        report = asyncio.run(
            capability_check(
                transport, model=openai.MODEL, reasoning_effort="none", temperature=1.0
            )
        )
    else:
        print(f"unknown provider {provider!r}; use bedrock, groq or openai")
        return 2
    print(json.dumps(report, indent=1))
    if not report["ok"]:
        print("capability check FAILED: " + "; ".join(report["problems"]))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
