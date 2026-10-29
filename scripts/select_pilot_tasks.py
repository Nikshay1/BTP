"""Choose the pilot's training tasks by fixed rules (decision D24) and print them as JSON.

Usage: uv run python scripts/select_pilot_tasks.py N_PER_SUBTYPE

Per family, the training tasks are split into two subtypes and N tasks are taken from each:

* planning: Blocksworld and Logistics;
* tools: the ``revenue`` and ``top_product`` templates;
* reasoning: GSM8K questions with at least the median training word count ("longer"), taking
  2N of them, because GSM8K has one subtype and the short half is likely at ceiling.

Within a subtype, tasks are taken in SHA-256 order of (seed, task id), skipping any whose group
was already taken, so no two chosen tasks share a group. Only public task text is used; answers
and model results play no part. The frozen split file is read, never changed.
"""

from __future__ import annotations

import hashlib
import json
import statistics
import sys
from pathlib import Path

from bioalloc.evaluation import gsm8k_data, splits

SEED = "pilot-d24"
ROOT = Path(__file__).resolve().parents[1]


def order(task_id: str) -> str:
    return hashlib.sha256(f"{SEED}:{task_id}".encode()).hexdigest()


def pick(pool: list[splits.Assignment], n: int, taken: set[str]) -> list[str]:
    out: list[str] = []
    for a in sorted(pool, key=lambda a: order(a.task_id)):
        if len(out) == n:
            break
        if a.group not in taken:
            taken.add(a.group)
            out.append(a.task_id)
    if len(out) < n:
        raise ValueError(f"only {len(out)} tasks with distinct groups")
    return out


def select(n: int) -> dict[str, list[str]]:
    train = [a for a in splits.load(ROOT / "data/splits/v2.json") if a.split == "train"]
    taken: set[str] = set()

    def of(prefix: str) -> list[splits.Assignment]:
        return [a for a in train if a.task_id.startswith(prefix)]

    words = {i.task.id: len(i.task.instruction.split()) for i in gsm8k_data.load()}
    reasoning = of("gsm8k-")
    median = statistics.median(words[a.task_id] for a in reasoning)
    longer = [a for a in reasoning if words[a.task_id] >= median]
    return {
        "planning": pick(of("planbench-blocksworld-"), n, taken)
        + pick(of("planbench-logistics-"), n, taken),
        "tools": pick(of("tools-v1-revenue-"), n, taken)
        + pick(of("tools-v1-top_product-"), n, taken),
        "reasoning": pick(longer, 2 * n, taken),
    }


if __name__ == "__main__":
    print(json.dumps(select(int(sys.argv[1])), indent=1))
