"""Freeze the AIME split (D31): 30 tuning problems, the other 148 test, by hashed order.

Usage: uv run python scripts/make_aime_split.py
"""

import hashlib
import json

from bioalloc.evaluation import aime_data
from bioalloc.evaluation.data import REPO_ROOT, verify_prepared
from bioalloc.evaluation.splits import Assignment, canonical

SEED = "aime-v1|20261111"
TUNE = 30


def main() -> None:
    items = aime_data.load()
    order = sorted(items, key=lambda i: hashlib.sha256(f"{SEED}|{i.group}".encode()).hexdigest())
    tune = {i.task.id for i in order[:TUNE]}
    assignments = [
        Assignment(
            task_id=i.task.id,
            family="reasoning",
            group=i.group,
            split="tune" if i.task.id in tune else "test",
        )
        for i in items
    ]
    text = canonical(assignments)
    payload = {
        "suite": "aime-v1",
        "seed": SEED,
        "provenance": {"aime": verify_prepared("aime")["prepared"]},
        "sha256": hashlib.sha256(text.encode()).hexdigest(),
        "assignments": json.loads(text),
    }
    path = REPO_ROOT / "data" / "splits" / "aime-v1.json"
    path.write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")
    print(path, payload["sha256"][:12], f"tune {len(tune)}, test {len(items) - len(tune)}")


if __name__ == "__main__":
    main()
