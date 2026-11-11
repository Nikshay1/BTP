"""Accuracy, cost and paired differences for a reasoning experiment (D31).

Usage: uv run python scripts/vote_report.py runs/aime-test-v1
Paired differences are over tasks (one repetition), with a percentile bootstrap (10,000 draws).
"""

import json
import random
import sys
from collections import defaultdict
from pathlib import Path

PAIRS = [
    ("bee_quorum5", "single_agent"),
    ("majority5", "single_agent"),
    ("bee_quorum5", "majority5"),
    ("bee_quorum3", "single_agent"),
    ("majority3", "single_agent"),
    ("bee_quorum3", "majority3"),
]


def boot(diffs: list[float], n: int = 10_000) -> tuple[float, float]:
    rng = random.Random(0)
    means = sorted(sum(rng.choice(diffs) for _ in diffs) / len(diffs) for _ in range(n))
    return means[int(0.025 * n)], means[int(0.975 * n)]


def main() -> None:
    rows = [json.loads(x) for x in (Path(sys.argv[1]) / "results.jsonl").open() if x.strip()]
    by: dict[str, dict[str, dict]] = defaultdict(dict)  # type: ignore[type-arg]
    for r in rows:
        by[r["system"]][r["task_id"]] = r
    print(f"{'system':14} {'correct':>9} {'acc':>6} {'tokens':>8} {'calls':>6} {'usd/run':>8}")
    for name, rs in sorted(by.items()):
        v = list(rs.values())
        k = sum(r["correct"] for r in v)
        tok = sum(r["tokens"] for r in v) / len(v)
        calls = sum(r["model_calls"] for r in v) / len(v)
        usd = sum(r["usd"] for r in v) / len(v)
        print(
            f"{name:14} {k:>4}/{len(v):<4} {k / len(v):>6.1%} {tok:>8.0f} {calls:>6.1f} {usd:>8.4f}"
        )
    print(f"total US${sum(r['usd'] for r in rows):.2f} over {len(rows)} runs\n")
    for a, b in PAIRS:
        if a not in by or b not in by:
            continue
        common = sorted(set(by[a]) & set(by[b]))
        if not common:
            continue
        acc = [float(by[a][t]["correct"]) - float(by[b][t]["correct"]) for t in common]
        tok = [by[a][t]["tokens"] / max(1, by[b][t]["tokens"]) for t in common]
        lo, hi = boot(acc)
        ta = sum(by[a][t]["tokens"] for t in common)
        tb = sum(by[b][t]["tokens"] for t in common)
        wins = sum(d > 0 for d in acc)
        losses = sum(d < 0 for d in acc)
        print(
            f"{a} - {b} on {len(common)} tasks: accuracy {sum(acc) / len(acc):+.3f} "
            f"(95% CI {lo:+.3f} to {hi:+.3f}; {wins} wins, {losses} losses), "
            f"tokens {ta / tb:.2f}x"
        )
        del tok


if __name__ == "__main__":
    main()
