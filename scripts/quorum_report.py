"""Split a bee_quorum5 experiment by how each bee run ended (exploratory, R25).

Usage: uv run python scripts/quorum_report.py runs/aime-test-v2

For each problem, the bee run ended in one of three ways: a quorum among the first scouts, the
token cap during the scouts, or D14's inhibition and recruitment rounds after the scouts
disagreed. For each group the script prints how many problems each system solved and the bee and
majority5 tokens, plus how often released scouts accepted another answer in recruitment.
"""

from __future__ import annotations

import collections
import json
import sys
from pathlib import Path

SYSTEMS = ("bee_quorum5", "majority5", "single_agent")


def outcome(trace: Path) -> tuple[str, list[bool]]:
    events = [json.loads(line) for line in trace.read_text().splitlines() if line.strip()]
    accepts = [
        bool(d["accept"])
        for e in events
        if e["kind"] == "bee_recruitment"
        for d in e["data"]["decisions"]
    ]
    if any(e["kind"] == "bee_round" for e in events):
        return "recruitment rounds", accepts
    stops = [e["data"] for e in events if e["kind"] == "executor_stop"]
    reason = stops[0].get("reason") if stops else "none"
    return ("quorum among scouts" if reason == "quorum" else f"stopped: {reason}"), accepts


def main(exp: Path) -> None:
    rows = [json.loads(x) for x in (exp / "results.jsonl").read_text().splitlines() if x.strip()]
    res = {(r["task_id"], r["system"]): r for r in rows}
    tasks = sorted({t for t, s in res if all((t, x) in res for x in SYSTEMS)})
    groups: dict[str, list[str]] = collections.defaultdict(list)
    accepts: list[bool] = []
    for t in tasks:
        group, acc = outcome(exp / "runs" / "bee_quorum5" / t / "rep0" / "trace.jsonl")
        groups[group].append(t)
        accepts += acc
    print(f"{len(tasks)} problems with all three systems")
    print(f"{'bee ended by':24} {'n':>4} " + " ".join(f"{s:>13}" for s in SYSTEMS) + "  bee/maj tokens")
    for group, ts in sorted(groups.items(), key=lambda kv: -len(kv[1])):
        solved = [sum(res[(t, s)]["correct"] for t in ts) for s in SYSTEMS]
        bee = sum(res[(t, "bee_quorum5")]["tokens"] for t in ts)
        maj = sum(res[(t, "majority5")]["tokens"] for t in ts)
        print(
            f"{group:24} {len(ts):4} "
            + " ".join(f"{n:>9} {n / len(ts):3.0%}" for n in solved)
            + f"  {bee / maj:.2f}x"
        )
    if accepts:
        print(f"recruitment: {sum(accepts)} of {len(accepts)} released scouts accepted another answer")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
