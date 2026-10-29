"""Compare answer diversity between the two diversity-check experiments (decision D25).

Usage: uv run python scripts/diversity_report.py runs/diversity-t06 runs/diversity-t10

For each task and temperature: how many of the repeated, identical requests gave distinct final
answers, how many distinct reply texts the solver produced, and how many runs were correct.
What matters for bee and for repetition statistics is whether errors differ, not wording.
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path


def summarize(run_dir: Path) -> dict[str, dict[str, int]]:
    answers: dict[str, set[str]] = defaultdict(set)
    texts: dict[str, set[str]] = defaultdict(set)
    correct: dict[str, int] = defaultdict(int)
    runs: dict[str, int] = defaultdict(int)
    for line in (run_dir / "results.jsonl").read_text().splitlines():
        rec = json.loads(line)
        task = rec["task_id"]
        runs[task] += 1
        correct[task] += int(rec["correct"])
        for ev in (Path(rec["run_dir"]) / "trace.jsonl").read_text().splitlines():
            e = json.loads(ev)
            if e["kind"] == "run_end":
                answers[task].add(json.dumps(e["data"].get("answer"), sort_keys=True))
            elif e["kind"] == "llm_reply":
                texts[task].add(e["data"]["reply"]["content"])
    return {
        t: {
            "runs": runs[t],
            "distinct_answers": len(answers[t]),
            "distinct_texts": len(texts[t]),
            "correct": correct[t],
        }
        for t in sorted(runs)
    }


def main() -> int:
    for arg in sys.argv[1:]:
        print(f"== {arg}")
        for task, row in summarize(Path(arg)).items():
            print(f"  {task:<30} " + "  ".join(f"{k}={v}" for k, v in row.items()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
