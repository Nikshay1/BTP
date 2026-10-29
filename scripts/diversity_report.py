"""Compare answer diversity between the two diversity-check experiments (decision D25).

Usage: uv run python scripts/diversity_report.py runs/diversity-t06 runs/diversity-t10

Both configs run Direct once per repetition on single-node tasks with one attempt, so every
repetition of a task sends the same first request and only the provider's sampling differs. Per
task and temperature the report gives: distinct first request bodies (must be 1), distinct
emitted answers after the task family's own normalization (accepted or not), unreadable ones,
distinct accepted answers (the options bee would count), distinct final reply texts, and correct
runs. What matters for bee and for
repetition statistics is whether errors differ, not wording.
"""

from __future__ import annotations

import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

from bioalloc.evaluation.registry import TaskLoader
from bioalloc.llm.base import LLMRequest
from bioalloc.llm.bedrock import wire_request


def trace(run_dir: str) -> list[dict[str, Any]]:
    lines = (Path(run_dir) / "trace.jsonl").read_text().splitlines()
    return [json.loads(line) for line in lines]


def summarize(exp_dir: Path, loader: TaskLoader) -> dict[str, dict[str, int]]:
    """Per task: first request bodies; emitted answers (what the solver proposed, read from each
    attempt's output, whether or not the checks accepted it) as normalized options; how many
    emitted answers were unreadable; how many passed the public checks (accepted runs, whose
    distinct options bee would count); distinct final reply texts; correct runs."""
    rows: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "bodies": set(),
            "emitted": set(),
            "accepted": set(),
            "texts": set(),
            "runs": 0,
            "unreadable": 0,
            "ok": 0,
        }
    )
    for line in (exp_dir / "results.jsonl").read_text().splitlines():
        rec = json.loads(line)
        task = loader.load(rec["task_id"])
        assert task.public is not None, "the diversity check uses single-node tasks"
        subtask = task.public.as_single_node().nodes[0]
        row = rows[rec["task_id"]]
        row["runs"] += 1
        row["ok"] += int(rec["correct"])
        events = trace(rec["run_dir"])
        # Only the first request is identical across repetitions: later ones carry the model's
        # own tool calls (provider-generated ids). Only the last reply is the final answer text.
        first = next(e for e in events if e["kind"] == "llm_request")
        body = wire_request(LLMRequest.model_validate(first["data"]["request"]))
        row["bodies"].add(hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest())
        replies = [e for e in events if e["kind"] == "llm_reply"]
        if replies:
            row["texts"].add(replies[-1]["data"]["reply"]["content"])
        for e in events:
            if e["kind"] == "attempt_end":
                option = task.task_family.normalize(subtask, e["data"]["result"].get("output"))
                if option is None:
                    row["unreadable"] += 1
                else:
                    row["emitted"].add(option)
        answer = next((e["data"].get("answer") for e in events if e["kind"] == "run_end"), None)
        accepted = task.task_family.normalize(subtask, answer)
        if accepted is not None:
            row["accepted"].add(accepted)
    return {
        t: {
            "runs": r["runs"],
            "request_bodies": len(r["bodies"]),
            "emitted_options": len(r["emitted"]),
            "unreadable": r["unreadable"],
            "accepted_options": len(r["accepted"]),
            "distinct_texts": len(r["texts"]),
            "correct": r["ok"],
        }
        for t, r in sorted(rows.items())
    }


def main() -> int:
    loader = TaskLoader()
    for arg in sys.argv[1:]:
        print(f"== {arg}")
        for task, row in summarize(Path(arg), loader).items():
            print(f"  {task:<30} " + "  ".join(f"{k}={v}" for k, v in row.items()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
