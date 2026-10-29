"""Unrecord runs that the provider throttled, so that a rerun redoes them (R18).

Usage: uv run python scripts/drop_throttled_runs.py runs/<experiment>

Before R18, a run refused by a daily quota (or 429s past every retry) was recorded as a task
failure. This removes every recorded run whose trace holds such a refusal from results.jsonl
(the old file is kept as results.jsonl.before-r18) and moves its directory aside as
``repN.throttled.K``, keeping its trace, so its spending stays counted. Running the experiment
again then redoes exactly those runs. Run it only while no experiment is running there.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path


def throttled(run_dir: Path) -> bool:
    trace = run_dir / "trace.jsonl"
    if not trace.exists():
        return False
    for line in trace.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            e = json.loads(line)
        except ValueError:
            continue
        if e.get("kind") in ("llm_error", "provider_throttled"):
            err = str(e.get("data", {}).get("error", ""))
            if "ThrottlingException" in err or "RateLimited" in err:
                return True
    return False


def main() -> int:
    exp = Path(sys.argv[1])
    results = exp / "results.jsonl"
    lines = results.read_text(encoding="utf-8").splitlines()
    keep, dropped = [], []
    for line in lines:
        rec = json.loads(line)
        (dropped if throttled(Path(rec["run_dir"])) else keep).append((line, rec))
    if not dropped:
        print("no throttled runs recorded")
        return 0
    shutil.copy2(results, results.with_name("results.jsonl.before-r18"))
    for _, rec in dropped:
        run_dir = Path(rec["run_dir"])
        n = 0
        while (aside := run_dir.with_name(f"{run_dir.name}.throttled.{n}")).exists():
            n += 1
        run_dir.rename(aside)
    tmp = results.with_suffix(".jsonl.tmp")
    tmp.write_text("".join(f"{line}\n" for line, _ in keep), encoding="utf-8")
    os.replace(tmp, results)
    print(f"unrecorded {len(dropped)} throttled runs; {len(keep)} kept")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
