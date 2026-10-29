"""Success and cost of the difficulty calibration by named subset (decision D26).

Usage: uv run python scripts/calibration_report.py runs/calibration-v1

Pooled rates can hide a hard subset (15 Logistics tasks against 15 others), so the report gives
each system's results per subset: each family of the D24 core panel, the 12 extra Logistics
problems, and all 15 Logistics problems. The D26 target region (single agent at roughly 60-85%)
is judged per subset, never on the pooled panel.
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from select_calibration_tasks import calibration  # noqa: E402


def subsets() -> dict[str, list[str]]:
    chosen = calibration()
    logistics = [t for t in chosen["planning"] if "logistics" in t] + chosen["logistics_hard"]
    return {
        "core planning": chosen["planning"],
        "core tools": chosen["tools"],
        "core reasoning": chosen["reasoning"],
        "extra logistics": chosen["logistics_hard"],
        "all logistics": logistics,
    }


def main() -> int:
    records = [
        json.loads(x) for x in (Path(sys.argv[1]) / "results.jsonl").read_text().splitlines()
    ]
    by_task: dict[tuple[str, str], list[dict]] = defaultdict(list)  # type: ignore[type-arg]
    for r in records:
        by_task[(r["system"], r["task_id"])].append(r)
    systems = sorted({r["system"] for r in records})
    print(f"{'subset':<16}{'system':<14}{'runs':>5}{'success':>9}{'mean_usd':>10}{'imputed':>9}")
    for name, tasks in subsets().items():
        for system in systems:
            rs = [r for t in tasks for r in by_task[(system, t)]]
            if not rs:
                continue
            ok = sum(bool(r["correct"]) for r in rs) / len(rs)
            usd = sum(r["usd"] for r in rs) / len(rs)
            imputed = sum(not r["usage_complete"] for r in rs)
            print(f"{name:<16}{system:<14}{len(rs):>5}{ok:>9.2f}{usd:>10.4f}{imputed:>9}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
