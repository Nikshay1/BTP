"""Choose the difficulty-calibration tasks by fixed rules (decision D26) and print them as JSON.

Usage: uv run python scripts/select_calibration_tasks.py

The panel is the 18 pilot tasks of D24 plus 12 more training Logistics problems: those with the
most goal atoms (a public measure of how many packages must be moved), ties broken by the D24
SHA-256 order, never two from one group or from a group the pilot tasks already use. Only public
task text is used; split v2 is read, never changed.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from bioalloc.evaluation import planbench_data, splits
from bioalloc.tasks.strips import parse_problem

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from select_pilot_tasks import order, select  # noqa: E402

EXTRA_LOGISTICS = 12


def goal_atoms(item: planbench_data.PlanItem) -> int:
    return len(parse_problem(str(item.task.inputs["problem"])).goal)


def calibration() -> dict[str, list[str]]:
    pilot = select(3)
    chosen = [t for ids in pilot.values() for t in ids]
    assignments = splits.load(ROOT / "data/splits/v2.json")
    group = {a.task_id: a.group for a in assignments}
    taken = {group[t] for t in chosen}
    items = {i.task.id: i for i in planbench_data.load()}
    pool = [
        a.task_id
        for a in assignments
        if a.split == "train" and a.task_id.startswith("planbench-logistics-")
    ]
    extra: list[str] = []
    for t in sorted(pool, key=lambda t: (-goal_atoms(items[t]), order(t))):
        if len(extra) == EXTRA_LOGISTICS:
            break
        if group[t] not in taken:
            taken.add(group[t])
            extra.append(t)
    if len(extra) < EXTRA_LOGISTICS:
        raise ValueError(f"only {len(extra)} extra Logistics tasks with distinct groups")
    return {**pilot, "logistics_hard": extra}


if __name__ == "__main__":
    print(json.dumps(calibration(), indent=1))
