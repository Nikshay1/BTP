"""Build the study's grouped splits once and save them to data/splits/v2.json.

Usage: uv run python scripts/make_splits.py   (needs the prepared GSM8K and PlanBench snapshots)
"""

from __future__ import annotations

from bioalloc.evaluation.data import REPO_ROOT, verify_prepared
from bioalloc.evaluation.splits import assign, save
from bioalloc.evaluation.study_tasks import inventory

OUT = REPO_ROOT / "data" / "splits" / "v2.json"


def main() -> int:
    provenance = {name: verify_prepared(name)["prepared"] for name in ("gsm8k", "planbench")}
    if OUT.exists():
        raise SystemExit(f"{OUT} exists; splits are made once and never regenerated in place")
    result = assign(inventory())
    digest = save(result, OUT, provenance)  # type: ignore[arg-type]
    print(f"wrote {OUT} ({len(result.assignments)} tasks, sha256 {digest[:12]})")
    for family, groups in result.groups_per_split.items():
        print(f"  {family}: groups per split {groups}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
