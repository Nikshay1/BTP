"""Fixed mechanisms against choosing by task type, on one backbone (R29, analysis after the tests).

Usage: uv run python scripts/allocation_report.py runs/luna-logistics-compare runs/luna-aime-test-v2

For each policy (always the single agent, always the bee, or the single agent on planning and the
bee on maths) it prints the solve rate and US$ per task in each family and their unweighted mean
over the two families, so the larger maths suite does not dominate. Planning uses `bee`, maths
`bee_quorum5`. The by-type rule was stated before the maths test (D31: votes help only where
answers cannot be checked), but these numbers come from the same runs that showed it, so they are
in-sample; a learned selector on held-out tasks is Phase 2.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

POLICIES = {
    "always single agent": {"planning": "single_agent", "maths": "single_agent"},
    "always bee": {"planning": "bee", "maths": "bee_quorum5"},
    "choose by task type": {"planning": "single_agent", "maths": "bee_quorum5"},
}


def family_stats(exp: Path) -> dict[str, tuple[float, float, int]]:
    """system -> (solve rate, mean US$ per run, runs)."""
    rows = [json.loads(x) for x in (exp / "results.jsonl").read_text().splitlines() if x.strip()]
    out: dict[str, tuple[float, float, int]] = {}
    for system in {r["system"] for r in rows}:
        runs = [r for r in rows if r["system"] == system]
        out[system] = (
            sum(r["correct"] for r in runs) / len(runs),
            sum(r["usd"] for r in runs) / len(runs),
            len(runs),
        )
    return out


def main(planning: Path, maths: Path) -> None:
    stats = {"planning": family_stats(planning), "maths": family_stats(maths)}
    print(f"{'policy':22} {'planning':>18} {'maths':>18} {'mean of the two':>22}")
    for name, pick in POLICIES.items():
        cells = []
        rates, costs = [], []
        for fam in ("planning", "maths"):
            rate, usd, n = stats[fam][pick[fam]]
            rates.append(rate)
            costs.append(usd)
            cells.append(f"{rate:6.1%} ${usd:.4f}")
        mean = f"{sum(rates) / 2:6.1%} ${sum(costs) / 2:.4f}"
        print(f"{name:22} {cells[0]:>18} {cells[1]:>18} {mean:>22}")


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
