"""Per-system and per-stratum success for a generated-Logistics experiment (D29).

Usage: uv run python scripts/tuning_report.py runs/repair-tune-v1 [single_agent]
Prints success, mean tokens and cost per system, then success per stratum (cities x packages),
then the D29 advancement rule against the named baseline.
"""

import json
import re
import sys
from collections import defaultdict
from pathlib import Path


def main() -> None:
    run_dir = Path(sys.argv[1])
    baseline = sys.argv[2] if len(sys.argv) > 2 else "single_agent"
    rows = [json.loads(line) for line in (run_dir / "results.jsonl").open() if line.strip()]
    by_sys: dict[str, list[dict]] = defaultdict(list)  # type: ignore[type-arg]
    strata: dict[tuple[str, str], list[bool]] = defaultdict(list)
    for r in rows:
        by_sys[r["system"]].append(r)
        m = re.search(r"-c(\d)-p(\d)-", r["task_id"])
        stratum = f"c{m.group(1)}-p{m.group(2)}" if m else "other"
        strata[(r["system"], stratum)].append(bool(r["correct"]))
    print(f"{'system':24} {'solved':>9} {'rate':>6} {'tokens':>8} {'usd/run':>8} {'usd/succ':>9}")
    stats = {}
    for name, rs in sorted(by_sys.items()):
        n, k = len(rs), sum(r["correct"] for r in rs)
        tok = sum(r["tokens"] for r in rs) / n
        usd = sum(r["usd"] for r in rs)
        stats[name] = (k, n, tok)
        per = f"{usd / k:.3f}" if k else "-"
        print(f"{name:24} {k:>4}/{n:<4} {k / n:>6.0%} {tok:>8.0f} {usd / n:>8.3f} {per:>9}")
    print(f"total spent: US${sum(r['usd'] for r in rows):.2f} over {len(rows)} runs")
    names = sorted(by_sys)
    keys = sorted({s for _, s in strata})
    print("\nper stratum (solved/runs):")
    print(f"{'system':24} " + " ".join(f"{s:>8}" for s in keys))
    for name in names:
        cells = []
        for s in keys:
            v = strata.get((name, s), [])
            cells.append(f"{sum(v)}/{len(v)}".rjust(8))
        print(f"{name:24} " + " ".join(cells))
    if baseline in stats:
        bk, bn, btok = stats[baseline]
        cands = [n for n in names if n != baseline]
        best = sorted(cands, key=lambda n: (-stats[n][0] / stats[n][1], stats[n][2]))[0]
        k, n, tok = stats[best]
        beats = k / n > bk / bn
        cheaper = bk > 0 and k / n >= bk / bn and tok <= 0.8 * btok
        verdict = "advances" if (beats or cheaper) else "does not advance"
        print(
            f"\nD29 rule: best candidate {best} ({k}/{n}, {tok:.0f} tokens) vs {baseline} "
            f"({bk}/{bn}, {btok:.0f} tokens): {verdict}"
        )


if __name__ == "__main__":
    main()
