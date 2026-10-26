"""Pilot and study analysis (research plan, Section 4.4; decisions D12, D13).

* ``summarize``: per system (and family) success rate, budget compliance, tokens by role, calls,
  dollars and time.
* ``paired``: the paired comparison of two systems. For task i, d_i is the mean over repetitions
  of the success difference; the interval comes from a bootstrap that resamples template groups
  (all tasks and repetitions of a group together). Also reported: the discordance rate q (share
  of task-repetition pairs on which the systems disagree) and the minimum detectable effect at 5%
  two-sided and 80% power, 2.8 * s_d / sqrt(n) (D13). These are sensitivity figures.
* ``cost_projection``: measured mean tokens and dollars per run, scaled to a planned number of runs.
"""

from __future__ import annotations

import math
import random
import statistics
from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

Record = Mapping[str, Any]


def summarize(records: Sequence[Record], by: tuple[str, ...] = ("system",)) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[Record]] = defaultdict(list)
    for r in records:
        groups[tuple(r[k] for k in by)].append(r)
    rows = []
    for key, rs in sorted(groups.items()):
        n = len(rs)
        rows.append(
            {
                **dict(zip(by, key, strict=True)),
                "runs": n,
                "success": sum(bool(r["correct"]) for r in rs) / n,
                "completed": sum(r["status"] == "completed" for r in rs) / n,
                "within_budget": sum(bool(r["within_budget"]) for r in rs) / n,
                "usage_incomplete_runs": sum(not r["usage_complete"] for r in rs),
                "mean_tokens": statistics.fmean(r["tokens"] for r in rs),
                "mean_coordinator_tokens": statistics.fmean(r["coordinator_tokens"] for r in rs),
                "mean_communication_tokens": statistics.fmean(
                    r["communication_tokens"] for r in rs
                ),
                "mean_reasoning_tokens": statistics.fmean(r["reasoning_tokens"] for r in rs),
                "mean_calls": statistics.fmean(r["model_calls"] for r in rs),
                "mean_usd": statistics.fmean(r["usd"] for r in rs),
                "mean_elapsed_s": statistics.fmean(r["elapsed_s"] for r in rs),
                "successes_per_million_tokens": (
                    1e6 * sum(bool(r["correct"]) for r in rs) / max(1, sum(r["tokens"] for r in rs))
                ),
            }
        )
    return rows


def paired(
    records: Sequence[Record],
    a: str,
    b: str,
    groups: Mapping[str, str] | None = None,
    resamples: int = 2000,
    seed: int = 0,
) -> dict[str, Any]:
    """Paired comparison of systems a and b on the tasks and repetitions both ran."""
    outcome: dict[tuple[str, int], dict[str, bool]] = defaultdict(dict)
    for r in records:
        if r["system"] in (a, b):
            outcome[(r["task_id"], r["repetition"])][r["system"]] = bool(r["correct"])
    pairs = {k: v for k, v in outcome.items() if a in v and b in v}
    if not pairs:
        raise ValueError(f"no paired runs of {a} and {b}")
    per_task: dict[str, list[float]] = defaultdict(list)
    discordant = 0
    for (task, _), v in pairs.items():
        per_task[task].append(float(v[a]) - float(v[b]))
        discordant += v[a] != v[b]
    d = {t: statistics.fmean(xs) for t, xs in per_task.items()}
    n = len(d)
    mean = statistics.fmean(d.values())
    s_d = statistics.stdev(d.values()) if n > 1 else 0.0
    group_of = groups or {t: t for t in d}
    by_group: dict[str, list[float]] = defaultdict(list)
    for t, diff in d.items():
        by_group[group_of.get(t, t)].append(diff)
    names = sorted(by_group)
    rng = random.Random(seed)
    boot = []
    for _ in range(resamples):
        sample = [x for g in rng.choices(names, k=len(names)) for x in by_group[g]]
        boot.append(statistics.fmean(sample))
    boot.sort()
    q = discordant / len(pairs)
    return {
        "a": a,
        "b": b,
        "tasks": n,
        "groups": len(names),
        "pairs": len(pairs),
        "difference": mean,
        "ci95": (boot[int(0.025 * resamples)], boot[int(0.975 * resamples) - 1]),
        "discordance": q,
        "s_d": s_d,
        "mde_from_sd": 2.8 * s_d / math.sqrt(n) if n else math.nan,
        "mde_from_discordance": 2.8 * math.sqrt(q / n) if n else math.nan,
    }


def mde_at(n: int, s_d: float) -> float:
    """Minimum detectable effect for n tasks at the pilot's s_d (D13)."""
    return 2.8 * s_d / math.sqrt(n)


def cost_projection(records: Sequence[Record], planned_runs: int) -> dict[str, float]:
    n = len(records)
    if n == 0:
        raise ValueError("no runs")
    tokens = statistics.fmean(r["tokens"] for r in records)
    usd = statistics.fmean(r["usd"] for r in records)
    return {
        "pilot_runs": n,
        "mean_tokens_per_run": tokens,
        "mean_usd_per_run": usd,
        "planned_runs": planned_runs,
        "projected_tokens": tokens * planned_runs,
        "projected_usd": usd * planned_runs,
        "projected_usd_with_30pct_allowance": 1.3 * usd * planned_runs,
    }
