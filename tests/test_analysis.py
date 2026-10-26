import math

import pytest

from bioalloc.evaluation.analysis import cost_projection, paired, summarize


def rec(task: str, system: str, rep: int, correct: bool, tokens: int = 100) -> dict:  # type: ignore[type-arg]
    return {
        "task_id": task,
        "system": system,
        "repetition": rep,
        "correct": correct,
        "status": "completed" if correct else "failed",
        "within_budget": True,
        "usage_complete": True,
        "tokens": tokens,
        "coordinator_tokens": 10,
        "communication_tokens": 5,
        "reasoning_tokens": 0,
        "model_calls": 2,
        "usd": 0.001,
        "elapsed_s": 1.0,
        "family": "f",
    }


def test_summary_rates_and_means() -> None:
    rs = [rec("t1", "a", 0, True, 100), rec("t2", "a", 0, False, 300), rec("t1", "b", 0, True, 50)]
    rows = {r["system"]: r for r in summarize(rs)}
    assert rows["a"]["success"] == 0.5 and rows["a"]["mean_tokens"] == 200
    assert rows["a"]["successes_per_million_tokens"] == pytest.approx(1e6 / 400)
    assert rows["b"]["runs"] == 1


def test_paired_difference_discordance_and_mde() -> None:
    rs = []
    for t in range(40):
        for rep in range(3):
            a_ok = t < 30
            b_ok = t < 20
            rs += [rec(f"t{t}", "a", rep, a_ok), rec(f"t{t}", "b", rep, b_ok)]
    out = paired(rs, "a", "b", resamples=500)
    assert out["tasks"] == 40 and out["pairs"] == 120
    assert out["difference"] == pytest.approx(0.25)
    assert out["discordance"] == pytest.approx(0.25)
    lo, hi = out["ci95"]
    assert lo < 0.25 < hi and lo > 0
    assert out["mde_from_discordance"] == pytest.approx(2.8 * math.sqrt(0.25 / 40))


def test_bootstrap_resamples_groups_not_tasks() -> None:
    # Ten tasks in two groups: resampling groups gives a much wider interval than tasks.
    rs = [rec(f"t{t}", "a", 0, t < 5) for t in range(10)] + [
        rec(f"t{t}", "b", 0, False) for t in range(10)
    ]
    by_task = paired(rs, "a", "b", resamples=1000)
    by_group = paired(
        rs, "a", "b", groups={f"t{t}": "g1" if t < 5 else "g2" for t in range(10)}, resamples=1000
    )
    w = lambda o: o["ci95"][1] - o["ci95"][0]  # noqa: E731
    assert by_group["groups"] == 2 and w(by_group) > w(by_task)


def test_cost_projection() -> None:
    p = cost_projection([rec("t", "a", 0, True, 1000)] * 4, planned_runs=1000)
    assert p["projected_tokens"] == 1_000_000 and p[
        "projected_usd_with_30pct_allowance"
    ] == pytest.approx(1.3)
