"""Generated Logistics suite (D29)."""

import json
from collections import Counter
from pathlib import Path

from bioalloc.evaluation import logistics_gen as gen
from bioalloc.evaluation import planbench_data, splits
from bioalloc.evaluation.experiment import TaskSelection, select_tasks
from bioalloc.tasks.strips import parse_domain, parse_problem, simulate

ROOT = Path(__file__).resolve().parents[1]


def test_the_frozen_suite_is_exactly_what_the_generator_builds() -> None:
    saved = splits.load(ROOT / gen.SUITE)
    assert splits.canonical(gen.build()) == splits.canonical(saved)
    payload = json.loads((ROOT / gen.SUITE).read_text())
    assert payload["generator_version"] == gen.GENERATOR_VERSION


def test_strata_splits_and_groups() -> None:
    saved = splits.load(ROOT / gen.SUITE)
    per = Counter((gen.parse_id(a.task_id)[:2], a.split) for a in saved)
    for stratum in gen.STRATA:
        assert {s: per[(stratum, s)] for s in gen.PER_STRATUM} == gen.PER_STRATUM
    digests = [a.group.split(":")[-1] for a in saved]
    assert len(set(digests)) == len(saved) == 180  # no group shared, so none crosses splits
    v2 = {
        planbench_data.structure_group("x", parse_problem(str(i.task.inputs["problem"]))).split(
            ":"
        )[-1]
        for i in planbench_data.load()
    }
    assert not set(digests) & v2  # nothing shared with PlanBench (split v2)


def test_every_goal_needs_work_and_every_witness_is_valid() -> None:
    domain = parse_domain(gen.domain_text())
    for a in splits.load(ROOT / gen.SUITE):
        inst = gen.instance(a.task_id)
        problem = parse_problem(inst.problem)
        assert not set(problem.goal) & set(problem.init), a.task_id
        cities, packages = gen.parse_id(a.task_id)[:2]
        assert len(problem.goal) == packages
        sim = simulate(domain, problem, list(inst.witness))
        assert sim.valid and sim.goal_reached, a.task_id


def test_experiments_select_from_the_suite() -> None:
    chosen = select_tasks(
        TaskSelection(split_file=str(gen.SUITE), split="tune", per_family={"planning": 30}), ROOT
    )
    assert len(chosen) == 30 and all(s == "tune" for _, s in chosen)
