"""Generated Logistics problems for harder planning tasks (decision D29; Sol's design in R14).

PlanBench's Logistics problems top out at seven goals and the single agent solves them all given
enough tokens (R12), so more and harder instances are generated in the classic IPC format: ``c``
cities with three locations each (location 0 is the airport), one truck per city, two airplanes
at airports, ``p`` packages. Every package starts somewhere other than its goal, so every goal
has to be achieved. Each instance is fixed by (cities, packages, index) and a pinned generator
version; nothing depends on model results.

Six strata (cities 3 or 4, packages 5, 7 or 9) of 30 instances each are split per stratum into
5 training, 5 tuning and 20 test instances (180 in all). Every instance has its own structure
group (the PlanBench Weisfeiler-Lehman hash), distinct from every other generated instance and
from every PlanBench group, so nothing is shared across splits or with split v2. Each instance is
checked for a valid plan by a simple constructive planner and the STRIPS simulator, and the
witness plan's length is recorded.
"""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass
from functools import cache
from pathlib import Path

from bioalloc.evaluation import planbench_data, splits
from bioalloc.tasks.strips import parse_domain, parse_problem, simulate

GENERATOR_VERSION = 1
LOCATIONS_PER_CITY = 3
AIRPLANES = 2
STRATA = tuple((c, p) for c in (3, 4) for p in (5, 7, 9))
PER_STRATUM = {"train": 5, "tune": 5, "test": 20}
SUITE = Path("data/splits/logistics-gen-v1.json")
PREFIX = f"logisticsgen-v{GENERATOR_VERSION}"


@dataclass(frozen=True)
class GenInstance:
    task_id: str
    cities: int
    packages: int
    problem: str
    witness: tuple[str, ...]


def task_id(cities: int, packages: int, index: int, attempt: int) -> str:
    return f"{PREFIX}-c{cities}-p{packages}-{index:02d}-{attempt}"


def parse_id(tid: str) -> tuple[int, int, int, int]:
    rest = tid.removeprefix(PREFIX + "-")
    c, p, index, attempt = rest.split("-")
    return int(c[1:]), int(p[1:]), int(index), int(attempt)


def _rng(cities: int, packages: int, index: int, attempt: int) -> random.Random:
    key = f"{GENERATOR_VERSION}|{cities}|{packages}|{index}|{attempt}"
    return random.Random(int(hashlib.sha256(key.encode()).hexdigest(), 16))


def _loc(city: int, k: int) -> str:
    return f"l{city}-{k}"


def problem_text(cities: int, packages: int, index: int, attempt: int) -> tuple[str, dict]:  # type: ignore[type-arg]
    """The PDDL problem and its initial placements (for the witness planner)."""
    rng = _rng(cities, packages, index, attempt)
    locs = [_loc(c, k) for c in range(cities) for k in range(LOCATIONS_PER_CITY)]
    trucks = {f"t{c}": _loc(c, rng.randrange(LOCATIONS_PER_CITY)) for c in range(cities)}
    planes = {f"a{i}": _loc(rng.randrange(cities), 0) for i in range(AIRPLANES)}
    start: dict[str, str] = {}
    goal: dict[str, str] = {}
    for i in range(packages):
        a = rng.choice(locs)
        b = rng.choice([x for x in locs if x != a])
        start[f"p{i}"], goal[f"p{i}"] = a, b
    objects = (
        " ".join(planes)
        + "\n          "
        + " ".join(f"c{c}" for c in range(cities))
        + "\n          "
        + " ".join(trucks)
        + "\n          "
        + " ".join(locs)
        + "\n          "
        + " ".join(start)
    )
    init = [f"(AIRPLANE {a})" for a in planes] + [f"(CITY c{c})" for c in range(cities)]
    init += [f"(TRUCK {t})" for t in trucks]
    for c in range(cities):
        for k in range(LOCATIONS_PER_CITY):
            init += [f"(LOCATION {_loc(c, k)})", f"(in-city  {_loc(c, k)} c{c})"]
    init += [f"(AIRPORT {_loc(c, 0)})" for c in range(cities)]
    init += [f"(OBJ {p})" for p in start]
    init += [f"(at {t} {loc})" for t, loc in trucks.items()]
    init += [f"(at {p} {loc})" for p, loc in start.items()]
    init += [f"(at {a} {loc})" for a, loc in planes.items()]
    name = f"logistics-gen-c{cities}-s{LOCATIONS_PER_CITY}-p{packages}-a{AIRPLANES}"
    text = (
        f"(define (problem {name})\n(:domain logistics-strips)\n(:objects {objects} \n)\n"
        "(:init\n    "
        + "\n    ".join(init)
        + "\n)\n(:goal\n    (and\n        "
        + "\n        ".join(f"(at {p} {loc})" for p, loc in goal.items())
        + "\n    )\n)\n)\n"
    )
    return text, {"trucks": trucks, "planes": planes, "start": start, "goal": goal}


def witness_plan(layout: dict) -> list[str]:  # type: ignore[type-arg]
    """One package at a time: truck to the airport, fly, truck to the goal."""
    trucks: dict[str, str] = dict(layout["trucks"])
    planes: dict[str, str] = dict(layout["planes"])
    plan: list[str] = []

    def city(loc: str) -> str:
        return "c" + loc[1:].split("-")[0]

    def airport(c: str) -> str:
        return f"l{c[1:]}-0"

    def truck_move(p: str, frm: str, to: str) -> None:
        t = f"t{city(frm)[1:]}"
        if trucks[t] != frm:
            plan.append(f"(drive-truck {t} {trucks[t]} {frm} {city(frm)})")
        plan.append(f"(load-truck {p} {t} {frm})")
        plan.append(f"(drive-truck {t} {frm} {to} {city(frm)})")
        plan.append(f"(unload-truck {p} {t} {to})")
        trucks[t] = to

    for p, a in layout["start"].items():
        b = layout["goal"][p]
        here = a
        if city(a) != city(b):
            if here != airport(city(a)):
                truck_move(p, here, airport(city(a)))
                here = airport(city(a))
            plane = "a0"
            if planes[plane] != here:
                plan.append(f"(fly-airplane {plane} {planes[plane]} {here})")
            plan.append(f"(load-airplane {p} {plane} {here})")
            dest = airport(city(b))
            plan.append(f"(fly-airplane {plane} {here} {dest})")
            plan.append(f"(unload-airplane {p} {plane} {dest})")
            planes[plane] = here = dest
        if here != b:
            truck_move(p, here, b)
    return plan


@cache
def domain_text() -> str:
    item = next(i for i in planbench_data.load() if i.domain_name == "logistics")
    return str(item.task.inputs["domain"])


def instance(tid: str) -> GenInstance:
    cities, packages, index, attempt = parse_id(tid)
    text, layout = problem_text(cities, packages, index, attempt)
    return GenInstance(tid, cities, packages, text, tuple(witness_plan(layout)))


def item(tid: str) -> planbench_data.PlanItem:
    inst = instance(tid)
    domain = domain_text()
    group = planbench_data.structure_group("logisticsgen", parse_problem(inst.problem))
    return planbench_data.PlanItem(
        task=planbench_data.public_task(tid, domain, inst.problem),
        group=group,
        domain_name="logistics",
        n_objects=0,
    )


def build() -> list[splits.Assignment]:
    """All 180 instances with their splits. Raises if any witness plan fails."""
    domain = parse_domain(domain_text())
    taken = {
        planbench_data.structure_group(
            "logistics", parse_problem(str(i.task.inputs["problem"]))
        ).split(":")[-1]
        for i in planbench_data.load()
    }
    out: list[splits.Assignment] = []
    for cities, packages in STRATA:
        chosen: list[tuple[str, str]] = []
        index = 0
        while len(chosen) < sum(PER_STRATUM.values()):
            attempt = 0
            while True:
                tid = task_id(cities, packages, index, attempt)
                inst = instance(tid)
                problem = parse_problem(inst.problem)
                group = planbench_data.structure_group("logisticsgen", problem)
                if group.split(":")[-1] not in taken:
                    break
                attempt += 1
            sim = simulate(domain, problem, list(inst.witness))
            if not (sim.valid and sim.goal_reached):
                raise ValueError(f"{tid}: witness plan fails: {sim.error}")
            taken.add(group.split(":")[-1])
            chosen.append((tid, group))
            index += 1
        it = iter(chosen)
        for split, n in PER_STRATUM.items():
            for _ in range(n):
                tid, group = next(it)
                out.append(
                    splits.Assignment(task_id=tid, family="planning", group=group, split=split)
                )
    return out


def save_suite(root: Path) -> str:
    """Write the suite file (same layout and hash as the split files)."""
    from bioalloc.evaluation.data import verify_prepared

    assignments = build()
    text = splits.canonical(assignments)
    digest = hashlib.sha256(text.encode()).hexdigest()
    payload = {
        "suite": PREFIX,
        # The domain text comes from the PlanBench snapshot; experiments verify it against this
        # hash before loading any generated task (R16).
        "provenance": {"planbench": verify_prepared("planbench")["prepared"]},
        "generator_version": GENERATOR_VERSION,
        "strata": [list(s) for s in STRATA],
        "per_stratum": PER_STRATUM,
        "sha256": digest,
        "assignments": json.loads(text),
    }
    path = root / SUITE
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")
    return digest
