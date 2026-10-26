"""PlanBench snapshot and grading (Valmeekam et al., NeurIPS 2023 Datasets and Benchmarks; MIT).

Source: github.com/karthikv792/LLMs-Planning, directory ``plan-bench/instances``, pinned commit
below; the main instance sets named by PlanBench's own configs (``configs/blocksworld.yaml`` and
``configs/logistics.yaml``: ``instance_dir`` = ``generated_basic``, instances ``start``..``end``).
The ``generated`` directories are PlanBench's plan-generalization sets (for Blocksworld, every
instance stacks all blocks from the table into one tower) and are not used. GitHub publishes no file
hashes, so the first preparation records them and later preparations must reproduce them.

Grading simulates the final plan on the domain and problem with the STRIPS interpreter (a valid
plan that reaches the goal is correct; optimality is not required). It is to be cross-checked
against the pinned VAL validator once VAL is built (decision D16).

Template groups: instances whose init-and-goal structure is identical up to renaming objects share
a group (Weisfeiler-Lehman hash of the atom graph). WL hashes can only merge non-isomorphic
instances, never split isomorphic ones, so the grouping errs on the safe side for splits.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import networkx as nx
from pydantic import JsonValue

from bioalloc.decomposition import PublicTask
from bioalloc.evaluation.data import (
    MANIFESTS,
    PREPARED,
    RAW,
    fetch_unpinned,
    read_jsonl,
    write_jsonl,
    write_manifest,
)
from bioalloc.evaluation.toy_data import Grade
from bioalloc.tasks.planbench import CONTRACT, INSTRUCTION, PLAN_SCHEMA, plan_of
from bioalloc.tasks.strips import Problem, parse_domain, parse_problem, simulate

REPO = "karthikv792/LLMs-Planning"
COMMIT = "fc638a1aff7df3fe7a1a1d289fa2c04cc24dc284"
BASE = f"https://raw.githubusercontent.com/{REPO}/{COMMIT}/plan-bench/instances"
# (start, end) from PlanBench's configs; instances live in <domain>/generated_basic/.
DOMAINS = {"blocksworld": (1, 500), "logistics": (1, 285)}
PUBLIC = Path("planbench/public.jsonl")


def structure_group(domain: str, problem: Problem) -> str:
    g = nx.Graph()
    for kind, atoms in (("init", problem.init), ("goal", problem.goal)):
        for i, atom in enumerate(sorted(atoms)):
            node = f"{kind}:{i}"
            g.add_node(node, label=f"{kind}:{atom[0]}")
            for pos, obj in enumerate(atom[1:]):
                g.add_node(f"obj:{obj}", label="obj")
                g.add_edge(node, f"obj:{obj}", label=str(pos))
    digest = nx.weisfeiler_lehman_graph_hash(g, node_attr="label", edge_attr="label", iterations=4)
    return f"planbench:{domain}:{digest[:16]}"


def prepare() -> Path:
    urls: dict[str, Path] = {}
    for domain, (start, end) in DOMAINS.items():
        urls[f"{BASE}/{domain}/generated_domain.pddl"] = RAW / "planbench" / domain / "domain.pddl"
        for i in range(start, end + 1):
            urls[f"{BASE}/{domain}/generated_basic/instance-{i}.pddl"] = (
                RAW / "planbench" / domain / f"instance-{i}.pddl"
            )
    hashes = fetch_unpinned(urls)
    old = MANIFESTS / "planbench.json"
    if old.exists():
        recorded = json.loads(old.read_text())["raw"]
        changed = [u for u, h in hashes.items() if recorded.get(u) not in (None, h)]
        if changed:
            raise ValueError(
                f"upstream files changed since the recorded preparation: {changed[:3]}"
            )
    rows: list[dict[str, JsonValue]] = []
    for domain, (start, end) in DOMAINS.items():
        domain_text = (RAW / "planbench" / domain / "domain.pddl").read_text()
        parse_domain(domain_text)
        for i in range(start, end + 1):
            problem_text = (RAW / "planbench" / domain / f"instance-{i}.pddl").read_text()
            problem = parse_problem(problem_text)
            rows.append(
                {
                    "id": f"planbench-{domain}-{i:03d}",
                    "domain_name": domain,
                    "instance": i,
                    "n_objects": len(problem.objects),
                    "group": structure_group(domain, problem),
                    "domain": domain_text,
                    "problem": problem_text,
                }
            )
    digest = write_jsonl(PREPARED / PUBLIC, rows)
    counts: dict[str, JsonValue] = {
        d: sum(1 for r in rows if r["domain_name"] == d) for d in DOMAINS
    }
    counts["groups"] = len({r["group"] for r in rows})
    return write_manifest(
        "planbench",
        {
            "source": f"https://github.com/{REPO}",
            "commit": COMMIT,
            "licence": "MIT",
            "raw": dict[str, JsonValue](hashes),
            "prepared": {str(PUBLIC): digest},
            "counts": counts,
        },
    )


@dataclass(frozen=True)
class PlanItem:
    task: PublicTask
    group: str
    domain_name: str
    n_objects: int


def public_task(task_id: str, domain_text: str, problem_text: str) -> PublicTask:
    return PublicTask(
        id=task_id,
        instruction=INSTRUCTION,
        inputs={"domain": domain_text, "problem": problem_text},
        output_schema=PLAN_SCHEMA,
        contract=CONTRACT,
    )


def load(root: Path = PREPARED) -> list[PlanItem]:
    return [
        PlanItem(
            task=public_task(str(r["id"]), str(r["domain"]), str(r["problem"])),
            group=str(r["group"]),
            domain_name=str(r["domain_name"]),
            n_objects=int(r["n_objects"]),  # type: ignore[arg-type]
        )
        for r in read_jsonl(root / PUBLIC)
    ]


def grade(item: PlanItem, answer: JsonValue) -> Grade:
    plan = plan_of(answer)
    if plan is None:
        return Grade(correct=False, detail="no plan")
    domain = parse_domain(str(item.task.inputs["domain"]))
    problem = parse_problem(str(item.task.inputs["problem"]))
    sim = simulate(domain, problem, plan)
    if sim.valid and sim.goal_reached:
        return Grade(correct=True)
    return Grade(correct=False, detail=str(sim.error))
