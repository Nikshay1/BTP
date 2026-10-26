"""Run one task through a system, write its trace and manifest, and grade the answer."""

from __future__ import annotations

import asyncio
import json
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from pydantic import JsonValue

from bioalloc import prompts
from bioalloc.budget import Account, MeteredClient, Pricing
from bioalloc.coordinator import Coordinator, CoordinatorConfig, RunResult
from bioalloc.evaluation.toy_data import Grade, ToyInstance, grade, make_instance, optimal_legs
from bioalloc.executors.direct import Direct
from bioalloc.graph import TaskGraph
from bioalloc.llm.base import RunMode, Transport
from bioalloc.llm.fake import FakeTransport
from bioalloc.policies import Constant
from bioalloc.schemas import Budget
from bioalloc.tasks.toy import ToyRoutes
from bioalloc.trace import Tracer, finish_manifest, sha256_text, start_manifest

SCENARIOS = ("good", "recover", "budget")


@dataclass(frozen=True)
class Outcome:
    result: RunResult
    grade: Grade
    run_dir: Path


def scripted_toy_transport(instance: ToyInstance, scenario: str) -> FakeTransport:
    """Fake model for the M1 demo.

    good:    both legs answered correctly at the first attempt;
    recover: leg 1 first uses an edge that does not exist, then answers correctly;
    budget:  as good, but the task budget allows only one model call.
    """
    leg1, leg2 = optimal_legs(instance)

    def is_leg(marker: str):  # type: ignore[no-untyped-def]
        return lambda req: marker in req.messages[1].content

    fake = FakeTransport()
    first = [json.dumps({"route": leg1})]
    if scenario == "recover":
        first.insert(0, json.dumps({"route": [leg1[0], "Z", leg1[-1]]}))
    fake.on(is_leg("Find the cheapest route from start"), *first)
    fake.on(is_leg("Continue from the node"), json.dumps({"route": leg2}))
    return fake


def run_toy(
    seed: int,
    scenario: str,
    out_dir: Path,
    *,
    transport: Transport | None = None,
    model: str = "fake-model",
    mode: RunMode = "debug",
    budget: Budget | None = None,
    pricing: Pricing | None = None,
) -> Outcome:
    if scenario not in SCENARIOS:
        raise ValueError(f"unknown scenario {scenario!r}; choose from {SCENARIOS}")
    instance = make_instance(seed)
    if budget is None:
        calls = 1 if scenario == "budget" else 20
        budget = Budget(max_tokens=50_000, max_model_calls=calls)
    transport = transport or scripted_toy_transport(instance, scenario)
    config = CoordinatorConfig(max_attempts=2)
    # Every system gets a fresh copy of the stored graph, so no run can alter another's inputs.
    stored = instance.graph.model_dump_json()
    graph = TaskGraph.model_validate_json(stored)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run_id = f"{graph.id}-{scenario}-{stamp}-{uuid.uuid4().hex[:6]}"
    run_dir = out_dir / run_id
    run_config: dict[str, JsonValue] = {
        "task": graph.id,
        "graph_sha256": sha256_text(stored),
        "family": "toy_routes",
        "scenario": scenario,
        "system": "constant:direct",
        "budget": budget.model_dump(mode="json"),
        "max_attempts": config.max_attempts,
        "accept_unknown": config.accept_unknown,
        "retry_feedback": config.retry_feedback,
    }
    manifest = start_manifest(
        run_dir,
        run_id,
        mode=mode,
        config=run_config,
        model=model,
        seeds={"task": seed, "coordinator": 0},
        prompts=prompts.all_prompts(),
    )
    tracer = Tracer(run_dir, run_id)
    llm = MeteredClient(
        transport,
        Account(budget),
        actor="coordinator",
        agent="coordinator",
        pricing=pricing or Pricing(),
        mode=mode,
    )
    coordinator = Coordinator({"direct": Direct(model=model)}, Constant("direct"), config)
    try:
        result = asyncio.run(coordinator.run(graph, ToyRoutes(), llm, tracer))
    except BaseException:
        tracer.close()
        finish_manifest(run_dir, manifest, "crashed", {})
        raise
    tracer.close()
    g = grade(instance, result.answer)
    finish_manifest(
        run_dir,
        manifest,
        result.status,
        {
            "correct": g.correct,
            "grade_detail": g.detail,
            "answer": result.answer,
            "within_budget": result.within_budget,
            "cost": result.cost.model_dump(mode="json"),
            "attempts": len(result.attempts),
        },
    )
    return Outcome(result=result, grade=g, run_dir=run_dir)
