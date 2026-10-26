"""Names to objects: study tasks (with their family and grader) and systems (executors + policy).

Systems are those of the main comparison (decision D12) plus the Table 3 controls:

* ``single_agent``; ``direct``, ``ant``, ``bee``, ``octopus`` alone; ``fixed``; ``random``;
  ``specialist``; ``conventional_adaptive`` (LinUCB over direct, act-observe, majority vote and
  plan-then-execute); ``bio_rules`` and ``bio_adaptive`` (rule table and LinUCB over the core
  pool, both trained in M3);
* controls: ``ant_no_deposit``, ``ant_no_evaporation``, ``ant_uniform``, ``bee_no_stop``,
  ``majority``, ``octopus_reporting``.

Every system gets the same model settings, coordinator configuration and budget; only the
executor pool and the policy differ.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from pydantic import JsonValue

from bioalloc.coordinator import CoordinatorConfig
from bioalloc.decomposition import PublicTask
from bioalloc.evaluation import gsm8k_data, planbench_data, toolsuite_data
from bioalloc.evaluation.toy_data import Grade
from bioalloc.executors.ant import ant_controls
from bioalloc.executors.base import Executor
from bioalloc.executors.bee import Bee, MajorityVote
from bioalloc.executors.conventional import PlanThenExecute, SingleAgent, act_observe
from bioalloc.executors.direct import Direct
from bioalloc.executors.octopus import Octopus
from bioalloc.executors.specialist import Specialist
from bioalloc.graph import TaskGraph
from bioalloc.policies import Constant, FixedByContract, LinUCB, Policy, RuleTable, Uniform
from bioalloc.tasks.base import TaskFamily
from bioalloc.tasks.gsm8k import CONTRACT as GSM8K_CONTRACT
from bioalloc.tasks.gsm8k import GSM8K
from bioalloc.tasks.planbench import CONTRACT as PLAN_CONTRACT
from bioalloc.tasks.planbench import PlanBench
from bioalloc.tasks.toolsuite import ANSWER_CONTRACT, LIST_CONTRACT, TEXT_CONTRACT, ToolSuite


@dataclass(frozen=True)
class StudyTask:
    task_id: str
    family: str  # planning | tools | reasoning
    task_family: TaskFamily
    grade: Callable[[JsonValue], Grade]
    # Either a natural graph (deterministic decomposition) or a public task for the decomposer.
    natural_graph: TaskGraph | None = None
    public: PublicTask | None = None


_TOOL_ID = re.compile(r"^tools-v\d+-(?P<template>[a-z_]+)-s(?P<seed>\d+)$")


class TaskLoader:
    """Loads study tasks by id from the prepared snapshots and generators (cached)."""

    def __init__(self) -> None:
        self._gsm8k: dict[str, gsm8k_data.GSM8KItem] | None = None
        self._plans: dict[str, planbench_data.PlanItem] | None = None

    def load(self, task_id: str) -> StudyTask:
        if task_id.startswith("gsm8k-"):
            if self._gsm8k is None:
                self._gsm8k = {i.task.id: i for i in gsm8k_data.load()}
            item = self._gsm8k[task_id]
            return StudyTask(
                task_id, "reasoning", GSM8K(), lambda a: gsm8k_data.grade(item, a), public=item.task
            )
        if task_id.startswith("planbench-"):
            if self._plans is None:
                self._plans = {i.task.id: i for i in planbench_data.load()}
            plan = self._plans[task_id]
            return StudyTask(
                task_id,
                "planning",
                PlanBench(),
                lambda a: planbench_data.grade(plan, a),
                public=plan.task,
            )
        m = _TOOL_ID.match(task_id)
        if m:
            inst = toolsuite_data.make_instance(m["template"], int(m["seed"]))
            return StudyTask(
                task_id,
                "tools",
                ToolSuite(),
                lambda a: toolsuite_data.grade(inst, a),
                natural_graph=inst.graph,
            )
        raise KeyError(f"unknown study task {task_id!r}")


@dataclass
class ModelSettings:
    model: str
    temperature: float = 0.6
    reasoning_effort: str | None = "low"
    max_completion_tokens: int = 2048

    def kw(self) -> dict[str, Any]:
        return {
            "temperature": self.temperature,
            "reasoning_effort": self.reasoning_effort,
            "max_completion_tokens": self.max_completion_tokens,
        }


@dataclass
class System:
    name: str
    executors: dict[str, Executor]
    policy: Policy
    config: CoordinatorConfig = field(default_factory=CoordinatorConfig)


CORE = ("direct", "ant", "bee", "octopus")
ORDINARY = ("direct", "act_observe", "majority", "plan_then_execute")
# Fixed allocation, locked before any test run (D18): planning -> ant, tools -> octopus,
# reasoning -> bee.
FIXED = {
    PLAN_CONTRACT: "ant",
    ANSWER_CONTRACT: "octopus",
    LIST_CONTRACT: "octopus",
    TEXT_CONTRACT: "octopus",
    GSM8K_CONTRACT: "bee",
}
SYSTEMS = (
    "single_agent",
    "direct",
    "ant",
    "bee",
    "octopus",
    "fixed",
    "random",
    "specialist",
    "conventional_adaptive",
    "bio_rules",
    "bio_adaptive",
    "ant_no_deposit",
    "ant_no_evaporation",
    "ant_uniform",
    "bee_no_stop",
    "majority",
    "octopus_reporting",
)


def executor_pool(m: ModelSettings) -> dict[str, Executor]:
    kw = m.kw()
    pool: dict[str, Executor] = {
        "direct": Direct(model=m.model, **kw),
        "bee": Bee(model=m.model, **kw),
        "bee_no_stop": Bee(model=m.model, sigma=0.0, name="bee_no_stop", **kw),
        "octopus": Octopus(model=m.model, **kw),
        "octopus_reporting": Octopus(
            model=m.model, report_every_step=True, name="octopus_reporting", **kw
        ),
        "act_observe": act_observe(m.model, **kw),
        "majority": MajorityVote(model=m.model, **kw),
        "plan_then_execute": PlanThenExecute(model=m.model, **kw),
        "single_agent": SingleAgent(model=m.model, **kw),
        "specialist": Specialist(model=m.model, **kw),
    }
    pool.update(ant_controls(m.model, **kw))
    return pool


def build_system(
    name: str,
    m: ModelSettings,
    base: CoordinatorConfig,
    seed: int,
    selector_state: dict[str, Any] | None = None,
    rules: RuleTable | None = None,
) -> System:
    pool = executor_pool(m)

    def only(*names: str) -> dict[str, Executor]:
        return {n: pool[n] for n in names}

    if name == "single_agent":
        cfg = CoordinatorConfig(**{**base.__dict__, "serial": True})
        return System(name, only("single_agent"), Constant("single_agent"), cfg)
    if name in pool and name not in ("act_observe", "plan_then_execute"):
        return System(name, only(name), Constant(name), base)
    if name == "fixed":
        return System(name, only(*CORE), FixedByContract(FIXED, default="direct"), base)
    if name == "random":
        return System(name, only(*CORE), Uniform(list(CORE), seed=seed), base)
    if name == "conventional_adaptive":
        policy = LinUCB.from_state(selector_state) if selector_state else LinUCB(list(ORDINARY))
        return System(name, only(*ORDINARY), policy, base)
    if name == "bio_adaptive":
        policy = LinUCB.from_state(selector_state) if selector_state else LinUCB(list(CORE))
        return System(name, only(*CORE), policy, base)
    if name == "bio_rules":
        return System(name, only(*CORE), rules or RuleTable([], default="direct"), base)
    raise KeyError(f"unknown system {name!r}")
