"""Allocation policies: which executor gets a subtask.

M1 has the non-learning policies used as baselines (constant, random). The rule table and the
LinUCB selector of the research plan (Section 4.3) are added in M3 behind the same interface. A
policy receives the full record of every finished attempt through ``update`` and defines its own
reward from it, so that acceptance by runtime checks, check strength, confidence and cost stay
separate signals rather than one number fixed by the coordinator.
"""

from __future__ import annotations

import random
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from bioalloc.schemas import CheckResult, ExecutorStatus, Subtask, Wire


class Attempt(Wire):
    """What a policy may know about a finished attempt. Never contains grading information."""

    subtask_id: str
    attempt: int
    executor: str
    status: ExecutorStatus
    verdict: str
    accepted: bool
    checks: tuple[CheckResult, ...] = ()
    confidence: float | None = None
    tokens: int = 0


class Policy(Protocol):
    name: str

    def select(self, task_id: str, subtask: Subtask, previous: Sequence[Attempt]) -> str: ...

    def update(self, subtask: Subtask, attempt: Attempt) -> None: ...


@dataclass
class Constant:
    """Always the same executor; used for 'each executor alone' and the single-agent baseline."""

    executor: str
    name: str = "constant"

    def select(self, task_id: str, subtask: Subtask, previous: Sequence[Attempt]) -> str:
        return self.executor

    def update(self, subtask: Subtask, attempt: Attempt) -> None:
        pass


@dataclass
class Uniform:
    """Random allocation baseline. Each draw depends only on the seed, the task, the subtask and the
    attempt number, so it does not change with the order in which concurrent subtasks finish, and
    subtasks with the same id in different tasks get independent draws."""

    executors: Sequence[str]
    seed: int = 0
    name: str = "random"

    def select(self, task_id: str, subtask: Subtask, previous: Sequence[Attempt]) -> str:
        rng = random.Random(f"{self.seed}:{task_id}:{subtask.id}:{len(previous)}")
        return rng.choice(list(self.executors))

    def update(self, subtask: Subtask, attempt: Attempt) -> None:
        pass


@dataclass
class FixedByContract:
    """Fixed allocation baseline: one executor per task family, chosen by hand and locked before
    any test run (D18). Families are recognized by the subtask's output contract; subtasks without
    a known contract (e.g. intermediate nodes of a model decomposition) get ``default``."""

    mapping: dict[str, str]
    default: str
    name: str = "fixed"

    def select(self, task_id: str, subtask: Subtask, previous: Sequence[Attempt]) -> str:
        return self.mapping.get(subtask.contract or "", self.default)

    def update(self, subtask: Subtask, attempt: Attempt) -> None:
        pass
