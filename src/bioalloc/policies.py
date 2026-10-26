"""Allocation policies: which executor gets a subtask.

M1 has the non-learning policies used as baselines (constant, random, fixed by family). The
rule table and the LinUCB selector of the research plan (Section 4.3) are added in M3 behind the
same interface.
"""

from __future__ import annotations

import random
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Protocol

from bioalloc.schemas import Subtask, Wire


class Attempt(Wire):
    """What a policy may know about an earlier attempt at the same subtask."""

    executor: str
    status: str
    verdict: str


class Policy(Protocol):
    name: str

    def select(self, subtask: Subtask, previous: Sequence[Attempt]) -> str: ...

    def update(self, subtask: Subtask, executor: str, reward: float) -> None: ...


@dataclass
class Constant:
    """Always the same executor; used for 'each executor alone' and the single-agent baseline."""

    executor: str
    name: str = "constant"

    def select(self, subtask: Subtask, previous: Sequence[Attempt]) -> str:
        return self.executor

    def update(self, subtask: Subtask, executor: str, reward: float) -> None:
        pass


@dataclass
class Uniform:
    """Random allocation baseline; a retry may draw the same executor again."""

    executors: Sequence[str]
    rng: random.Random = field(default_factory=lambda: random.Random(0))
    name: str = "random"

    def select(self, subtask: Subtask, previous: Sequence[Attempt]) -> str:
        return self.rng.choice(list(self.executors))

    def update(self, subtask: Subtask, executor: str, reward: float) -> None:
        pass
