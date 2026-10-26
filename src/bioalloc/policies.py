"""Allocation policies: which executor gets a subtask.

M1 has the non-learning policies used as baselines (constant, random). The rule table and the
LinUCB selector of the research plan (Section 4.3) are added in M3 behind the same interface. A
policy receives the full record of every finished attempt through ``update`` and defines its own
reward from it, so that acceptance by runtime checks, check strength, confidence and cost stay
separate signals rather than one number fixed by the coordinator.
"""

from __future__ import annotations

import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

from pydantic import JsonValue

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


def provisional_reward(
    attempt: Attempt, cost_weight: float = 0.1, token_scale: float = 20_000
) -> float:
    """Runtime reward from public evidence only (D9; final definition fixed in M3): 1 for a result
    accepted with goal-level evidence, 0.5 for one accepted on weaker evidence, 0 for a rejection,
    minus a cost term."""
    from bioalloc.checks import goal_verified

    strong = goal_verified(attempt.checks)
    success = (1.0 if strong else 0.5) if attempt.accepted else 0.0
    return success - cost_weight * attempt.tokens / token_scale


@dataclass
class LinUCB:
    """Disjoint LinUCB over executors (Li et al., 2010), as used for subtask routing by
    Symphony-Coord. Each arm keeps A = ridge * I + sum x x^T and b = sum r x; it picks the arm
    maximizing theta^T x + alpha * sqrt(x^T A^-1 x), ties to the earlier arm. ``frozen`` stops all
    learning (the frozen protocol, Section 4.3)."""

    arms: Sequence[str]
    alpha: float = 1.0
    ridge: float = 1.0
    frozen: bool = False
    reward: Callable[[Attempt], float] = provisional_reward
    name: str = "linucb"
    _A: dict[str, Any] = field(default_factory=dict, init=False)
    _b: dict[str, Any] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        import numpy as np

        from bioalloc.features import NAMES

        d = len(NAMES)
        for arm in self.arms:
            self._A.setdefault(arm, self.ridge * np.eye(d))
            self._b.setdefault(arm, np.zeros(d))

    def _x(self, subtask: Subtask) -> Any:
        import numpy as np

        from bioalloc.features import vector

        return np.array(vector(subtask.features))

    def scores(self, subtask: Subtask) -> dict[str, float]:
        import numpy as np

        x = self._x(subtask)
        out = {}
        for arm in self.arms:
            a_inv = np.linalg.inv(self._A[arm])
            theta = a_inv @ self._b[arm]
            out[arm] = float(theta @ x + self.alpha * np.sqrt(x @ a_inv @ x))
        return out

    def select(self, task_id: str, subtask: Subtask, previous: Sequence[Attempt]) -> str:
        scores = self.scores(subtask)
        return max(self.arms, key=lambda a: (scores[a], -list(self.arms).index(a)))

    def learn(self, subtask: Subtask, arm: str, reward: float) -> None:
        """One update with an explicit reward, e.g. a training-only label from the harness."""
        if self.frozen:
            return
        x = self._x(subtask)
        self._A[arm] = self._A[arm] + _outer(x)
        self._b[arm] = self._b[arm] + reward * x

    def update(self, subtask: Subtask, attempt: Attempt) -> None:
        self.learn(subtask, attempt.executor, self.reward(attempt))

    def state(self) -> dict[str, JsonValue]:
        return {
            "arms": list(self.arms),
            "alpha": self.alpha,
            "ridge": self.ridge,
            "A": {a: self._A[a].tolist() for a in self.arms},
            "b": {a: self._b[a].tolist() for a in self.arms},
        }

    @classmethod
    def from_state(cls, state: dict[str, Any], *, frozen: bool = True) -> LinUCB:
        import numpy as np

        policy = cls(arms=state["arms"], alpha=state["alpha"], ridge=state["ridge"], frozen=frozen)
        for a in policy.arms:
            policy._A[a] = np.array(state["A"][a])
            policy._b[a] = np.array(state["b"][a])
        return policy


def _outer(x: Any) -> Any:
    import numpy as np

    return np.outer(x, x)


class Rule(Wire):
    """Fires when every condition holds: feature name -> (operator, threshold)."""

    when: dict[str, tuple[str, float]]
    executor: str


@dataclass
class RuleTable:
    """Rule-based allocation (Section 4.3, first stage): the first matching rule wins. The table is
    serializable; its contents are derived from development profiles in M3."""

    rules: Sequence[Rule]
    default: str
    name: str = "rules"

    def select(self, task_id: str, subtask: Subtask, previous: Sequence[Attempt]) -> str:
        ops = {
            ">": float.__gt__,
            ">=": float.__ge__,
            "<": float.__lt__,
            "<=": float.__le__,
            "==": float.__eq__,
        }
        for rule in self.rules:
            if all(
                ops[op](float(subtask.features.get(f, 0.0)), float(v))
                for f, (op, v) in rule.when.items()
            ):
                return rule.executor
        return self.default

    def update(self, subtask: Subtask, attempt: Attempt) -> None:
        pass
