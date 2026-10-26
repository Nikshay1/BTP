"""Subtask features for the selector, computed in trusted code before execution (Section 4.3).

Only information available before a subtask runs is used: its family (from its output contract),
whether it allows tools, whether its family offers a goal-level check for it, its number of
dependencies, how many sibling subtasks are ready at the same moment, the size of its inputs and
instruction, and the attempt number. Nothing comes from model output or from reference answers.
The names and their order are fixed; a selector's parameters are only valid for this order.
"""

from __future__ import annotations

import json
import math

from bioalloc.schemas import Subtask
from bioalloc.tasks.base import TaskFamily

FAMILY_PREFIXES = {
    "planning": ("pddl_plan",),
    "tools": ("toolsuite_",),
    "reasoning": ("gsm8k_", "math_"),
    "synthetic": ("synthetic_",),
}
NAMES: tuple[str, ...] = (
    "bias",
    *(f"family_{f}" for f in FAMILY_PREFIXES),
    "family_other",
    "has_tools",
    "goal_checkable",
    "dependencies",
    "ready_siblings",
    "log_input_chars",
    "log_instruction_chars",
    "attempt",
)


def family_of(subtask: Subtask) -> str:
    contract = subtask.contract or ""
    for family, prefixes in FAMILY_PREFIXES.items():
        if contract.startswith(prefixes):
            return family
    return "other"


def subtask_features(
    subtask: Subtask, family: TaskFamily, ready_siblings: int, attempt: int
) -> dict[str, float]:
    fam = family_of(subtask)
    goal = any(getattr(c, "evidence", None) == "goal" for c in family.checks(subtask))
    values = {name: 0.0 for name in NAMES}
    values["bias"] = 1.0
    values[f"family_{fam}"] = 1.0
    values["has_tools"] = float(bool(subtask.allowed_tools))
    values["goal_checkable"] = float(goal)
    values["dependencies"] = min(len(subtask.depends_on), 8) / 4
    values["ready_siblings"] = min(ready_siblings, 8) / 4
    values["log_input_chars"] = math.log1p(len(json.dumps(subtask.inputs))) / 10
    values["log_instruction_chars"] = math.log1p(len(subtask.instruction)) / 10
    values["attempt"] = float(attempt)
    return values


def vector(features: dict[str, float]) -> list[float]:
    return [features.get(n, 0.0) for n in NAMES]
