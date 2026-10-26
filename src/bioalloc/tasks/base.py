from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Protocol

from pydantic import JsonValue

from bioalloc.actions import ActionSpace
from bioalloc.checks import RuntimeCheck
from bioalloc.graph import TaskGraph
from bioalloc.schemas import Subtask


class TaskFamily(Protocol):
    name: str

    def checks(self, subtask: Subtask) -> Sequence[RuntimeCheck]:
        """Runtime checks that apply to a subtask of this family."""
        ...

    def aggregate(self, graph: TaskGraph, outputs: Mapping[str, JsonValue]) -> JsonValue:
        """Deterministically combine accepted subtask outputs into the final answer."""
        ...


class HasActions(Protocol):
    """Optional part of a family: a typed action space for step-wise executors (D10)."""

    def action_space(self, subtask: Subtask) -> ActionSpace | None: ...
