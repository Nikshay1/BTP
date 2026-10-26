"""Typed action spaces: the fixed choice vocabulary a family offers to step-wise executors (D10).

An action space turns a subtask into a small environment: a start state built from public inputs, a
fixed set of action kinds, a legality check, a transition that returns an observation, and the
candidate output produced by ``FINISH``. Ant keeps its weights over the action kinds, so what it
learns on one path applies to the next.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Protocol

from pydantic import JsonValue

from bioalloc.schemas import Subtask, Wire

FINISH = "FINISH"


class Action(Wire):
    kind: str
    args: dict[str, JsonValue] = {}


class ActionSpace(Protocol):
    kinds: tuple[str, ...]  # includes FINISH

    def start(self, subtask: Subtask, upstream: Mapping[str, JsonValue]) -> Any: ...

    def legal(self, state: Any, action: Action) -> str | None:
        """None if the action is legal in this state, else the reason it is not."""
        ...

    def apply(self, state: Any, action: Action) -> tuple[Any, str]:
        """New state and an observation, for a legal non-FINISH action."""
        ...

    def finish(self, state: Any, action: Action) -> JsonValue:
        """The candidate output for a legal FINISH action."""
        ...

    def describe(self, state: Any) -> str:
        """Public description of the state shown to the model."""
        ...
