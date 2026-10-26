"""Typed action spaces: the fixed choice vocabulary a family offers to step-wise executors (D10).

An action space turns a subtask into a small environment: a start state built from public inputs, a
fixed set of action kinds, a legality check, a transition that returns an observation, and the
candidate output produced by ``FINISH``. Ant keeps its weights over the action kinds, so what it
learns on one path applies to the next.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, Protocol

from pydantic import JsonValue

from bioalloc.llm.base import ToolCall
from bioalloc.schemas import Subtask, Wire

if TYPE_CHECKING:
    from bioalloc.tools import ToolResult

FINISH = "FINISH"


class Action(Wire):
    kind: str
    args: dict[str, JsonValue] = {}


class ActionSpace(Protocol):
    """Some action kinds may be tool-backed: for those ``tool_call`` returns the call to make,
    which the executor sends through the metered tool service (so limits, counting and tracing are
    the same as for every other executor), and ``observe`` turns the result into the next state.
    ``legal`` only checks an action's structure and must not run anything."""

    kinds: tuple[str, ...]  # includes FINISH

    def start(self, subtask: Subtask, upstream: Mapping[str, JsonValue]) -> Any: ...

    def legal(self, state: Any, action: Action) -> str | None:
        """None if the action is legal in this state, else the reason it is not."""
        ...

    def apply(self, state: Any, action: Action) -> tuple[Any, str]:
        """New state and an observation, for a legal local (not tool-backed) action."""
        ...

    def tool_call(self, state: Any, action: Action, call_id: str) -> ToolCall | None:
        """The tool call a tool-backed action makes, or None for a local action."""
        ...

    def observe(self, state: Any, action: Action, result: ToolResult) -> tuple[Any, str]:
        """New state and an observation after a tool-backed action's call returned."""
        ...

    def describe_actions(self) -> str:
        """Each action kind and the arguments it takes, FINISH included, shown to the model."""
        ...

    def finish(self, state: Any, action: Action) -> JsonValue:
        """The candidate output for a legal FINISH action."""
        ...

    def describe(self, state: Any) -> str:
        """Public description of the state shown to the model."""
        ...
