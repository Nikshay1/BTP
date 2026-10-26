"""Task graphs: subtasks with dependencies, validated as a DAG."""

from __future__ import annotations

from collections.abc import Collection

import networkx as nx
from pydantic import Field, model_validator

from bioalloc.schemas import Subtask, Wire


class TaskGraph(Wire):
    id: str = Field(min_length=1)
    nodes: tuple[Subtask, ...] = Field(min_length=1)
    # Subtasks whose outputs form the answer, in order.
    final_nodes: tuple[str, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _validate(self) -> TaskGraph:
        ids = [n.id for n in self.nodes]
        if len(set(ids)) != len(ids):
            dupes = sorted({i for i in ids if ids.count(i) > 1})
            raise ValueError(f"duplicate subtask ids: {dupes}")
        known = set(ids)
        for node in self.nodes:
            missing = [d for d in node.depends_on if d not in known]
            if missing:
                raise ValueError(f"subtask {node.id!r} depends on unknown {missing}")
            if node.id in node.depends_on:
                raise ValueError(f"subtask {node.id!r} depends on itself")
        unknown_final = [f for f in self.final_nodes if f not in known]
        if unknown_final:
            raise ValueError(f"unknown final nodes: {unknown_final}")
        if not nx.is_directed_acyclic_graph(self.to_networkx()):
            cycle = nx.find_cycle(self.to_networkx())
            raise ValueError(f"dependency cycle: {[edge[0] for edge in cycle]}")
        return self

    def to_networkx(self) -> nx.DiGraph:
        g = nx.DiGraph()
        for node in self.nodes:
            g.add_node(node.id)
            for dep in node.depends_on:
                g.add_edge(dep, node.id)
        return g

    def node(self, subtask_id: str) -> Subtask:
        for n in self.nodes:
            if n.id == subtask_id:
                return n
        raise KeyError(subtask_id)

    def ready(self, done: Collection[str], started: Collection[str] = ()) -> list[Subtask]:
        """Subtasks not yet started whose dependencies are all done, in declaration order."""
        return [
            n
            for n in self.nodes
            if n.id not in done and n.id not in started and all(d in done for d in n.depends_on)
        ]

    def descendants(self, subtask_id: str) -> set[str]:
        return set(nx.descendants(self.to_networkx(), subtask_id))
