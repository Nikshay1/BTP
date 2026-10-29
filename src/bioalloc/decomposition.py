"""Stored decompositions (decision D17).

Every task x repetition gets exactly one task graph, made before any system runs, stored as
canonical JSON with its SHA-256, and replayed to every system as a fresh copy. Tasks that come with
a natural work graph use it directly (``decomposer = "deterministic"``, no cost). Otherwise a model
decomposer (actor ``coordinator``, agent ``decomposer``) proposes a DAG from the public task alone.
An invalid proposal gets one repair call with the validation errors; if that fails too the task
becomes a single node holding the original task. Both calls are charged, no task is dropped, and
the stored cost is imported into every system's budget before it starts.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Literal

import networkx as nx
from pydantic import Field, JsonValue, ValidationError

from bioalloc import prompts
from bioalloc.budget import Account, MeteredClient
from bioalloc.executors.base import parse_json
from bioalloc.graph import TaskGraph
from bioalloc.llm.base import LLMRequest, Message
from bioalloc.schemas import Cost, Subtask, Wire

MAX_NODES = 8
MAX_DEPTH = 6


class PublicTask(Wire):
    """What a decomposer may see: the task as given to a solver (instruction, input values,
    output schema, tools), never its answer."""

    id: str
    instruction: str
    inputs: dict[str, JsonValue] = {}
    output_schema: dict[str, JsonValue] | None = None
    allowed_tools: tuple[str, ...] = ()
    contract: str | None = None

    def as_single_node(self) -> TaskGraph:
        node = Subtask(
            id="task",
            instruction=self.instruction,
            inputs=self.inputs,
            output_schema=self.output_schema,
            allowed_tools=self.allowed_tools,
            contract=self.contract,
        )
        return TaskGraph(id=self.id, nodes=(node,), final_nodes=("task",))


Status = Literal["deterministic", "ok", "repaired", "fallback"]


class Decomposition(Wire):
    task_id: str
    repetition: int = Field(ge=0)
    status: Status
    graph_json: str
    graph_sha256: str
    cost: Cost = Cost()
    prompt_sha256: str | None = None
    errors: tuple[str, ...] = ()

    def graph(self) -> TaskGraph:
        """A fresh copy of the stored graph, after verifying its hash."""
        if sha256(self.graph_json) != self.graph_sha256:
            raise ValueError(f"decomposition of {self.task_id} does not match its hash")
        return TaskGraph.model_validate_json(self.graph_json)


def canonical_json(graph: TaskGraph) -> str:
    return json.dumps(
        graph.model_dump(mode="json"),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _record(
    task_id: str, repetition: int, status: Status, graph: TaskGraph, **kw: object
) -> Decomposition:
    text = canonical_json(graph)
    return Decomposition(
        task_id=task_id,
        repetition=repetition,
        status=status,
        graph_json=text,
        graph_sha256=sha256(text),
        **kw,  # type: ignore[arg-type]
    )


def deterministic(graph: TaskGraph, repetition: int) -> Decomposition:
    return _record(graph.id, repetition, "deterministic", graph)


def graph_from_proposal(task: PublicTask, proposal: JsonValue) -> TaskGraph:
    """Validate a decomposer's proposal and build the task graph. Raises ValueError with every
    problem found. Routing features are never taken from the proposal."""
    if not isinstance(proposal, dict):
        raise ValueError("the reply is not a JSON object")
    nodes = proposal.get("nodes")
    final = proposal.get("final")
    if not isinstance(nodes, list) or not nodes:
        raise ValueError("'nodes' must be a non-empty list")
    if len(nodes) > MAX_NODES:
        raise ValueError(f"at most {MAX_NODES} subtasks are allowed, got {len(nodes)}")
    problems: list[str] = []
    subtasks: list[Subtask] = []
    for i, n in enumerate(nodes):
        if not isinstance(n, dict) or not isinstance(n.get("id"), str):
            problems.append(f"node {i} has no string id")
            continue
        wanted = n.get("inputs", [])
        if not isinstance(wanted, list) or not all(isinstance(x, str) for x in wanted):
            problems.append(f"node {n['id']}: 'inputs' must be a list of input names")
            wanted = []
        unknown = [x for x in wanted if x not in task.inputs]
        if unknown:
            problems.append(f"node {n['id']}: unknown inputs {unknown}")
        deps = n.get("depends_on", [])
        if not isinstance(deps, list) or not all(isinstance(d, str) for d in deps):
            problems.append(f"node {n['id']}: 'depends_on' must be a list of ids")
            deps = []
        instruction = n.get("instruction")
        if not isinstance(instruction, str) or not instruction.strip():
            problems.append(f"node {n['id']}: empty instruction")
            instruction = ""
        node_id = str(n["id"])
        is_final = node_id == final
        if is_final:
            # The final node always keeps the whole public task, so a plan can never lose the
            # question or weaken the goal (D17).
            instruction = f"{instruction}\n\nThe overall task: {task.instruction}"
            wanted = list(task.inputs)
        subtasks.append(
            Subtask(
                id=node_id,
                instruction=instruction,
                inputs={k: task.inputs[k] for k in wanted if k in task.inputs},
                depends_on=tuple(str(d) for d in deps),
                allowed_tools=task.allowed_tools,
                output_schema=task.output_schema if is_final else None,
                contract=task.contract if is_final else None,
                written_by="decomposer",
            )
        )
    if not isinstance(final, str) or final not in {s.id for s in subtasks}:
        problems.append("'final' must name one of the subtasks")
    if problems:
        raise ValueError("; ".join(problems))
    try:
        graph = TaskGraph(id=task.id, nodes=tuple(subtasks), final_nodes=(str(final),))
    except ValidationError as exc:
        raise ValueError("; ".join(e["msg"] for e in exc.errors())) from exc
    g = graph.to_networkx()
    depth = nx.dag_longest_path_length(g) + 1
    if depth > MAX_DEPTH:
        raise ValueError(f"dependency depth {depth} exceeds {MAX_DEPTH}")
    unused = set(g.nodes) - nx.ancestors(g, final) - {final}
    if unused:
        raise ValueError(f"subtasks {sorted(unused)} do not lead to the final subtask")
    return graph


async def model_decomposition(
    task: PublicTask,
    repetition: int,
    llm: MeteredClient,
    *,
    model: str,
    max_completion_tokens: int = 2048,
    temperature: float = 0.6,
    reasoning_effort: Literal["low", "medium", "high"] | None = None,
    seed: int | None = None,
) -> Decomposition:
    """Ask the decomposer once, repair once if needed, else fall back to a single node. ``llm``
    should be a coordinator client with agent ``decomposer``; its account holds the cost."""
    template = prompts.load("decompose")
    task_text = json.dumps(
        {
            "instruction": task.instruction,
            "inputs": task.inputs,
            "output_schema": task.output_schema,
            "tools": list(task.allowed_tools),
        },
        indent=1,
    )
    messages = [Message(role="system", content=template), Message(role="user", content=task_text)]
    errors: list[str] = []
    for attempt in range(2):
        request = LLMRequest(
            model=model,
            messages=tuple(messages),
            max_completion_tokens=max_completion_tokens,
            temperature=temperature,
            reasoning_effort=reasoning_effort,
            seed=seed,
            sample_id=f"decomposer:{task.id}:{repetition}:{attempt}",
        )
        reply = await llm.complete(request)
        try:
            graph = graph_from_proposal(task, parse_json(reply.content, envelope="nodes"))
        except ValueError as exc:
            errors.append(str(exc))
            messages += [
                Message(role="assistant", content=reply.content, source="decomposer"),
                Message(
                    role="user", content=f"That plan is invalid: {exc}. Return a corrected plan."
                ),
            ]
            continue
        status: Status = "ok" if attempt == 0 else "repaired"
        return _record(
            task.id,
            repetition,
            status,
            graph,
            cost=llm.account.snapshot(),
            prompt_sha256=sha256(template),
            errors=tuple(errors),
        )
    return _record(
        task.id,
        repetition,
        "fallback",
        task.as_single_node(),
        cost=llm.account.snapshot(),
        prompt_sha256=sha256(template),
        errors=tuple(errors),
    )


class DecompositionStore:
    """One JSON file per task and repetition. Files are written once and never overwritten."""

    def __init__(self, root: Path) -> None:
        self.root = root

    def path(self, task_id: str, repetition: int) -> Path:
        return self.root / task_id / f"rep{repetition}.json"

    def save(self, record: Decomposition) -> Path:
        path = self.path(record.task_id, record.repetition)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            raise FileExistsError(path)
        # Atomic: a crash leaves either no file or the whole file, never part of one.
        tmp = path.with_suffix(".json.tmp")
        with tmp.open("w", encoding="utf-8") as f:
            f.write(record.model_dump_json(indent=1) + "\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
        dir_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(dir_fd)
        finally:
            os.close(dir_fd)
        return path

    def load(self, task_id: str, repetition: int) -> Decomposition:
        record = Decomposition.model_validate_json(self.path(task_id, repetition).read_text())
        record.graph()  # verifies the hash
        return record


def start_from(record: Decomposition, account: Account) -> TaskGraph:
    """Charge the stored decomposition cost to a system's budget and return its fresh graph copy."""
    account.import_cost(record.cost)
    return record.graph()
