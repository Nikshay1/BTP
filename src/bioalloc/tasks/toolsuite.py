"""Our deterministic tool-use suite, public side (decision D16).

Each task owns a small generated sales database (customers, orders, products), identified by a
dataset id and rebuilt deterministically from it, so tools need no files or network. Tools:
``list_tables``, ``query`` (rows of a table matching equality conditions, optional columns) and the
shared ``calculate``. Tasks come with a natural workflow graph (decision D17: deterministic
decomposer), e.g. customers in a city -> their orders in a month -> revenue. Only formats are
publicly checkable (no goal check), so acceptance is weak evidence (D15). Ant's action kinds are
QUERY, CALCULATE and FINISH, the same operations the other executors get as tools.
"""

from __future__ import annotations

import json
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any

from pydantic import JsonValue

from bioalloc.actions import FINISH, Action
from bioalloc.checks import RuntimeCheck, SchemaCheck
from bioalloc.graph import TaskGraph
from bioalloc.schemas import Subtask
from bioalloc.tasks.common import CALCULATOR, CalcError, calculate, normalize_number
from bioalloc.tasks.gsm8k import NumericAnswerCheck
from bioalloc.tools import Tool, ToolSpec

DB_VERSION = 1
CITIES = ("Agra", "Bhopal", "Cochin", "Delhi", "Indore", "Jaipur")
TIERS = ("basic", "plus", "pro")
PRODUCTS = {
    "lamp": "home",
    "kettle": "home",
    "pen": "office",
    "stapler": "office",
    "ball": "sport",
    "racket": "sport",
}
PRICES = {"lamp": 30, "kettle": 25, "pen": 2, "stapler": 8, "ball": 12, "racket": 45}
MONTHS = ("jan", "feb", "mar", "apr")
MAX_ROWS = 60

Table = list[dict[str, Any]]


@lru_cache(maxsize=256)
def database(dataset: str) -> dict[str, Table]:
    """The tables of a dataset id such as 'shop-v1-17'. Deterministic in the id."""
    if not dataset.startswith(f"shop-v{DB_VERSION}-"):
        raise KeyError(f"unknown dataset {dataset!r}")
    rng = random.Random(dataset)
    customers = [
        {"id": i, "name": f"C{i:02d}", "city": rng.choice(CITIES), "tier": rng.choice(TIERS)}
        for i in range(1, 16)
    ]
    orders = []
    for i in range(1, 51):
        product = rng.choice(list(PRODUCTS))
        orders.append(
            {
                "id": i,
                "customer_id": rng.randint(1, 15),
                "product": product,
                "qty": rng.randint(1, 5),
                "unit_price": PRICES[product],
                "month": rng.choice(MONTHS),
            }
        )
    products = [{"name": p, "category": c} for p, c in PRODUCTS.items()]
    return {"customers": customers, "orders": orders, "products": products}


def run_query(
    dataset: str, table: str, where: Mapping[str, Any], columns: Sequence[str] | None
) -> Table:
    tables = database(dataset)
    if table not in tables:
        raise KeyError(f"unknown table {table!r}; tables are {sorted(tables)}")
    rows = tables[table]
    known = set(rows[0])
    for col in [*where, *(columns or [])]:
        if col not in known:
            raise KeyError(f"unknown column {col!r} in {table}; columns are {sorted(known)}")
    out = []
    for row in rows:
        ok = True
        for col, want in where.items():
            if isinstance(want, list):
                ok &= row[col] in want
            else:
                ok &= row[col] == want
        if ok:
            out.append({c: row[c] for c in columns} if columns else dict(row))
    if len(out) > MAX_ROWS:
        raise ValueError(f"{len(out)} rows match; narrow the query (at most {MAX_ROWS})")
    return out


_DATASET: dict[str, JsonValue] = {
    "type": "string",
    "description": "dataset id given in the task inputs",
}


def _json(value: Any) -> JsonValue:
    """Plain JSON copy of a tool result (no shared references into the cached database)."""
    result: JsonValue = json.loads(json.dumps(value))
    return result


QUERY = Tool(
    ToolSpec(
        name="query",
        description="Rows of a table whose columns equal the given values (a list means any of). "
        "Optionally return only some columns.",
        parameters={
            "type": "object",
            "properties": {
                "dataset": _DATASET,
                "table": {"type": "string"},
                "where": {"type": "object"},
                "columns": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["dataset", "table"],
            "additionalProperties": False,
        },
    ),
    lambda a: _json(
        run_query(str(a["dataset"]), str(a["table"]), a.get("where", {}), a.get("columns"))
    ),
)
LIST_TABLES = Tool(
    ToolSpec(
        name="list_tables",
        description="Table names and their columns.",
        parameters={
            "type": "object",
            "properties": {"dataset": _DATASET},
            "required": ["dataset"],
            "additionalProperties": False,
        },
    ),
    lambda a: _json({t: sorted(rows[0]) for t, rows in database(str(a["dataset"])).items()}),
)
TOOLS = {"query": QUERY, "list_tables": LIST_TABLES, "calculate": CALCULATOR}

ANSWER_CONTRACT = "toolsuite_number"
LIST_CONTRACT = "toolsuite_list"
TEXT_CONTRACT = "toolsuite_text"
SCHEMAS: dict[str, dict[str, JsonValue]] = {
    ANSWER_CONTRACT: {
        "type": "object",
        "properties": {"answer": {"type": ["number", "string"]}},
        "required": ["answer"],
    },
    LIST_CONTRACT: {
        "type": "object",
        "properties": {"items": {"type": "array"}},
        "required": ["items"],
    },
    TEXT_CONTRACT: {
        "type": "object",
        "properties": {"answer": {"type": "string"}},
        "required": ["answer"],
    },
}


@dataclass
class _State:
    results: dict[str, JsonValue] = field(default_factory=dict)


class ToolActions:
    kinds: tuple[str, ...] = ("QUERY", "CALCULATE", FINISH)

    def __init__(self, subtask: Subtask) -> None:
        self.dataset = str(subtask.inputs.get("dataset", ""))

    def start(self, subtask: Subtask, upstream: Mapping[str, JsonValue]) -> _State:
        return _State({f"upstream_{k}": v for k, v in upstream.items()})

    def legal(self, state: _State, action: Action) -> str | None:
        a = action.args
        if action.kind == FINISH:
            return None if "output" in a else "FINISH needs 'output' (the subtask's JSON answer)"
        if not isinstance(a.get("name"), str):
            return "needs a string 'name' for the result"
        try:
            self._run(state, action)
        except (KeyError, ValueError, CalcError) as exc:
            return str(exc)
        return None

    def _run(self, state: _State, action: Action) -> JsonValue:
        a = action.args
        if action.kind == "QUERY":
            where = a.get("where", {})
            columns = a.get("columns")
            if not isinstance(where, dict) or (
                columns is not None and not isinstance(columns, list)
            ):
                raise ValueError("'where' must be an object and 'columns' a list")
            return _json(run_query(self.dataset, str(a.get("table")), where, columns))  # type: ignore[arg-type]
        if action.kind == "CALCULATE":
            return calculate(str(a.get("expression", "")))
        raise ValueError(f"unknown action {action.kind}")

    def apply(self, state: _State, action: Action) -> tuple[_State, str]:
        value = self._run(state, action)
        name = str(action.args["name"])
        return _State({**state.results, name: value}), f"{name} = {json.dumps(value)[:500]}"

    def finish(self, state: _State, action: Action) -> JsonValue:
        return action.args.get("output")

    def describe(self, state: _State) -> str:
        return f"Dataset: {self.dataset}. Results so far: {json.dumps(state.results)[:3000]}"


class ToolSuite:
    name = "toolsuite"

    def checks(self, subtask: Subtask) -> Sequence[RuntimeCheck]:
        if subtask.contract == ANSWER_CONTRACT:
            return [SchemaCheck(), NumericAnswerCheck()]
        if subtask.contract in SCHEMAS:
            return [SchemaCheck()]
        return []

    def aggregate(self, graph: TaskGraph, outputs: Mapping[str, JsonValue]) -> JsonValue:
        return outputs[graph.final_nodes[-1]]

    def normalize(self, subtask: Subtask, output: JsonValue) -> str | None:
        if not isinstance(output, dict):
            return None
        if subtask.contract == ANSWER_CONTRACT:
            return normalize_number(output.get("answer"))
        if subtask.contract == TEXT_CONTRACT:
            answer = output.get("answer")
            return answer.strip().lower() if isinstance(answer, str) and answer.strip() else None
        items = output.get("items")
        if isinstance(items, list):
            return json.dumps(sorted(json.dumps(i, sort_keys=True) for i in items))
        return None

    def action_space(self, subtask: Subtask) -> ToolActions:
        return ToolActions(subtask)

    def tools(self) -> dict[str, Tool]:
        return dict(TOOLS)
