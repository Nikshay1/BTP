"""Generator, references and grader for our deterministic tool-use suite.

Three workflow templates, each with a natural graph (decision D17, deterministic decomposer):

* ``revenue``: customers in a city -> their orders in a month -> total revenue (a chain of 3);
* ``compare``: the revenue chain for two cities in parallel, then their difference (width 2);
* ``top_product``: customers of a tier -> their orders -> the product with the largest quantity.

Parameters are drawn so that answers are not degenerate (non-empty selections, a unique top
product). The oracle computes answers from the same deterministic database the tools serve.
"""

from __future__ import annotations

import random
from collections import Counter
from dataclasses import dataclass

from pydantic import JsonValue

from bioalloc.evaluation.toy_data import Grade
from bioalloc.graph import TaskGraph
from bioalloc.schemas import Subtask
from bioalloc.tasks.common import normalize_number
from bioalloc.tasks.toolsuite import (
    ANSWER_CONTRACT,
    CITIES,
    DB_VERSION,
    LIST_CONTRACT,
    MONTHS,
    SCHEMAS,
    TEXT_CONTRACT,
    TIERS,
    database,
)

GENERATOR_VERSION = 1
TOOLS = ("query", "list_tables", "calculate")
TEMPLATES = ("revenue", "compare", "top_product")


@dataclass(frozen=True)
class ToolInstance:
    graph: TaskGraph
    template: str
    answer: str  # normalized
    group: str


def _node(
    node_id: str, instruction: str, dataset: str, contract: str, deps: tuple[str, ...] = ()
) -> Subtask:
    return Subtask(
        id=node_id,
        instruction=instruction,
        inputs={"dataset": dataset},
        depends_on=deps,
        allowed_tools=TOOLS,
        output_schema=SCHEMAS[contract],
        contract=contract,
    )


def _revenue(dataset: str, city: str, month: str) -> int:
    db = database(dataset)
    ids = {c["id"] for c in db["customers"] if c["city"] == city}
    return sum(
        o["qty"] * o["unit_price"]
        for o in db["orders"]
        if o["customer_id"] in ids and o["month"] == month
    )


def _revenue_chain(prefix: str, dataset: str, city: str, month: str) -> list[Subtask]:
    p = f"{prefix}-" if prefix else ""
    return [
        _node(
            f"{p}customers",
            f'Find the ids of all customers in the city {city}. Reply as {{"items": [ids]}}.',
            dataset,
            LIST_CONTRACT,
        ),
        _node(
            f"{p}orders",
            f"Find all orders placed in month '{month}' by the customers found in "
            f"subtask {p}customers. Reply with each order's qty and unit_price as "
            '{"items": [{"qty": ..., "unit_price": ...}, ...]}.',
            dataset,
            LIST_CONTRACT,
            (f"{p}customers",),
        ),
        _node(
            f"{p}revenue",
            f"Compute the total revenue, the sum of qty x unit_price, of the orders "
            f'found in subtask {p}orders. Reply as {{"answer": <number>}}.',
            dataset,
            ANSWER_CONTRACT,
            (f"{p}orders",),
        ),
    ]


def make_instance(template: str, seed: int) -> ToolInstance:
    rng = random.Random(f"toolsuite:v{GENERATOR_VERSION}:{template}:{seed}")
    dataset = f"shop-v{DB_VERSION}-{seed}"
    db = database(dataset)
    task_id = f"tools-v{GENERATOR_VERSION}-{template}-s{seed}"
    if template == "revenue":
        while True:
            city, month = rng.choice(CITIES), rng.choice(MONTHS)
            value = _revenue(dataset, city, month)
            if value > 0:
                break
        nodes = _revenue_chain("", dataset, city, month)
        answer = str(value)
    elif template == "compare":
        while True:
            a, b = rng.sample(CITIES, 2)
            month = rng.choice(MONTHS)
            ra, rb = _revenue(dataset, a, month), _revenue(dataset, b, month)
            if ra > 0 and rb > 0 and ra != rb:
                break
        nodes = _revenue_chain("a", dataset, a, month) + _revenue_chain("b", dataset, b, month)
        nodes.append(
            _node(
                "difference",
                f"Subtask a-revenue is the revenue of {a} and subtask b-revenue the "
                f"revenue of {b}. Compute the revenue of {a} minus the revenue of {b}. "
                'Reply as {"answer": <number>}.',
                dataset,
                ANSWER_CONTRACT,
                ("a-revenue", "b-revenue"),
            )
        )
        answer = str(ra - rb)
    elif template == "top_product":
        while True:
            tier = rng.choice(TIERS)
            ids = {c["id"] for c in db["customers"] if c["tier"] == tier}
            qty: Counter[str] = Counter()
            for o in db["orders"]:
                if o["customer_id"] in ids:
                    qty[o["product"]] += o["qty"]
            top = qty.most_common(2)
            if top and (len(top) == 1 or top[0][1] > top[1][1]):
                break
        nodes = [
            _node(
                "customers",
                f"Find the ids of all customers of tier '{tier}'. Reply as {{\"items\": [ids]}}.",
                dataset,
                LIST_CONTRACT,
            ),
            _node(
                "orders",
                "Find all orders placed by the customers found in subtask customers. "
                'Reply with each order\'s product and qty as {"items": [{"product": ..., '
                '"qty": ...}, ...]}.',
                dataset,
                LIST_CONTRACT,
                ("customers",),
            ),
            _node(
                "top",
                "Which product has the largest total qty among the orders found in "
                'subtask orders? Reply as {"answer": "<product name>"}.',
                dataset,
                TEXT_CONTRACT,
                ("orders",),
            ),
        ]
        answer = top[0][0]
    else:
        raise ValueError(f"unknown template {template!r}")
    graph = TaskGraph(id=task_id, nodes=tuple(nodes), final_nodes=(nodes[-1].id,))
    return ToolInstance(graph, template, answer, group=f"toolsuite:{task_id}")


def grade(instance: ToolInstance, answer: JsonValue) -> Grade:
    value = answer.get("answer") if isinstance(answer, dict) else None
    if instance.template == "top_product":
        got = value.strip().lower() if isinstance(value, str) else None
    else:
        got = normalize_number(value)
    if got == instance.answer:
        return Grade(correct=True)
    return Grade(correct=False, detail=f"expected {instance.answer}, got {got}")
