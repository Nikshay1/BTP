import asyncio
import json
from pathlib import Path

import pytest

from bioalloc.actions import Action
from bioalloc.budget import Account, MeteredClient
from bioalloc.coordinator import Coordinator
from bioalloc.evaluation.toolsuite_data import TEMPLATES, grade, make_instance
from bioalloc.executors.direct import Direct
from bioalloc.llm.base import LLMReply, LLMRequest, ToolCall, Usage
from bioalloc.llm.fake import FakeTransport
from bioalloc.policies import Constant
from bioalloc.schemas import Budget
from bioalloc.tasks.toolsuite import ToolSuite, database, run_query
from bioalloc.trace import Tracer


def test_database_is_deterministic_per_dataset() -> None:
    assert database("shop-v1-3") == database("shop-v1-3")
    assert database("shop-v1-3") != database("shop-v1-4")
    with pytest.raises(KeyError):
        database("other-1")


def test_query_tool_validates_and_bounds() -> None:
    rows = run_query("shop-v1-3", "customers", {"city": "Agra"}, ["id"])
    assert all(set(r) == {"id"} for r in rows)
    with pytest.raises(KeyError, match="unknown column"):
        run_query("shop-v1-3", "customers", {"town": "Agra"}, None)
    with pytest.raises(KeyError, match="unknown table"):
        run_query("shop-v1-3", "people", {}, None)
    assert len(run_query("shop-v1-3", "orders", {"customer_id": [1, 2]}, None)) >= 0


@pytest.mark.parametrize("template", TEMPLATES)
def test_templates_are_reproducible_with_non_degenerate_answers(template: str) -> None:
    a, b = make_instance(template, 7), make_instance(template, 7)
    assert a.graph == b.graph and a.answer == b.answer and a.answer not in ("", "0")
    assert grade(a, {"answer": a.answer}).correct
    assert not grade(a, {"answer": "nonsense"}).correct


def test_compare_has_two_parallel_chains_joined_at_the_end() -> None:
    g = make_instance("compare", 2).graph
    assert [n.id for n in g.ready(done=set())] == ["a-customers", "b-customers"]
    assert g.node("difference").depends_on == ("a-revenue", "b-revenue")
    assert g.final_nodes == ("difference",)


def reply(content: str = "", calls: tuple[ToolCall, ...] = ()) -> LLMReply:
    return LLMReply(
        content=content,
        tool_calls=calls,
        model="m",
        usage=Usage(prompt_tokens=10, completion_tokens=5),
        finish_reason="tool_calls" if calls else "stop",
    )


def test_revenue_workflow_end_to_end_through_real_tools(tmp_path: Path) -> None:
    inst = make_instance("revenue", 4)
    dataset = str(inst.graph.nodes[0].inputs["dataset"])
    city = inst.graph.nodes[0].instruction.split("city ")[1].split(".")[0]
    month = inst.graph.nodes[1].instruction.split("'")[1]

    def respond(req: LLMRequest) -> LLMReply:
        task = req.messages[1].content
        last = req.messages[-1]
        if last.role == "tool":  # read the tool result back and answer
            data = json.loads(last.content)
            if "ids of all customers" in task:
                return reply(json.dumps({"items": [r["id"] for r in data]}))
            if "Find all orders" in task:
                return reply(json.dumps({"items": data}))
            return reply(json.dumps({"answer": data}))
        if "ids of all customers" in task:
            args = {
                "dataset": dataset,
                "table": "customers",
                "where": {"city": city},
                "columns": ["id"],
            }
            return reply(calls=(ToolCall(id="q1", name="query", arguments=json.dumps(args)),))
        if "Find all orders" in task:
            ids = json.loads(req.messages[-1].content)["items"]
            args = {
                "dataset": dataset,
                "table": "orders",
                "where": {"customer_id": ids, "month": month},
                "columns": ["qty", "unit_price"],
            }
            return reply(calls=(ToolCall(id="q2", name="query", arguments=json.dumps(args)),))
        items = json.loads(req.messages[-1].content)["items"]
        expr = " + ".join(f"{o['qty']} * {o['unit_price']}" for o in items)
        return reply(
            calls=(ToolCall(id="c1", name="calculate", arguments=json.dumps({"expression": expr})),)
        )

    fake = FakeTransport().respond(lambda r: True, respond)
    llm = MeteredClient(
        fake, Account(Budget(max_tokens=10**6, max_model_calls=30)), "coordinator", "coordinator"
    )
    result = asyncio.run(
        Coordinator({"direct": Direct(model="m")}, Constant("direct")).run(
            inst.graph, ToolSuite(), llm, Tracer(tmp_path, "r")
        )
    )
    assert result.status == "completed", [a.checks for a in result.attempts]
    assert grade(inst, result.answer).correct
    assert result.cost.tool_calls == 3 and result.cost.model_calls == 6


def test_ant_actions_query_and_calculate() -> None:
    from bioalloc.tools import ToolService

    inst = make_instance("revenue", 4)
    node = inst.graph.nodes[0]
    space = ToolSuite().action_space(node)
    s = space.start(node, {})
    q = Action(
        kind="QUERY", args={"name": "c", "table": "customers", "where": {}, "columns": ["id"]}
    )
    assert space.legal(s, q) is None
    call = space.tool_call(s, q, "q1")
    assert call is not None and call.name == "query"
    assert json.loads(call.arguments)["dataset"] == "shop-v1-4"
    svc = ToolService(
        ToolSuite().tools(), ("query",), Account(Budget(max_tokens=1, max_model_calls=1))
    )
    s, obs = space.observe(s, q, asyncio.run(svc.call(call)))
    assert obs.startswith("c = [")
    # Legality is structural only: an unknown table is legal until the tool says otherwise.
    assert space.legal(s, Action(kind="QUERY", args={"name": "x", "table": "nope"})) is None
    assert space.legal(s, Action(kind="QUERY", args={"name": "x"}))
    assert space.finish(s, Action(kind="FINISH", args={"output": {"items": [1]}})) == {"items": [1]}


def test_ant_tool_actions_are_metered_and_limited(tmp_path: Path) -> None:
    from bioalloc.coordinator import CoordinatorConfig
    from bioalloc.executors.ant import Ant, AntParams

    inst = make_instance("revenue", 4)
    first = inst.graph.nodes[0]
    graph = inst.graph.model_copy(update={"nodes": (first,), "final_nodes": (first.id,)})
    query = {
        "candidates": [
            {"kind": "QUERY", "args": {"name": "c", "table": "customers", "columns": ["id"]}}
        ]
    }
    finish = {"candidates": [{"kind": "FINISH", "args": {"output": {"items": [1]}}}]}
    for limit, expect_calls in ((None, 1), (0, 0)):
        replies = iter([query, finish] * 20)
        fake = FakeTransport().respond(lambda r: True, lambda r, it=replies: json.dumps(next(it)))
        budget = Budget(max_tokens=10**6, max_model_calls=40, max_tool_calls=limit)
        llm = MeteredClient(fake, Account(budget), "coordinator", "coordinator")
        ant = Ant(model="m", params=AntParams(max_paths=1))
        result = asyncio.run(
            Coordinator({"ant": ant}, Constant("ant"), CoordinatorConfig(max_attempts=1)).run(
                graph, ToolSuite(), llm, Tracer(tmp_path / str(limit), "r")
            )
        )
        assert result.cost.tool_calls == expect_calls
        if limit == 0:
            assert result.status == "budget_exhausted"
