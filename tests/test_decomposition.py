import asyncio
import json
from pathlib import Path

import pytest

from bioalloc.budget import Account, MeteredClient
from bioalloc.decomposition import (
    DecompositionStore,
    PublicTask,
    deterministic,
    graph_from_proposal,
    model_decomposition,
    start_from,
)
from bioalloc.evaluation.toy_data import make_instance
from bioalloc.llm.fake import FakeTransport
from bioalloc.schemas import Budget
from bioalloc.trace import Tracer, read_trace, tokens_from_trace

TASK = PublicTask(
    id="t1",
    instruction="Compute the total price of the order.",
    inputs={"items": [["pen", 2, 1.5]], "tax": 0.1},
    output_schema={"type": "object", "properties": {"total": {"type": "number"}}},
    contract="total",
)
GOOD = {
    "nodes": [
        {"id": "s1", "instruction": "Sum item prices.", "depends_on": [], "inputs": ["items"]},
        {"id": "s2", "instruction": "Apply tax.", "depends_on": ["s1"], "inputs": ["tax"]},
    ],
    "final": "s2",
}


def decomposer(*replies: str) -> tuple[MeteredClient, FakeTransport]:
    fake = FakeTransport().always(*replies)
    account = Account(Budget(max_tokens=50_000, max_model_calls=5))
    return MeteredClient(fake, account, "coordinator", "decomposer"), fake


def test_valid_proposal_builds_graph_with_contract_on_final_node_only() -> None:
    g = graph_from_proposal(TASK, GOOD)
    s1, s2 = g.nodes
    assert s1.inputs == {"items": [["pen", 2, 1.5]]} and s1.contract is None
    assert s1.output_schema is None and s1.written_by == "decomposer"
    assert s2.contract == "total" and s2.output_schema == TASK.output_schema
    # The final node always keeps the whole public task.
    assert set(s2.inputs) == set(TASK.inputs) and TASK.instruction in s2.instruction
    assert g.final_nodes == ("s2",)


@pytest.mark.parametrize(
    ("proposal", "problem"),
    [
        ([], "not a JSON object"),
        ({"nodes": [], "final": "x"}, "non-empty"),
        (
            {"nodes": [{"id": f"s{i}", "instruction": "x"} for i in range(9)], "final": "s0"},
            "at most",
        ),
        (
            {"nodes": [{"id": "a", "instruction": "x", "inputs": ["secret"]}], "final": "a"},
            "unknown inputs",
        ),
        ({"nodes": [{"id": "a", "instruction": "x"}], "final": "b"}, "'final'"),
        (
            {
                "nodes": [
                    {"id": "a", "instruction": "x", "depends_on": ["b"]},
                    {"id": "b", "instruction": "y", "depends_on": ["a"]},
                ],
                "final": "b",
            },
            "cycle",
        ),
        (
            {
                "nodes": [{"id": "a", "instruction": "x"}, {"id": "b", "instruction": "y"}],
                "final": "b",
            },
            "do not lead",
        ),
        (
            {
                "nodes": [
                    {"id": f"s{i}", "instruction": "x", "depends_on": [f"s{i - 1}"] if i else []}
                    for i in range(7)
                ],
                "final": "s6",
            },
            "depth",
        ),
        ({"nodes": [{"id": "a", "instruction": " "}], "final": "a"}, "empty instruction"),
    ],
)
def test_invalid_proposals_are_rejected(proposal: object, problem: str) -> None:
    with pytest.raises(ValueError, match=problem):
        graph_from_proposal(TASK, proposal)  # type: ignore[arg-type]


def test_first_valid_proposal_is_ok() -> None:
    llm, fake = decomposer(json.dumps(GOOD))
    rec = asyncio.run(model_decomposition(TASK, 0, llm, model="m"))
    assert rec.status == "ok" and rec.cost.coordinator.total > 0 and rec.cost.worker.total == 0
    assert len(rec.graph().nodes) == 2
    # The decomposer sees the public task, input values included.
    sent = fake.requests[0].messages[1].content
    assert "items" in sent and "pen" in sent


def test_invalid_then_repaired_then_fallback() -> None:
    llm, fake = decomposer("not json", json.dumps(GOOD))
    rec = asyncio.run(model_decomposition(TASK, 0, llm, model="m"))
    assert rec.status == "repaired" and len(rec.errors) == 1 and rec.cost.model_calls == 2
    assert "invalid" in fake.requests[1].messages[-1].content

    llm, _ = decomposer("nope", "still nope")
    rec = asyncio.run(model_decomposition(TASK, 0, llm, model="m"))
    assert rec.status == "fallback" and rec.cost.model_calls == 2
    (node,) = rec.graph().nodes
    assert node.instruction == TASK.instruction and node.contract == "total"


def test_store_writes_once_verifies_hash_and_replays_fresh_copies(tmp_path: Path) -> None:
    store = DecompositionStore(tmp_path)
    rec = deterministic(make_instance(0).graph, repetition=0)
    store.save(rec)
    with pytest.raises(FileExistsError):
        store.save(rec)
    loaded = store.load("toy-0", 0)
    a, b = loaded.graph(), loaded.graph()
    assert a == b and a is not b
    a.nodes[0].inputs["edges"].append(["X", "Y", 1])  # type: ignore[union-attr]
    assert loaded.graph() == b  # mutating one copy never affects another
    tampered = loaded.model_copy(update={"graph_json": loaded.graph_json.replace("leg1", "legX")})
    with pytest.raises(ValueError, match="hash"):
        tampered.graph()


def test_every_system_starts_with_the_same_imported_cost(tmp_path: Path) -> None:
    llm, _ = decomposer(json.dumps(GOOD))
    rec = asyncio.run(model_decomposition(TASK, 0, llm, model="m"))
    for system in ("a", "b"):
        account = Account(Budget(max_tokens=50_000, max_model_calls=5))
        tracer = Tracer(tmp_path / system, "r")
        span = tracer.span()
        graph = start_from(rec, account)
        tracer.emit(
            span,
            "cost_import",
            {"source": rec.graph_sha256, "cost": rec.cost.model_dump(mode="json")},
        )
        tracer.close()
        assert account.snapshot().tokens == rec.cost.tokens
        assert account.snapshot().model_calls == rec.cost.model_calls
        assert account.remaining().max_model_calls == 5 - rec.cost.model_calls
        assert (
            tokens_from_trace(list(read_trace(tmp_path / system)))["tokens"]
            == rec.cost.tokens.total
        )
        assert graph.final_nodes == ("s2",)


def test_a_plan_that_drops_the_inputs_still_keeps_them_on_the_final_node() -> None:
    lossy = {
        "nodes": [
            {"id": "s1", "instruction": "Think.", "inputs": []},
            {"id": "s2", "instruction": "Answer.", "depends_on": ["s1"], "inputs": []},
        ],
        "final": "s2",
    }
    g = graph_from_proposal(TASK, lossy)
    assert g.node("s1").inputs == {} and g.node("s2").inputs == TASK.inputs


def test_imported_cost_counts_against_calls_tools_and_time() -> None:
    from bioalloc.schemas import Cost

    now = [0.0]
    account = Account(
        Budget(max_tokens=10**6, max_model_calls=1, max_tool_calls=1, max_elapsed_s=10),
        clock=lambda: now[0],
    )
    account.import_cost(Cost(model_calls=2, tool_calls=2, elapsed_s=20.0))
    assert account.overrun and account.snapshot().overrun and account.exhausted()
    fresh = Account(
        Budget(max_tokens=10**6, max_model_calls=5, max_elapsed_s=10), clock=lambda: now[0]
    )
    fresh.import_cost(Cost(model_calls=1, elapsed_s=4.0))
    assert not fresh.overrun and fresh.time_left() == 6.0 and fresh.snapshot().elapsed_s == 4.0
