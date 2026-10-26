import pytest
from pydantic import ValidationError

from bioalloc.graph import TaskGraph
from bioalloc.schemas import Subtask


def st(id: str, *deps: str) -> Subtask:
    return Subtask(id=id, instruction=f"do {id}", depends_on=deps)


def test_ready_respects_dependencies() -> None:
    g = TaskGraph(
        id="t", nodes=(st("a"), st("b", "a"), st("c"), st("d", "b", "c")), final_nodes=("d",)
    )
    assert [n.id for n in g.ready(done=set())] == ["a", "c"]
    assert [n.id for n in g.ready(done={"a"}, started={"c"})] == ["b"]
    assert [n.id for n in g.ready(done={"a", "b", "c"})] == ["d"]
    assert g.ready(done={"a", "b", "c", "d"}) == []
    assert g.descendants("a") == {"b", "d"}


def test_rejects_duplicate_ids() -> None:
    with pytest.raises(ValidationError, match="duplicate"):
        TaskGraph(id="t", nodes=(st("a"), st("a")), final_nodes=("a",))


def test_rejects_unknown_dependency() -> None:
    with pytest.raises(ValidationError, match="unknown"):
        TaskGraph(id="t", nodes=(st("a", "zz"),), final_nodes=("a",))


def test_rejects_self_dependency() -> None:
    with pytest.raises(ValidationError, match="itself"):
        TaskGraph(id="t", nodes=(st("a", "a"),), final_nodes=("a",))


def test_rejects_cycle() -> None:
    with pytest.raises(ValidationError, match="cycle"):
        TaskGraph(id="t", nodes=(st("a", "c"), st("b", "a"), st("c", "b")), final_nodes=("c",))


def test_rejects_unknown_final_node() -> None:
    with pytest.raises(ValidationError, match="final"):
        TaskGraph(id="t", nodes=(st("a"),), final_nodes=("b",))


def test_round_trips_through_json() -> None:
    g = TaskGraph(id="t", nodes=(st("a"), st("b", "a")), final_nodes=("b",))
    assert TaskGraph.model_validate_json(g.model_dump_json()) == g
