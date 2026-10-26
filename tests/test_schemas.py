import pytest
from pydantic import ValidationError

from bioalloc.schemas import Budget, Cost, ExecutorResult, Subtask, Tokens


def test_cost_views_add_up() -> None:
    cost = Cost(
        coordinator=Tokens(prompt=100, completion=20),
        worker=Tokens(prompt=300, completion=80),
        communication=150,
    )
    assert cost.tokens.total == 500
    assert cost.local + cost.communication == cost.tokens.total


def test_contracts_reject_unknown_fields() -> None:
    # A misspelled or smuggled field (e.g. a reference answer) must not pass silently.
    with pytest.raises(ValidationError):
        Subtask(id="a", instruction="x", answer="42")  # type: ignore[call-arg]


def test_contracts_are_immutable() -> None:
    b = Budget(max_tokens=10, max_model_calls=1)
    with pytest.raises(ValidationError):
        b.max_tokens = 20


def test_budget_rejects_negative_limits() -> None:
    with pytest.raises(ValidationError):
        Budget(max_tokens=-1, max_model_calls=1)


def test_confidence_range() -> None:
    with pytest.raises(ValidationError):
        ExecutorResult(
            executor="direct",
            subtask_id="a",
            output="x",
            confidence=1.5,
            cost=Cost(),
            status="completed",
            span_id="s",
        )
