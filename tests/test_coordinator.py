import asyncio
import json
from pathlib import Path

from bioalloc.budget import Account, MeteredClient
from bioalloc.coordinator import Coordinator, CoordinatorConfig, RunResult
from bioalloc.evaluation.toy_data import grade, make_instance, optimal_legs
from bioalloc.executors.direct import Direct
from bioalloc.llm.base import LLMRequest, TransportTimeout
from bioalloc.llm.fake import FakeTransport, Script
from bioalloc.policies import Constant
from bioalloc.schemas import Budget
from bioalloc.tasks.toy import ToyRoutes
from bioalloc.trace import Tracer, read_trace, tokens_from_trace

INSTANCE = make_instance(seed=0)
LEG1, LEG2 = optimal_legs(INSTANCE)


def leg(n: int):  # type: ignore[no-untyped-def]
    marker = "Find the cheapest route from start" if n == 1 else "Continue from the node"
    return lambda req: marker in req.messages[1].content


def route(r: list[str]) -> str:
    return json.dumps({"route": r})


def run(
    fake: FakeTransport,
    tmp: Path,
    budget: Budget | None = None,
    config: CoordinatorConfig | None = None,
) -> RunResult:
    tracer = Tracer(tmp, "run")
    account = Account(budget or Budget(max_tokens=50_000, max_model_calls=20))
    llm = MeteredClient(fake, account, actor="coordinator", agent="coordinator")
    coord = Coordinator({"direct": Direct(model="fake")}, Constant("direct"), config)
    result = asyncio.run(coord.run(INSTANCE.graph, ToyRoutes(), llm, tracer))
    tracer.close()
    return result


def script(fake: FakeTransport, n: int, *replies: Script) -> FakeTransport:
    return fake.on(leg(n), *replies)


def test_good_run_completes_and_passes_leg1_result_to_leg2(tmp_path: Path) -> None:
    fake = script(script(FakeTransport(), 1, route(LEG1)), 2, route(LEG2))
    result = run(fake, tmp_path)
    assert result.status == "completed"
    assert grade(INSTANCE, result.answer).correct
    assert [r.messages[1].content.split()[1] for r in fake.requests] == ["Find", "Continue"]
    leg2_request: LLMRequest = fake.requests[1]
    assert leg2_request.messages[-1].content == f"Result of subtask leg1:\n{route(LEG1)}"
    assert result.cost.model_calls == 2 and result.cost.coordinator.total == 0


def test_invalid_first_answer_is_reallocated_and_recovers(tmp_path: Path) -> None:
    illegal = [LEG1[0], "Z", LEG1[-1]]
    fake = script(script(FakeTransport(), 1, route(illegal), route(LEG1)), 2, route(LEG2))
    result = run(fake, tmp_path)
    assert result.status == "completed" and grade(INSTANCE, result.answer).correct
    assert [(a.result.subtask_id, a.accepted) for a in result.attempts] == [
        ("leg1", False),
        ("leg1", True),
        ("leg2", True),
    ]
    assert result.attempts[0].verdict == "fail"
    # The failed attempt still costs budget.
    assert result.cost.model_calls == 3


def test_failed_subtask_blocks_descendants_without_looping(tmp_path: Path) -> None:
    fake = script(FakeTransport(), 1, "no idea", "still no idea")
    result = run(fake, tmp_path)
    assert result.status == "failed" and result.answer is None
    assert len(result.attempts) == 2
    assert all(not leg(2)(r) for r in fake.requests)
    kinds = [e.kind for e in read_trace(tmp_path)]
    assert "subtask_failed" in kinds and kinds[-1] == "run_end"


def test_small_budget_ends_with_budget_exhausted_and_a_complete_trace(tmp_path: Path) -> None:
    fake = script(script(FakeTransport(), 1, route(LEG1)), 2, route(LEG2))
    result = run(fake, tmp_path, budget=Budget(max_tokens=50_000, max_model_calls=1))
    assert result.status == "budget_exhausted"
    assert result.cost.model_calls == 1
    events = list(read_trace(tmp_path))
    assert events[-1].kind == "run_end" and events[-1].data["status"] == "budget_exhausted"


def test_token_cap_too_small_for_any_call(tmp_path: Path) -> None:
    fake = FakeTransport()
    result = run(fake, tmp_path, budget=Budget(max_tokens=100, max_model_calls=10))
    assert result.status == "budget_exhausted"
    assert fake.requests == []


def test_timeout_is_retried_and_charged(tmp_path: Path) -> None:
    fake = script(script(FakeTransport(), 1, TransportTimeout("slow"), route(LEG1)), 2, route(LEG2))
    result = run(fake, tmp_path)
    assert result.status == "completed"
    assert result.attempts[0].result.status == "timeout"
    assert result.cost.usage_complete is False


def test_trace_matches_ledger_after_a_run(tmp_path: Path) -> None:
    illegal = [LEG1[0], "Z", LEG1[-1]]
    fake = script(script(FakeTransport(), 1, route(illegal), route(LEG1)), 2, route(LEG2))
    result = run(fake, tmp_path)
    totals = tokens_from_trace(list(read_trace(tmp_path)))
    assert totals["model_calls"] == result.cost.model_calls
    assert totals["tokens"] == result.cost.tokens.total
    assert totals["communication"] == result.cost.communication > 0
    per_attempt = sum(a.result.cost.tokens.total for a in result.attempts)
    assert per_attempt == result.cost.tokens.total
