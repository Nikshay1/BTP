import asyncio
import json
import re
from pathlib import Path

from bioalloc.budget import Account, MeteredClient
from bioalloc.coordinator import Coordinator, CoordinatorConfig, RunResult
from bioalloc.executors.base import ExecutionContext
from bioalloc.executors.bee import Bee, MajorityVote
from bioalloc.graph import TaskGraph
from bioalloc.llm.base import LLMRequest
from bioalloc.llm.fake import FakeTransport
from bioalloc.policies import Constant
from bioalloc.schemas import Budget, CheckResult, Subtask
from bioalloc.trace import Tracer, read_trace

ASSESS = lambda r: "has proposed an answer" in r.messages[0].content  # noqa: E731
SOLVE = lambda r: not ASSESS(r)  # noqa: E731
GRAPH = TaskGraph(id="q", nodes=(Subtask(id="a", instruction="pick"),), final_nodes=("a",))


class Family:
    """Answers {"x": n}; {"x": -1} is rejected by a public check."""

    name = "f"

    class Valid:
        name = "valid"
        evidence = "operational"

        def evaluate(self, subtask, output, upstream):  # type: ignore[no-untyped-def]
            ok = isinstance(output, dict) and output.get("x", -1) >= 0
            return CheckResult(
                check="valid",
                verdict="pass" if ok else "fail",
                evidence="operational",
                reason="" if ok else "negative",
            )

    def checks(self, s):  # type: ignore[no-untyped-def]
        return [self.Valid()]

    def aggregate(self, g, outputs):  # type: ignore[no-untyped-def]
        return outputs["a"]


def scout_of(req: LLMRequest) -> tuple[int, int]:
    """(scout or voter index, round) from the request's sample id."""
    m = re.search(r"/(?:scout|voter)(\d+)/(?:r(\d+)|vote)", req.sample_id)
    assert m, req.sample_id
    return int(m.group(1)), int(m.group(2) or 0)


def answers(table: dict[tuple[int, int], int]):  # type: ignore[no-untyped-def]
    return lambda req: json.dumps({"x": table[scout_of(req)]})


def run(fake: FakeTransport, tmp: Path, executor, calls: int = 200) -> tuple[RunResult, list]:  # type: ignore[no-untyped-def, type-arg]
    llm = MeteredClient(
        fake, Account(Budget(max_tokens=10**7, max_model_calls=calls)), "coordinator", "coordinator"
    )
    coord = Coordinator(
        {executor.name: executor}, Constant(executor.name), CoordinatorConfig(max_attempts=1)
    )
    tracer = Tracer(tmp, "r")
    result = asyncio.run(coord.run(GRAPH, Family(), llm, tracer))
    tracer.close()
    return result, list(read_trace(tmp))


def stop_of(events: list) -> dict:  # type: ignore[type-arg]
    return next(e.data for e in events if e.kind == "executor_stop")


def test_unanimous_first_round_reaches_quorum_without_assessments(tmp_path: Path) -> None:
    fake = FakeTransport().respond(SOLVE, answers({(i, 0): 7 for i in range(5)}))
    result, events = run(fake, tmp_path, Bee(model="m"))
    assert result.answer == {"x": 7} and len(fake.requests) == 5
    assert stop_of(events)["reason"] == "quorum"


def test_uncommitted_scouts_are_recruited_to_quorum(tmp_path: Path) -> None:
    table = {(0, 0): 3, (1, 0): 3, (2, 0): 3, (3, 0): -1, (4, 0): -1}
    fake = FakeTransport().respond(SOLVE, answers(table))
    fake.respond(ASSESS, lambda r: json.dumps({"accept": True, "confidence": 0.9}))
    result, events = run(fake, tmp_path, Bee(model="m", sigma=0.0))
    assert result.answer == {"x": 3} and stop_of(events)["reason"] == "quorum"
    assessments = [r for r in fake.requests if ASSESS(r)]
    assert len(assessments) == 2
    shown = assessments[0].messages[-1]
    assert json.loads(shown.content) == {"x": 3} and shown.source.endswith("/scout0")  # type: ignore[union-attr]


def test_sigma_zero_with_all_committed_is_absorbing(tmp_path: Path) -> None:
    table = {(0, 0): 1, (1, 0): 1, (2, 0): 1, (3, 0): 2, (4, 0): 2}
    fake = FakeTransport().respond(SOLVE, answers(table))
    result, events = run(fake, tmp_path, Bee(model="m", sigma=0.0))
    assert result.answer == {"x": 1} and len(fake.requests) == 5
    assert stop_of(events) == {
        "reason": "absorbing",
        "support": {'{"x":1}': 3, '{"x":2}': 2},
        "quorum_reached": False,
        "from_released": False,
    }


def test_full_release_leads_to_independent_rediscovery(tmp_path: Path) -> None:
    table = {(0, 0): 1, (1, 0): 1, (2, 0): 1, (3, 0): 2, (4, 0): 2}
    table |= {(i, 1): 2 for i in range(5)}
    fake = FakeTransport().respond(SOLVE, answers(table))
    result, events = run(fake, tmp_path, Bee(model="m", sigma=1.0))
    assert result.answer == {"x": 2} and stop_of(events)["reason"] == "quorum"
    released = next(e.data for e in events if e.kind == "bee_inhibition")["released"]
    assert released == [0, 1, 2, 3, 4] and len(fake.requests) == 10


def test_round_limit_falls_back_to_plurality(tmp_path: Path) -> None:
    table = {(0, 0): 5, (1, 0): 5, (2, 0): 6, (3, 0): -1, (4, 0): -1}
    fake = FakeTransport().respond(SOLVE, answers(table))
    fake.respond(ASSESS, lambda r: json.dumps({"accept": False, "confidence": 0.1}))
    result, events = run(fake, tmp_path, Bee(model="m", sigma=0.0, max_rounds=2))
    assert result.answer == {"x": 5}
    stop = stop_of(events)
    assert stop["reason"] == "round_limit" and not stop["quorum_reached"]
    assert len([r for r in fake.requests if ASSESS(r)]) == 4  # 2 uncommitted x 2 rounds


def test_majority_vote_and_canonical_tie_break(tmp_path: Path) -> None:
    table = {(0, 0): 4, (1, 0): 9, (2, 0): 9, (3, 0): 4, (4, 0): -1}
    fake = FakeTransport().respond(SOLVE, answers(table))
    result, events = run(fake, tmp_path, MajorityVote(model="m"))
    assert result.answer == {"x": 4}  # 2-2 tie, broken by canonical order
    assert stop_of(events)["tally"] == {'{"x":4}': 2, '{"x":9}': 2}


def test_call_yoked_majority_spends_exactly_its_allowance(tmp_path: Path) -> None:
    fake = FakeTransport().respond(SOLVE, lambda r: json.dumps({"x": scout_of(r)[0] % 2}))
    result, _ = run(fake, tmp_path, MajorityVote(model="m", max_calls=7, name="majority_yoked"))
    assert len(fake.requests) == 7 and result.answer == {"x": 0}


def test_budget_before_any_usable_answer_is_budget_exhaustion(tmp_path: Path) -> None:
    fake = FakeTransport().respond(SOLVE, answers({(i, 0): 1 for i in range(5)}))
    result, _ = run(fake, tmp_path, Bee(model="m"), calls=0)
    assert result.status == "budget_exhausted"


def test_scout_answers_count_as_communication_only_when_shown_to_another_scout() -> None:
    table = {(0, 0): 3, (1, 0): 3, (2, 0): 3, (3, 0): -1, (4, 0): -1}
    fake = FakeTransport().respond(SOLVE, answers(table))
    fake.respond(ASSESS, lambda r: json.dumps({"accept": True, "confidence": 1}))
    account = Account(Budget(max_tokens=10**7, max_model_calls=100))
    ctx = ExecutionContext(
        llm=MeteredClient(fake, account, "worker", "bee:a:0"), span_id="s", checks=(Family.Valid(),)
    )
    asyncio.run(Bee(model="m", sigma=0.0).execute(GRAPH.nodes[0], ctx))
    cost = account.snapshot()
    assert 0 < cost.communication < cost.tokens.total


def test_released_answers_survive_a_budget_cut_after_inhibition(tmp_path: Path) -> None:
    table = {(0, 0): 1, (1, 0): 1, (2, 0): 1, (3, 0): 2, (4, 0): 2}
    fake = FakeTransport().respond(SOLVE, answers(table))
    # 5 calls for the scouts, then nothing left: sigma = 1 releases everyone and rediscovery is refused.
    result, events = run(fake, tmp_path, Bee(model="m", sigma=1.0), calls=5)
    assert result.answer == {"x": 1} and result.status == "completed"
    stop = stop_of(events)
    assert stop["reason"] == "budget" and stop["from_released"] is True


def test_yoked_majority_never_overshoots_with_multi_call_episodes(tmp_path: Path) -> None:
    from bioalloc.llm.base import LLMReply, ToolCall, Usage

    tool_turn = LLMReply(
        content="",
        model="m",
        finish_reason="tool_calls",
        tool_calls=(ToolCall(id="t", name="calculate", arguments='{"expression": "1+1"}'),),
        usage=Usage(prompt_tokens=5, completion_tokens=5),
    )
    calls = iter(range(10_000))

    def respond(req: LLMRequest):  # type: ignore[no-untyped-def]
        # Every voter first calls the tool once, then answers: two calls per vote.
        return tool_turn if next(calls) % 2 == 0 else json.dumps({"x": 1})

    graph = TaskGraph(
        id="q",
        nodes=(Subtask(id="a", instruction="pick", allowed_tools=("calculate",)),),
        final_nodes=("a",),
    )
    from bioalloc.tasks.common import CALCULATOR

    fake = FakeTransport().respond(SOLVE, respond)
    llm = MeteredClient(
        fake, Account(Budget(max_tokens=10**7, max_model_calls=100)), "coordinator", "coordinator"
    )
    voter = MajorityVote(model="m", max_calls=5, name="majority_yoked")
    coord = Coordinator(
        {voter.name: voter},
        Constant(voter.name),
        CoordinatorConfig(max_attempts=1),
        tools={"calculate": CALCULATOR},
    )
    asyncio.run(coord.run(graph, Family(), llm, Tracer(tmp_path, "r")))
    assert len(fake.requests) == 5  # 2 + 2 + 1 (the last episode is cut at the allowance)


def test_yoked_majority_stops_when_every_request_is_rejected(tmp_path: Path) -> None:
    from bioalloc.llm.base import RequestRejected

    fake = FakeTransport().respond(SOLVE, lambda r: RequestRejected("bad request"))
    result, _ = run(fake, tmp_path, MajorityVote(model="m", max_calls=4, name="majority_yoked"))
    assert len(fake.requests) == 4 and result.status == "failed"


def test_bee_votes_on_intermediate_nodes_without_a_contract(tmp_path: Path) -> None:
    from bioalloc.tasks.gsm8k import GSM8K

    graph = TaskGraph(
        id="i",
        nodes=(Subtask(id="a", instruction="pick", written_by="decomposer"),),
        final_nodes=("a",),
    )
    fake = FakeTransport().respond(SOLVE, lambda r: json.dumps({"partial": 12}))
    llm = MeteredClient(
        fake, Account(Budget(max_tokens=10**6, max_model_calls=50)), "coordinator", "coordinator"
    )
    coord = Coordinator({"bee": Bee(model="m")}, Constant("bee"), CoordinatorConfig(max_attempts=1))
    result = asyncio.run(coord.run(graph, GSM8K(), llm, Tracer(tmp_path, "r")))
    assert result.status == "completed" and result.answer == {"partial": 12}


class GoalFamily(Family):
    """As Family, plus a goal-level check that confirms only {"x": 42} (unknown otherwise)."""

    class Goal:
        name = "goal"
        evidence = "goal"

        def evaluate(self, subtask, output, upstream):  # type: ignore[no-untyped-def]
            ok = isinstance(output, dict) and output.get("x") == 42
            return CheckResult(
                check="goal", verdict="pass" if ok else "unknown", evidence="goal", reason=""
            )

    def checks(self, s):  # type: ignore[no-untyped-def]
        return [self.Valid(), self.Goal()]


def run_goal(fake: FakeTransport, tmp: Path, executor) -> tuple[RunResult, list]:  # type: ignore[no-untyped-def, type-arg]
    llm = MeteredClient(
        fake, Account(Budget(max_tokens=10**7, max_model_calls=200)), "coordinator", "coordinator"
    )
    coord = Coordinator(
        {executor.name: executor}, Constant(executor.name), CoordinatorConfig(max_attempts=1)
    )
    tracer = Tracer(tmp, "r")
    result = asyncio.run(coord.run(GRAPH, GoalFamily(), llm, tracer))
    tracer.close()
    return result, list(read_trace(tmp))


MINORITY_CERTIFIED = {(0, 0): 5, (1, 0): 42, (2, 0): 5, (3, 0): 5, (4, 0): 7}


def test_bee_returns_a_certified_answer_at_once(tmp_path: Path) -> None:
    # D27: one scout's answer is goal-verified in round 0; no recruitment or quorum is needed.
    fake = FakeTransport().respond(SOLVE, answers(MINORITY_CERTIFIED))
    result, events = run_goal(fake, tmp_path, Bee(model="m"))
    assert result.answer == {"x": 42} and len(fake.requests) == 5
    assert stop_of(events)["reason"] == "certificate" and stop_of(events)["scout"] == 1


def test_bee_without_certificate_stopping_keeps_the_d14_rule(tmp_path: Path) -> None:
    table = {**MINORITY_CERTIFIED, **{(i, r): 5 for i in range(5) for r in range(1, 6)}}
    fake = (
        FakeTransport()
        .respond(SOLVE, answers(table))
        .respond(ASSESS, lambda r: json.dumps({"accept": True, "confidence": 0.9}))
    )
    _, events = run_goal(fake, tmp_path, Bee(model="m", stop_on_certificate=False))
    assert stop_of(events)["reason"] != "certificate" and len(fake.requests) > 5


def test_majority_returns_a_certified_vote_over_the_plurality(tmp_path: Path) -> None:
    fake = FakeTransport().respond(SOLVE, answers(MINORITY_CERTIFIED))
    result, events = run_goal(fake, tmp_path, MajorityVote(model="m"))
    assert result.answer == {"x": 42} and stop_of(events)["reason"] == "certificate"
    plain = FakeTransport().respond(SOLVE, answers(MINORITY_CERTIFIED))
    result, _ = run_goal(
        plain, tmp_path / "plain", MajorityVote(model="m", stop_on_certificate=False)
    )
    assert result.answer == {"x": 5}


def test_a_certificate_found_in_the_last_round_is_returned(tmp_path: Path) -> None:
    # Round 0: all rejected; round 1 (the last, max_rounds=1): scout 3 is goal-verified.
    table = {(i, 0): -1 for i in range(5)} | {(i, 1): 5 for i in range(5)} | {(3, 1): 42}
    fake = FakeTransport().respond(SOLVE, answers(table))
    result, events = run_goal(fake, tmp_path, Bee(model="m", max_rounds=1))
    assert result.answer == {"x": 42} and stop_of(events)["reason"] == "certificate"
