"""Scouts that repair their own rejected answers (D28)."""

import json
import re
from pathlib import Path

from test_bee import run_goal, stop_of

from bioalloc.executors.repair import ALL_START_THREE, SHORT_MEMORY, STAGED_TWO, ScoutRepair
from bioalloc.llm.base import LLMReply, LLMRequest, Usage
from bioalloc.llm.fake import FakeTransport

SOLVE = lambda r: "has proposed an answer" not in r.messages[0].content  # noqa: E731


def episode_of(req: LLMRequest) -> tuple[int, int]:
    m = re.search(r"/scout(\d+)/e(\d+)(?:/|$)", req.sample_id)
    assert m, req.sample_id
    return int(m.group(1)), int(m.group(2))


def answers(table: dict[tuple[int, int], int]):  # type: ignore[no-untyped-def]
    return lambda req: json.dumps({"x": table[episode_of(req)]})


def order(fake: FakeTransport) -> list[tuple[int, int]]:
    return [episode_of(r) for r in fake.requests]


def test_a_revision_sees_only_its_latest_answer_and_feedback(tmp_path: Path) -> None:
    fake = FakeTransport().respond(SOLVE, answers({(0, 0): -1, (0, 1): -2, (0, 2): 42}))
    result, events = run_goal(fake, tmp_path, ScoutRepair(model="m", schedule=SHORT_MEMORY))
    assert result.answer == {"x": 42} and order(fake) == [(0, 0), (0, 1), (0, 2)]
    third = fake.requests[2].messages
    assert third[-2].role == "assistant" and third[-2].content == '{"x": -2}'
    assert third[-1].content.startswith("The checks rejected that answer: valid: negative")
    assert not any(m.content == '{"x": -1}' for m in third)  # the older attempt is gone
    assert fake.requests[0].messages[-1].role == "user"  # the first answer is independent
    assert stop_of(events)["reason"] == "certificate"


def test_the_staged_order_launches_the_second_scout_after_two_failures(tmp_path: Path) -> None:
    fake = FakeTransport().respond(SOLVE, answers({(0, 0): -1, (0, 1): -1, (1, 0): 42}))
    result, _ = run_goal(fake, tmp_path, ScoutRepair(model="m", schedule=STAGED_TWO))
    assert result.answer == {"x": 42}
    assert order(fake) == [(0, 0), (0, 1), (1, 0)]  # nothing is launched after the certificate
    second_scout = fake.requests[2].messages
    assert not any(m.role == "assistant" for m in second_scout)  # no other scout's attempts


def test_bee_stops_at_quorum_and_the_plurality_control_runs_the_schedule(tmp_path: Path) -> None:
    # Two committed, unverified answers of 5 reach quorum ceil(0.6 * 3) = 2 for bee.
    table = {(0, 0): 5, (1, 0): 5, (2, 0): 7}
    bee = FakeTransport().respond(SOLVE, answers(table))
    result, events = run_goal(
        bee, tmp_path / "bee", ScoutRepair(model="m", schedule=ALL_START_THREE)
    )
    assert result.answer == {"x": 5} and len(bee.requests) == 2
    assert stop_of(events)["reason"] == "quorum"
    plain = FakeTransport().respond(SOLVE, answers(table))
    result, events = run_goal(
        plain,
        tmp_path / "plain",
        ScoutRepair(model="m", schedule=ALL_START_THREE, consensus="plurality", name="mj"),
    )
    # Every scout committed after its first episode, so the rest of the schedule is skipped.
    assert result.answer == {"x": 5} and len(plain.requests) == 3
    assert stop_of(events) == {"reason": "votes", "tally": {'{"x":5}': 2, '{"x":7}': 1}}


def test_a_cut_off_reply_gets_the_last_step_failed_feedback(tmp_path: Path) -> None:
    replies = iter(
        [
            LLMReply(
                content='{"x": ',
                finish_reason="length",
                usage=Usage(prompt_tokens=5, completion_tokens=5),
                model="m",
            ),
            LLMReply(
                content='{"x": 42}', usage=Usage(prompt_tokens=5, completion_tokens=5), model="m"
            ),
        ]
    )
    fake = FakeTransport().respond(SOLVE, lambda r: next(replies))
    result, _ = run_goal(fake, tmp_path, ScoutRepair(model="m", schedule=SHORT_MEMORY))
    assert result.answer == {"x": 42}
    assert fake.requests[1].messages[-1].content == (
        "Your last step failed: the reply was cut off at the completion limit."
    )


def test_every_registered_candidate_has_a_matched_control() -> None:
    from bioalloc.executors.repair import repair_executors

    pool = repair_executors("m")
    assert pool["bee_staged"].schedule == pool["majority_staged"].schedule == STAGED_TWO
    assert pool["bee_feedback3"].schedule == pool["majority_feedback3"].schedule
    assert {e.consensus for n, e in pool.items() if n.startswith("majority")} == {"plurality"}
    assert pool["bee_staged"].quorum == 2 and pool["bee_feedback3"].quorum == 2


class Tracking:
    """Wraps a fake transport and records the most calls in flight at once."""

    def __init__(self, fake: FakeTransport) -> None:
        self.fake, self.inflight, self.peak = fake, 0, 0

    async def send(self, req: LLMRequest) -> LLMReply:
        import asyncio

        self.inflight += 1
        self.peak = max(self.peak, self.inflight)
        await asyncio.sleep(0.005)
        try:
            return await self.fake.send(req)
        finally:
            self.inflight -= 1


ASSESS = lambda r: "has proposed an answer" in r.messages[0].content  # noqa: E731


def test_recruitment_assessments_run_one_at_a_time(tmp_path: Path) -> None:
    # R15 (Sol): inherited recruitment gathered assessments concurrently.
    rejected = {(i, e): -1 for i in (1, 2) for e in (0, 1)}
    fake = (
        FakeTransport()
        .respond(SOLVE, answers({(0, 0): 5, **rejected}))
        .respond(ASSESS, lambda r: json.dumps({"accept": True, "confidence": 0.9}))
    )
    tracking = Tracking(fake)
    result, events = run_goal(
        tracking,  # type: ignore[arg-type]
        tmp_path,
        ScoutRepair(model="m", schedule=ALL_START_THREE, sigma=0.0),
    )
    assessments = [r for r in fake.requests if ASSESS(r)]
    assert len(assessments) == 2 and tracking.peak == 1
    assert result.answer == {"x": 5} and stop_of(events)["reason"] == "quorum"


def test_releasing_every_supporter_ends_the_run(tmp_path: Path) -> None:
    # R15: D14 would restart all scouts; D28 has no episodes left, so it stops explicitly.
    table = {(0, 0): 5, (1, 0): 7, (2, 0): -1, (2, 1): -1}
    fake = FakeTransport().respond(SOLVE, answers(table))
    result, events = run_goal(
        fake, tmp_path, ScoutRepair(model="m", schedule=ALL_START_THREE, sigma=1.0)
    )
    stop = stop_of(events)
    assert stop["reason"] == "all_released" and stop["from_released"]
    assert result.answer in ({"x": 5}, {"x": 7})


def test_configs_using_repair_scouts_run_one_attempt_per_task() -> None:
    # R15: the six-episode ceiling is per dispatch; one outer attempt makes it per task.
    from bioalloc.executors.repair import repair_executors

    names = set(repair_executors("m"))
    for path in (Path(__file__).resolve().parents[1] / "configs").glob("*.json"):
        cfg = json.loads(path.read_text())
        if names & set(cfg.get("systems", [])):
            assert cfg.get("max_attempts") == 1, path.name
