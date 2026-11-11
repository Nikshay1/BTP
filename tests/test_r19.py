"""R19: a provider throttle must never be recorded as a task failure, and a resume accepts only
changes from an allow list or pinned by a review."""

import asyncio
import json
from pathlib import Path

import httpx
import pytest
from test_groq import ok_body, req, transport

from bioalloc.budget import Account, BudgetExceeded, MeteredClient
from bioalloc.executors.bee import _gather_budgeted
from bioalloc.llm.base import LLMError, ProviderThrottled
from bioalloc.schemas import Budget


def test_bee_gather_raises_a_throttle_at_once_and_settles_hung_scouts() -> None:
    cancelled: list[str] = []

    async def throttled() -> None:
        raise ProviderThrottled("Too many tokens per day")

    async def hung() -> None:
        try:
            await asyncio.sleep(3600)
        except asyncio.CancelledError:
            cancelled.append("hung")
            raise

    async def go() -> None:
        with pytest.raises(ProviderThrottled):
            await asyncio.wait_for(_gather_budgeted([hung(), throttled()]), 5)

    asyncio.run(go())
    assert cancelled == ["hung"]


def test_bee_gather_still_turns_budget_and_llm_errors_into_no_answer() -> None:
    async def budget() -> None:
        raise BudgetExceeded("x")

    async def err() -> None:
        raise LLMError("x")

    async def ok() -> int:
        return 7

    out, hit = asyncio.run(_gather_budgeted([budget(), err(), ok()]))
    assert out == [None, None, 7] and hit


class Clock:
    def __init__(self) -> None:
        self.t = 0.0

    def __call__(self) -> float:
        return self.t


def _client(handler, clock: Clock, sleep):  # type: ignore[no-untyped-def]
    t, _ = transport(handler)
    account = Account(Budget(max_tokens=10_000, max_model_calls=5, max_elapsed_s=10), clock=clock)
    return MeteredClient(t, account, "worker", "w", sleep=sleep), account


def test_a_backoff_that_overruns_the_deadline_is_a_throttle_not_a_budget_failure() -> None:
    clock = Clock()

    async def late_sleep(s: float) -> None:
        clock.t += 11  # scheduling delay carries the wait past the 10 s deadline

    replies = iter(
        [httpx.Response(429, headers={"retry-after": "1"}), httpx.Response(200, json=ok_body())]
    )
    client, account = _client(lambda r: next(replies), clock, late_sleep)
    with pytest.raises(ProviderThrottled):
        asyncio.run(client.complete(req()))
    assert account.throttled is not None


def test_cancellation_during_a_backoff_marks_the_run_throttled() -> None:
    clock = Clock()

    async def go() -> None:
        client, account = _client(
            lambda r: httpx.Response(429, headers={"retry-after": "1"}), clock, asyncio.sleep
        )
        task = asyncio.ensure_future(client.complete(req()))
        await asyncio.sleep(0.2)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert account.throttled is not None

    asyncio.run(go())


def test_a_successful_retry_leaves_the_run_unmarked() -> None:
    clock = Clock()

    async def sleep(s: float) -> None:
        clock.t += s

    replies = iter(
        [httpx.Response(429, headers={"retry-after": "1"}), httpx.Response(200, json=ok_body())]
    )
    client, account = _client(lambda r: next(replies), clock, sleep)
    asyncio.run(client.complete(req()))
    assert account.throttled is None


def test_resume_guard_is_an_allow_list_with_pinned_reviews(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    from test_experiment import ROOT, config

    from bioalloc.evaluation import experiment as ex
    from bioalloc.evaluation.experiment import Experiment, ExperimentMismatch
    from bioalloc.evaluation.rehearsal import RehearsalTransport

    cfg = config(["direct"])
    asyncio.run(Experiment(cfg, RehearsalTransport(), tmp_path, root=ROOT).run())
    prov = tmp_path / "t" / "provenance.json"
    recorded = json.loads(prov.read_text())
    recorded["code"]["git_commit"] = "0" * 40
    prov.write_text(json.dumps(recorded))
    reviews = {"fix": {"files": {"src/bioalloc/budget.py": "b" * 40}}}

    def fake_git(changed: str, blob: str):  # type: ignore[no-untyped-def]
        def git(*a: str) -> str | None:
            if a[0] == "diff":
                return changed
            if a[0] == "show":
                return json.dumps(reviews)
            if a[0] == "rev-parse":
                return blob
            return None

        return git

    def resume(reason: str) -> None:
        exp = Experiment(cfg, RehearsalTransport(), tmp_path, root=ROOT, accept_code_change=reason)
        asyncio.run(exp.run())

    # An unlisted file (the old deny list missed these) is refused.
    for f in ("src/bioalloc/evaluation/splits.py", "pyproject.toml", "src/bioalloc/new_module.py"):
        monkeypatch.setattr(ex, "_git", fake_git(f"{f}\ndocs/x.md", "b" * 40))
        with pytest.raises(ExperimentMismatch, match="how tasks are solved"):
            resume("fix")
    # A pinned file is accepted only at its reviewed content and under its review's name.
    monkeypatch.setattr(ex, "_git", fake_git("src/bioalloc/budget.py", "c" * 40))
    with pytest.raises(ExperimentMismatch, match="how tasks are solved"):
        resume("fix")
    monkeypatch.setattr(ex, "_git", fake_git("src/bioalloc/budget.py", "b" * 40))
    with pytest.raises(ExperimentMismatch, match="how tasks are solved"):
        resume("another reason")
    resume("fix")
    monkeypatch.setattr(ex, "_git", fake_git("docs/decisions.md\ntests/test_x.py", "b" * 40))
    resume("docs")


class LimitedThenHangs:
    """429 on the first request; the retried request never returns."""

    def __init__(self) -> None:
        self.calls = 0

    async def send(self, request):  # type: ignore[no-untyped-def]
        from bioalloc.llm.base import RateLimited

        self.calls += 1
        if self.calls == 1:
            raise RateLimited("Too many requests", retry_after=1)
        await asyncio.sleep(3600)


def test_a_retry_that_runs_into_the_deadline_after_a_429_is_a_throttle() -> None:
    # R20 (found by Sol): the retried call was sent and then timed out at the deadline.
    import time

    offset = [0.0]

    def clock() -> float:
        return time.monotonic() + offset[0]

    async def sleep(s: float) -> None:
        offset[0] += 9.7  # the backoff leaves 0.3 s of a 10 s budget

    account = Account(Budget(max_tokens=10_000, max_model_calls=5, max_elapsed_s=10), clock=clock)
    client = MeteredClient(LimitedThenHangs(), account, "worker", "w", sleep=sleep)  # type: ignore[arg-type]
    with pytest.raises(ProviderThrottled):
        asyncio.run(client.complete(req()))
    assert account.throttled is not None


def test_cancelling_a_retried_call_after_a_429_marks_the_run_throttled() -> None:
    async def go() -> None:
        async def sleep(s: float) -> None:
            return None

        account = Account(Budget(max_tokens=10_000, max_model_calls=5))
        client = MeteredClient(LimitedThenHangs(), account, "worker", "w", sleep=sleep)  # type: ignore[arg-type]
        task = asyncio.ensure_future(client.complete(req()))
        await asyncio.sleep(0.2)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert account.throttled is not None

    asyncio.run(go())


_AIME = Path(__file__).resolve().parents[1] / "data/prepared/aime/public.jsonl"


@pytest.mark.skipif(not _AIME.exists(), reason="AIME snapshot not prepared")
def test_aime_tasks_load_and_grade_exactly() -> None:
    from test_experiment import ROOT

    from bioalloc.evaluation import aime_data
    from bioalloc.evaluation.experiment import select_tasks
    from bioalloc.evaluation.registry import TaskLoader
    from bioalloc.evaluation.splits import load as load_split

    items = aime_data.load()
    assert len(items) == 178 and all(0 <= int(i.reference) <= 999 for i in items)
    split = load_split(ROOT / "data/splits/aime-v1.json")
    assert sum(a.split == "tune" for a in split) == 30
    task = TaskLoader().load(items[0].task.id)
    assert task.grade({"answer": int(items[0].reference)}).correct
    assert not task.grade({"answer": int(items[0].reference) + 1}).correct
    assert select_tasks  # imported for the config check below


def test_quorum_bee_matches_its_plurality_control_on_the_schedule() -> None:
    from bioalloc.executors.repair import repair_executors

    ex = repair_executors("m")
    assert ex["bee_quorum5"].schedule == ex["majority5"].schedule == (0, 1, 2, 3, 4)
    assert ex["bee_quorum5"].quorum == 3 and ex["bee_quorum3"].quorum == 2
