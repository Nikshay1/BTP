"""The coordinator: select, dispatch, check, then aggregate or reallocate.

The task graph comes in already decomposed (decision D5).

A subtask is dispatched as soon as all its dependencies are accepted. The policy picks an
executor; the executor runs under a child budget account that draws from the task's shared pool
(decision D7) and within the task's deadline. Its output is checked with the family's runtime
checks and either accepted or rejected. A rejected subtask is reallocated, with the rejected
output and the public reasons for rejection passed to the next attempt (decision D8), up to an
attempt allowance shared by every system. A subtask refused for lack of budget is not retried.
"""

from __future__ import annotations

import asyncio
import hashlib
import random
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Literal

from pydantic import JsonValue

from bioalloc.actions import ActionSpace
from bioalloc.budget import BudgetExceeded, MeteredClient
from bioalloc.checks import combine
from bioalloc.executors.base import (
    ExecutionContext,
    Executor,
    Feedback,
    Upstream,
    canonical_option,
)
from bioalloc.graph import TaskGraph
from bioalloc.llm.base import LLMError, TransportTimeout
from bioalloc.policies import Attempt, Policy
from bioalloc.schemas import (
    Budget,
    CheckResult,
    Cost,
    ExecutorResult,
    ExecutorStatus,
    Subtask,
    Wire,
)
from bioalloc.tasks.base import TaskFamily
from bioalloc.tools import Tool, ToolService
from bioalloc.trace import Tracer

RunStatus = Literal["completed", "failed", "budget_exhausted"]


class AttemptRecord(Wire):
    attempt: int
    result: ExecutorResult
    checks: tuple[CheckResult, ...]
    verdict: str
    accepted: bool


class RunResult(Wire):
    task_id: str
    status: RunStatus
    answer: JsonValue
    cost: Cost
    attempts: tuple[AttemptRecord, ...]
    span_id: str

    @property
    def within_budget(self) -> bool:
        return not self.cost.overrun


@dataclass
class CoordinatorConfig:
    max_attempts: int = 2
    # Cap for one dispatch. None: the dispatch draws from the task's shared pool (D7).
    dispatch_budget: Budget | None = None
    # Accept results whose checks can only say "unknown" (research plan, Section 4.4).
    accept_unknown: bool = True
    # Pass rejected attempts and the reasons to the next attempt (D8).
    retry_feedback: bool = True
    # Dispatch one subtask at a time in declaration (topological) order, e.g. for the single agent.
    serial: bool = False


def _action_space(family: TaskFamily, subtask: Subtask) -> ActionSpace | None:
    get = getattr(family, "action_space", None)
    return get(subtask) if get is not None else None


def _normalizer(family: TaskFamily, subtask: Subtask) -> Callable[[JsonValue], str | None]:
    get = getattr(family, "normalize", None)
    if get is None:
        return canonical_option
    return lambda output: get(subtask, output)


def derive_seed(*parts: object) -> int:
    digest = hashlib.sha256(":".join(map(str, parts)).encode()).digest()
    return int.from_bytes(digest[:4], "big")


@dataclass(frozen=True)
class _Solved:
    accepted: Upstream | None
    records: list[AttemptRecord]
    out_of_budget: bool


class Coordinator:
    def __init__(
        self,
        executors: dict[str, Executor],
        policy: Policy,
        config: CoordinatorConfig | None = None,
        seed: int = 0,
        tools: Mapping[str, Tool] | None = None,
    ) -> None:
        self.executors = executors
        self.policy = policy
        self.config = config or CoordinatorConfig()
        self.seed = seed
        self.tools = dict(tools or {})

    def _tools_for(self, family: TaskFamily) -> dict[str, Tool]:
        """The family's tools plus any given to the coordinator; every executor sees the same."""
        get = getattr(family, "tools", None)
        return {**(get() if get is not None else {}), **self.tools}

    async def run(
        self, graph: TaskGraph, family: TaskFamily, llm: MeteredClient, tracer: Tracer
    ) -> RunResult:
        root = tracer.span()
        tracer.emit(root, "run_start", {"task_id": graph.id, "policy": self.policy.name})
        done: dict[str, Upstream] = {}
        given_up: set[str] = set()
        records: list[AttemptRecord] = []
        out_of_budget = False
        active: dict[asyncio.Task[_Solved], Subtask] = {}
        task_state: dict[str, Any] = {}  # private to this run (D10)

        def launch() -> None:
            busy = given_up | {s.id for s in active.values()}
            ready = graph.ready(done, started=busy)
            if self.config.serial:
                ready = ready[:1] if not active else []
            for s in ready:
                task = asyncio.create_task(
                    self._solve(graph.id, s, family, done, llm, tracer, root, task_state)
                )
                active[task] = s

        try:
            launch()
            while active:
                finished, _ = await asyncio.wait(active, return_when=asyncio.FIRST_COMPLETED)
                # Retrieve every finished task before raising, so no exception goes unobserved.
                outcomes = [(active.pop(t), t) for t in finished]
                errors: list[BaseException] = []
                for sub, t in outcomes:
                    if t.cancelled():
                        errors.append(asyncio.CancelledError(f"worker for {sub.id} was cancelled"))
                    elif (exc := t.exception()) is not None:
                        errors.append(exc)
                if errors:
                    # Prefer a real failure over a cancellation when both happened together.
                    errors.sort(key=lambda e: isinstance(e, asyncio.CancelledError))
                    raise errors[0]
                for subtask, task in outcomes:
                    solved = task.result()
                    records += solved.records
                    out_of_budget |= solved.out_of_budget
                    if solved.accepted is None:
                        given_up.add(subtask.id)
                        blocked = sorted(graph.descendants(subtask.id))
                        tracer.emit(
                            root, "subtask_failed", {"subtask": subtask.id, "blocked": blocked}
                        )
                    else:
                        done[subtask.id] = solved.accepted
                launch()
        finally:
            # On cancellation or error, stop every worker and let it settle its accounting.
            for task in active:
                task.cancel()
            if active:
                await asyncio.gather(*active, return_exceptions=True)

        complete = all(f in done for f in graph.final_nodes)
        status: RunStatus = (
            "completed" if complete else "budget_exhausted" if out_of_budget else "failed"
        )
        answer = (
            family.aggregate(graph, {k: v.output for k, v in done.items()}) if complete else None
        )
        cost = llm.account.snapshot()
        tracer.emit(
            root,
            "run_end",
            {"status": status, "answer": answer, "cost": cost.model_dump(mode="json")},
        )
        order = {n.id: i for i, n in enumerate(graph.nodes)}
        records.sort(key=lambda r: (order[r.result.subtask_id], r.attempt))
        return RunResult(
            task_id=graph.id,
            status=status,
            answer=answer,
            cost=cost,
            attempts=tuple(records),
            span_id=root,
        )

    async def _solve(
        self,
        task_id: str,
        subtask: Subtask,
        family: TaskFamily,
        done: dict[str, Upstream],
        llm: MeteredClient,
        tracer: Tracer,
        root: str,
        task_state: dict[str, Any],
    ) -> _Solved:
        previous: list[Attempt] = []
        feedback: list[Feedback] = []
        records: list[AttemptRecord] = []
        upstream = {d: done[d] for d in subtask.depends_on}
        plain_upstream = {k: v.output for k, v in upstream.items()}
        for attempt in range(self.config.max_attempts):
            if llm.account.exhausted():
                return _Solved(None, records, True)
            name = self.policy.select(task_id, subtask, previous)
            executor = self.executors[name]
            span = tracer.span(parent=root)
            # A persistent executor (the single agent) keeps one identity across the task.
            agent = (
                name if getattr(executor, "persistent", False) else f"{name}:{subtask.id}:{attempt}"
            )
            limits = self.config.dispatch_budget or llm.account.limits
            child = llm.child(limits, actor="worker", agent=agent)
            child.on_event = tracer.hook(span)
            seed = derive_seed(self.seed, task_id, subtask.id, attempt)
            tracer.emit(
                span,
                "dispatch",
                {
                    "subtask": subtask.id,
                    "executor": name,
                    "attempt": attempt,
                    "seed": seed,
                    "limits": limits.model_dump(),
                    "feedback_items": len(feedback),
                },
            )
            ctx = ExecutionContext(
                llm=child,
                span_id=span,
                upstream=upstream,
                feedback=tuple(feedback) if self.config.retry_feedback else (),
                seed=seed,
                rng=random.Random(seed),
                emit=tracer.hook(span),
                checks=tuple(family.checks(subtask)),
                tools=(
                    ToolService(
                        self._tools_for(family),
                        subtask.allowed_tools,
                        child.account,
                        tracer.hook(span),
                    )
                    if subtask.allowed_tools
                    else None
                ),
                actions=_action_space(family, subtask),
                task_state=task_state,
                normalize=_normalizer(family, subtask),
            )
            output: JsonValue = None
            confidence: float | None = None
            incomplete: str | None = None
            status: ExecutorStatus
            try:
                outcome = await asyncio.wait_for(
                    executor.execute(subtask, ctx), child.account.time_left()
                )
                output, status, confidence = outcome.output, outcome.status, outcome.confidence
                incomplete = outcome.incomplete
            except BudgetExceeded:
                status = "budget_exhausted"
            except (TransportTimeout, TimeoutError):
                status = "timeout"
            except LLMError:
                status = "error"
            result = ExecutorResult(
                executor=name,
                subtask_id=subtask.id,
                output=output,
                confidence=confidence,
                cost=child.account.snapshot(),
                status=status,
                span_id=span,
            )
            checks: tuple[CheckResult, ...] = ()
            if status == "completed":
                checks = tuple(
                    c.evaluate(subtask, output, plain_upstream) for c in family.checks(subtask)
                )
                if incomplete is not None:
                    checks += (
                        CheckResult(
                            check="generation",
                            verdict="fail",
                            reason=incomplete,
                            evidence="operational",
                        ),
                    )
                verdict = combine(list(checks))
            else:
                verdict = "fail"
            accepted = verdict == "pass" or (verdict == "unknown" and self.config.accept_unknown)
            record = AttemptRecord(
                attempt=attempt, result=result, checks=checks, verdict=verdict, accepted=accepted
            )
            records.append(record)
            tracer.emit(span, "attempt_end", record.model_dump(mode="json"))
            summary = Attempt(
                subtask_id=subtask.id,
                attempt=attempt,
                executor=name,
                status=status,
                verdict=verdict,
                accepted=accepted,
                checks=checks,
                confidence=confidence,
                tokens=result.cost.tokens.total,
            )
            self.policy.update(subtask, summary)
            if accepted:
                return _Solved(
                    Upstream(subtask_id=subtask.id, agent=agent, output=output), records, False
                )
            if status == "budget_exhausted":
                return _Solved(None, records, True)
            previous.append(summary)
            if status == "completed":
                reasons = tuple(
                    f"{c.check}: {c.reason or c.verdict}" for c in checks if c.verdict != "pass"
                )
                feedback.append(Feedback(agent=agent, output=output, reasons=reasons))
        return _Solved(None, records, llm.account.exhausted())
