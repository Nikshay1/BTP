"""The coordinator: select, dispatch, check, then aggregate or reallocate.

The task graph comes in already decomposed. The decomposition of each task is generated once,
stored, and replayed to every system so that all of them solve identical subtasks (decision D5).

For each ready subtask the policy picks an executor. The executor runs under a child budget
account, its output is checked with the family's runtime checks, and the result is accepted or the
subtask is reallocated, with the failed attempt passed to the policy. Every system gets the same
attempt allowance. Ready subtasks run concurrently.
"""

from __future__ import annotations

import asyncio
import random
from dataclasses import dataclass
from typing import Literal

from pydantic import JsonValue

from bioalloc.budget import BudgetExceeded, MeteredClient
from bioalloc.checks import combine
from bioalloc.executors.base import ExecutionContext, Executor, Upstream
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


@dataclass
class CoordinatorConfig:
    max_attempts: int = 2
    # Cap for one dispatch; the remaining task budget caps it further.
    dispatch_budget: Budget | None = None
    # Accept results whose checks can only say "unknown" (research plan, Section 4.4).
    accept_unknown: bool = True
    # Weight of normalized token cost in the policy reward s - lambda * c (Section 4.3).
    cost_weight: float = 0.0


class Coordinator:
    def __init__(
        self,
        executors: dict[str, Executor],
        policy: Policy,
        config: CoordinatorConfig | None = None,
        seed: int = 0,
    ) -> None:
        self.executors = executors
        self.policy = policy
        self.config = config or CoordinatorConfig()
        self.seed = seed

    async def run(
        self, graph: TaskGraph, family: TaskFamily, llm: MeteredClient, tracer: Tracer
    ) -> RunResult:
        root = tracer.span()
        tracer.emit(root, "run_start", {"task_id": graph.id, "policy": self.policy.name})
        done: dict[str, Upstream] = {}
        given_up: set[str] = set()
        records: list[AttemptRecord] = []
        out_of_budget = False

        while True:
            ready = graph.ready(done, started=given_up)
            if not ready:
                break
            results = await asyncio.gather(
                *(self._solve(s, family, done, llm, tracer, root) for s in ready)
            )
            for subtask, (accepted, attempts, exhausted) in zip(ready, results, strict=True):
                records += attempts
                out_of_budget |= exhausted
                if accepted is None:
                    given_up.add(subtask.id)
                    blocked = sorted(graph.descendants(subtask.id))
                    tracer.emit(root, "subtask_failed", {"subtask": subtask.id, "blocked": blocked})
                else:
                    done[subtask.id] = accepted

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
        subtask: Subtask,
        family: TaskFamily,
        done: dict[str, Upstream],
        llm: MeteredClient,
        tracer: Tracer,
        root: str,
    ) -> tuple[Upstream | None, list[AttemptRecord], bool]:
        previous: list[Attempt] = []
        records: list[AttemptRecord] = []
        upstream = {d: done[d] for d in subtask.depends_on}
        for attempt in range(self.config.max_attempts):
            if llm.account.exhausted():
                return None, records, True
            name = self.policy.select(subtask, previous)
            executor = self.executors[name]
            span = tracer.span(parent=root)
            agent = f"{name}:{subtask.id}:{attempt}"
            limits = self.config.dispatch_budget or llm.account.remaining()
            child = llm.child(limits, actor="worker", agent=agent)
            child.on_event = tracer.hook(span)
            tracer.emit(
                span,
                "dispatch",
                {
                    "subtask": subtask.id,
                    "executor": name,
                    "attempt": attempt,
                    "limits": limits.model_dump(),
                },
            )
            ctx = ExecutionContext(
                llm=child,
                span_id=span,
                upstream=upstream,
                rng=random.Random(f"{self.seed}:{subtask.id}:{attempt}"),
                emit=tracer.hook(span),
            )
            output: JsonValue = None
            confidence: float | None = None
            status: ExecutorStatus
            try:
                outcome = await executor.execute(subtask, ctx)
                output, status, confidence = outcome.output, outcome.status, outcome.confidence
            except BudgetExceeded:
                status = "budget_exhausted"
            except TransportTimeout:
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
                plain_upstream = {k: v.output for k, v in upstream.items()}
                checks = tuple(
                    c.evaluate(subtask, output, plain_upstream) for c in family.checks(subtask)
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
            self.policy.update(subtask, name, self._reward(accepted, result.cost, llm))
            if accepted:
                return Upstream(subtask_id=subtask.id, agent=agent, output=output), records, False
            previous.append(Attempt(executor=name, status=status, verdict=verdict))
        # A dispatch refused for lack of tokens also means the task ran out of budget.
        refused = any(r.result.status == "budget_exhausted" for r in records)
        return None, records, refused or llm.account.exhausted()

    def _reward(self, accepted: bool, cost: Cost, llm: MeteredClient) -> float:
        cap = llm.account.limits.max_tokens or 1
        return float(accepted) - self.config.cost_weight * cost.tokens.total / cap
