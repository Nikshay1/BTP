"""Experiment runner: tasks x repetitions x systems, with shared stored decompositions.

For each task and repetition the decomposition is prepared once (natural graph, single node, or
the model decomposer, decision D17), stored, and replayed to every system as a fresh copy whose
cost is imported into the system's budget. Systems run on the same task in an order shuffled per
task and repetition, to spread provider drift over time. Every run writes its own trace and
manifest; one line per run goes to ``results.jsonl``.
"""

from __future__ import annotations

import asyncio
import fcntl
import json
import os
import random
from collections.abc import Callable
from pathlib import Path
from typing import Literal

from pydantic import Field, JsonValue

from bioalloc import prompts
from bioalloc.budget import Account, MeteredClient, Pricing
from bioalloc.coordinator import Coordinator, CoordinatorConfig, derive_seed
from bioalloc.decomposition import (
    Decomposition,
    DecompositionStore,
    deterministic,
    model_decomposition,
    start_from,
)
from bioalloc.evaluation import splits
from bioalloc.evaluation.registry import (
    NEEDS_TRAINING,
    ModelSettings,
    StudyTask,
    TaskLoader,
    build_system,
)
from bioalloc.llm.base import ProviderThrottled, RunMode, Transport
from bioalloc.schemas import Budget, Wire
from bioalloc.trace import (
    Tracer,
    code_revision,
    finish_manifest,
    fsync_dir,
    read_trace,
    sha256_text,
    start_manifest,
    usd_from_trace,
)

DecomposeMode = Literal["natural", "single", "model"]


class TaskSelection(Wire):
    split_file: str = "data/splits/v2.json"
    split: str = "train"
    per_family: dict[str, int] = {}  # e.g. {"planning": 10, "tools": 10, "reasoning": 10}
    task_ids: list[str] = []  # explicit ids override the per-family selection


class ExperimentConfig(Wire):
    name: str
    mode: RunMode = "measurement"
    model: str
    temperature: float = 0.6
    reasoning_effort: str | None = "low"
    max_completion_tokens: int = 2048
    pricing: dict[str, float] = {"input_per_m": 0.0, "output_per_m": 0.0}
    systems: list[str]
    tasks: TaskSelection
    repetitions: int = Field(default=3, ge=1)
    seed: int = 0
    budget: Budget
    max_attempts: int = 2
    # How each family's graph is made: tools come with a natural graph; for planning a single node
    # is preferred to an unvalidated model split (D21); reasoning uses the model decomposer.
    decompose: dict[str, DecomposeMode] = {
        "planning": "single",
        "tools": "natural",
        "reasoning": "model",
    }
    concurrency: int = Field(default=4, ge=1)


class RunRecord(Wire):
    experiment: str
    task_id: str
    family: str
    split: str
    system: str
    repetition: int
    status: str
    correct: bool
    grade_detail: str
    within_budget: bool
    usage_complete: bool
    tokens: int
    coordinator_tokens: int
    worker_tokens: int
    communication_tokens: int
    reasoning_tokens: int
    model_calls: int
    tool_calls: int
    usd: float
    elapsed_s: float
    decomposition_status: str
    run_dir: str


def select_tasks(sel: TaskSelection, root: Path) -> list[tuple[str, str]]:
    """(task id, split) pairs, in a fixed order."""
    assignments = splits.load(root / sel.split_file)
    if sel.task_ids:
        known = {a.task_id: a.split for a in assignments}
        return [(t, known[t]) for t in sel.task_ids]
    chosen: list[tuple[str, str]] = []
    for family, n in sorted(sel.per_family.items()):
        pool = sorted(a.task_id for a in assignments if a.family == family and a.split == sel.split)
        if len(pool) < n:
            raise ValueError(f"only {len(pool)} {family} tasks in split {sel.split}")
        chosen += [(t, sel.split) for t in pool[:n]]
    return chosen


class ExperimentMismatch(RuntimeError):
    """An experiment directory holds runs made under a different configuration or code."""


class ProviderUnavailable(RuntimeError):
    """The provider throttled or a quota ran out (R18). Runs it interrupted were not recorded;
    running the experiment again later resumes them, and finished runs are kept."""


class SpendingLimitReached(RuntimeError):
    pass


class Experiment:
    """Resumable: finished runs are skipped. Runs interrupted before their result was recorded are
    moved aside (their trace kept) and run again. Resuming requires the same configuration,
    prompts, split file and code; a populated directory without that record is refused. One runner
    process at a time may use a directory (``.lock``).

    Actual spending is computed from the per-call traces, which are written as calls happen: every
    model call that returned (or was charged as unknown usage) in any decomposition or run trace,
    interrupted ones included, counted once. Imported decomposition costs are not calls in a run's
    trace, so each decomposition is paid once. With ``max_usd``, no new decomposition or run starts
    once actual spending reaches the limit; runs already in flight finish (overshoot at most
    ``concurrency`` runs), and the run then stops with ``SpendingLimitReached``. A call in flight at
    a crash has no reply event and is not counted, so a crash can undercount by those calls."""

    def __init__(
        self,
        config: ExperimentConfig,
        transport: Transport,
        out: Path,
        root: Path | None = None,
        max_usd: float | None = None,
    ) -> None:
        from bioalloc.evaluation.data import REPO_ROOT

        self.config = config
        self.transport = transport
        self.root = root or REPO_ROOT
        self.out = out / config.name
        self.max_usd = max_usd
        split_payload = json.loads((self.root / config.tasks.split_file).read_text())
        self.split_sha256 = str(split_payload["sha256"])
        self.loader = TaskLoader(expected=split_payload.get("provenance"))
        self.store = DecompositionStore(self.out / "decompositions")
        self.models = ModelSettings(
            config.model, config.temperature, config.reasoning_effort, config.max_completion_tokens
        )
        self.pricing = Pricing(**config.pricing)
        self._lock = asyncio.Lock()
        self._spent: float | None = None
        self._stopped = False
        self._throttled = False

    def _client(
        self, account: Account, agent: str, tracer: Tracer | None, span: str | None
    ) -> MeteredClient:
        return MeteredClient(
            self.transport,
            account,
            actor="coordinator",
            agent=agent,
            pricing=self.pricing,
            mode=self.config.mode,
            on_event=tracer.hook(span) if tracer and span else None,
        )

    # -- provenance and spending ------------------------------------------------------------

    def provenance(self) -> dict[str, JsonValue]:
        return {
            "config": json.loads(self.config.model_dump_json()),
            "code": code_revision(),
            "prompts": {k: sha256_text(v) for k, v in sorted(prompts.all_prompts().items())},
            "split_sha256": self.split_sha256,
        }

    def check_provenance(self) -> None:
        now = self.provenance()
        code = now["code"]
        assert isinstance(code, dict)
        if self.config.mode == "measurement" and code.get("git_dirty"):
            raise ExperimentMismatch("measured runs need committed code (the work tree is dirty)")
        path = self.out / "provenance.json"
        if path.exists():
            before = json.loads(path.read_text())
            diff = sorted(k for k in now if before.get(k) != now[k])
            if diff:
                raise ExperimentMismatch(
                    f"{self.out} was run with a different {', '.join(diff)}; use a new experiment "
                    "name instead of mixing runs"
                )
        else:
            populated = [
                n for n in ("results.jsonl", "runs", "decompositions") if (self.out / n).exists()
            ]
            if populated:
                raise ExperimentMismatch(
                    f"{self.out} already holds {', '.join(populated)} but no provenance record; "
                    "its runs cannot be shown to match this configuration"
                )
            path.write_text(json.dumps(now, indent=1) + "\n")

    def _trace_usd(self, trace_dir: Path) -> float:
        if not (trace_dir / "trace.jsonl").exists():
            return 0.0
        events = list(read_trace(trace_dir, tolerate_torn_tail=True))
        return usd_from_trace(events, self.pricing.input_per_m, self.pricing.output_per_m)

    def spent(self) -> float:
        """Actual spending so far, from every trace under the experiment directory."""
        if self._spent is None:
            self._spent = sum(self._trace_usd(p.parent) for p in self.out.rglob("trace.jsonl"))
        return self._spent

    def _add_spending(self, trace_dir: Path) -> None:
        # The total was taken from a scan at the start of the run (before this trace existed).
        assert self._spent is not None, "spending must be scanned before new traces are written"
        self._spent += self._trace_usd(trace_dir)

    def _over_limit(self) -> bool:
        if self.max_usd is not None and self.spent() >= self.max_usd:
            self._stopped = True
        return self._stopped

    def _set_aside(self, run_dir: Path) -> None:
        """Move an interrupted run out of the way, keeping its trace (its calls stay counted)."""
        n = 0
        while (aside := run_dir.with_name(f"{run_dir.name}.interrupted.{n}")).exists():
            n += 1
        run_dir.rename(aside)
        fsync_dir(aside.parent)

    def _acquire(self) -> None:
        """An OS-level exclusive lock on ``.lock`` for the life of this runner. The kernel releases
        it if the process dies, so there is no stale-lock guessing and nothing is unlinked."""
        self._lock_file = (self.out / ".lock").open("a+", encoding="utf-8")
        try:
            fcntl.flock(self._lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            self._lock_file.seek(0)
            holder = self._lock_file.read().strip() or "unknown"
            self._lock_file.close()
            raise ExperimentMismatch(f"another runner (pid {holder}) is using {self.out}") from None
        self._lock_file.seek(0)
        self._lock_file.truncate()
        self._lock_file.write(str(os.getpid()))
        self._lock_file.flush()

    def _release(self) -> None:
        fcntl.flock(self._lock_file, fcntl.LOCK_UN)
        self._lock_file.close()

    # -- decompositions and runs --------------------------------------------------------------

    async def decomposition(self, task: StudyTask, rep: int) -> Decomposition:
        """Prepared once per task and repetition, before any system runs on it."""
        path = self.store.path(task.task_id, rep)
        if path.exists():
            try:
                return self.store.load(task.task_id, rep)
            except ValueError:  # left corrupt by a crash in an older version: redo it
                self._set_aside(path)
        trace_dir = self.out / "decompositions" / task.task_id / f"rep{rep}-trace"
        if trace_dir.exists():
            self._set_aside(trace_dir)
        mode = self.config.decompose.get(task.family, "single")
        if task.natural_graph is not None:
            record = deterministic(task.natural_graph, rep)
        elif mode == "model" and task.public is not None:
            run_dir = self.out / "decompositions" / task.task_id / f"rep{rep}-trace"
            tracer = Tracer(run_dir, f"{task.task_id}-decompose-{rep}")
            span = tracer.span()
            account = Account(self.config.budget)
            client = self._client(account, "decomposer", tracer, span)
            record = await model_decomposition(
                task.public,
                rep,
                client,
                model=self.config.model,
                max_completion_tokens=self.config.max_completion_tokens,
                temperature=self.config.temperature,
                reasoning_effort=self.config.reasoning_effort,  # type: ignore[arg-type]
                seed=derive_seed(self.config.seed, task.task_id, rep, "decompose"),
            )
            tracer.close()
        else:
            assert task.public is not None
            record = deterministic(task.public.as_single_node(), rep)
        self.store.save(record)
        self._add_spending(trace_dir)
        return record

    async def run_one(
        self, task: StudyTask, split: str, rep: int, system_name: str, record: Decomposition
    ) -> RunRecord:
        cfg = self.config
        base = CoordinatorConfig(max_attempts=cfg.max_attempts)
        seed = derive_seed(cfg.seed, rep)
        system = build_system(system_name, self.models, base, seed)
        run_dir = self.out / "runs" / system_name / task.task_id / f"rep{rep}"
        run_id = f"{cfg.name}/{system_name}/{task.task_id}/rep{rep}"
        if run_dir.exists():  # started before but never recorded: interrupted
            self._set_aside(run_dir)
        manifest = start_manifest(
            run_dir,
            run_id,
            mode=cfg.mode,
            config={
                "experiment": json.loads(cfg.model_dump_json()),
                "system": system_name,
                "task": task.task_id,
                "repetition": rep,
                "graph_sha256": record.graph_sha256,
                "decomposition_status": record.status,
            },
            model=cfg.model,
            seeds={"coordinator": seed, "repetition": rep},
            prompts=prompts.all_prompts(),
        )
        tracer = Tracer(run_dir, run_id)
        account = Account(cfg.budget)
        graph = start_from(record, account)
        tracer.emit(
            tracer.span(),
            "cost_import",
            {"source": record.graph_sha256, "cost": record.cost.model_dump(mode="json")},
        )
        llm = self._client(account, "coordinator", None, None)
        coordinator = Coordinator(system.executors, system.policy, system.config, seed=seed)
        try:
            result = await coordinator.run(graph, task.task_family, llm, tracer)
        except BaseException:
            tracer.close()
            finish_manifest(run_dir, manifest, "crashed", {})
            raise
        tracer.close()
        self._add_spending(run_dir)
        grade = task.grade(result.answer)
        c = result.cost
        finish_manifest(
            run_dir,
            manifest,
            result.status,
            {"correct": grade.correct, "within_budget": result.within_budget},
        )
        return RunRecord(
            experiment=cfg.name,
            task_id=task.task_id,
            family=task.family,
            split=split,
            system=system_name,
            repetition=rep,
            status=result.status,
            correct=grade.correct,
            grade_detail=grade.detail,
            within_budget=result.within_budget,
            usage_complete=c.usage_complete,
            tokens=c.tokens.total,
            coordinator_tokens=c.coordinator.total,
            worker_tokens=c.worker.total,
            communication_tokens=c.communication,
            reasoning_tokens=c.reasoning,
            model_calls=c.model_calls,
            tool_calls=c.tool_calls,
            usd=c.usd,
            elapsed_s=c.elapsed_s,
            decomposition_status=record.status,
            run_dir=str(run_dir),
        )

    async def run(self, progress: Callable[[RunRecord], None] | None = None) -> list[RunRecord]:
        cfg = self.config
        if cfg.mode == "measurement":
            untrained = sorted(set(cfg.systems) & set(NEEDS_TRAINING))
            if untrained:
                raise ValueError(f"{untrained} need a trained selector (M3); not measurable yet")
        self.out.mkdir(parents=True, exist_ok=True)
        self._acquire()
        try:
            return await self._run(progress)
        finally:
            self._release()

    async def _run(self, progress: Callable[[RunRecord], None] | None) -> list[RunRecord]:
        cfg = self.config
        self.check_provenance()
        self._spent = None
        self.spent()  # scan existing traces once, before any new trace is written
        (self.out / "config.json").write_text(cfg.model_dump_json(indent=1) + "\n")
        results_path = self.out / "results.jsonl"
        done = set()
        if results_path.exists():  # resume: skip runs already recorded
            kept = []
            for raw in results_path.read_bytes().splitlines():
                try:
                    line = raw.decode("utf-8")
                    r = json.loads(line)
                except ValueError:  # torn by a crash (JSON or UTF-8): set aside and rerun
                    continue
                kept.append(line)
                done.add((r["task_id"], r["system"], r["repetition"]))
            # Rewrite atomically without the torn line, so new lines never join onto it.
            tmp = results_path.with_suffix(".jsonl.tmp")
            with tmp.open("w", encoding="utf-8") as f:
                f.write("".join(f"{line}\n" for line in kept))
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, results_path)
            fsync_dir(self.out)
        tasks = select_tasks(cfg.tasks, self.root)
        sem = asyncio.Semaphore(cfg.concurrency)
        records: list[RunRecord] = []

        async def job(task_id: str, split: str, rep: int) -> None:
            async with sem:
                if self._over_limit() or self._throttled:
                    return
                task = self.loader.load(task_id)
                try:
                    record = await self.decomposition(task, rep)
                except ProviderThrottled:
                    self._throttled = True
                    return
                order = list(cfg.systems)
                random.Random(f"{cfg.seed}:{task_id}:{rep}").shuffle(order)
                for system in order:
                    if (task_id, system, rep) in done:
                        continue
                    if self._over_limit() or self._throttled:
                        return
                    try:
                        rec = await self.run_one(task, split, rep, system, record)
                    except ProviderThrottled:
                        # Not a task outcome: leave the run unrecorded (its directory is set aside
                        # on resume, its spending stays counted) and start nothing new (R18).
                        self._throttled = True
                        return
                    async with self._lock:
                        with results_path.open("a", encoding="utf-8") as f:
                            f.write(rec.model_dump_json() + "\n")
                            f.flush()
                            os.fsync(f.fileno())
                    records.append(rec)
                    if progress is not None:
                        progress(rec)

        await asyncio.gather(*(job(t, s, rep) for rep in range(cfg.repetitions) for t, s in tasks))
        if self._throttled:
            raise ProviderUnavailable(
                "the provider throttled or a quota ran out; finished runs are kept and the rest "
                "resume when the experiment is run again"
            )
        if self._stopped:
            raise SpendingLimitReached(
                f"spending limit ${self.max_usd} reached (spent ${self.spent():.4f})"
            )
        return records


def load_results(path: Path) -> list[dict[str, JsonValue]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
