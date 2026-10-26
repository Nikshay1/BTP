"""Offline rehearsal of the experiment pipeline: every system, real runner, stand-in model."""

import asyncio
import json
from pathlib import Path

import pytest

from bioalloc.cli import main
from bioalloc.evaluation.analysis import summarize
from bioalloc.evaluation.experiment import Experiment, ExperimentConfig, TaskSelection, load_results
from bioalloc.evaluation.registry import SYSTEMS
from bioalloc.evaluation.rehearsal import RehearsalTransport
from bioalloc.schemas import Budget
from bioalloc.trace import read_manifest, read_trace, tokens_from_trace

ROOT = Path(__file__).resolve().parents[1]
TASKS = ["tools-v1-revenue-s0", "tools-v1-top_product-s0"]


def config(systems: list[str], reps: int = 1) -> ExperimentConfig:
    return ExperimentConfig(
        name="t",
        mode="debug",
        model="rehearsal",
        systems=systems,
        tasks=TaskSelection(task_ids=TASKS),
        repetitions=reps,
        budget=Budget(max_tokens=200_000, max_model_calls=80, max_tool_calls=40),
    )


def test_every_system_runs_end_to_end_offline(tmp_path: Path) -> None:
    exp = Experiment(config(list(SYSTEMS)), RehearsalTransport(), tmp_path, root=ROOT)
    records = asyncio.run(exp.run())
    assert len(records) == len(SYSTEMS) * len(TASKS)
    assert {r.system for r in records} == set(SYSTEMS)
    for r in records:
        manifest = read_manifest(Path(r.run_dir))
        assert manifest.status == r.status and manifest.mode == "debug"
        events = list(read_trace(Path(r.run_dir)))
        assert tokens_from_trace(events)["tokens"] == r.tokens  # ledger and trace agree
    rows = summarize(load_results(tmp_path / "t" / "results.jsonl"))
    assert len(rows) == len(SYSTEMS)


def test_decomposition_is_shared_and_runs_resume(tmp_path: Path) -> None:
    cfg = config(["direct", "octopus"], reps=2)
    asyncio.run(Experiment(cfg, RehearsalTransport(), tmp_path, root=ROOT).run())
    decomp = list((tmp_path / "t" / "decompositions").rglob("rep*.json"))
    assert len(decomp) == len(TASKS) * 2  # once per task and repetition, not per system
    # Running again records nothing new: completed runs are skipped.
    again = asyncio.run(Experiment(cfg, RehearsalTransport(), tmp_path, root=ROOT).run())
    assert again == []
    assert len(load_results(tmp_path / "t" / "results.jsonl")) == 2 * len(TASKS) * 2


def test_cli_rehearsal_and_analysis(tmp_path: Path, capsys) -> None:  # type: ignore[no-untyped-def]
    path = tmp_path / "c.json"
    path.write_text(config(["direct", "bee"]).model_dump_json())
    assert main(["experiment", "run", str(path), "--rehearsal", "--out", str(tmp_path)]) == 0
    assert (
        main(["experiment", "analyze", str(tmp_path / "t-rehearsal"), "--compare", "direct", "bee"])
        == 0
    )
    out = capsys.readouterr().out
    assert "direct" in out and '"discordance"' in out


def test_live_run_refuses_without_a_key(tmp_path: Path, monkeypatch, capsys) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.chdir(tmp_path)
    path = tmp_path / "c.json"
    path.write_text(config(["direct"]).model_dump_json())
    assert main(["experiment", "run", str(path), "--out", str(tmp_path)]) == 2
    assert "nothing was run" in capsys.readouterr().out


def test_pilot_config_is_valid() -> None:
    cfg = ExperimentConfig.model_validate(json.loads((ROOT / "configs" / "pilot.json").read_text()))
    assert cfg.mode == "measurement" and set(cfg.systems) <= set(SYSTEMS)
    assert cfg.tasks.split == "train"


def priced(systems: list[str], reps: int = 1) -> ExperimentConfig:
    return config(systems, reps).model_copy(
        update={"pricing": {"input_per_m": 1.0, "output_per_m": 2.0}}
    )


def test_interrupted_run_is_set_aside_counted_once_and_rerun(tmp_path: Path) -> None:
    from bioalloc.llm.base import LLMReply, Usage
    from bioalloc.trace import Tracer as T

    cfg = priced(["direct"])
    run_dir = tmp_path / "t" / "runs" / "direct" / TASKS[0] / "rep0"
    # A first runner made provenance and started a run that never finished.
    first = Experiment(cfg, RehearsalTransport(), tmp_path, root=ROOT)
    first.out.mkdir(parents=True)
    first.check_provenance()
    tracer = T(run_dir, "half")
    reply = LLMReply(content="x", model="m", usage=Usage(prompt_tokens=1000, completion_tokens=500))
    tracer.emit(
        tracer.span(),
        "llm_reply",
        {
            "agent": "a",
            "actor": "worker",
            "communication_tokens": 0,
            "reply": reply.model_dump(mode="json"),
        },
    )
    tracer.close()
    interrupted_usd = (1000 * 1.0 + 500 * 2.0) / 1e6
    exp = Experiment(cfg, RehearsalTransport(), tmp_path, root=ROOT)
    records = asyncio.run(exp.run())
    assert len(records) == len(TASKS)
    assert (run_dir.parent / "rep0.interrupted.0" / "trace.jsonl").exists()
    assert exp.spent() == pytest.approx(interrupted_usd + sum(r.usd for r in records))
    # Recovery is idempotent: a fresh look at the directory gives the same total.
    again = Experiment(cfg, RehearsalTransport(), tmp_path, root=ROOT)
    assert again.spent() == pytest.approx(exp.spent())


def test_spending_counts_each_decomposition_once(tmp_path: Path) -> None:
    from bioalloc.decomposition import PublicTask
    from bioalloc.evaluation.registry import StudyTask, TaskLoader

    real = TaskLoader()

    def as_reasoning(task_id: str) -> StudyTask:
        t = real.load(task_id)
        first = t.natural_graph.nodes[0]  # type: ignore[union-attr]
        public = PublicTask(
            id=task_id,
            instruction=first.instruction,
            inputs=first.inputs,
            output_schema=first.output_schema,
            contract=first.contract,
        )
        return StudyTask(task_id, "reasoning", t.task_family, t.grade, public=public)

    systems = ["direct", "octopus", "bee"]
    exp = Experiment(priced(systems), RehearsalTransport(), tmp_path, root=ROOT)
    exp.loader.load = as_reasoning  # type: ignore[method-assign]
    records = asyncio.run(exp.run())
    from bioalloc.decomposition import DecompositionStore

    store = DecompositionStore(tmp_path / "t" / "decompositions")
    decomp_usd = [store.load(t, 0).cost.usd for t in TASKS]
    assert all(u > 0 for u in decomp_usd)
    attributed = sum(r.usd for r in records)  # each system is charged the decomposition
    paid_once = attributed - (len(systems) - 1) * sum(decomp_usd)
    assert exp.spent() == pytest.approx(paid_once)


def test_spending_limit_stops_new_runs(tmp_path: Path) -> None:
    from bioalloc.evaluation.experiment import SpendingLimitReached

    cfg = priced(["direct", "octopus", "bee"]).model_copy(update={"concurrency": 1})
    exp = Experiment(cfg, RehearsalTransport(), tmp_path, root=ROOT, max_usd=0.0005)
    with pytest.raises(SpendingLimitReached):
        asyncio.run(exp.run())
    done = load_results(tmp_path / "t" / "results.jsonl")
    assert 0 < len(done) < 3 * len(TASKS)
    # Resuming with a higher limit finishes the rest and keeps the ledger.
    exp2 = Experiment(cfg, RehearsalTransport(), tmp_path, root=ROOT, max_usd=1.0)
    asyncio.run(exp2.run())
    assert len(load_results(tmp_path / "t" / "results.jsonl")) == 3 * len(TASKS)


def test_resume_with_a_different_config_is_refused(tmp_path: Path) -> None:
    from bioalloc.evaluation.experiment import ExperimentMismatch

    asyncio.run(Experiment(config(["direct"]), RehearsalTransport(), tmp_path, root=ROOT).run())
    changed = config(["direct"]).model_copy(update={"temperature": 0.1})
    with pytest.raises(ExperimentMismatch, match="config"):
        asyncio.run(Experiment(changed, RehearsalTransport(), tmp_path, root=ROOT).run())


def test_measurement_refuses_untrained_selectors(tmp_path: Path) -> None:
    cfg = config(["bio_adaptive"]).model_copy(update={"mode": "measurement"})
    with pytest.raises(ValueError, match="trained selector"):
        asyncio.run(Experiment(cfg, RehearsalTransport(), tmp_path, root=ROOT).run())


def test_loader_refuses_a_snapshot_that_differs_from_the_split_provenance() -> None:
    from bioalloc.evaluation.registry import TaskLoader

    loader = TaskLoader(expected={"gsm8k": {"gsm8k/public.jsonl": "0" * 64}})
    with pytest.raises((ValueError, FileNotFoundError)):
        loader.load("gsm8k-test-00001")


def test_torn_result_line_reruns_that_run_and_leaves_valid_results(tmp_path: Path) -> None:
    cfg = config(["direct"])
    asyncio.run(Experiment(cfg, RehearsalTransport(), tmp_path, root=ROOT).run())
    results = tmp_path / "t" / "results.jsonl"
    lines = results.read_text().splitlines()
    results.write_text(lines[0] + "\n" + lines[1][: len(lines[1]) // 2])  # crash mid-write
    rerun = asyncio.run(Experiment(cfg, RehearsalTransport(), tmp_path, root=ROOT).run())
    assert len(rerun) == 1
    assert len(load_results(results)) == 2  # every line parses; analysis works
    again = asyncio.run(Experiment(cfg, RehearsalTransport(), tmp_path, root=ROOT).run())
    assert again == []  # and the run is not repeated on later resumes


def test_torn_trace_tail_and_corrupt_decomposition_do_not_block_recovery(tmp_path: Path) -> None:
    cfg = config(["direct"])
    asyncio.run(Experiment(cfg, RehearsalTransport(), tmp_path, root=ROOT).run())
    out = tmp_path / "t"
    # A run trace whose last event was cut off, inside an archived interrupted directory.
    aside = out / "runs" / "direct" / TASKS[0] / "rep0.interrupted.0"
    aside.mkdir(parents=True)
    (aside / "trace.jsonl").write_text('{"schema_version": 1, "run_id": "x", "seq')
    # A decomposition file left half-written by an older version.
    decomp = out / "decompositions" / TASKS[1] / "rep0.json"
    decomp.write_text(decomp.read_text()[:40])
    results = out / "results.jsonl"
    lines = results.read_text().splitlines()
    results.write_text(lines[0] + "\n")  # the second task's run is to be redone
    exp = Experiment(cfg, RehearsalTransport(), tmp_path, root=ROOT)
    rerun = asyncio.run(exp.run())
    assert len(rerun) == 1 and exp.spent() >= 0
    assert (out / "decompositions" / TASKS[1] / "rep0.json.interrupted.0").exists()


def test_one_runner_per_directory(tmp_path: Path) -> None:
    import fcntl

    from bioalloc.evaluation.experiment import ExperimentMismatch

    out = tmp_path / "t"
    out.mkdir()
    holder = (out / ".lock").open("a+")
    fcntl.flock(holder, fcntl.LOCK_EX | fcntl.LOCK_NB)  # another runner holds the lock
    with pytest.raises(ExperimentMismatch, match="another runner"):
        asyncio.run(Experiment(config(["direct"]), RehearsalTransport(), tmp_path, root=ROOT).run())
    holder.close()  # that runner exits (or dies): the kernel releases the lock
    (out / ".lock").write_text("")  # even an empty leftover file does not block
    asyncio.run(Experiment(config(["direct"]), RehearsalTransport(), tmp_path, root=ROOT).run())


def test_populated_directory_without_provenance_is_refused(tmp_path: Path) -> None:
    from bioalloc.evaluation.experiment import ExperimentMismatch

    asyncio.run(Experiment(config(["direct"]), RehearsalTransport(), tmp_path, root=ROOT).run())
    (tmp_path / "t" / "provenance.json").unlink()
    with pytest.raises(ExperimentMismatch, match="no provenance"):
        asyncio.run(Experiment(config(["direct"]), RehearsalTransport(), tmp_path, root=ROOT).run())
