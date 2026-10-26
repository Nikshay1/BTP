"""Offline rehearsal of the experiment pipeline: every system, real runner, stand-in model."""

import asyncio
import json
from pathlib import Path

from bioalloc.cli import main
from bioalloc.evaluation.analysis import summarize
from bioalloc.evaluation.experiment import Experiment, ExperimentConfig, TaskSelection, load_results
from bioalloc.evaluation.registry import SYSTEMS
from bioalloc.evaluation.rehearsal import RehearsalTransport
from bioalloc.schemas import Budget
from bioalloc.trace import read_manifest, read_trace, tokens_from_trace

ROOT = Path(__file__).resolve().parents[1]
TASKS = ["tools-v1-revenue-s105", "tools-v1-top_product-s119"]


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
