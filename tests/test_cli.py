import pytest

from bioalloc import __version__
from bioalloc.cli import main


def test_version(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        main(["--version"])
    assert exc.value.code == 0
    assert __version__ in capsys.readouterr().out


def test_no_command_prints_help(capsys: pytest.CaptureFixture[str]) -> None:
    assert main([]) == 0
    assert "usage: bioalloc" in capsys.readouterr().out


@pytest.mark.parametrize(
    ("scenario", "status", "correct", "attempts"),
    [
        ("good", "completed", True, 2),
        ("recover", "completed", True, 3),
        ("budget", "budget_exhausted", False, 1),
    ],
)
def test_demo_scenarios(
    tmp_path,
    scenario: str,
    status: str,
    correct: bool,
    attempts: int,  # type: ignore[no-untyped-def]
) -> None:
    from bioalloc.evaluation.runner import run_toy
    from bioalloc.trace import read_manifest, read_trace

    outcome = run_toy(0, scenario, tmp_path)
    assert outcome.result.status == status
    assert outcome.grade.correct is correct
    assert len(outcome.result.attempts) == attempts
    manifest = read_manifest(outcome.run_dir)
    assert manifest.status == status and manifest.mode == "debug"
    assert "solve_subtask" in manifest.prompt_sha256
    assert list(read_trace(outcome.run_dir))[-1].kind == "run_end"


def test_demo_command_prints_summary(tmp_path, capsys) -> None:  # type: ignore[no-untyped-def]
    assert main(["demo", "--scenario", "recover", "--out", str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert "status      completed" in out and "rejected" in out and "not a result" in out
