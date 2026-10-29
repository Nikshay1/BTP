"""Command-line entry point."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

from bioalloc import __version__

if TYPE_CHECKING:
    from bioalloc.llm.base import Transport


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="bioalloc", description=__doc__)
    parser.add_argument("--version", action="version", version=f"bioalloc {__version__}")
    sub = parser.add_subparsers(dest="command")

    demo = sub.add_parser(
        "demo",
        help="run the toy route task end to end with the scripted fake model (no API key)",
    )
    demo.add_argument("--scenario", choices=["good", "recover", "budget"], default="good")
    demo.add_argument("--seed", type=int, default=0)
    demo.add_argument("--out", type=Path, default=Path("runs/demo"))

    exp = sub.add_parser("experiment", help="run or analyze an experiment")
    exp_sub = exp.add_subparsers(dest="action", required=True)
    run = exp_sub.add_parser("run", help="run an experiment config")
    run.add_argument("config", type=Path)
    run.add_argument("--out", type=Path, default=Path("runs"))
    run.add_argument(
        "--rehearsal",
        action="store_true",
        help="offline dress rehearsal with a stand-in model (no network, no key)",
    )
    run.add_argument(
        "--max-usd",
        type=float,
        default=None,
        help="stop starting new runs once this much has been spent (live runs)",
    )
    an = exp_sub.add_parser("analyze", help="summarize results.jsonl of an experiment")
    an.add_argument("run_dir", type=Path)
    an.add_argument("--compare", nargs=2, action="append", default=[], metavar=("A", "B"))
    return parser


def demo(scenario: str, seed: int, out: Path) -> int:
    from bioalloc.evaluation.runner import run_toy

    outcome = run_toy(seed, scenario, out)
    r = outcome.result
    print(f"task        {r.task_id}  (scenario: {scenario})")
    print(f"status      {r.status}")
    print(f"answer      {r.answer}")
    print(f"graded      {'correct' if outcome.grade.correct else 'wrong'} {outcome.grade.detail}")
    for a in r.attempts:
        mark = "accepted" if a.accepted else "rejected"
        reasons = "; ".join(c.reason for c in a.checks if c.reason)
        print(
            f"  attempt   {a.result.subtask_id}#{a.attempt} {a.result.executor:<8} "
            f"{a.result.status:<16} {a.verdict:<7} {mark} {reasons}"
        )
    c = r.cost
    print(
        f"cost        {c.model_calls} calls, {c.tokens.total} tokens "
        f"(coordinator {c.coordinator.total}, worker {c.worker.total}, "
        f"communication {c.communication})"
    )
    print(f"trace       {outcome.run_dir / 'trace.jsonl'}")
    print("note        fake model: this checks the plumbing and is not a result")
    return 0


def read_env_key(name: str) -> str | None:
    """A key from the environment or from a .env file in the working directory."""
    import os

    if os.environ.get(name):
        return os.environ[name]
    env = Path(".env")
    if env.exists():
        for line in env.read_text().splitlines():
            key, _, value = line.partition("=")
            if key.strip() == name:
                return value.strip().strip("\"'")
    return None


AWS_ENV = (
    "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY",
    "AWS_SESSION_TOKEN",
    "AWS_BEARER_TOKEN_BEDROCK",
    "AWS_REGION",
    "AWS_PROFILE",
)


def is_bedrock_model(model: str) -> bool:
    return "anthropic." in model


def bedrock_transport() -> Transport:
    """A Bedrock transport. AWS settings in .env are copied into the environment for boto3, which
    otherwise falls back to its usual chain (~/.aws/credentials, profiles)."""
    import os

    from bioalloc.llm.bedrock import BedrockTransport

    for name in AWS_ENV:
        value = read_env_key(name)
        if value and not os.environ.get(name):
            os.environ[name] = value
    return BedrockTransport(region=os.environ.get("AWS_REGION", "us-east-1"))


def experiment_run(config_path: Path, out: Path, rehearsal: bool, max_usd: float | None) -> int:
    import asyncio
    import json

    from bioalloc.evaluation.experiment import (
        Experiment,
        ExperimentConfig,
        ExperimentMismatch,
        RunRecord,
        SpendingLimitReached,
    )

    config = ExperimentConfig.model_validate(json.loads(config_path.read_text()))
    transport: Transport
    if rehearsal:
        from bioalloc.evaluation.rehearsal import RehearsalTransport

        config = config.model_copy(update={"mode": "debug", "name": f"{config.name}-rehearsal"})
        transport = RehearsalTransport()
    elif is_bedrock_model(config.model):
        transport = bedrock_transport()
    else:
        from bioalloc.llm.groq import GroqTransport

        key = read_env_key("GROQ_API_KEY")
        if not key:
            print("GROQ_API_KEY is not set (environment or .env); nothing was run.")
            return 2
        transport = GroqTransport(key)
    experiment = Experiment(config, transport, out, max_usd=max_usd)

    def progress(rec: RunRecord) -> None:
        mark = "ok " if rec.correct else "-- "
        print(
            f"{mark}{rec.system:<22} {rec.task_id:<32} rep{rec.repetition} "
            f"{rec.status:<16} {rec.tokens:>7} tok  spent ${experiment.spent():.4f}"
        )

    try:
        asyncio.run(experiment.run(progress))
    except SpendingLimitReached as exc:
        print(f"{exc}; stopped. Finished runs are kept; rerun to resume with a higher limit.")
        return 3
    except ExperimentMismatch as exc:
        print(f"refused: {exc}")
        return 4
    print(
        f"results: {out / config.name / 'results.jsonl'}"
        f"  (actual spending ${experiment.spent():.4f})"
    )
    return 0


def experiment_analyze(run_dir: Path, comparisons: list[list[str]]) -> int:
    import json

    from bioalloc.evaluation.analysis import paired, summarize
    from bioalloc.evaluation.experiment import load_results

    records = load_results(run_dir / "results.jsonl")
    # Template groups and families come from the split file the experiment used.
    from bioalloc.evaluation.data import REPO_ROOT
    from bioalloc.evaluation.splits import load as load_split

    config = json.loads((run_dir / "config.json").read_text())
    assignments = load_split(REPO_ROOT / config["tasks"]["split_file"])
    groups = {x.task_id: x.group for x in assignments}
    families = {x.task_id: x.family for x in assignments}
    cols = ("system", "runs", "success", "within_budget", "mean_tokens", "mean_calls", "mean_usd")
    print("  ".join(f"{c:>14}" for c in cols))
    for row in summarize(records):
        print(
            "  ".join(
                f"{row[c]:>14.3f}" if isinstance(row[c], float) else f"{row[c]:>14}" for c in cols
            )
        )
    for a, b in comparisons:
        print(json.dumps(paired(records, a, b, groups=groups, families=families), indent=1))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "demo":
        return demo(args.scenario, args.seed, args.out)
    if args.command == "experiment" and args.action == "run":
        return experiment_run(args.config, args.out, args.rehearsal, args.max_usd)
    if args.command == "experiment" and args.action == "analyze":
        return experiment_analyze(args.run_dir, args.compare)
    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
