"""Command-line entry point."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from bioalloc import __version__


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


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "demo":
        return demo(args.scenario, args.seed, args.out)
    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
