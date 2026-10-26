# BTP

B.Tech final-year project (2026-27), Department of Computer Science and Engineering, Delhi Technological University.

**Adaptive Task Allocation for Heterogeneous Bio-Inspired Agents.** A coordinator decomposes a task and assigns each subtask to one of several coordination mechanisms (ant, bee, octopus, crow, starfish, and others), and the project measures whether this beats fixed workflows at an equal budget.

## Status

Building milestone M1 (mid-November 2026): the coordinator, the common executor interface, runtime
checks and logging, working end to end with one executor on a toy task. See `docs/decisions.md` for
every design decision and the reason for it.

## Development

Requires [uv](https://docs.astral.sh/uv/). Python 3.12 is installed by uv if missing.

```sh
uv sync               # create the environment from uv.lock
uv run pytest         # tests (offline, no API key needed)
uv run ruff check     # lint
uv run mypy           # type check
```

## Contents

- `research-plan/main.tex` – detailed research plan (single file, compiles with pdfLaTeX on Overleaf)
- `research-plan/main.pdf` – compiled copy of the plan
- `BTP_Synopsis (1).pdf` – approved project synopsis
- `docs/research-plan-explained.md` – the research plan explained in simple words
- `docs/mentor-questions.md` – questions for the guide, with their answers and the decisions that follow
- `docs/decisions.md` – log of design and research decisions, with dates and reasons
- `src/bioalloc/` – the framework (Python package)
- `tests/` – unit and end-to-end tests
