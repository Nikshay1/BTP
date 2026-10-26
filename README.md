# BTP

B.Tech final-year project (2026-27), Department of Computer Science and Engineering, Delhi Technological University.

**Adaptive Task Allocation for Heterogeneous Bio-Inspired Agents.** A coordinator decomposes a task and assigns each subtask to one of several coordination mechanisms (ant, bee, octopus, crow, starfish, and others), and the project measures whether this beats fixed workflows at an equal budget.

## Status

M1 done; M2 built up to the pilot (26 October 2026). All executors, controls and baselines, four
task suites with frozen splits, the Groq adapter, the experiment runner and analysis work offline;
the pilot on the real backbone waits for an API key. See `docs/progress.md`, `docs/pilot.md` and
`docs/decisions.md`.

## Development

Requires [uv](https://docs.astral.sh/uv/). Python 3.12 is installed by uv if missing.

```sh
uv sync               # create the environment from uv.lock
uv run pytest         # tests (offline, no API key needed)
uv run ruff check     # lint
uv run mypy           # type check
```

Run the M1 toy task end to end with the scripted fake model (no API key needed):

```sh
uv run bioalloc demo --scenario good      # both subtasks solved at the first attempt
uv run bioalloc demo --scenario recover   # a bad first answer is rejected and reallocated
uv run bioalloc demo --scenario budget    # the run stops cleanly when the budget runs out
```

Each run writes `runs/demo/<run-id>/trace.jsonl` and `manifest.json`.

Prepare the datasets once (needs the network), then rehearse the pilot offline:

```sh
uv run --group prep python scripts/prepare_data.py gsm8k planbench
uv run bioalloc experiment run configs/pilot.json --rehearsal
uv run bioalloc experiment analyze runs/pilot-v1-rehearsal
```

## Contents

- `research-plan/main.tex` – detailed research plan (single file, compiles with pdfLaTeX on Overleaf)
- `research-plan/main.pdf` – compiled copy of the plan
- `BTP_Synopsis (1).pdf` – approved project synopsis
- `docs/research-plan-explained.md` – the research plan explained in simple words
- `docs/mentor-questions.md` – questions for the guide, with their answers and the decisions that follow
- `docs/decisions.md` – log of design and research decisions, with dates and reasons
- `docs/architecture.md` – how the framework works, written for reuse in the report and paper
- `docs/progress.md` – milestone status and what carries over
- `docs/pilot.md` – what the pilot runs, its readiness checklist and how to run it
- `configs/` – experiment configurations
- `data/manifests/`, `data/splits/` – dataset hashes and the frozen study splits
- `scripts/` – data preparation and split building
- `src/bioalloc/` – the framework (Python package)
- `tests/` – unit and end-to-end tests
