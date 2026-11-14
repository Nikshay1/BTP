# BTP

B.Tech final-year project (2026-27), Department of Computer Science and Engineering, Delhi Technological University.

**Adaptive Task Allocation for Heterogeneous Bio-Inspired Agents.** A coordinator decomposes a task and assigns each subtask to one of several coordination mechanisms (ant, bee, octopus, crow, starfish, and others), and the project measures whether this beats fixed workflows at an equal budget.

## Status

Phase 1 done (November 2026): executors, controls and baselines, task suites with frozen splits,
the experiment runner and analysis, and measured runs on Claude Sonnet 4.6 (Bedrock). New work runs
on GPT-6 Luna through OpenAI's Responses API (decision D32). See `docs/progress.md` and
`docs/decisions.md`.

## Backbone and keys

The model in an experiment config picks the adapter: `gpt-*` names go to OpenAI (Responses API,
`OPENAI_API_KEY`), `*anthropic.*` names to Bedrock (AWS settings), anything else to Groq
(`GROQ_API_KEY`). Keys are read from the environment or from `.env` (git-ignored). Check a provider
for well under a cent with `uv run python scripts/capability_check.py openai`. Luna configs are
`configs/luna-*.json` (temperature 1, reasoning effort low, US$0.10 / US$0.50 per million tokens).

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
uv run bioalloc experiment analyze runs/pilot-v2-rehearsal
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
