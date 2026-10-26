# Pilot

The pilot is the first run on the real backbone (GPT-OSS-120B on Groq, D3). Its purpose is
feasibility, not results: measure what a task run costs, check that the runtime checks behave,
look for floor and ceiling effects, and estimate how often systems disagree, which sets the power
of the main study (D13). It uses training tasks only.

## What it runs

`configs/pilot.json`: 30 training tasks from split v1 (10 planning, 10 tools, 10 reasoning),
3 repetitions, 8 systems (single agent, Direct, ant, bee and octopus alone, fixed, random,
specialist), cap of 120,000 tokens, 60 model calls, 40 tool calls and 10 minutes per task.
720 task runs in total. The learned selectors are not in it; they are trained in M3.

## Before running it (readiness checklist)

| Item | Status |
|---|---|
| Design decisions recorded (D10-D21), system list fixed | done |
| Executors and all Table 3 controls pass offline tests | done |
| Ledger and trace agree on every run; tool calls and imported costs traced | done |
| Strong (goal) and weak acceptance kept apart; acceptance never reported as correctness | done |
| Stored decomposition replayed identically to every system | done |
| Pilot tasks from training groups only; data snapshots and split file hashed | done |
| Public task files contain no answers (GSM8K solutions only in the private file) | done |
| Offline CI: lint, strict types, tests, no network or key | done |
| Full offline rehearsal of the pilot config (720 runs) | done |
| Cross-check of the plan simulator against the pinned VAL binary | **open**: VAL build needs Nikshay (see below) |
| Hand-labelled sample of checker and grader verdicts | **open**: after the first live runs |
| Independent code review of the M2 batch | **in progress** |
| Groq key with a spend cap | **open**: Nikshay |

## How to run it

1. Put the key in `.env` at the repository root (never committed): `GROQ_API_KEY=...`, and set a
   monthly spend limit in the Groq console.
2. First a tiny capability check (a few calls) to confirm the API behaves as the adapter assumes
   (seed, reasoning-token reporting, limits). This step is still to be written as a script.
3. Then the pilot, with a hard spending stop:

   ```sh
   uv run bioalloc experiment run configs/pilot.json --max-usd 5
   uv run bioalloc experiment analyze runs/pilot-v1 --compare bee single_agent
   ```

   The run can be stopped and restarted; finished runs are skipped.

## Building VAL (optional, for the grader cross-check)

The environment would not let Claude build external code, so this is for Nikshay to run if wanted:

```sh
git clone https://github.com/KCL-Planning/VAL.git ~/Projects/btp-tools/VAL
cd ~/Projects/btp-tools/VAL && git checkout 3c7a1f330bdab0ba28a4762bb45c3f06c27fb6d4
mkdir -p build && cd build
uv tool run --from cmake cmake .. -DCMAKE_BUILD_TYPE=Release && make -j8 Validate
```
