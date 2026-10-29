# Pilot

The pilot is the first run on a real model: Claude Sonnet 4.6 on Amazon Bedrock at low effort
(D23; D3's GPT-OSS-120B is suspended until the pilot shows what a run costs). Its purpose is
feasibility, not results: measure what a task run costs, check that the runtime checks behave,
look for floor and ceiling effects, and estimate how often systems disagree, which sets the power
of the main study (D13). It uses training tasks only.

## What it runs

1. `configs/pilot-smoke.json` (pilot-smoke-2): 6 training tasks (one Blocksworld, one Logistics,
   one `revenue`, one `top_product`, two longer GSM8K questions; D24), one repetition, 8 systems:
   48 runs, about US$4, stop at US$8. The first smoke test (24 runs, US$1.26, `runs/pilot-smoke`)
   found the parsing bugs in R5.
2. `configs/pilot.json` (pilot-v2): 18 training tasks chosen by D24 (6 per family), 2
   repetitions, 8 systems (single agent, Direct, ant, bee and octopus alone, fixed, random,
   specialist): 288 runs, about US$25, stop at US$40. Cap of 120,000 tokens, 60 model calls, 40
   tool calls and 10 minutes per task.

The learned selectors are not in it; they are trained in M3. After the pilot, its cost per run
decides the main-study backbone and size (D23).

## Before running it (readiness checklist)

| Item | Status |
|---|---|
| Design decisions recorded (D10-D23), system list fixed | done |
| Executors and all Table 3 controls pass offline tests | done |
| Ledger and trace agree on every run; tool calls and imported costs traced | done |
| Strong (goal) and weak acceptance kept apart; acceptance never reported as correctness | done |
| Stored decomposition replayed identically to every system | done |
| Pilot tasks from training groups only; data snapshots and split file hashed | done |
| Public task files contain no answers (GSM8K solutions only in the private file) | done |
| Offline CI: lint, strict types, tests, no network or key | done |
| Offline rehearsal of both pilot configs | done |
| Independent code review of the M2 batch (R2, R3) | done |
| Bedrock adapter tested offline (stand-in boto3 client) | done |
| AWS access to Sonnet 4.6 and credits that cover it | **open**: Nikshay |
| Cross-check of the plan simulator against the pinned VAL binary | **open**: VAL build needs Nikshay (see below) |
| Hand-labelled sample of checker and grader verdicts | **open**: after the first live runs |

## How to run it

1. AWS setup (once):
   - In the Bedrock console, open **Model access** and enable Anthropic Claude Sonnet 4.6 (the
     first time, AWS asks for a short use-case form).
   - In **Billing → Credits**, check that the credits list Amazon Bedrock (or the Claude models)
     under the services they apply to. Some promotional credits exclude third-party models.
   - Set a budget alert in **Billing → Budgets** (for example at US$50).
   - The IAM user needs `bedrock:InvokeModel` on the model and the inference profile.
2. Put the credentials in `.env` at the repository root (never committed). Either an access key:

   ```sh
   AWS_ACCESS_KEY_ID=...
   AWS_SECRET_ACCESS_KEY=...
   AWS_REGION=us-east-1
   ```

   or a Bedrock API key: `AWS_BEARER_TOKEN_BEDROCK=...` with `AWS_REGION`.
3. Capability check (four calls, well under a cent). It confirms that usage is reported, that
   low effort is accepted, that tool calls and their read-back work, and that the ledger's
   prompt estimate is not below the real prompt count. It exits with an error if a step fails:

   ```sh
   uv run python scripts/capability_check.py
   ```
4. Diversity check (D25; 32 short Direct runs, about US$1):

   ```sh
   uv run bioalloc experiment run configs/diversity-t06.json --max-usd 2
   uv run bioalloc experiment run configs/diversity-t10.json --max-usd 2
   uv run python scripts/diversity_report.py runs/diversity-t06 runs/diversity-t10
   ```
5. Smoke test, then look at the results before going on:

   ```sh
   uv run bioalloc experiment run configs/pilot-smoke.json --max-usd 8
   uv run bioalloc experiment analyze runs/pilot-smoke-2
   ```
6. The pilot (after a difficulty check, R6):

   ```sh
   uv run bioalloc experiment run configs/pilot.json --max-usd 40
   uv run bioalloc experiment analyze runs/pilot-v2 --compare bee single_agent
   ```

   Both can be stopped and restarted; finished runs are skipped.

   `--max-usd` is a stop threshold, not an exact ceiling: once spending passes it no new run
   starts, but the (up to four) runs in flight finish. At Sonnet prices a run at its 120,000-token
   cap costs at most about US$1.80, so spending can end up to about US$7 above the threshold.

## Building VAL (optional, for the grader cross-check)

The environment would not let Claude build external code, so this is for Nikshay to run if wanted:

```sh
git clone https://github.com/KCL-Planning/VAL.git ~/Projects/btp-tools/VAL
cd ~/Projects/btp-tools/VAL && git checkout 3c7a1f330bdab0ba28a4762bb45c3f06c27fb6d4
mkdir -p build && cd build
uv tool run --from cmake cmake .. -DCMAKE_BUILD_TYPE=Release && make -j8 Validate
```
