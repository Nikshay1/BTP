# Framework architecture

This document describes the `bioalloc` framework as built. It is written so that its sections can
be lifted into the system and method sections of the report and the paper. The research plan
(`research-plan/main.pdf`) states what is studied; this file states how the code does it. Design
choices are justified in `docs/decisions.md` (cited as D1, D2, ...).

Last updated: 26 October 2026 (M2, up to the pilot).

## Overview

A task arrives as a dependency graph of subtasks. The coordinator repeatedly takes the subtasks
whose dependencies are done, asks an allocation policy which executor should handle each one,
dispatches it under a share of the task budget, checks the result with runtime checks, and either
accepts it or reallocates the subtask. When every final subtask is accepted, a deterministic
family-specific aggregator builds the answer. The evaluation harness, which alone holds reference
answers, grades that answer.

```
TaskGraph ──> Coordinator ──select──> Policy
                  │  dispatch (child budget)
                  ▼
              Executor ──calls──> MeteredClient ──> Transport (provider or fake)
                  │ output
                  ▼
           Runtime checks ── pass / unknown ──> accept ──> Aggregate ──> answer ──> Grader
                  │ fail
                  └──> reallocate (attempt record passed to the policy)
```

## Components

| Module | Role |
|---|---|
| `schemas.py` | Frozen data contracts: `Subtask`, `Budget`, `Tokens`, `Cost`, `CheckResult`, `ExecutorResult`. Unknown fields are rejected. |
| `graph.py` | `TaskGraph`: validates unique ids, known dependencies and acyclicity; yields ready subtasks. |
| `llm/` | `LLMRequest`/`LLMReply` contracts, the `Transport` protocol, a scripted `FakeTransport`, and a debug-only response cache. |
| `budget.py` | Budget accounts (a tree, one node per dispatch) and the `MeteredClient`, the only path from an executor to a model. |
| `trace.py` | Append-only JSON Lines trace grouped in spans, and the run manifest. |
| `executors/` | The common interface and the executors: `Direct`, `Octopus`, `Ant`, `Bee`, their Table 3 controls, the conventional workers (`act_observe`, `PlanThenExecute`, `MajorityVote`), `SingleAgent` and `Specialist`. |
| `actions.py` | Typed action spaces: the fixed choice vocabulary a family offers to step-wise executors (D10). |
| `tools.py` | The restricted tool service: allowed tools only, validated arguments, bounded time and output, every attempt counted and traced. |
| `decomposition.py` | Stored decompositions: one graph per task and repetition, hashed and replayed with its cost imported (D17). |
| `features.py` | Selector features computed in trusted code before execution. |
| `llm/groq.py` | The Groq transport for GPT-OSS-120B, with error classes mapped to the ledger (D19). |
| `checks.py` | Runtime checks (`pass`, `fail`, `unknown`) and how verdicts combine. |
| `policies.py` | Allocation policies: `Constant`, `Uniform` (random), `FixedByContract`, `RuleTable`, `LinUCB`. |
| `coordinator.py` | Select, dispatch, check, accept or reallocate, aggregate. |
| `tasks/` | Public side of each family (checks with evidence strength, aggregator, normalizer, action space, tools): toy routes, synthetic routes, GSM8K, PlanBench (with a STRIPS simulator), our tool suite. |
| `evaluation/` | Data preparation, generators, references, graders, splits, the task and system registry, the experiment runner, analysis and the rehearsal model. Nothing outside it may import it (tested). |
| `prompts/` | Shared prompt templates; their hashes go into every manifest. |

## Executor interface

```python
class Executor(Protocol):
    name: str

    async def execute(self, subtask: Subtask, ctx: ExecutionContext) -> ExecutorOutcome: ...
```

`ExecutionContext` holds the accepted outputs of the subtask's dependencies (with the agent that
produced each), any rejected earlier attempts with the reasons for rejection (D8), a
`MeteredClient` whose account is the budget of this dispatch, a seed derived from the run seed,
task, subtask and attempt (used for the random generator and for model sampling), and an event
hook into the trace. The coordinator turns
the outcome into an `ExecutorResult` by adding the dispatch's cost and its trace span.

Executors differ only in control flow, communication rules, memory and stopping conditions. The
prompt for solving a subtask, the runtime checks, the backbone model and the decomposition are
shared by every system, including the baselines (research plan, Section 4.1).

## Budget accounting

- Every model call reserves an estimate of its prompt (three characters per token over all
  messages, tool definitions and response format) plus its maximum completion tokens before it is
  sent. If the reservation does not fit in what the dispatch and all its ancestors have left, the
  call is not sent.
- If it would fit once calls already in flight have settled, it waits for them; concurrent
  subtasks share one pool and waiting calls are admitted in arrival order (D7). It is refused
  (`BudgetExceeded`) only when the budget itself is spent.
- After the reply the reservation is released and the provider's reported usage is charged to the
  dispatch and all its ancestors. Reasoning tokens are recorded as a subset of completion tokens.
- A call that reached the transport but got no reply (timeout, deadline, cancellation, transport
  error) is charged its full reservation, split into prompt and completion, and the cost is marked
  `usage_complete=False`. A `RequestRejected` error, or a call cut off before it was sent, is
  released without charge.
- With a wall-time limit, waiting for admission, for a reply and for a whole executor are all
  bounded by the time left on the tightest account (dispatch cap or task).
- Reservations are estimates. If reported usage still crosses a limit, of the task or of one
  dispatch, `Cost.overrun` is set on that account and its ancestors, no further calls are admitted
  under it, and the run is reported with `within_budget = false`.
- In `measurement` mode a cached reply is refused (D5.4).
- Failed attempts, retries and (from M2) decomposition all count against the same task budget.

## Token attribution (D6)

Each call's tokens go to its actor, coordinator or worker. Separately, the share of a call's
prompt spent reading content written by another agent is counted as communication. The share is
estimated by characters, using the `source` field that every message carries. Communication is a
subset of the total, so `coordinator + worker = local + communication = total`.

## Scheduling and reallocation

A subtask is dispatched as soon as all its dependencies are accepted, independently of other
branches. A rejected result is reallocated: the policy chooses again, with the record of every
earlier attempt, and the executor receives the rejected output and the public reasons for
rejection (D8). The attempt allowance is the same for every system. A subtask refused for lack of
budget is not retried. After each attempt the policy receives a structured record (D9) and defines
its own reward from it. If the run is cancelled or an executor raises, the remaining workers are
cancelled and settle their accounting before the run returns.

## Runtime checks and acceptance

A runtime check may use only what a deployed system would have: the output schema, tool status,
constraint checkers that are part of the task's public statement, tests supplied with the task and
agreement between samples. Checks are total: any output, however malformed, gets a verdict.
Verdicts combine as: any `fail` fails; otherwise any `pass` passes;
otherwise `unknown`. `unknown` results are accepted by default (Section 4.4), but the verdict is
recorded separately so that acceptance is never reported as correctness.

## Traces and manifests

Each run directory holds `trace.jsonl` (append-only, flushed per event) and `manifest.json`.
Event kinds so far: `run_start`, `dispatch`, `llm_request`, `llm_reply`, `llm_error`,
`attempt_end`, `subtask_failed`, `run_end`. Every run gets its own directory (task, scenario, time and a random
suffix) and the trace file is created exclusively. The manifest records the git commit and whether
the tree was dirty, the `uv.lock` hash, the resolved configuration and its hash, the hash of the
task graph, the hash of every prompt template, the model identifier, seeds, the run mode (`debug` or `measurement`) and the final
status with a summary. A run that crashed or was interrupted is identifiable by its status.
Token totals rebuilt from the trace alone are tested to equal the ledger.

## Toy task (M1 only)

Two-leg route planning on a random weighted digraph of eight nodes (D4). Leg 1: cheapest route from
`A` to a waypoint. Leg 2: from wherever leg 1 ended to `H`, so it depends on leg 1's accepted
output. The public route check verifies endpoints and edges; only the grader knows whether the
route is cheapest. `bioalloc demo` runs three scripted scenarios: `good`, `recover` (an invalid
first answer is rejected and reallocated) and `budget` (the run stops cleanly when the call budget
is spent).

## Executors (M2)

| Executor | Control flow | Calls per dispatch |
|---|---|---|
| Direct | one completion, or one tool episode of at most four completions (D11) | 1-4 |
| Octopus | private act-observe loop; each step a tool episode; check-driven correction; stops on goal verification, an unrejected answer, or the step limit | up to 12 steps |
| Octopus, reporting | the same, with a metered coordinator call after every step | + 1 per step |
| Ant | paths of typed actions; model ranks candidates; sampling by tau^alpha (1/rank)^beta; reward only for goal verification; evaporation and once-per-kind deposit; task-private weights (D10) | 1 per choice, up to 8 paths |
| Bee | k = 5 independent scouts; synchronous rounds of inhibition then recruitment; quorum 4 (D14) | 5 + assessments |
| MajorityVote | independent answers, plurality vote; call-yoked variant | k |
| PlanThenExecute | one planning call, then Direct's episode | 2-5 |
| SingleAgent | one persistent act-observe worker over the whole graph, serial, one identity | per subtask |
| Specialist | planner advice once per task, act-observe executor, critic with at most one revision | varies |

## Experiments

`bioalloc experiment run CONFIG` runs tasks x repetitions x systems from a JSON config. For each
task and repetition one decomposition is prepared and stored, then every system runs on a fresh
copy with the decomposition's cost already charged, in an order shuffled per task. Each run has
its own trace and manifest; one line per run goes to `results.jsonl`, and a stopped experiment
resumes where it left off. `--rehearsal` swaps in a stand-in model for an offline dress
rehearsal; `--max-usd` stops a live run at a spending limit. `bioalloc experiment analyze` prints
per-system summaries and paired comparisons (D13).
