# Framework architecture

This document describes the `bioalloc` framework as built. It is written so that its sections can
be lifted into the system and method sections of the report and the paper. The research plan
(`research-plan/main.pdf`) states what is studied; this file states how the code does it. Design
choices are justified in `docs/decisions.md` (cited as D1, D2, ...).

Last updated: 26 October 2026 (milestone M1).

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
| `executors/` | The common executor interface and the executors. M1 has `Direct`. |
| `checks.py` | Runtime checks (`pass`, `fail`, `unknown`) and how verdicts combine. |
| `policies.py` | Allocation policies. M1 has `Constant` and `Uniform` (random). |
| `coordinator.py` | Select, dispatch, check, accept or reallocate, aggregate. |
| `tasks/` | Public side of each task family: its runtime checks and aggregator. |
| `evaluation/` | Task generators, reference answers, graders and the run harness. Nothing outside it may import it (tested). |
| `prompts/` | Shared prompt templates; their hashes go into every manifest. |

## Executor interface

```python
class Executor(Protocol):
    name: str
    async def execute(self, subtask: Subtask, ctx: ExecutionContext) -> ExecutorOutcome: ...
```

`ExecutionContext` holds the accepted outputs of the subtask's dependencies (with the agent that
produced each), a `MeteredClient` whose account is the budget of this dispatch, a random generator
seeded by run seed, subtask and attempt, and an event hook into the trace. The coordinator turns
the outcome into an `ExecutorResult` by adding the dispatch's cost and its trace span.

Executors differ only in control flow, communication rules, memory and stopping conditions. The
prompt for solving a subtask, the runtime checks, the backbone model and the decomposition are
shared by every system, including the baselines (research plan, Section 4.1).

## Budget accounting

- Every model call reserves `ceil(chars/3) + 4` tokens per prompt message plus its maximum
  completion tokens before it is sent. If the reservation does not fit in the remaining budget of
  the dispatch or any ancestor, the call is not sent (`BudgetExceeded`).
- After the reply the reservation is released and the provider's reported usage is charged to the
  dispatch and all its ancestors.
- A timed-out call is charged its full reservation and the cost is marked `usage_complete=False`.
  A call that failed with no reply is released without charge.
- Limits: total tokens (prompt plus completion), number of model calls, and optionally wall time.
- Failed attempts, retries and (from M2) decomposition all count against the same task budget.

## Token attribution (D6)

Each call's tokens go to its actor, coordinator or worker. Separately, the share of a call's
prompt spent reading content written by another agent is counted as communication. The share is
estimated by characters, using the `source` field that every message carries. Communication is a
subset of the total, so `coordinator + worker = local + communication = total`.

## Runtime checks and acceptance

A runtime check may use only what a deployed system would have: the output schema, tool status,
constraint checkers that are part of the task's public statement, tests supplied with the task and
agreement between samples. Verdicts combine as: any `fail` fails; otherwise any `pass` passes;
otherwise `unknown`. `unknown` results are accepted by default (Section 4.4), but the verdict is
recorded separately so that acceptance is never reported as correctness.

## Traces and manifests

Each run directory holds `trace.jsonl` (append-only, flushed per event) and `manifest.json`.
Event kinds so far: `run_start`, `dispatch`, `llm_request`, `llm_reply`, `llm_error`,
`attempt_end`, `subtask_failed`, `run_end`. The manifest records the git commit and whether the
tree was dirty, the `uv.lock` hash, the resolved configuration and its hash, the hash of every
prompt template, the model identifier, seeds, the run mode (`debug` or `measurement`) and the final
status with a summary. A run that crashed or was interrupted is identifiable by its status.
Token totals rebuilt from the trace alone are tested to equal the ledger.

## Toy task (M1 only)

Two-leg route planning on a random weighted digraph of eight nodes (D4). Leg 1: cheapest route from
`A` to a waypoint. Leg 2: from wherever leg 1 ended to `H`, so it depends on leg 1's accepted
output. The public route check verifies endpoints and edges; only the grader knows whether the
route is cheapest. `bioalloc demo` runs three scripted scenarios: `good`, `recover` (an invalid
first answer is rejected and reallocated) and `budget` (the run stops cleanly when the call budget
is spent).
