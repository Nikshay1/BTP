# Progress

Milestones are those of the research plan (Table 6). Each entry says what was delivered, how it
was checked, and what carries over.

## M1. Coordinator, interface, checks and logging, end to end on a toy task

- **Planned:** mid-November 2026 · **Done:** 26 October 2026
- **Delivered:** the `bioalloc` package: data contracts and task graphs, the model interface with
  a scripted fake model and a debug-only cache, budget accounts and the metered client, run traces
  and manifests, the common executor interface with the Direct executor, runtime checks, the
  coordinator with constant and random policies, and the two-leg route-planning toy task.
  `bioalloc demo` runs three scenarios: a clean run, recovery from a rejected answer, and a run
  that stops cleanly when its budget is spent. See `docs/architecture.md`.
- **How it was checked:** 78 offline tests in CI (ruff, strict mypy, pytest). Three rounds of
  review by GPT-6.1 Sol found 15 issues and then 7 more; all were fixed, each with a regression
  test that fails when its fix is reverted (`docs/decisions.md`, R1). Final review verdict: M1 done.
- **Limits:** all M1 runs use the scripted fake model. They show that the plumbing is correct and
  are not results.

## M2. Core executors, task suites, baselines, pilot (December 2026)

**Status (26 October 2026): everything up to the pilot is built; the pilot waits for the API key.**

- Decisions D10-D21 settle O1-O5 and the M2 design (`docs/decisions.md`).
- Executors: octopus, ant and bee with all Table 3 controls; the conventional workers, the single
  agent and the specialist coordinator; Direct's tool episode.
- Tools, typed tool calls, typed action spaces, stored decompositions, the Groq adapter with
  rate-limit retries (mocked HTTP only), selector features, LinUCB and a rule table.
- Task suites: GSM8K and PlanBench (pinned, hashed snapshots), our deterministic tool suite, the
  controlled synthetic suite. Split v2 of the 450 study tasks is frozen (D22; v1 was replaced after the review found tool subtasks shared across splits).
- Experiment runner, analysis and an offline rehearsal of the full pilot (720 runs).
- Still open before the pilot: the Groq key, the VAL cross-check, a hand-labelled sample of checker
  and grader verdicts, and the independent review of the M2 batch (`docs/pilot.md`). MATH is
  deferred to a later split version.

Carried over from M1 (all now addressed above):

- Stored LLM decomposition, replayed identically to every system, with its cost charged equally.
- Tool and runtime-check services in the execution context; typed tool-call records and tool
  events; finish reasons on replies.
- Groq adapter for GPT-OSS-120B (D3) with usage normalisation (reasoning tokens counted once).
- Open questions O1-O5 in `docs/decisions.md` (ant weights under the frozen protocol, Direct versus
  single agent, compute table units, detectable effect, bee update order).
- Executors: octopus, ant, bee, each with its plain counterparts (Table 3 of the plan).
- Task suites with fixed, hashed splits; baselines; a pilot on training tasks to measure the real
  cost of a task run. The pilot needs a Groq API key.
