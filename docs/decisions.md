# Decision log

Every decision that shapes the code, the experiments or the paper is recorded here with its date,
who made it and why. A paper reviewer may ask why something was done a certain way; this file is
where the answer lives. When a decision changes the research plan, the entry names the section.

Process: the guide asked the group on 26 October 2026 to decide the details themselves. Each
decision is made by the group after an independent second opinion from a separate model
(GPT-6.1 Sol, high reasoning effort), and disagreements are written down.

Status values: **active**, **superseded** (by a later entry), **open** (needs a decision before the
named milestone).

---

## D1. Build M1 fully offline, with a scripted fake model

- **Date:** 2026-10-26 · **Status:** active
- **Decision:** Milestone M1 (coordinator, common interface, runtime checks and logging, end to end
  with one executor on a toy task) is built and tested without any API key. A deterministic fake
  LLM returns scripted replies with scripted token usage.
- **Why:** M1 tests orchestration correctness, not model quality. Offline tests are free, repeatable
  and run in CI. The M1 demo is described as evidence of plumbing only and never as a result.
- **Second opinion:** Sol proposed the same; both plans had near-identical first steps.

## D2. Package name `bioalloc`, Python 3.12, uv, MIT licence

- **Date:** 2026-10-26 · **Status:** active
- **Decision:** One Python package `bioalloc` under `src/`. Python pinned to 3.12 (the system Python
  is 3.15, too new for parts of the scientific stack). Dependencies locked with uv (`uv.lock` is
  committed). Ruff, mypy (strict) and pytest run in CI. Code released under MIT so that it can
  accompany a paper.
- **Why:** a locked environment is the minimum for a reproducible paper artifact. The orchestration
  layer is written by us, not taken from an agent framework, because every message must be metered
  (research plan, Section 5).

## D3. Backbone model: GPT-OSS-120B served by Groq

- **Date:** 2026-10-26 · **Status:** active (chosen by Nikshay)
- **Decision:** the primary hosted backbone from the December pilot onward is `openai/gpt-oss-120b`
  on Groq. Listed price at the time of the decision: US$0.15 per million input tokens and US$0.60
  per million output tokens (Groq model page, checked 2026-10-26).
- **Why:** an open-weight model whose weights do not change under us, at a low price. The 20B
  variant (half the price) was considered and rejected by Nikshay in favour of the stronger model,
  to keep task success away from the floor where mechanisms cannot be told apart.
- **Consequence for the budget:** at the plan's own token estimate (165-310 million tokens, four
  fifths input) the hosted cost is about US$40-75, or up to about US$97 with the plan's 30%
  allowance, roughly twice the figure in Section 5 of the plan. GPT-OSS also emits reasoning
  tokens, billed as output; the pilot measures them. Section 5 will be revised after the pilot.
- **Open follow-up (before M2):** the plan's open-weight replication on a free T4 cannot run a 120B
  model. Options: replicate the primary comparison on GPT-OSS-20B locally on a cloud T4, or on a
  second provider serving the same 120B weights. To be decided with the pilot data.
- **Settings to fix before the pilot:** reasoning effort (start at `low`), temperature, maximum
  completion tokens. These are held equal across all systems.

## D4. Toy task for M1: two-leg route planning

- **Date:** 2026-10-26 · **Status:** active (chosen by Nikshay, proposed by Sol)
- **Decision:** the M1 toy task is route planning on a small directed graph in two dependent legs
  (S to M, then M to T) followed by a deterministic join. A public checker verifies edges and
  endpoints.
- **Why:** it exercises dependencies, context passing, runtime checks, retries, aggregation and
  budget exhaustion, and its `move(node)` choices form a fixed vocabulary that the ant executor can
  reuse. It is excluded from all reported results.

## D5. Validity rules built into the code from the start

- **Date:** 2026-10-26 · **Status:** active
- **Decision:** the following are enforced by code and tests, not by convention.
  1. Executors never see the provider client directly. Every call goes through a metered client
     that reserves budget before the call and settles it from the provider's reported usage after.
     A call that could exceed the remaining budget is not sent.
  2. Tokens are recorded by actor (coordinator or worker) and by purpose (local or inter-agent
     communication), as Section 4.4 of the plan requires.
  3. Reference answers live only in the evaluation package. Subtasks and execution contexts have no
     field that could carry them.
  4. The response cache is for debugging and replay only. A run in measurement mode refuses to use
     it, and every run manifest records the mode.
  5. Every run writes an append-only JSON Lines trace and a manifest with the code revision,
     lockfile hash, configuration, prompt hashes, model identifier and seeds.
  6. One decomposition per task and repetition is generated once, stored, and replayed to every
     system, so that all systems solve identical subtask graphs (Section 4.1, shared decomposer).
- **Second opinion:** points 1, 3, 4 and 6 were raised independently by both plans; Sol added the
  reservation-then-settle accounting and the separate purpose view of tokens.

## D6. How tokens are attributed

- **Date:** 2026-10-26 · **Status:** active
- **Decision:** each model call is tagged with its actor, coordinator or worker, and all of its
  tokens go to that actor. Separately, the prompt tokens a call spends on content written by
  another agent (another worker's output, or the coordinator's instructions when they were written
  by a model) are counted as **communication**. Communication is a subset of the actor totals, so
  both views add up to the same total and nothing is counted twice. Generating a message counts as
  the sender's local work; each time another agent reads it, the reading is communication.
- **How the split is computed:** providers report prompt tokens per call, not per message. The
  communication share of a call's prompt is estimated from message provenance in proportion to the
  characters of the messages written by other agents. The residual error of this estimate is
  documented alongside results.
- **Why:** Section 4.4 of the plan reports coordinator, worker and communication tokens separately,
  and RQ5 depends on those numbers being well defined.
- **Second opinion:** Sol recommended an actor view and a disjoint purpose view, with generation and
  each reading billed as separate occurrences; this entry follows that.

## Open questions found while planning (to settle before M2)

These came out of Sol's critique of the plan. They need a decision and, where noted, a change to
the research plan.

- **O1. Ant weights and the frozen protocol.** Section 4.3 freezes the ant weights after
  development, but the ant mechanism (Section 4.2) updates its weights after each path. Proposal:
  freeze the learned snapshot across tasks, let each task update a private copy, and discard it
  when the task ends. Also, setting the deposit to zero does not reduce sampling to the prior if the
  starting weights are not uniform; the ablation must reset them to uniform.
- **O2. Direct versus single agent.** "One model call, with tools" for Direct needs a precise
  meaning when a tool result must be read back. The single-agent baseline also needs a definition of
  how it uses the shared task graph.
- **O3. Mixed units in the compute table.** Table 5 counts 2,400 replay subtask runs as 600 task
  runs. Budgeting will be done from measured tokens per experiment instead.
- **O4. Detectable effect size.** The "about eight points" figure in Section 4.4 depends on how often
  systems disagree on the same task, not only on the number of tasks. It will be re-estimated from
  pilot data.
- **O5. Bee with all scouts committed.** If every scout commits in round one, recruitment has no
  uncommitted scouts to act on unless inhibition runs first; with the sigma = 0 ablation it never
  does. The update order must be fixed before tuning.
