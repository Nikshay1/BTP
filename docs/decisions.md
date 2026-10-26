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

- **Date:** 2026-10-26 · **Status:** active (wording corrected after the M1 review, see R1)
- **Decision:** the following are enforced by code and tests, not by convention. Where a rule is
  only partly implemented so far, the entry says so.
  1. Executors never see the provider transport. Every call goes through a metered client that
     reserves budget before the call and settles it from the provider's reported usage after. A
     call whose reservation does not fit is not sent. The reservation is a conservative estimate,
     not a guarantee: if the provider reports more than estimated and a limit is crossed (the
     task's or a single dispatch's), the overrun is recorded in the cost of that account and every
     ancestor, no further calls are admitted under the account that overran, and the run is
     reported as over budget (`within_budget = false`). A call cut off by a deadline before it
     reached the transport is not charged.
  2. Tokens are recorded by actor (coordinator or worker) and by purpose (local or inter-agent
     communication), as Section 4.4 of the plan requires (D6).
  3. Reference answers live only in the evaluation package; `Subtask` has no field meant for them
     and unknown fields are rejected. A test forbids system modules from importing the evaluation
     package. This is a guard against mistakes, not a proof: a generator could still put reference
     information into `Subtask.inputs`, so every task generator is reviewed for this.
  4. Measured runs never use cached replies. The metered client of a run in `measurement` mode
     refuses any reply marked as cached, whatever transport produced it, and the cache itself
     refuses to be built in that mode. Every run manifest records the mode.
  5. Every run writes an append-only JSON Lines trace and a manifest with the code revision,
     lockfile hash, configuration, the hash of the task graph, prompt hashes, model identifier and
     seeds. Each run gets its own directory and a trace file is never appended to by another run.
  6. All systems solve identical subtask graphs: each system receives a fresh copy parsed from the
     stored graph, and the graph's hash goes into the manifest. *Partly implemented:* for the toy
     family the graph comes from a deterministic generator. The protocol for LLM-generated
     decompositions (generate once per task and repetition, store, replay, and charge its cost
     equally to every system) is built in M2.
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

## D7. Concurrent subtasks share one budget pool, first come first served

- **Date:** 2026-10-26 · **Status:** active
- **Decision:** a dispatch is not given a fixed share of the task budget. All dispatches of a task
  draw from the task's pool. A call that does not fit only because other calls are in flight (their
  reservations are worst cases) waits for them to settle; it is refused only when the budget
  itself is spent. Waiting calls are admitted strictly in arrival order. A subtask refused for lack
  of budget is not retried. Ready subtasks are dispatched as soon as their dependencies are
  accepted, not in waves. When a run is cancelled or an executor raises, every other worker of the
  run is cancelled and settles its accounting before the run returns.
- **Why:** the M1 review (R1, findings 1 and 7) showed that the first version could abandon a
  subtask with 98% of the budget unspent, because sibling reservations were mistaken for spent
  budget, and could delay a ready subtask behind an unrelated slow one. The rule is the same for
  every system compared, so it does not favour any of them. A fixed per-subtask split is possible
  (`CoordinatorConfig.dispatch_budget`) if an experiment needs it.

## D8. Retries are told why the previous answer was rejected

- **Date:** 2026-10-26 · **Status:** active
- **Decision:** when a result is rejected, the next attempt at that subtask receives the rejected
  output and the public reasons the runtime checks gave, including the reasons of checks that could
  only say `unknown` when unknown results are not accepted. Reading the rejected output counts as
  communication; the fixed wrapper text and the check reasons do not (no agent wrote them). The
  same holds for upstream results: only the output itself is attributed to the upstream agent. Every system, baselines included,
  gets the same attempt allowance and the same feedback (`CoordinatorConfig.retry_feedback`).
- **Why:** blind resampling would make "recovery" depend only on sampling luck, and a deployed
  system would pass the error back. The review (R1, finding 10) pointed out that the first version
  retried blindly while the demo appeared to recover.

## D9. Policies receive structured feedback and define their own reward

- **Date:** 2026-10-26 · **Status:** active
- **Decision:** after each attempt the policy receives a record with the executor, status, every
  check result, the overall verdict, whether the result was accepted, the executor's confidence and
  the tokens spent. The coordinator does not collapse these into one number. The reward used by the
  LinUCB selector (M3) will be defined from these fields, keeping three things apart: acceptance by
  a strong check (constraint checker, tests), acceptance by a weak check (schema only, or
  `unknown`), and the training-only graded outcome of the parent task.
- **Why:** the review (R1, finding 9) showed that the first version gave the same reward of 1.0 to
  an unverified answer with confidence 0.01 as to a verified one; a learned selector would then
  optimise for passing weak checks rather than for being right. Acceptance is never reported as
  correctness.

## R1. Review of the M1 implementation (2026-10-26)

GPT-6.1 Sol (high effort) reviewed the eight M1 commits and reported 15 findings, verified with
probes. All were checked against the code and accepted. A second review of the fixes found five of
them only partly fixed and six new problems; those were fixed in a second round. A third review
found one remaining case (a batch holding both a cancelled and a failed worker left the failure
unobserved), fixed in a third round. Regression tests
are in `tests/test_review_regressions.py`. Each was mutation-checked: the fix was reverted and the
test confirmed to fail. The "every task exception is retrieved" property is tested by capturing
asyncio's unobserved-exception handler.

| # | Finding | Fix | Round |
|---|---|---|---|
| 1 | Sibling reservations treated as spent budget | Admission waits for in-flight calls (D7) | 1 |
| 2 | Wall-time limit not enforced while waiting | Waits bounded by the tightest deadline, dispatch caps included | 1, 2 |
| 3 | Cancellation leaked reservations | Calls that reached the transport settle as unknown usage; cancelling a run cancels its workers | 1, 2 |
| 4 | Empty route crashed the run | Checks made total over any output | 1 |
| 5 | Overrun reported as a normal success | `Cost.overrun` propagates from any dispatch to the run; `RunResult.within_budget` | 1, 2 |
| 6 | Reused run directory mixed two runs' traces | Unique run ids; traces opened exclusively | 1 |
| 7 | Subtasks dispatched in waves | Event-driven dispatch (D7) | 1 |
| 8 | Measurement mode did not stop cached replies | Enforced in the metered client (D5.4) | 1 |
| 9 | Reward collapsed weak acceptance into success | Structured feedback (D9) | 1 |
| 10 | Retries were blind | Feedback-assisted retries (D8); `unknown` reasons kept | 1, 2 |
| 11 | Model seeds unset; random policy order-dependent | Seeds derived from task, subtask and attempt | 1 |
| 12 | Unknown usage priced at the input rate | Charged as prompt estimate plus maximum completion | 1 |
| 13 | Raw inputs counted as communication | One message per source; wrappers separate from agent text | 1, 2 |
| N1 | Executor exception left siblings unobserved | Every finished task retrieved (cancelled ones included), others cancelled, a real error raised before a cancellation | 2, 3 |
| N2 | Call cut off before sending was charged | Charged only if it reached the transport | 2 |
| N3 | Admission not first come first served | FIFO admission queue per task | 2 |

Deferred to M2, before any comparative result is collected: the stored-decomposition protocol
(14), tool and check services in the execution context, typed tool-call records and tool events,
finish reasons on replies, and Groq usage normalisation (15).

## Plan amendments for M2 (2026-10-26)

The submitted research plan (`research-plan/main.tex`) is left as submitted. The entries below amend
it; each names the section it changes. They settle the open questions O1-O5 raised while planning
M1. Process as before: independent M2 plans from Claude Opus 5.5 and GPT-6.1 Sol (high), compared
and settled; where they differed, the entry says which was taken and why.

## D10. Ant: frozen initial weights, task-local learning (settles O1)

- **Date:** 2026-10-26 · **Status:** active · **Amends:** Sections 4.2 and 4.3
- **Decision:** the ant weights tau are a frozen development snapshot. Each task gets a private
  copy per subtask, updated by deposit and evaporation as paths finish, and discarded when the task
  ends; nothing is carried across tasks, systems or repetitions. In M2 the snapshot is uniform
  (tau = 1); a learned snapshot, if any, is fitted on training data in M3.
- **Ablations:** *no deposit* (delta = 0, same initial weights) isolates in-task reinforcement;
  *prior only* (delta = 0 and uniform weights) removes both. With a uniform snapshot the two coincide;
  if M3 learns a snapshot, both are reported. *No evaporation* is rho = 0. *Uniform sampler* is
  alpha = beta = 0 over the proposed legal candidates, keeping the proposal call so that the control
  removes the sampling bias, not the model computation.
- **Choice vocabulary:** weights are kept over a fixed set of typed action kinds per task family
  (e.g. Blocksworld operators, `CALCULATE`, `FINISH`), not over free-form steps, so that what is
  learned on one path applies to the next. Reinforcement transfers across action kinds, not across
  grounded plans; the paper states this.
- **Reward for a path:** 1 only when a public check verifies the goal (D15), else 0. On families
  without a goal check (GSM8K, MATH) ant therefore gets little reinforcement; that is measured and
  reported, not hidden behind a schema-based reward.
- **Second opinion:** both plans proposed the frozen-snapshot, private-copy rule; Sol added the
  distinction between no-deposit and prior-only and the per-distinct-action deposit.

## D11. Direct and the single agent (settles O2)

- **Date:** 2026-10-26 · **Status:** active · **Amends:** Sections 1.2, 4.2 (Table 3) and 4.4
- **Direct:** without tools, exactly one completion. With tools, one uninterrupted tool episode of
  at most four completions, including reading tool results back. No resampling, critique or
  check-driven revision inside it; the coordinator's outer retries still apply.
- **Single agent:** one persistent worker that works through the stored task graph serially in a
  fixed topological order, keeping its own history across nodes, with an act-observe loop, the same
  tools, public checks, retry allowance and task budget, and no roles or selector.
- **Why:** "one call with tools" cannot work when a tool result must be read. My first proposal
  (the single agent takes the whole task without the decomposition) would have made decomposition a
  confound; Sol's version keeps it identical across systems, so it was taken.

## D12. Units in the compute table (settles O3)

- **Date:** 2026-10-26 · **Status:** active · **Amends:** Section 5, Table 5
- **Decision:** three units are counted separately: task runs (system x task x repetition),
  subtask replays (executor x stored subtask context x repetition) and decompositions (task x
  repetition, shared by all systems). The replay set is 2,400 subtask executions, not 600 task runs.
  With D11, the main comparison has eleven systems (single agent, four executors alone including
  Direct, fixed, random, specialist, conventional adaptive, rule-based and learned adaptive), so its
  row is 11 x 210 x 3 = 6,930 task runs. Budgets are estimated from measured tokens per experiment,
  and actual spending (each decomposition paid once) is kept apart from the cost charged to each
  system.

## D13. Power analysis from paired pilot data (settles O4)

- **Date:** 2026-10-26 · **Status:** active · **Amends:** Section 4.4 (statistics, threats)
- **Decision:** the "about eight points" figure is replaced by a paired, clustered analysis. For
  systems A and B, d_i is the mean over repetitions of the success difference on task i; the minimum
  detectable effect at 5% two-sided and 80% power is about 2.8 s_d / sqrt(n). For one binary pair
  per task with discordance q this is about 2.8 sqrt(q/n): at n = 210, q = 0.1, 0.3, 0.5 gives about
  6, 11 and 14 points. These are sensitivity figures. Power curves are simulated with the intended
  analysis (paired bootstrap over template groups, three seeds, equal family weights) from pilot
  discordance, and refreshed with development data in M3, since the learned selector does not exist
  at the pilot. The cost claim needs the lower confidence bound of the success difference above
  -5 points together with lower cost.

## D14. Bee: inhibition before recruitment, synchronous rounds (settles O5)

- **Date:** 2026-10-26 · **Status:** active · **Amends:** Section 4.2 and Table 3
- **Decision:** k scouts answer independently; malformed or publicly invalid answers leave a scout
  uncommitted. Quorum is tested. Each later round, on a snapshot: (1) each committed scout with a
  rival receives one stop signal and is released with probability sigma; releases apply together;
  (2) uncommitted scouts are recruited to an option drawn in proportion to the remaining support,
  shown its best argument, and decide with one metered assessment call; decisions apply together;
  (3) quorum is tested. With no supporters left, uncommitted scouts answer independently again.
  Defaults k = 5, theta = 0.8 (quorum 4), sigma = 0.2, at most five rounds. Fallback: most support,
  then summed confidence, then a fixed order; whether quorum was reached is traced.
- **sigma = 0:** if every scout commits in round one the state is absorbing and the module returns
  at once; the no-stop-signal control then equals the initial k-sample vote. This degeneration is
  reported, not hidden.
- **Majority-vote control:** independent full answers until the bee run's call allowance or the
  token cap is spent (a call-yoked diagnostic). Equal calls do not mean equal tokens; both are
  reported, and the main comparison uses equal token caps.
- **Second opinion:** my first proposal used a confidence threshold for commitment; Sol's version
  (validity-based commitment, explicit absorbing state) is simpler and was taken.

## D15. Weak acceptance is not goal verification

- **Date:** 2026-10-26 · **Status:** active
- **Decision:** each runtime check declares the strength of its evidence: `format` (schema, type),
  `operational` (tool status, legal actions), `agreement` (between samples) or `goal` (a public
  checker that verifies the subtask's goal, such as a plan validator or supplied tests). Acceptance
  keeps the D5 rules, but stopping rules inside executors (octopus, ant) and policy rewards use only
  `goal` evidence. A schema pass alone never counts as a verified answer. An executor that has a
  finished candidate when its budget runs out returns it as completed and traces the stopping
  reason; `budget_exhausted` means there was no candidate.

## D16. Task families for the first pilot, and the rest later

- **Date:** 2026-10-26 · **Status:** active · **Amends:** Section 4.4 (tasks), if made permanent
- **Decision:** the pilot uses PlanBench Blocksworld (and Logistics if time allows), GSM8K, a small
  MATH subset, our own deterministic tool workflows and the controlled synthetic tasks. TravelPlanner
  and GAIA are deferred: TravelPlanner's evaluator re-downloads data and assumes full-split
  denominators, and GAIA is gated, restricts redistribution, and many "text" questions need the web.
  If they are left out of the main 450 tasks, that is recorded here as a scope amendment.
- **Data rule:** one preparation step may use the network; it pins upstream revisions, hashes every
  file and stores a local snapshot. Runs and CI never download. Pinned so far: GSM8K
  `openai/gsm8k@740312add88f781978c0658806c59bc2815b9866` (MIT).
- **Planning checks:** a Python simulator of the pinned PDDL domains gives the public checks (legal
  actions, goal reached). Grading uses the pinned VAL validator where it can be built; the simulator
  is cross-checked against VAL on fixtures.

## D17. Stored decomposition

- **Date:** 2026-10-26 · **Status:** active · **Amends:** Section 4.1; completes D5.6
- **Decision:** after splitting and before any system runs, each task x repetition gets one graph.
  Tasks that come with a natural work graph (toy, synthetic, our tool workflows) use it directly
  (`decomposer = deterministic`, no cost). Otherwise an LLM decomposer (actor coordinator, agent
  `decomposer`) sees only public task material and returns a DAG of at most 8 nodes and depth 6 whose
  final node keeps the original task's output contract. An invalid graph gets one repair call with
  the validation errors; if still invalid, the task becomes a single node. No task is dropped or
  regenerated for convenience. The graph is stored as canonical JSON with its SHA-256 and a
  preparation record (requests, replies, cost). Every system parses a fresh copy, verifies the hash,
  and starts with the decomposition's cost already charged to its budget; experiment-wide spending
  counts it once. GSM8K and MATH subtasks are never derived from reference solutions.

## D18. Baselines

- **Date:** 2026-10-26 · **Status:** active · **Amends:** Section 4.4 (systems compared)
- **Decision:** single agent (D11); each of Direct, ant, bee and octopus alone; fixed allocation
  (planning to ant, tools to octopus, reasoning to bee, locked before test); random allocation over
  the four, seeded by task, subtask and attempt; specialist coordinator (a metered planner gives
  bounded per-node advice, an act-observe worker executes, a metered critic may ask for one
  revision; all role costs count; no graph rewriting); conventional adaptive (the same LinUCB over
  Direct, act-observe, majority vote and plan-then-execute workers); rule-based and learned bio
  allocation. The act-observe worker shares code with octopus wherever their behaviour coincides, so
  that the conventional baseline is not artificially weak. Octopus and an ordinary bounded
  act-observe worker are close computationally; the paper says so, and RQ3 rests on the
  conventional adaptive comparison and the component controls.

## D19. Groq adapter (no key rotation)

- **Date:** 2026-10-26 · **Status:** active
- **Decision:** one API key, Developer tier with a spend cap (pending Nikshay's decision on the
  key). No multi-account key rotation: limits are per organization, and the provider's acceptable
  use policy reportedly forbids circumventing them with several accounts. A 429 is a
  `RateLimited` rejection (nothing charged); the client waits at least `Retry-After` (otherwise
  exponential backoff from 1 s, capped at 30 s), stays within the deadline, re-enters FIFO admission,
  and gives up after three retries. Timeouts and 5xx errors that may have consumed tokens are not
  resent automatically. Reasoning tokens are read from `completion_tokens_details` and counted once;
  missing usage is treated as unknown, not zero. The seed is best effort on Groq. Tested offline with
  a mocked HTTP transport; the first live step is a small charged capability check.

## Open questions found while planning (settled by D10-D14)

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
