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

- **Date:** 2026-10-26 · **Status:** suspended for the pilot by D23; the main-study backbone is
  decided after the pilot
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
- **Update 2026-10-26 (PlanBench subset):** PlanBench at commit `fc638a1` is prepared from the
  instance sets its own configs name as the main ones (`generated_basic`: Blocksworld 1-500,
  Logistics 1-285; 523 template groups). The first preparation used the `generated` directories by
  mistake; a check of the template groups showed that every Blocksworld instance there stacks all
  blocks from the table into one tower, so the 501 files hold only 10 distinct problems (one per
  block count). They are PlanBench's plan-generalization set and are not used. Grouping was verified
  by exact isomorphism tests on sampled groups (122 pairs, no mismatch).
- **Consequence for the test-only variant:** the main Blocksworld set has only 4 and 5 blocks
  (445 and 55 instances), so "a larger-object-count bucket absent from development" means the
  5-block instances. To be fixed with the splits.
- **VAL status:** building VAL from source (KCL-Planning/VAL, commit `3c7a1f3`) was blocked by
  this environment's permission rules for running external code. Until Nikshay builds it, the
  simulator is both the public check and the grader, and the paper must say so; the cross-check
  against VAL is a pilot-readiness item.

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

## D20. Grouped splits v1 and the test-only variants

- **Date:** 2026-10-26 · **Status:** superseded by D22 (v1 leaked tool subtasks across splits)
- **Amends:** Section 4.4 (tasks)
- **Decision:** the 450 study tasks are split once into `data/splits/v1.json` (SHA-256
  `8b38f35b966e...`, recorded in the file): per family 50 training, 30 tuning, 50 ordinary test and
  20 test-only variant tasks. Whole template groups are assigned in SHA-256 order of (split
  version, seed 20261026, group id); no group crosses splits. The variants, fixed before any model
  run:
  - planning: the 5-block Blocksworld instances (development has only 4-block Blocksworld and
    Logistics);
  - tools: the `compare` template (two parallel chains joined at the end), a composition topology
    absent from development;
  - reasoning: the longest fifth of GSM8K test questions by word count.
- **Pools:** PlanBench main sets (Blocksworld 1-500, Logistics 1-285); tool-suite seeds 0-299 per
  template; GSM8K upstream test split (1,319 problems).
- **Trade-off:** with `compare` held out, every development tool task is a chain of three
  subtasks; parallel siblings are studied through the synthetic suite (width factor) and appear in
  tools only at test time. MATH is not in v1; adding it later means a new split version, never an
  edit to v1.
- **Statistics:** planning has more tasks than groups (isomorphic duplicates stay together), so the
  bootstrap resamples groups, as Section E of Sol's plan requires.

## D21. Pilot design

- **Date:** 2026-10-26 · **Status:** active
- **Graphs per family:** tools use their natural workflow graphs; reasoning (GSM8K) uses the model
  decomposer from the question alone; planning (PlanBench) is a single node, because a split of a
  planning problem into goals cannot be validated and an invalid split would hurt every system
  equally but uninformatively (Sol's advice in the M2 plan).
- **Budget cap:** the plan's primary cap is twice the single agent's mean cost, which is unknown
  until the pilot. The pilot uses a generous provisional cap (120,000 tokens, 60 model calls, 40
  tool calls, 10 minutes per task), records actual use, and the main study's cap is fixed from the
  pilot's single-agent runs before any test task is touched.
- **Systems:** single agent, the four core executors alone, fixed, random and specialist; the
  learned selectors and the Table 3 controls are left out of the pilot to keep its cost to a few
  dollars, and are rehearsed offline.
- **Spending stop:** live runs take `--max-usd`; the pilot is run with `--max-usd 5`.

## D22. Split v2 replaces v1

- **Date:** 2026-10-27 · **Status:** active · **Supersedes:** D20's split file (its variants and
  quotas stay)
- **Problem:** in v1 the three tool templates used the same database seeds and every tool task was
  its own template group. Tasks on the same seed share a database, so the first subtask of a
  held-out `compare` task could be identical to a subtask of a training task; a check found three
  such subtasks. Found by the M2 review (finding 7).
- **Decision:** `compare` uses database seeds 1000-1299, which no other template uses; revenue and
  top_product keep seeds 0-299; tool tasks are grouped by database, so the two tasks on one
  database always share a split. Split v2 (`data/splits/v2.json`, SHA-256 `bae17926c64d...`) is
  frozen with the same quotas and variants as D20. A test asserts that no database and no subtask
  appears in two splits. No model run ever used v1.
- **Consequence:** the tool family's development tasks now come in pairs on 25 + 15 databases.

## R2. Review of the M2 implementation (2026-10-26/27)

GPT-6.1 Sol (high effort) reviewed the M2 commits up to the offline rehearsal (a first attempt was
cut off by model capacity after four findings, fixed in `114538f`). The full review reported 19
findings, each with a probe. All were checked against the code and accepted.

| # | Finding | Fix | Commit |
|---|---|---|---|
| 1 | Decomposer saw input names, not values; a plan could drop the question | Decomposer sees the public task; final node keeps all inputs and the original instruction | `fc7f7cc` |
| 2 | Bee/majority found no usable answer on intermediate nodes | Intermediate nodes compared by canonical JSON | `fc7f7cc` |
| 3 | Ant's 12-choice paths too short for a 12-action plan in the pool | Up to 30 choices; the task budget binds | `26ac1f1` |
| 4 | Ant's prompt said FINISH takes no arguments | Every action space describes its kinds and arguments | `26ac1f1` |
| 5 | Ant ran tools directly, outside the tool service | Tool-backed actions go through the metered ToolService; legality is structural | `26ac1f1` |
| 6 | Calculator errors and complex results crashed runs | All become CalcError or failed tool observations | `1f1b809` |
| 7 | Tool templates shared databases across splits | Split v2 (D22) | `aff17c5` |
| 8 | Interrupted runs could not be resumed | Interrupted runs set aside with their spending, then rerun | `4afbd1a` |
| 9 | Resume could mix configurations | Provenance check; dirty trees refused in measurement | `4afbd1a` |
| 10 | Reporting control skipped successful tool episodes | Octopus step = one completion; report after every non-final step | `62ddd0f` |
| 11 | `--max-usd` counted attributed, not actual, spending | Persistent spending ledger, decomposition paid once | `4afbd1a` |
| 12 | CLI analysis ignored groups and family weights | Family-stratified group bootstrap from the split file | `4afbd1a` |
| 13 | Imported costs ignored call, tool and time limits | All limits and deadline checked | `a0bcabb` |
| 14 | Runs did not verify data snapshots | Loader verifies manifests and split provenance | `4afbd1a` |
| 15 | Ant ignored the configured completion limit | Passed through | `26ac1f1` |
| 16 | Specialist lost its answer if the critic could not run | Finished answer kept | `a0bcabb` |
| 17 | Learned systems ran untrained | Refused in measurement mode until M3 artifacts exist | `4afbd1a` |
| 18 | `majority` control not call-yoked to bee | **Gated**: to wire before the Table 3 control experiments | open |
| 19 | Typed PDDL accepted without type checks | Typed lists refused | `62ddd0f` |

## R3. Re-review of the R2 fixes (2026-10-27)

Sol found findings 8, 9, 11 and 13 only partly fixed and five new problems in the fixes: spending
appended to a ledger was not crash-safe or idempotent; the spending stop cancelled runs in flight;
two runner processes could set aside each other's runs; a populated directory without provenance
was adopted; an imported time overrun was reported as within budget; octopus ran tools on its last
step. Fixes: actual spending is computed from the per-call traces (scanned once per run, then
updated), the stop lets runs in flight finish, a pid lock allows one runner per directory, a
populated directory without provenance is refused, time overruns are reported, and the last step
runs no tools. A test now checks the frozen split file itself for tool leaks.

A final check found four remaining crash-safety gaps, fixed in turn: the results file is rewritten
atomically without a torn line on resume; spending scans tolerate a torn last trace line and
stored decompositions are written atomically; the directory lock is an OS file lock (released by
the kernel if the runner dies); trace events are fsynced so spending survives a power loss. A last
pass handled a crash that cuts a multi-byte UTF-8 character, fsyncs the rewritten results file and
directory entries, and limits fsync to spending-relevant events (about 14 ms per run on btrfs).
Remaining risk is limited to a power cut at an exact moment; it is accepted for the pilot.

## D23. Pilot backbone: Claude Sonnet 4.6 on Amazon Bedrock, low effort

- **Date:** 2026-10-29 · **Status:** active (backbone, effort and budget chosen by Nikshay)
- **Amends:** D3, D19 (provider), D21 (size and spending stop)
- **Decision:** the pilot runs on Claude Sonnet 4.6 through Amazon Bedrock, paid from about
  US$200-250 of AWS credit. Model ID `global.anthropic.claude-sonnet-4-6` (global cross-region
  profile, no regional surcharge). Listed price US$3 per million input tokens and US$15 per million
  output tokens. Effort `low` (`output_config.effort`), thinking off, temperature 0.6, at most 2,048
  output tokens per call, the same for every system.
- **Why thinking is off:** Claude requires temperature 1 when thinking is on; keeping it off keeps
  the D3 temperature and spends no hidden output tokens. Low effort was Nikshay's choice.
- **Budget:** at Sonnet prices the plan's 165-310 million tokens would cost about US$1,000-1,900,
  far beyond the credit. Nikshay chose to run a smaller pilot first and decide the main-study
  backbone from its measured cost per run: either Sonnet at a reduced study size, or a cheaper model.
- **Pilot size:** a 24-run smoke test (`configs/pilot-smoke.json`: one task per family, one
  repetition, stop at US$5) to catch broken prompts cheaply, then `configs/pilot.json` (pilot-v2):
  6 tasks per family, 2 repetitions, 8 systems, 288 runs, stop at US$40. The offline rehearsal
  gives a lower bound of US$5.2; real replies are longer, so the expected cost is about US$15-40.
- **Adapter:** `bioalloc.llm.bedrock` calls `invoke_model` through boto3 with the Anthropic
  Messages body (`anthropic_version: bedrock-2023-05-31`). The Anthropic SDK is not used because its
  current releases send an API version that Bedrock rejects for Sonnet 4.6. botocore retries are
  off; throttling and quota errors are `RateLimited` and retried by the metered client as in D19
  (Bedrock sends no retry-after, so backoff starts at 1 s). The seed is not sent (Claude has none),
  so repeated requests are independent samples. Prompt caching is off, so the ledger's two prices
  are exact. Tested offline with a stand-in boto3 client; the first live step is the capability
  check.
- **Spending stops** are thresholds: runs in flight finish after the threshold is passed, so the
  pilot can end up to about US$7 above US$40 (four runs at their token cap). Accepted: the total
  stays far below the credit.

## R4. Review of the Bedrock switch (2026-10-29)

GPT-6.1 Sol (high effort) reviewed commit 9b2b6ce and found no invalid request reachable from the
executors, but five issues, handled as follows:

1. The decomposer ignored the configured effort and completion limit, so reasoning-task
   decompositions ran at the provider's default effort. Fixed; regression test fails without it.
2. The capability check answered only the first of several tool calls, which the provider rejects.
   It now answers every call, and fails (exit code 1) when the tool is not called, the read-back
   is wrong, or a reply is cut off. Regression test with two parallel calls.
3. Spending stops are thresholds, not ceilings. Documented (D23, `docs/pilot.md`).
4. When usage is unknown the ledger charges its prompt reservation, which left out the hidden
   tool-use prompt that providers bill. Reservations now add 600 tokens when tools are offered
   (Anthropic documents about 350), and the capability check prints the estimate next to the
   real prompt count.
5. Money tests did not check the ledger. Added: exact dollar charge after throttling, full
   reservation charged and never resent for 424, 408, 500 and read timeouts.

## R5. First live smoke test and its review (2026-10-29)

`configs/pilot-smoke.json` (24 runs, US$1.26) on Sonnet 4.6. Reviewed independently by Claude
(Opus 5.5) and GPT-6.1 Sol (high); both found that every failure came from how replies were read,
not from the mechanisms. Ledger, traces and results agreed on every run (Sol reconstructed all
totals and regraded every output).

1. Ant (0/3, and every failed fixed/random run, which had picked ant): the model wrote candidates
   flat (`{"kind": "QUERY", "name": ...}`) or as a bare line (`FINISH {"answer": 25}`, the right
   answer), copying how the allowed actions were listed; the parser dropped them silently. Fix:
   the actions are now listed in the exact `{"kind": ..., "args": {...}}` form, the prompt says so,
   the unambiguous flat form is read as kind plus args, and every rejected candidate is traced
   (`ant_rejected`) with its reason.
2. Bee on planning (60 calls, budget exhausted) and Direct inside random: with thinking off the
   model often answers, writes "Wait, c is not clear...", and answers again. Reading from the
   first "{" to the last "}" gave invalid JSON. All 60 bee replies ended with a plan that passes
   the simulator. Fix: `parse_json` takes the last top-level JSON object, a fixed rule for every
   system that never looks at whether an object passes a check.
3. Prompt reservations fell below real prompt tokens on 117 of 354 calls (worst 7.4%, planning).
   Fix: 2.5 characters per token instead of 3; a test replays all 183 distinct smoke prompts and
   requires a 10% margin.
4. Task selection took the first ids in order: only Blocksworld, only `revenue`, short GSM8K
   questions, and two planning tasks from one group. Fixed by D24.
5. Ant could not call `list_tables`, which every other executor may call. Added as LIST_TABLES.

Regression tests use the real replies and prompts from the traces (`tests/fixtures/`), and each
fails when its fix is reverted. Observed but not yet acted on: replies to identical requests are
nearly identical at temperature 0.6, so bee's scouts and repeated runs may be less independent
than planned; measured in the second smoke test.

## D24. Pilot task selection by fixed rules

- **Date:** 2026-10-29 · **Status:** active
- **Decision:** the pilot's tasks are listed explicitly in the configs, chosen from training by
  `scripts/select_pilot_tasks.py`: per family two subtypes with N tasks each (planning:
  Blocksworld and Logistics; tools: `revenue` and `top_product`), and for reasoning 2N GSM8K
  questions at or above the training median word count, since short questions are likely at
  ceiling for Sonnet. Within a subtype, tasks are taken in SHA-256 order of a fixed seed and the
  task id, never two from one group. N = 3 for the pilot (18 tasks), N = 1 for the second smoke
  test (6 tasks, a subset of the pilot's). Only public text is used; split v2 is unchanged.
  A test checks that both configs hold exactly this selection.

## R6. Review of the R5 fixes and a research plan (2026-10-29)

GPT-6.1 Sol (high) reviewed commit 613c386 and proposed a plan towards the project goal (bio
systems that answer better than a single agent, preferably with fewer tokens).

1. D24's "longer GSM8K" rule counted the instruction, which is the same 18 words for every task,
   so every question passed. Fixed: words of the problem text. The pilot's reasoning tasks changed
   (four of six had been below the median). The test now checks the rule independently.
2. The first "last object" parser could take a fragment from inside a malformed correction, or
   split a fenced array into its last element. Replaced by a scanner that tracks brackets and
   strings, takes only top-level values, never looks inside one that fails to parse, and gives no
   answer when the last top-level object is malformed.
3. "Last object" cannot tell a correction from a trailing example. Every JSON prompt now asks for
   the corrected object last and no example objects; ant reads the last object holding its
   `candidates` envelope, so a quoted action after it no longer replaces it. The remaining case
   is pinned by a test as a known limit.
4. Ant now traces cut-off replies and empty candidate lists.

Plan adopted (in order): a small diversity check (D25); the repaired smoke test at temperature 1.0;
then a difficulty check before the 288-run pilot, because GSM8K and Blocksworld may be at ceiling
for Sonnet. Later, as separately versioned changes: bee with three scouts, ant loop detection and
path limits, stopping on a public goal certificate, split v3 with harder tasks (MATH, longer
Logistics, branching tool workflows) if the difficulty check confirms the ceiling, and evidence of
complementarity (oracle versus best fixed executor) before more work on the selector. Noted
structural limit: ant receives no reward where no goal check exists (tools, GSM8K), so its
deposit and evaporation controls cannot differ there.

## D25. Sampling: temperature 1.0 for every system; thinking stays off, effort low

- **Date:** 2026-10-29 · **Status:** active from the second smoke test; checked by the diversity
  check
- **Amends:** D23 (temperature 0.6)
- **Decision:** temperature 1.0 (Claude's default) for every system. Thinking stays off and
  effort low.
- **Why:** in the first smoke test, identical requests returned near-identical replies at 0.6,
  and all 60 bee planning scouts reached the same plan. Bee's scouts and repeated runs are only
  useful if their errors differ. Claude has no seed, so temperature is the only sampling control.
  Thinking stays off: the model already corrects itself in visible text (which the parser now
  reads), thinking would require temperature 1 anyway and adds hidden output cost and adapter
  work; it can be compared later on harder tasks if failures turn out to be reasoning failures.
- **Check:** `configs/diversity-t06.json` and `configs/diversity-t10.json` run Direct 4 times on
  each of 4 pilot tasks at each temperature (32 runs); `scripts/diversity_report.py` compares
  distinct answers and accuracy. If 1.0 lowers accuracy, the choice is revisited before the pilot.

## R7. Second review of the parser and the diversity check (2026-10-29)

GPT-6.1 Sol (high) reviewed commit 25ce3d4: "fix first". All seven findings fixed:

1. A mismatched bracket let recovery resume inside the broken object and expose a nested
   fragment. Recovery now skips to where the outer bracket's own kind balances, and a container
   that never closes ends the scan.
2. The diversity check mixed sampling with per-repetition model decompositions and check-driven
   retries. Both configs now use single-node tasks and one attempt; the report verifies that every
   repetition sent the same request body.
3. A valid envelope followed by a broken correction returned the earlier envelope. A malformed
   container after the last envelope now means no answer.
4. Prose braces (`\frac{1}{2}`, `{n}`) were treated as malformed answers. Only containers that
   look like JSON count: an object opening with a key or `}`, an array opening with `{`, `[` or a
   string. `[1]` or `[note]` in prose are ignored.
5. The report counted raw answer objects; it now counts the family's normalized options (what
   bee counts), unreadable answers separately, and reply texts per run.
6. Other protocol replies had the trailing-object problem fixed for ant: advice, decomposition,
   reviews and assessments, and supervision now read their envelope (`advice`, `nodes`,
   `accept`, `decision`). Solver answers keep the general rule.
7. A huge integer literal raised an uncaught error that could stop an experiment; it is now an
   unreadable answer.

## R8. Third review of the parser (2026-10-29)

GPT-6.1 Sol (high) re-checked the R7 fixes on all 354 smoke replies (the JSON-looking rule
rejected no real answer) and found two remaining problems, both fixed:

1. A stray closer inside a broken correction (`{"steps":[},"final":{"answer":0}}`) still ended
   recovery early and exposed a fragment, for answers and all five envelope keys. Bad nesting now
   ends the scan: where such a container ends cannot be known, so nothing after it is read.
2. The diversity report compared every request, including tool continuations that carry
   provider-generated ids, and counted intermediate replies as answer texts. It now compares each
   run's first request and its final reply.

Final re-review of commit 5043a9f: no new findings (213 adversarial assertions, all 354 smoke
replies parsed exactly as before, diversity report verified on repetitions with tool
continuations). Cleared for the diversity check and the second smoke test.

## R9. Diversity check and second smoke test (2026-10-29)

Runs: `runs/diversity-t06`, `runs/diversity-t10` (32 Direct runs, about US$0.33) and
`runs/pilot-smoke-2` (48 runs, US$3.55 counting shared decompositions once, of which US$0.18 is
imputed for five timed-out calls). Reviewed by Claude and GPT-6.1 Sol (high); traces, results
and recomputed grades agree on all 80 records.

- **No parser-induced failures observed.** The failures are substantive: Direct's Logistics plans
  break action preconditions (0 of 10 valid), ant answers GSM8K by guessing (no calculator, no
  reward signal; 75 instead of 315) and exhausts its reservations on Logistics (US$0.51), and five
  bee scout calls hung for 120 s (one, then four together). Successful replies never took more
  than 9 s (1,688 tokens in 8.8 s), so the hang is not slow generation; cause unknown.
- **Ceiling on these samples.** The single agent solved 6/6 at US$0.023 per run. Octopus 6/6 at
  1.09 times its cost, bee 6/6 at 4.4 times, ant 4/6 at 5.7 times. Even a hindsight oracle
  picking the cheapest correct bio run per task costs more than the single agent: no present
  reason to train a selector. Single-node tasks give octopus nothing to exploit over the single
  agent, which uses the same worker.
- **Temperature 1.0:** no observed accuracy loss (12/16 at both); on Logistics both temperatures
  emit 3 distinct invalid plans, with somewhat different errors at 1.0. Kept (D25).
- **Savings found in traces:** fixed's ant found a verified Logistics plan after 24 calls and spent
  11 more (36% of the run); bee had a verified plan in round 0 and paid four recruitment calls to
  reach quorum. Stopping on a public goal certificate is the first mechanism change to try.

Fixed: the diversity report now reads every attempt's output (emitted, unreadable and accepted
options separately; it had reported 0 distinct Logistics plans instead of 3); reservations use
1.75 characters per token, which keeps all 580 distinct live prompts at least 10% above their real
count (2.5 fell short on 114 Logistics prompts, worst 2,225 reserved against 2,711); the analysis
table shows runs with imputed usage; paired comparisons flag a degenerate result (no discordant
pairs), which is not evidence of equivalence. Planning checks and grading share one simulator,
so their agreement is not independent validation (VAL remains open).

Decision: pilot-v2 (288 runs) is not run as configured. Next is a baseline-only calibration (D26).

## D26. Difficulty calibration before any mechanism comparison

- **Date:** 2026-10-29 · **Status:** active
- **Replaces:** running pilot-v2 as configured (D24 tasks stay as the panel's core)
- **Decision:** `configs/calibration.json` runs only the single agent and Direct, once, on 30
  training tasks: the 18 D24 pilot tasks plus the 12 training Logistics problems with the most
  goal atoms (6 or 7 packages, against 5 in the smoke test), never two from one group, chosen by
  `scripts/select_calibration_tasks.py`; a test checks the rule independently. 60 runs, about
  US$2-3, stop at US$6.
- **Why:** the comparison only means something where the single agent itself sometimes fails.
  The target development region is a single-agent success rate of roughly 60-85%, judged per
  named subset with `scripts/calibration_report.py` (each core family, the 12 extra Logistics,
  all 15 Logistics), never on the pooled panel. One repetition only screens; a promising subset
  is confirmed with more samples before any comparison is run on it. If the
  calibration finds it (likely in Logistics), the next step compares single agent, octopus and a
  bee with certificate stopping there; if not, harder task families (branching tool workflows,
  MATH) go into a split v3 before more paid runs.

## R10. Review of the R9 changes (2026-10-29)

GPT-6.1 Sol (high) cleared the calibration run and raised two points, both handled:

1. The wider prompt margin (1.75 characters per token) can refuse a call the budget could pay
   for: replaying octopus on Logistics-095 at twice the smoke single-agent mean (10,570 tokens),
   its third, successful call no longer fit (5,658 spent + 5,718 reserved). New rule for every
   system: when a call's full reservation does not fit but its prompt estimate plus 256 tokens
   does, it is sent with its completion limit cut to what is left (`completion_fitted` in the
   trace). An outcome-level test replays octopus's recorded replies at that cap and requires the
   same correct answer at 9,173 tokens.
2. A pooled success rate can hide a hard subset (15 Logistics against 15 others). The calibration
   is reported per named subset (`scripts/calibration_report.py`) and D26 judges its target that
   way.

## R11. Re-review of the fitting rule (2026-10-29)

GPT-6.1 Sol (high) found one new problem: the completion limit was fitted once, before a call
queued for admission, so a call waiting behind a sibling kept a limit that no longer fit after
the sibling settled and spent budget, and was refused (with 1,000 tokens, two calls of 100 + 800:
the second failed with 800 left, while the same call arriving later would have been fitted to
700). Fitting now happens inside FIFO admission and is recomputed each time the budget changes;
the call keeps its place in line, and the admitted limit is used for the reservation, the request
and the trace alike. A concurrent test reproduces the case and fails on the old code.

Re-review of commit c73f66e: no new findings (FIFO fairness, sibling and ancestor budgets,
wakeups, retries, deadlines, call limits, small completion limits, ledger consistency, and the
D26 config, tasks and snapshot hashes). Cleared for the calibration run.

## R12. Difficulty calibration results (2026-10-29)

`runs/calibration-v1`: 60 runs, US$2.49, no errors, timeouts or imputed usage. Reviewed by Claude
only (GPT-6.1 Sol reviews were paused for the rest of the session at Nikshay's request, to save
his Codex usage; a Sol review of this and D27 is owed).

| Subset | Single agent | Direct |
|---|---|---|
| Core planning, tools, reasoning (6 each) | 6/6 each | 6/6 each |
| Extra Logistics (12) | 12/12, US$0.096/run | 5/12, US$0.044/run |
| All Logistics (15) | 15/15, US$0.087/run | 8/15 |

At the generous 120,000-token cap the single agent is at ceiling everywhere. On Logistics it gets
there by revising against the simulator's rejection: only 2 of 15 first plans were valid, and the
hardest problems took 4 to 8 calls and up to 47,000 tokens. Truncating the recorded runs at a
token cap (a counterfactual, one repetition) gives the single agent 9/15 Logistics at 12,000
tokens, 10/15 (67%) at 16,000-25,000, 13/15 at 30,000 and 15/15 at 50,000. The research plan's
primary cap, twice the single agent's mean (10,532 tokens over the panel), is about 21,000 tokens,
where the single agent sits at 67% on Logistics: inside the D26 target region. Direct stays at
8/15 at every cap.

## D27. Stop on a certificate; Logistics comparison at the primary cap

- **Date:** 2026-10-29 · **Status:** active (implemented; Sol review owed)
- **Amends:** D10 (ant stopping), D14 (bee stopping), D18 (majority vote)
- **Decision 1:** bee, its no-stop control and majority vote return a goal-verified answer
  (D15: a goal-level check passed, none failed) at the next round boundary (when the scouts
  launched together have returned; R14), and ant stops at its first
  goal-verified path (its weights are still updated). Octopus and the single agent already stop
  this way. Each has `stop_on_certificate` (default on) to restore the earlier rule. Why: in the
  second smoke test, fixed's ant spent 36% of its run searching after a verified plan, and bee
  paid four recruitment calls after a verified plan in round 0. The certificate is public (the
  plan simulator), so this uses no private information; it only matters where a goal check
  exists (planning), and ant's learning signal is unchanged.
- **Decision 2:** the primary token cap is 21,000 per task (twice the calibration single-agent
  mean, rounded), fixed now, before any comparison. `configs/logistics-compare.json` runs single
  agent, Direct, octopus, bee and ant on the 15 calibration Logistics problems, two repetitions,
  at that cap: 150 runs, about US$8-14, stop at US$15.
- **Question it answers:** at equal tokens, does any mechanism reach a verified Logistics plan
  more often than the single agent (67% counterfactually)? The single agent revises one growing
  conversation (about 6,000 tokens per call by its seventh plan); bee samples independent
  short-context plans, which at temperature 1.0 differ (3 distinct plans of 4 in D25) but get no
  feedback; octopus acts locally with check feedback. Speculative until measured; development
  tasks only.

## R13. Logistics comparison at the primary cap (2026-10-29)

`runs/logistics-compare-v1`: 150 runs, US$12.37 recorded (including one ant call that timed out
and was charged at its reservation, 5,529 tokens, US$0.04), 15 Logistics training problems, 2
repetitions, 21,000 tokens per task (D27). Corrected after GPT-6.1 Sol's review (R14).

| System | Valid plans | Mean tokens | US$/run | US$ per success |
|---|---|---|---|---|
| single agent | 22/30 (73%) | 11,138 | 0.066 | 0.090 |
| octopus | 20/30 (67%) | 12,085 | 0.071 | 0.106 |
| bee (certificate stop) | 18/30 (60%) | 17,213 | 0.139 | 0.231 |
| Direct | 17/30 (57%) | 5,703 | 0.040 | 0.071 |
| ant | 0/30 | 19,053 | 0.096 | - |

- **No mechanism beats the single agent at equal tokens.** Paired differences (group bootstrap,
  `bioalloc experiment analyze`): bee -0.13 (95% CI -0.30 to 0.07; a plain percentile bootstrap
  over the 15 task differences gives an upper bound of 0.03), octopus -0.07 (-0.20 to 0.07).
  Single agent and octopus run the same solving loop on a single-node graph, so their difference
  here is sampling noise, not a memory effect. With 15 tasks the detectable effect is
  about 0.25, so small differences cannot be resolved.
- **Some complementarity:** choosing the best system per task (on its two repetitions together)
  would solve 25/30 against the single agent's 22/30 (bee solves 095 and 250 more often, octopus
  235); choosing per task and repetition gives 27/30. Too few tasks and
  repetitions to rely on.
- **Bee** stopped on a certificate in 18 runs and ran out of budget in 12: its first five answers
  cost 15,632 tokens on average, and scouts get no feedback on a rejected plan, while the single
  agent's strength is revision against the simulator's message. 17 of the 18 successes came from
  the first five answers. Bee's own mechanism never ran: every rejected plan left its scout
  uncommitted and every accepted plan was a certificate, so there were no recruitment calls and
  inhibition released nobody. Here bee was independent sampling with certificate selection.
- **Ant is infeasible here:** in 30 runs it finished one path (an immediate, rejected FINISH).
  Each step is one call carrying the whole problem (about 3,500 tokens), the valid plans found by
  the other systems have 14 to 45 actions (median 27), and the cap buys 5 to 7 calls. A
  step-per-call ant cannot do long-horizon planning at a realistic budget.

## R14. Review of D27 and R13, and the next design (2026-10-29)

GPT-6.1 Sol (high) reviewed commits 3c303e7 and 569620b against all 150 records and traces (final
plans revalidated, costs reconciled) and proposed the next design.

1. With D10's whole-task reward and uniform initial weights, ant's first positive deposit comes
   with the first certificate, so stopping there (D27) leaves reinforcement no later choice to
   affect: full ant, no-deposit and no-evaporation behave identically. Recorded, not changed:
   ant's learning can only matter with rewards before task success (milestones) or a learned
   prior, which is the macro-and-milestone ant planned below.
2. Bee checks for a certificate when a batch of scouts has returned, not as soon as one exists:
   seven requests were admitted after a certificate (US$0.18). D27 now says "at the round
   boundary"; the new staged executors launch one episode at a time and check after each.
3. Logistics does not exercise recruitment, inhibition or quorum (R13). Bee there is independent
   sampling with certificate selection, and bee with feedback would be a portfolio of repair
   trajectories. A real Table 3 bee experiment needs tasks where several accepted but unverified
   answers can coexist.
4. R13 corrected (interval method, ant's one finished path, plan lengths 14-45, bee's first
   cohort, the imputed timeout).
5. A certificate stop reported `quorum_reached` false even when support reached the quorum. Fixed,
   with a test.

Completion fitting reconciled: bee 8 fitted calls (6 ended at the length limit), ant 19 (8),
single agent 2 (0), octopus 3 (1). Fitting stays, and fitted and truncated replies are reported
separately.

## D28. Scouts that repair their own rejected answers

- **Date:** 2026-10-29 · **Status:** implemented, not yet run (design by GPT-6.1 Sol, R14)
- **Why:** the single agent's strength on Logistics is revision against the simulator's
  rejection, inside one growing conversation (about 6,000 tokens per call by its seventh plan);
  bee's scouts got no feedback. Three questions are kept apart: does short memory help (full
  memory single agent against one short-memory scout), does spreading repair over scouts help
  (one scout against several), and does the bee mechanism help (bee against the matched plurality
  control). The last needs tasks with several accepted but unverified answers; on Logistics the
  two consensus modes are identical (R14), so a Logistics win would be a staged scout-search
  result, not evidence for inhibition or recruitment.
- **Mechanism** (`bioalloc.executors.repair.ScoutRepair`): a fixed schedule gives one solve
  episode at a time to a named scout. A scout's first episode is independent; later ones see only
  its own latest answer and the latest feedback (octopus's wording; a cut-off reply gets "Your
  last step failed: ..."). A goal-verified answer ends the search after any episode; an accepted,
  unverified answer commits the scout, which takes no more episodes. Bee consensus stops at a
  quorum of ceil(theta * k) and otherwise runs D14's rounds; plurality consensus runs the full
  schedule and returns the plurality.
- **Pre-registered variants (six solve episodes at most, per-task model concurrency 1):**
  `scout_short_memory` (one scout), `bee_staged` (two scouts, order A0 A1 B0 A2 B1 A3: B starts
  only after A's first two answers fail), `bee_feedback3` (three scouts, A0 B0 C0 A1 B1 C1),
  each bee with theta 0.6 (quorum 2), sigma 0.2, five consensus rounds; `majority_staged` and
  `majority_feedback3` are the matched plurality controls.
- **Limit:** only an episode's final reply is remembered, not the tool calls inside it, so tool
  tasks need their own memory rule before these variants are used there.

## D29. Generated Logistics suite and the repair tuning run

- **Date:** 2026-10-29 · **Status:** suite frozen; run configured, not yet run (design by GPT-6.1
  Sol, R14)
- **Why:** 15 PlanBench Logistics problems give a detectable effect of about 0.25 and have been
  looked at repeatedly; PlanBench has no harder Logistics. Split v2 stays as it is.
- **Suite:** `bioalloc.evaluation.logistics_gen` (generator version 1) builds IPC-format
  Logistics problems: 3 or 4 cities with 3 locations each, one truck per city, 2 airplanes, 5, 7
  or 9 packages, every package starting away from its goal. Six strata of 30 instances, each split
  5 training, 5 tuning, 20 test: 180 instances in `data/splits/logistics-gen-v1.json` (SHA-256
  `eb34314f485a...`). Every instance has its own structure group, distinct from every other one
  and from every PlanBench group. A constructive witness plan (25 to 93 actions) passes the
  simulator for each. Tests rebuild the suite and require the frozen file exactly. Test
  instances were never selected or filtered by any model result.
- **Tuning run:** `configs/repair-tune.json` runs single agent, `scout_short_memory`,
  `bee_staged` and `bee_feedback3` (D28) on the 30 tuning instances, two repetitions, one outer
  attempt each (the executors' own loops do the repair), 21,000 tokens, 2,048 completion tokens,
  60 calls, 600 s per task: 240 runs, about US$25 (Sol's estimate from the R13 traces), US$35
  stop. Octopus is left out: on one-node graphs it is the single agent's own loop.
- **Pre-registered rule for what follows:** the candidate with the highest tuning success
  advances (fewer mean tokens break ties), if it either beats the single agent's observed success
  or, with the single agent above 0 successes, matches or exceeds it while saving at least 20% of
  its mean tokens (a screening rule, not a significance test; a cheaper failure never advances,
  R16). It is then compared with the single agent on the 120 test instances, two repetitions (480
  runs, about US$45-57), with power re-estimated from the tuning data first.
- **Pre-registered response to floor effects:** results are reported per stratum. If the single
  agent solves fewer than 15% of its 60 tuning runs, or no candidate solves any, there is no test
  run; the cap and strata are reconsidered from the tuning data first. A stratum where every
  system solves nothing is reported as a floor and kept in the test set unchanged (test instances
  are never dropped by results). Sol's rough expectation for the single agent at 21,000 tokens:
  60-80% with 5 packages, 35-60% with 7, 15-40% with 9 (R16).
- **Snapshot:** the suite file records the PlanBench snapshot hash its domain text comes from, and
  experiments verify it before loading a generated task (R16).

## R15. Review of the D28 implementation (2026-10-29)

GPT-6.1 Sol (high) found three problems, all fixed with tests that fail on the old code:

1. The six-episode ceiling is per dispatch, and the coordinator's default two attempts could give
   twelve solve episodes. D28 now states the ceiling per dispatch, experiments using these
   executors run one outer attempt, and a test checks every config that lists them.
2. Recruitment inherited bee's concurrent assessments, breaking per-task concurrency 1. It now
   assesses one scout at a time, with draws and representatives fixed before the round and
   decisions applied together, as in D14.
3. When inhibition released every supporter, D28 stopped silently with "round_limit", while D14
   lets every scout answer afresh. With the episodes spent, D28 now ends explicitly with
   `all_released` and the answers held before release (a stated difference from D14).

Sol also replayed the recorded Logistics-095 replies offline: the staged bee and its plurality
control sent identical requests and returned the same verified plan after three calls, so their
equivalence on Logistics holds under coupled replies.

## D30. Macro ant with milestone rewards (amends D10 for planning)

- **Date:** 2026-10-29 · **Status:** implemented and revised after review (R17), not yet run
  (design by GPT-6.1 Sol, R14)
- **Why:** the step-per-call ant cannot finish long plans at the primary cap (R13), and with a
  whole-task reward its learning cannot act before the task is solved (R14).
- **Mechanism** (`bioalloc.executors.ant_macro.MacroAnt`): at most 5 decisions; each is one call
  proposing up to 2 ranked segments of at most 12 ground actions from the current state; every
  segment is simulated and an inapplicable one is dropped whole, with the reasons shown at the
  next decision; one legal segment is sampled with probability tau[shape]^alpha * (1/rank)^beta,
  where the shape is the segment's action kinds in order of first use (a fixed public
  vocabulary); weights evaporate after every decision (alpha 1, beta 2, rho 0.1, tau0 1,
  tau_min 0.01) and the chosen shape receives delta 1 only when the segment makes a goal atom
  true that was not before (a delivered package; legality alone earns nothing); the search stops
  on the goal certificate.
- **Controls:** `ant_macro_no_deposit`, `ant_macro_no_evaporation`, `ant_macro_uniform` (still
  paying for proposals) and `macro_greedy`, the ordinary macro planner that applies the best
  ranked legal segment, all sharing proposer, simulation, feedback, budgets and stopping. Every
  decision traces each candidate's shape, weight and probability, so it can be checked whether
  deposits change later choices; with only two candidates of possibly identical shape they may
  not, which the first run must report.
- **Feasibility on paper:** a macro prompt is about 2,000-2,600 real tokens on the generated
  problems (3,300-3,900 reserved), segments a few hundred tokens, so five decisions fit 21,000
  tokens and can place up to 60 actions (witness plans 25-93; good plans are shorter). Not yet
  in a paid run: first a Sol review, then a small run with the controls on tuning instances.

## R16. Review of the R15 fixes and the generated suite (2026-10-29)

GPT-6.1 Sol (high): no new findings on R15. Two on D29, both fixed: the suite file had no
provenance, so experiments never verified the PlanBench snapshot that supplies the domain text
(now recorded and checked, with a test that a wrong hash is refused); and the advancement rule
let a candidate advance on fewer tokens when everything solved nothing (now excluded, with the
response to floor effects fixed in advance). Sol's difficulty check: tuning witness plans average
36, 52 and 64 actions for 5, 7 and 9 packages; from R13, roughly 60-80%, 35-60% and 15-40%
single-agent success at 21,000 tokens, with four cities at the lower end (uncertain: no comparable
four-city problems were run before). The US$35 stop is the `--max-usd 35` flag, not the config.

## R17. Review of the R16 fixes and the macro ant (2026-10-29)

GPT-6.1 Sol (high): no new findings on R16; the repair tuning run is cleared. Five problems in the
macro ant (D30), all fixed with executor-level tests that fail on the old code:

1. A legal candidate that reaches the goal could be left untaken by the sampler (reproduced on
   `logisticsgen-v1-c3-p7-07-0`). A completing segment is now taken at once, by every variant.
2. A milestone compared with the previous state, so a package delivered, picked up and
   delivered again earned two rewards. A milestone is now a goal atom made true for the first
   time in the run.
3. Shapes seen for the first time started at tau0 while older ones had evaporated, a hidden
   novelty bias (0.7826/0.2174 instead of 0.8/0.2 with no deposits). New shapes now start at
   tau0 decayed as often as every other weight; a failed call also evaporates.
4. Rejection reasons reached the next decision only when every candidate was rejected; now all
   are passed on.
5. The greedy control's trace reported sampling probabilities; it now reports the probabilities
   actually used (1 and 0), and a completing choice is marked.

Still a design risk to measure, not fix: two candidates with the same shape cancel the pheromone
factor, and one legal candidate gives no choice. The first run reports how often candidates
differ in shape, milestones before the last decision, and later probability changes due to
deposits; ant work stops there if reinforcement has no room to act.

Macro check (Sol's sizing): `configs/macro-check.json`, the six tuning instances
`logisticsgen-v1-c{3,4}-p{5,7,9}-07-0`, ant_macro and its four controls, two repetitions, the
repair run's settings and one outer attempt: 60 runs, at most 300 calls, about US$3.60-5.94
(centrally US$4.32), stop at US$8. A mechanism check, not an efficacy comparison.

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
