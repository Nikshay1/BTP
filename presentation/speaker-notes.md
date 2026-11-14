# Speaker notes: Phase 1 panel presentation, 17 November 2026

About 15 minutes: Aryan slides 1-4 (about 3.5 min), Madan slides 5-7 (about 3.5 min), Nikshay slides 8-18 (about 8.5 min; keep slides 9-13 brisk to leave time for 14-16). The same notes are in the deck under each slide.

## 1. Adaptive Task Allocation for Heterogeneous Bio-Inspired Agents

Speaker: Aryan. Good morning. We are group GP105. Our project asks one question: when several AI agents work together on a task, should they always work together the same way, or should a coordinator choose a different way of working for each part of the task? We borrow those ways of working from animals: ant colonies, honeybee swarms and the octopus. I will introduce the problem, Madan will explain the system we built, and Nikshay will present the results and the plan.

## 2. The problem: one way of working for every subtask

Speaker: Aryan. Today most multi-agent systems built on large language models choose one way for the agents to work together, and they use it for the whole task. But a task has different parts. Some parts need searching, some need a vote, and some are easy and need just one worker. A controlled study by Kim and others in 2026 compared a single agent with four multi-agent designs under the same model and compute. The change ranged from plus 81 percent on one task to minus 70 percent on another. Cemri and others studied more than 1,600 failed runs and found fourteen failure modes, many caused by the coordination design itself. So the way agents coordinate should match the work.

## 3. Our question and how we test it

Speaker: Aryan. Here is the problem in one line. A task is broken into a graph of subtasks. For each subtask, our coordinator chooses which mechanism will solve it, or sends it to one direct model call. The goal is the right answer inside a fixed token budget. The coordinator never sees the true answer, only automatic checks such as a plan simulator. We have five research questions from the approved synopsis. This review covers the first phase: the framework, the mechanisms and their first measurements, which answer parts of RQ1, RQ3 and RQ5. The learned selector and failure recovery come in the second phase.

## 4. Related work and the gap

Speaker: Aryan. Related work falls into three groups. First, adaptive routing: systems like MasRouter and GPTSwarm, and newer ones like AdaptOrch and Symphony-Coord, route each task to an agent, a role, a model or a graph shape. Second, swarm ideas: two recent systems use ant colony rules for routing, and Wu and others survey LLMs with evolutionary methods. Third, a skeptical view: some animal-named algorithms are old ideas with new names, and one study found an LLM version of a flocking algorithm needed about 300 times more compute. So the gap we address is this: nobody chooses between whole coordination mechanisms for each subtask, and tests each mechanism against plain versions without the biological parts, at equal compute. I hand over to Madan.

## 5. Three mechanisms, written as plain algorithms

Speaker: Madan. We implemented three mechanisms first. The ant colony builds a solution step by step; steps that led to accepted solutions get more weight, and weights fade over time. The honeybee swarm lets scouts propose answers, recruit others, and stop rivals with stop signals, until a quorum agrees. The octopus hands a bounded job to an arm that works on its own and only reports the result. Following the skeptical literature, we write each one as an ordinary algorithm, and we built plain versions without the biological part: for example, bee without stop signals, or a majority vote with exactly the same number of calls. If the mechanism wins only because it uses more compute, these controls will show it.

## 6. The bioalloc framework

Speaker: Madan. This is the framework we built, called bioalloc. A task arrives as a graph of subtasks. The coordinator takes the ready subtasks, asks the allocation policy which executor should handle each one, and gives it a share of the budget. The executor is the mechanism: ant, bee, octopus, or a direct call. Every model call goes through the metered client, which counts tokens and refuses a call that would break the budget. The result is checked by runtime checks, for example a plan simulator. If the check fails, the subtask is reallocated. Only a separate grader holds the true answers, and the coordinator can never see them. Every run is written to a trace, so any number we report can be audited.

## 7. Built so the comparison is fair

Speaker: Madan. Multi-agent papers are often criticised because the multi-agent system simply used more compute. We built the framework to prevent that. Every system gets the same model, the same prompt for solving, the same task decomposition and the same checks. Every system gets the same token budget, and the budget is enforced before a call is sent, not after. The task splits are frozen and hashed before any run, and the rule for which variant goes to the final test was written down before the tuning run. Every design decision and every review round is logged in the repository, and we have 354 automated tests. Now Nikshay will present the experiments.

## 8. Experimental setup

Speaker: Nikshay. Phase 1 ran on Claude Sonnet 4.6 through Amazon Bedrock; we then moved to GPT-6 Luna, about 30 times cheaper per token, for the maths test and to repeat the planning comparison. Within any one experiment every system uses the same model and settings. The budget is 21,000 tokens per task, which is twice what the single agent uses on average. Our main baseline is a strong single agent that can revise its answer when the checker rejects it. The tasks come from standard benchmarks, GSM8K and PlanBench, plus our own tool and synthetic suites, and we generated 180 harder Logistics planning problems, split into tuning and test sets before any model saw them. For maths we added AIME competition problems from 2020 to 2025. We went step by step: smoke tests, then difficulty calibration, then the comparison, then tuning of improved versions, with a held-out test reserved for any version that qualified.

## 9. Finding 1: easy tasks leave no room to improve

Speaker: Nikshay. Our first finding is that on the standard maths, tool and easy planning tasks, a modern model is already at the ceiling: the single agent solved all 30 calibration tasks. No mechanism can beat 100 percent, so testing there would tell us nothing. The only place with room was Logistics planning, where plans are 20 to 90 steps long. There, the single agent's first plan was valid only 2 times out of 15. It succeeds by revising after the simulator rejects the plan. With our cap of 21,000 tokens it solves about two thirds. That is the region where a better mechanism could show a difference, so we focused there.

## 10. Finding 2: on planning, the single agent is hard to beat

Speaker: Nikshay. Here is the main comparison on Logistics planning at equal budget. On Claude Sonnet the single agent solved 73 percent, octopus 67, bee 60, the one-call direct route 57, and the ant nothing. With 15 problems those differences were not statistically clear. So we repeated exactly the same comparison on a second model, GPT-6 Luna, and the order came out the same, more sharply: single agent 90 percent, octopus 87, bee 47, direct 47, ant nothing. Against the bee the gap is 43 points with a confidence interval well above zero. So on planning, no mechanism beats a strong single agent. That matches the literature we cited, and the traces told us why, which is the next slide.

## 11. Why each mechanism fell short, and the redesign

Speaker: Nikshay. Because every run is traced, we could see why each mechanism lost. The ant made one model call per action, and each call carried the whole problem, so the budget allowed only five to seven calls while good plans need 14 to 45 actions. The bee's scouts never saw why their plan was rejected, so it was really just independent sampling. Octopus on a single-step graph is the same loop as the single agent. The common lesson is that the single agent wins through feedback. So we redesigned. The macro ant proposes whole plan segments and rewards a segment only if it actually delivers a package. The repair scouts let each scout fix its own rejected plan using the simulator's error message, with a short memory, so it stays cheap.

## 12. Finding 3: repair helps the scouts, not enough to win

Speaker: Nikshay. We then tested the redesigned scouts on 30 harder generated problems, two runs each. The single agent solved 40 percent. One scout that remembers only its last answer and the error message got 38 percent, almost the same, which tells us a short memory is enough. But splitting the repair across two or three bee scouts was worse, because each new scout pays for a fresh first plan. Before this run we wrote down a rule: a variant goes to the held-out test only if it beats the single agent, or matches it with 20 percent fewer tokens. None did. So we did not run the test, and the 120 test problems are still unseen. We keep them for evaluating the selector in Phase 2.

## 13. Finding 4: the macro ant has nothing to choose from

Speaker: Nikshay. We also tested the macro ant, which proposes plan segments, together with its four controls. Together they solved only 2 of 60 runs. The traces show why. In 70 percent of decisions, every segment the model proposed broke a precondition, so there was nothing legal to choose. Only 15 of about 300 decisions had two legal options, and that is the only situation where the pheromone can change anything. So the ant's learning never really got to act. Our conclusion is that ant-style search belongs where there are several cheap, checkable options, for example choosing between tools or sub-plans, not single plan steps. That shapes how we use it in Phase 2.

## 14. Finding 5: on maths, voting beats the single agent

Speaker: Nikshay. Phase 1 showed where a single agent wins, so we asked where the mechanisms should win: problems where the model cannot check its own answer. Competition maths is that case. We took AIME problems from 2020 to 2025, kept 148 as a held-out test and wrote the test down before running it. On GPT-6 Luna the single agent solved 71 percent, five independent answers with a majority vote 84.5, and the quorum bee 86.5. Against the single agent the bee gains 15.5 points, with a confidence interval from 8 to 23 points: it won 30 problems and lost 7. To be clear about where that comes from: most of it is the known effect of voting over independent answers, called self-consistency, because plain majority gets almost the same. It also costs about five times as much per correct answer, though still well under a cent. The next slide shows what the bee itself adds. We see the same order on Claude Sonnet.

## 15. What the bee adds: a quorum makes voting cheaper

Speaker: Nikshay. So what does the bee add beyond voting? We split the test by how each bee run ended. On 120 problems three of the five scouts agreed early, the bee stopped there, and it was right 97 percent of the time; where no quorum formed it was right only 43 percent. So the quorum is a confidence signal for free. The bee's recruitment rounds, where scouts judge each other's answers, cost almost twice the tokens and did not beat a plain vote. If we keep the quorum stop and replace recruitment by a plain vote, the same recorded answers give 126 correct against the majority's 125, using three quarters of the tokens. On Sonnet the saving is the same, but accuracy fell below the majority, so we treat equal accuracy at lower cost as a hypothesis for Phase 2, not a result. This is analysis of runs we already had, not something we fixed in advance.

## 16. Same model, the winner flips with the task

Speaker: Nikshay. Putting the two findings side by side, on the same model, GPT-6 Luna. On planning, where a simulator checks every plan, the single agent that revises from feedback beat the bee, 90 against 47 percent. On maths, where nothing checks the answer, the quorum bee beat the single agent, 87 against 71. Both differences have confidence intervals clear of zero, and Claude Sonnet shows the same order on both. So which way of working is best really depends on the kind of task. Inside one task family the differences were only extra sampling, but across kinds of task the winner flips, and that is exactly what per-subtask allocation needs. In Phase 2 a learned selector, already implemented as a rule table and a LinUCB bandit, makes that choice inside task graphs that mix both kinds.

## 17. Status and plan to May 2027

Speaker: Nikshay. To summarise where we are. Phase 1 is done: the framework, the three mechanisms with all their controls, the baselines, the task suites, a full cycle of experiments on planning, and a held-out maths test where the quorum bee won. Moving to GPT-6 Luna makes the full study affordable: at about a thirtieth of the price per token it fits in well under a hundred dollars instead of more than a thousand. In Phase 2 we move to task graphs that mix planning and maths subtasks, test the cheaper quorum-stop vote properly, and learn the selector from recorded outcomes. In February and March we add the crow and starfish modules and test recovery from injected failures. The full study and ablations run in March and April, and we finish with the thesis and a conference paper draft in May.

## 18. What we take from Phase 1

Speaker: Nikshay. To close, three points. First, on planning tasks where the system gets feedback, a strong single agent is the real bar, and none of our mechanisms cleared it on either model; we measured that fairly instead of assuming the opposite. Second, on maths, where nothing checks the answer, voting beats the single agent by 13 to 16 points on a held-out test, and the bee's quorum rule makes that voting cheaper and tells us when to trust the answer. Third, on the same model the winner flips with the kind of task, which is exactly why the coordinator should choose per subtask; that is Phase 2. The code, every decision and every run are in our repository. Thank you, we are happy to take questions.

## Likely panel questions

**"Your mechanisms lose to a single agent on planning. So what is the contribution?"**
They lose on planning, where the simulator gives the single agent feedback, and we explain why from the traces. On maths without a checker, voting wins by 13 to 16 points on a held-out test fixed in advance. On the same model, GPT-6 Luna, the winner flips: single agent 90% against bee 47% on planning, bee 87% against single agent 71% on maths. That flip is the evidence that per-subtask allocation (RQ2, the selector) is worth building, and it comes from a fair, enforced-budget comparison that most multi-agent papers skip.

**"Why not use a weaker or open model, where coordination may help more?"**
Possible in Phase 2. We chose a strong model on purpose: a gain that only appears with a weak model may disappear as models improve. The framework is model-agnostic (Bedrock, OpenAI and Groq adapters exist), and the maths test already runs on a second model, GPT-6 Luna.

**"Is 15 or 30 problems enough?"**
No, and we say so: with 15 problems we detect only about 25-point differences. That is why we generated 180 problems and kept 120 as a held-out test.

**"Why did you not run the held-out test?"**
We wrote the rule before the tuning run: a variant goes to the test only if it beats the single agent, or matches it with 20% fewer tokens. None did. Running the test anyway would spend the unseen problems on a variant we already know does not qualify.

**"How do you make sure the comparison is fair?"**
Same model, prompt, decomposition and checks for every system; a ledger that refuses a call before it is sent if it would break the budget; frozen, hashed splits; and controls that remove only the biological part (for example bee without stop signals).

**"Isn't this just self-consistency? Voting over samples is known to beat one sample."**
Yes, most of the maths gain is self-consistency (Wang et al., 2023, which we cite), and we say so on the slide: plain majority of five gets 84.5% against the bee's 86.5%. We do not claim the bee discovers voting. Our claims are two. First, the task-type flip: the same voting that wins on maths loses badly on planning, where one agent with feedback wins, so voting is the right tool only for some subtasks; that is what the selector is for. Second, the bee's quorum rule makes voting cheaper and tells you when to trust it.

**"Isn't 'bee' just majority voting?"**
Partly, and we measured which part is more. The quorum stop is the part that helps: where three of five scouts agree, the bee stops early and is right 97% of the time, against 43% where no quorum forms. The recruitment rounds, where scouts judge each other's answers, cost almost twice the tokens and did not beat a plain vote. If we keep the quorum and replace recruitment by a plain vote, the same recorded answers give 126 correct against the majority's 125 at 0.75 times the tokens. On Sonnet the saving was the same (0.78 times) but accuracy fell below the majority, so "same accuracy, cheaper" is a hypothesis we will test properly in Phase 2. That analysis was done after the test and we label it so.

**"Is +15 points worth five times the cost?"**
Depends on the task, which is our point. A correct AIME answer costs US$0.0006 from the single agent and US$0.0029 from the bee, both well under a cent; when correctness matters and nothing can check the answer, that is cheap. On planning, the extra tokens buy nothing: the single agent is both better and cheaper. A selector should spend votes only where answers cannot be checked, and the quorum tells it when to stop spending.

**"What did it cost?"**
About US$125 of AWS credit on Sonnet for everything in Phase 1, and about US$1.60 on GPT-6 Luna for the whole AIME tuning and test plus the planning replication. The maths test itself (444 runs) cost 77 cents, the planning replication (150 runs) 39 cents.

**"Why did you change the model in the middle?"**
Cost and quota. Sonnet costs US$3/US$15 per million tokens and Bedrock's daily token quota stopped every long run; Luna costs US$0.10/US$0.50, about 30 times less, with no daily quota. Every Phase 1 result is reported with its model, and no experiment mixes models. Having both is also a strength: the planning comparison gives the same order on both models. The full study now fits well under US$100 instead of US$1,000-1,900.

**"Isn't the planning-versus-maths contrast just a difference of models?"**
No: we ran both on the same model. On GPT-6 Luna the single agent beats the bee on planning by 43 points (CI +20 to +67) and the bee beats the single agent on maths by 15.5 points (CI +8 to +23). Claude Sonnet shows the same order on both families. The remaining limit is that each family is one suite (Logistics, AIME).

**"Didn't your own rule say the Luna planning run was at ceiling?"**
Yes, and we report that. Before the run we wrote that if the single agent solved 90% or more, the problems would be at Luna's ceiling and could not separate the systems. It solved exactly 90%. But the rule was meant for the case where everyone is near the top, and here the bee and the direct call were at 47% and the ant at 0, with the single-agent-minus-bee interval well clear of zero. We show the flag and the numbers rather than quietly reinterpret the rule.

**"Did you look at the test before deciding?"**
No. The maths test, its two questions and a gate were written in the decision log before it ran, and so was the planning replication on Luna. Two earlier test runs were stopped because we found bugs that were unfair to the voters (scouts cut off by a call limit, and recruitment replies cut off by a 512-token limit that Luna's hidden reasoning used up). Both are disclosed in the log, fixed with tests, and the reported runs are fresh.

**"What exactly is new compared with Symphony-Coord or MasRouter?"**
They route among agents, roles, models or graph shapes. We route among coordination mechanisms with their own control flow, and test each against controls without its biological component.
