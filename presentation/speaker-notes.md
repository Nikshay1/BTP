# Speaker notes: Phase 1 panel presentation, 17 November 2026

About 15 minutes: Aryan slides 1-4 (about 3.5 min), Madan slides 5-7 (about 3.5 min), Nikshay slides 8-16 (about 8 min). The same notes are in the deck under each slide.

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

Speaker: Nikshay. For the experiments we use Claude Sonnet 4.6 through Amazon Bedrock, with the same settings for every system. The budget is 21,000 tokens per task, which is twice what the single agent uses on average. Our main baseline is a strong single agent that can revise its answer when the checker rejects it. The tasks come from standard benchmarks, GSM8K and PlanBench, plus our own tool and synthetic suites, and we generated 180 harder Logistics planning problems, split into tuning and test sets before any model saw them. We went step by step: smoke tests, then difficulty calibration, then the comparison, then tuning of improved versions, with a held-out test reserved for any version that qualified.

## 9. Finding 1: easy tasks leave no room to improve

Speaker: Nikshay. Our first finding is that on the standard maths, tool and easy planning tasks, a modern model is already at the ceiling: the single agent solved all 30 calibration tasks. No mechanism can beat 100 percent, so testing there would tell us nothing. The only place with room was Logistics planning, where plans are 20 to 90 steps long. There, the single agent's first plan was valid only 2 times out of 15. It succeeds by revising after the simulator rejects the plan. With our cap of 21,000 tokens it solves about two thirds. That is the region where a better mechanism could show a difference, so we focused there.

## 10. Finding 2: the single agent is hard to beat

Speaker: Nikshay. Here is the main comparison on Logistics at equal budget. The single agent solved 73 percent. Octopus was close at 67, bee 60, the one-call direct route 57 but with the lowest cost per success, and the ant solved nothing. None of the differences against the single agent is statistically clear: the confidence intervals include zero, and with 15 tasks we can only detect a difference of about 25 points. So, honestly: in this first version, no mechanism beats a strong single agent. That matches the literature we cited, and it told us exactly what to fix, which is the next slide.

## 11. Why each mechanism fell short, and the redesign

Speaker: Nikshay. Because every run is traced, we could see why each mechanism lost. The ant made one model call per action, and each call carried the whole problem, so the budget allowed only five to seven calls while good plans need 14 to 45 actions. The bee's scouts never saw why their plan was rejected, so it was really just independent sampling. Octopus on a single-step graph is the same loop as the single agent. The common lesson is that the single agent wins through feedback. So we redesigned. The macro ant proposes whole plan segments and rewards a segment only if it actually delivers a package. The repair scouts let each scout fix its own rejected plan using the simulator's error message, with a short memory, so it stays cheap.

## 12. Finding 3: repair helps the scouts, not enough to win

Speaker: Nikshay. We then tested the redesigned scouts on 30 harder generated problems, two runs each. The single agent solved 40 percent. One scout that remembers only its last answer and the error message got 38 percent, almost the same, which tells us a short memory is enough. But splitting the repair across two or three bee scouts was worse, because each new scout pays for a fresh first plan. Before this run we wrote down a rule: a variant goes to the held-out test only if it beats the single agent, or matches it with 20 percent fewer tokens. None did. So we did not run the test, and the 120 test problems are still unseen. We keep them for evaluating the selector in Phase 2.

## 13. Finding 4: the macro ant has nothing to choose from

Speaker: Nikshay. We also tested the macro ant, which proposes plan segments, together with its four controls. Together they solved only 2 of 60 runs. The traces show why. In 70 percent of decisions, every segment the model proposed broke a precondition, so there was nothing legal to choose. Only 15 of about 300 decisions had two legal options, and that is the only situation where the pheromone can change anything. So the ant's learning never really got to act. Our conclusion is that ant-style search belongs where there are several cheap, checkable options, for example choosing between tools or sub-plans, not single plan steps. That shapes how we use it in Phase 2.

## 14. Where allocation can pay off

Speaker: Nikshay. Does choosing a mechanism per task help? On the 15 PlanBench problems, picking the best system per problem would raise 22 to 25 out of 30. But we checked this on the larger generated suite: every problem that any system solved was also solved by the single agent at least once. So inside one task family the gain mostly comes from taking more samples, not from mechanisms being good at different problems. We think that is expected: our hypothesis is that mechanisms differ across kinds of subtask, such as search, reconciling competing answers, and routine tool work. So Phase 2 moves to task graphs that mix these kinds, where a learned selector can actually choose. The selector, a rule table and a LinUCB bandit, is already implemented.

## 15. Status and plan to May 2027

Speaker: Nikshay. To summarise where we are. Phase 1 is done: the framework, the three mechanisms with all their controls, the baselines, the task suites, and a full cycle of experiments from calibration to a held-out test. In Phase 2 we move to multi-step tool and reasoning tasks, where subtasks really differ, then we learn the selector from recorded outcomes. In February and March we add the crow and starfish modules and test recovery from injected failures. The full study and ablations run in March and April, and we finish with the thesis and a conference paper draft in May.

## 16. What we take from Phase 1

Speaker: Nikshay. To close, three points. First, a strong single agent is the real bar, and most bio-inspired designs do not clear it as first written; we measured that fairly instead of assuming the opposite. Second, on long planning, what decides the winner is whether the system uses feedback, not the animal metaphor. Third, inside one task family the differences between systems are mostly sampling noise, so the selector we build next has to choose across different kinds of subtask. The code, every decision and every run are in our repository. Thank you, we are happy to take questions.
