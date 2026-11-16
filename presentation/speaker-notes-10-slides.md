# Speaker notes: 10-slide panel deck (one presenter)

About one minute per slide, roughly 10 minutes in total, leaving time for questions.

## 1. Title

Good morning. I'm Nikshay Yadav. With Madan Raj Murugesan and Aryan Varshney, we are group GP105,
supervised by Prof. Shailender Kumar. Our project is Adaptive Task Allocation for Heterogeneous
Bio-Inspired Agents. In one sentence: when several AI agents work together on a task, should they
always work together the same way, or should a coordinator choose a different way of working for each
part of the task? We borrow those ways of working from animals: ant colonies, honeybee swarms and the
octopus.

## 2. Problem and research question

Today, most systems where several language-model agents work together pick one pattern, such as a
debate, a pipeline of roles, or a manager with workers, and use it for the whole task. But real tasks
are mixed. One part may be a planning problem where a simulator can check the answer. Another may be a
calculation where nothing can check it. A third may be routine. Using one pattern everywhere wastes
tokens, which means money and time, on parts that don't need it. And the choice matters a lot: a
controlled study by Kim and others found that, with the same model and the same compute, switching to a
multi-agent design changed results by anywhere from plus 80.8 percent to minus 70 percent, depending on
the task.

So our question is: if a coordinator picks a mechanism for each subtask, can it beat one well-prompted
agent when both get exactly the same token budget? The gap in existing work is that other systems
choose which agent or which model to use, but none chooses between whole coordination mechanisms per
subtask and tests each one against plain versions of itself at equal compute.

## 3. The three mechanisms

We built three mechanisms, each written as an ordinary algorithm. The ant colony builds a solution
step by step; steps that led to good solutions get more weight, like pheromone, and old weight fades
away. The honeybee swarm sends out independent scouts that each propose an answer; scouts can recruit
others or stop rivals, and the swarm commits when a quorum, here three of five scouts, agrees. The
octopus hands a bounded job to one "arm" that works on its own and reports back only the result.

The important part: for each mechanism we also built plain versions without the animal-inspired part,
for example a simple majority vote instead of the bee. So if a mechanism wins, we can tell whether it is
the idea itself or just extra computation.

## 4. A fair framework

To compare fairly, we built our own framework, called bioalloc: about 9,400 lines of Python with 372
automated tests. The diagram shows the flow. A task comes in as a graph of subtasks. The coordinator
asks the allocation policy which mechanism should take each subtask. The executor runs it, and runtime
checks either accept the result or send it back to be reallocated.

Two design points make the comparison fair. First, every system uses the same model, prompt and checks,
and every model call goes through a metered client that refuses any call that would go over the token
budget, so no system can quietly use more compute. Second, only a separate grader knows the correct
answers; the system itself never sees them. Every run is logged, so any number in this talk can be
traced back to the runs that produced it.

## 5. Experimental setup

We used two language models: Claude Sonnet 4.6, and GPT-6 Luna, which is about thirty times cheaper,
so we could repeat the key experiments on a second model. Budgets were fixed in advance: 21,000 tokens
per planning task, about twice what a single agent normally needs, and 60,000 tokens per maths problem.

We chose two kinds of tasks on purpose. Logistics planning, moving packages with trucks and planes,
where a plan simulator checks every plan. And AIME competition maths, where the answer is a single
number and nothing can check it during the run. We kept 148 AIME problems aside as a held-out test and
wrote the test down before running it.

## 6. Finding: planning

First result: planning. The chart shows the share of valid plans for each system, light bars on Claude
and dark bars on Luna. The single agent is best on both models, at 73 and 90 percent, with the octopus
close behind. The bee gets 60 and 47 percent, the one-call direct route 57 and 47, and the ant solved
nothing. On Luna, the single agent beats the bee by 43 points, with a confidence interval of 20 to 67
points, so this is not luck.

Because every run is logged, we could see why. The single agent reads the simulator's error message
and fixes its plan, while the mechanisms did not use that feedback. The ant spent one model call per
action, so its budget ran out long before a plan of 14 to 45 actions was finished. So when a task has a
checker, one agent that learns from feedback is very hard to beat.

## 7. Finding: maths

Second result: maths, where nothing checks the answer. Here the picture flips. The single agent solved
70.9 percent of the 148 held-out problems; a majority vote over five independent answers got 84.5
percent; our quorum bee got 86.5 percent. The bee beats the single agent by 15.5 points, with a
confidence interval from 8.1 to 23 points.

To be honest about where this comes from: most of the gain is from voting over independent answers, a
known effect called self-consistency, because a plain majority vote gets almost the same score. It also
costs about five times more per correct answer, though that is still well under a cent per problem.

## 8. What the bee's quorum adds

So what does the bee add beyond voting? Its quorum. On 120 of the 148 problems, three scouts agreed
early. The bee stopped there and was right 97 percent of the time, using fewer tokens than a full vote.
So the quorum is a cheap and reliable signal of confidence.

On the other 28 problems the scouts disagreed. There, the bee's recruitment rounds, where scouts judge
each other's answers, cost about 1.87 times the tokens and did not beat a plain vote. That suggests a
simpler rule for Phase II: stop at a quorum, and otherwise just take a vote, which would use about
three quarters of a full vote's tokens. We have not tested this yet; it is a hypothesis.

## 9. Key result: choose by task type

Putting the two results together gives our key point. On the same model, always using the single agent
gives 90 percent on planning but only 70.9 on maths. Always using the bee gives 86.5 on maths but only
46.7 on planning. Simply choosing by task type, the single agent for planning and the bee for maths,
keeps the best of both and averages 88.2 percent, better than either fixed choice.

One caveat: this rule is evaluated on the same runs, so it is a demonstration, not yet a test. In Phase
II we will learn this choice automatically, for each subtask, and test it on tasks it has never seen.

## 10. Conclusions and Phase II

To conclude, three takeaways. One: when a task has a checker, a strong single agent that learns from
feedback is the bar, and our mechanisms did not beat it. Two: when nothing can check the answer, voting,
as in the quorum bee, beats the single agent by 13 to 16 points, and the quorum tells us when to trust
the answer. Three: the best mechanism flips with the kind of task, which is exactly why allocation per
subtask matters.

In Phase II, from December to May, we will build tasks that mix both kinds of subtask, learn the
selector with a rule table and a LinUCB bandit, add two more mechanisms and failure tests, run the full
study, and write the thesis and a paper. All our code and results are on GitHub. Thank you; I'm happy
to take questions.
