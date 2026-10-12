# The research plan in simple words

This is a plain-language explanation of the detailed research plan in `research-plan/main.pdf`.
If this file and the plan ever disagree, the plan is the correct one.

Last updated: 12 October 2026

## The idea

When several AI agents work together on a big task, they usually follow one fixed style of teamwork for everything. That wastes money (tokens), and sometimes a single AI working alone does just as well.

Our project builds a "manager" AI, called the coordinator. It breaks a task into small pieces and picks the best working style for each piece.

## The working styles

We call them mechanisms. Each is named after an animal, but each is just an ordinary algorithm.

| Name | What it does in plain words | Good for |
|---|---|---|
| Direct | Asks the AI once | Easy pieces |
| Ant | Tries several paths and remembers which ones worked | Pieces with many possible routes |
| Bee | Several AIs answer separately, then convince each other until most agree | Choosing between a few answers |
| Octopus | One worker gets a goal and a budget, works alone, reports only at the end | Clear, independent pieces |
| Crow | Plans which tool is needed before which | Multi-step tool use |
| Starfish | Saves progress, and if a worker fails, hands the work to another | Recovering from failures |
| Wolf | Several explorers search together without repeating each other | Only if time allows |
| Elephant | Remembers past tasks and uses them | Only if time allows |

There are two kinds:

- **Executors** solve a piece of the task. The coordinator chooses among them. These are Direct, Ant, Bee, Octopus, Crow and Wolf.
- **Wrappers** cannot solve a piece alone. They add something to whichever executor is running. These are Starfish (recovery) and Elephant (memory).

Build order: Direct, Ant, Bee and Octopus first (the "core pool"), then Starfish, then Crow. Wolf and Elephant only if we are on schedule.

## How the coordinator works

It does five steps for every piece:

1. **Decompose:** break the task into pieces and note which piece depends on which.
2. **Select:** choose an executor for the piece.
3. **Dispatch:** give the piece and a token budget to that executor.
4. **Check:** test the result. Did the tool work? Is the format right? Do the answers agree?
5. **Aggregate or reallocate:** if the check passes, add the result to the final answer. If it fails, choose again.

## How the coordinator chooses

Two stages:

1. **Rules.** We write simple rules by hand, based on what we learn about each executor.
2. **Learning.** We use a small learning method called LinUCB. It remembers which executor worked on which kind of piece and gets better at choosing.

It decides using things it can see before running the piece, for example how many possible answers there are, which tools are needed, and how long the piece looks.

## What is new and what is not

Choosing things adaptively is **not** new. A 2026 paper called Symphony-Coord already routes pieces of a task with the same learning method, and also studies recovery.

Our new part is narrower:

- We put several different working styles behind one common interface.
- We run them on the same pieces with the same budget.
- We find which style suits which kind of piece, and which part of each style actually helps.

We use the same learning method as the earlier paper on purpose. That way, if our results differ, it is because of *what* we choose among, not *how* we choose.

## The five research questions

1. **RQ1:** Are the styles really good at different things?
2. **RQ2:** Does choosing per piece beat using one style, at the same cost?
3. **RQ3:** Do the animal-inspired styles add anything over normal, well-designed agents?
4. **RQ4:** Can the system recover when something fails or when the tasks change?
5. **RQ5:** When is teamwork a waste, and a single AI better?

## How we test it

**Tasks.** 450 tasks of three types, 150 each:

- Planning (PlanBench, TravelPlanner)
- Tool use (GAIA, plus a small set we build ourselves)
- Maths reasoning (GSM8K, MATH)

All have answers a program can check. We also make artificial tasks where we control the difficulty ourselves.

**Split.** 150 tasks for training, 90 for tuning, and 210 kept hidden for the final test.

**What we compare against:**

1. A single AI
2. Each executor used alone
3. A fixed choice (one executor per task type, picked by hand)
4. A random choice
5. A normal "specialist team" (planner, executor, critic)
6. A normal adaptive system without the animal styles
7. Our system, with rules and with learning

Number 6 is the most important comparison for RQ3. It has the same chooser as ours but ordinary workers, so it shows whether the animal styles themselves matter.

**Fairness rules.**

- Everyone gets the same AI model, tools, token limit and retries.
- The correct answers are hidden from all systems while they run. They are used only to grade the final answer.
- Each setup is run three times.

**Main comparison, decided in advance.** Our learning system against the best single executor, on the 210 hidden tasks.

**Ablations.** We switch off one part of a style to see if that part mattered. For example, we remove the bee's "stop signals", or the ant's memory.

**Fault tests.** We break things on purpose: a tool fails once, a worker stops responding, or a result gets corrupted. Then we see who recovers.

**Honesty rule.** If our system is not better, we report that. With 210 test tasks we can only detect big differences, about 8 points or more.

## What we need

- **Hardware:** our own laptops, and free GPUs on Kaggle and Google Colab.
- **Software:** Python, and standard open-source libraries. We write the coordinator ourselves.
- **Models:** one small paid AI model through an API, and one free open model as a second check.
- **Size of the experiments:** about 20,700 test runs.
- **Cost:** roughly Rs. 2,000 to 5,500. This is an estimate. We will correct it after a small trial in December.
- **If money runs short:** a smaller version with about 11,800 runs.

## Timeline

| When | What |
|---|---|
| October to December 2026 | Build the coordinator, the four core executors and the test setup. Run a small trial. |
| January to February 2027 | Build Starfish. Train the chooser. |
| March to April 2027 | Main experiments, ablations, fault tests. Crow as an add-on. |
| May 2027 | Final report, paper draft, code release. |

Milestones:

- **M1, mid-November:** coordinator works end to end with one executor on a toy task.
- **M2, December:** core executors, tasks and baselines ready; trial results for the seventh-semester evaluation.
- **M3, end of February:** executor profiles done, chooser trained, Starfish ready.
- **M4, end of April:** all main experiments done.
- **M5, May:** final report, code and paper draft.

## Possible results

- **Best case:** choosing per piece gives more correct answers at the same cost, or the same answers at lower cost.
- **Middle case:** choosing helps, but the normal adaptive system does equally well. Then the gain comes from choosing, not from the animal styles.
- **Negative case:** a single AI or one fixed style is as good. That is still a valid finding, and it answers RQ5.

## Small glossary

- **Token:** the unit AI models are billed in. Roughly a piece of a word.
- **Budget:** the maximum number of tokens a system may spend on a task.
- **Baseline:** a simpler system we compare ours against.
- **Benchmark:** a standard set of tasks used by many researchers.
- **Ablation:** removing one part of a system to see if that part mattered.
- **Held-out test set:** tasks kept hidden until the final evaluation, so the result is honest.
- **Seed / run:** one repetition of an experiment. AI answers vary, so we repeat.
- **Subtask:** one piece of a bigger task.
