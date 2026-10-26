"""The task inventory of the main study and its preregistered test-only variants.

Families and pools (decision D16 and D20):

* planning: PlanBench Blocksworld and Logistics (main instance sets). Variant: the 5-block
  Blocksworld instances, larger than anything in development.
* tools: our deterministic tool suite, seeds 0-299 of the revenue and top_product templates and
  of the compare template. Variant: the compare template (two parallel chains joined at the end),
  a composition topology never seen in development.
* reasoning: GSM8K problems from the upstream test split. Variant: the longest fifth of those
  questions by word count (more quantities and steps), fixed before any run.

The synthetic suite is a separate diagnostic panel with its own seed ranges and is not split here.
"""

from __future__ import annotations

from bioalloc.evaluation import gsm8k_data, planbench_data, toolsuite_data
from bioalloc.evaluation.splits import Candidate

TOOL_SEEDS = range(300)


def gsm8k_long_threshold(questions: list[str]) -> int:
    """Word count at the 80th percentile: questions above it form the reasoning variant."""
    counts = sorted(len(q.split()) for q in questions)
    return counts[int(0.8 * len(counts))]


def inventory() -> list[Candidate]:
    cands: list[Candidate] = []
    for item in planbench_data.load():
        variant = item.domain_name == "blocksworld" and item.n_objects == 5
        cands.append(
            Candidate(task_id=item.task.id, family="planning", group=item.group, variant=variant)
        )
    for template in toolsuite_data.TEMPLATES:
        for seed in TOOL_SEEDS:
            task_id = f"tools-v{toolsuite_data.GENERATOR_VERSION}-{template}-s{seed}"
            cands.append(
                Candidate(
                    task_id=task_id,
                    family="tools",
                    group=f"toolsuite:{task_id}",
                    variant=template == "compare",
                )
            )
    problems = [p for p in gsm8k_data.load() if p.upstream_split == "test"]
    threshold = gsm8k_long_threshold([str(p.task.inputs["question"]) for p in problems])
    for problem in problems:
        long = len(str(problem.task.inputs["question"]).split()) > threshold
        cands.append(
            Candidate(
                task_id=problem.task.id, family="reasoning", group=problem.group, variant=long
            )
        )
    return cands
