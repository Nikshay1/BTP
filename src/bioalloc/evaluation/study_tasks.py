"""The task inventory of the main study and its preregistered test-only variants.

Families and pools (decision D16 and D20):

* planning: PlanBench Blocksworld and Logistics (main instance sets). Variant: the 5-block
  Blocksworld instances, larger than anything in development.
* tools: our deterministic tool suite: revenue and top_product on database seeds 0-299, compare
  on seeds 1000-1299 (no shared databases with development). Variant: the compare template (two
  parallel chains joined at the end), a composition topology never seen in development. Tasks on
  the same database share a template group.
* reasoning: GSM8K problems from the upstream test split. Variant: the longest fifth of those
  questions by word count (more quantities and steps), fixed before any run.

The synthetic suite is a separate diagnostic panel with its own seed ranges and is not split here.
"""

from __future__ import annotations

from bioalloc.evaluation import gsm8k_data, planbench_data, toolsuite_data
from bioalloc.evaluation.splits import Candidate

# Database seeds. Templates on the same seed share a database (and can share identical first
# subtasks), so the held-out compare template gets seeds no other template uses, and tasks are
# grouped by database, not by task (split v2, decision D22).
TOOL_SEEDS = {"revenue": range(300), "top_product": range(300), "compare": range(1000, 1300)}


def gsm8k_long_threshold(questions: list[str]) -> int:
    """Word count at the 80th percentile: questions above it form the reasoning variant."""
    counts = sorted(len(q.split()) for q in questions)
    return counts[int(0.8 * len(counts))]


def tool_candidates() -> list[Candidate]:
    cands: list[Candidate] = []
    for template in toolsuite_data.TEMPLATES:
        for seed in TOOL_SEEDS[template]:
            task_id = f"tools-v{toolsuite_data.GENERATOR_VERSION}-{template}-s{seed}"
            cands.append(
                Candidate(
                    task_id=task_id,
                    family="tools",
                    group=f"toolsuite:shop-v{toolsuite_data.GENERATOR_VERSION}-{seed}",
                    variant=template == "compare",
                )
            )
    return cands


def inventory() -> list[Candidate]:
    cands: list[Candidate] = []
    for item in planbench_data.load():
        variant = item.domain_name == "blocksworld" and item.n_objects == 5
        cands.append(
            Candidate(task_id=item.task.id, family="planning", group=item.group, variant=variant)
        )
    cands += tool_candidates()
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
