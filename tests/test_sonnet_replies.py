"""Regressions from the first live smoke test (Claude Sonnet 4.6, 2026-10-29).

The fixtures are real replies and requests copied from the smoke-test traces. Every failure in
that run came from how replies were read, not from the mechanisms (decisions.md, R5).
"""

import gzip
import json
from pathlib import Path

from bioalloc.actions import Action
from bioalloc.budget import reserve_prompt_tokens
from bioalloc.evaluation.registry import TaskLoader
from bioalloc.executors.ant import normalize_candidate
from bioalloc.executors.base import parse_json
from bioalloc.llm.base import LLMRequest

FIXTURES = Path(__file__).parent / "fixtures"
REPLIES = json.loads((FIXTURES / "sonnet46_smoke_replies.json").read_text())


def test_a_self_corrected_plan_is_read_as_its_final_answer() -> None:
    # The model answered, wrote "Wait, c is not clear...", and answered again. Reading from the
    # first "{" to the last "}" gave invalid JSON, so all 60 bee replies were rejected.
    plan = TaskLoader().load("planbench-blocksworld-007")
    for key in ("bee_planning_self_correction", "direct_planning_self_correction"):
        parsed = parse_json(REPLIES[key])
        assert isinstance(parsed, dict) and len(parsed["plan"]) == 8, key
        assert plan.grade(parsed).correct, key
        # The first (wrong) plan is not what is read.
        assert parsed["plan"] != ["(unstack a d)", "(stack a c)"]


def test_flat_candidates_are_read_as_kind_and_args() -> None:
    parsed = parse_json(REPLIES["ant_tools_flat_candidates"])
    assert isinstance(parsed, dict)
    action = Action.model_validate(normalize_candidate(parsed["candidates"][0]))
    assert action.kind == "QUERY"
    assert action.args == {
        "name": "r",
        "table": "customers",
        "where": {"city": "Cochin"},
        "columns": ["id"],
    }


def test_candidates_after_prose_are_found() -> None:
    parsed = parse_json(REPLIES["ant_planning_prose_then_candidates"])
    assert isinstance(parsed, dict) and parsed["candidates"][0]["kind"] == "UNSTACK"


def test_a_bare_action_line_is_not_taken_for_a_candidates_object() -> None:
    # 'FINISH {"answer": 25}' holds no candidates list; ant records the rejection instead of
    # guessing. The prompt now shows the exact envelope, so this form should not recur.
    parsed = parse_json(REPLIES["ant_reasoning_bare_action"])
    assert parsed == {"answer": 25}
    assert normalize_candidate({"answer": 25}) == {"answer": 25}


def test_normalization_leaves_well_formed_and_kindless_items_alone() -> None:
    good = {"kind": "FINISH", "args": {"answer": 3}}
    assert normalize_candidate(good) == good
    assert normalize_candidate("FINISH") == "FINISH"


def test_reservations_cover_every_real_prompt_of_the_smoke_test() -> None:
    with gzip.open(FIXTURES / "sonnet46_smoke_prompts.json.gz", "rt") as f:
        rows = json.load(f)
    assert len(rows) == 183
    short = [
        (reserve_prompt_tokens(LLMRequest.model_validate(r["request"])), r["prompt_tokens"])
        for r in rows
    ]
    assert all(est >= 1.1 * actual for est, actual in short), min(e / a for e, a in short)
