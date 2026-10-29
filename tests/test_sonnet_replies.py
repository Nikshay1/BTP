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


# Adversarial cases from Sol's review of the first parser fix (R6). Each would have produced a
# wrong but check-passing answer, or replaced a valid envelope.


def test_a_malformed_final_correction_gives_no_answer() -> None:
    text = '{"answer":25}\nCorrection: {"answer":30, "note":{"answer":0}, broken}'
    assert parse_json(text) == text  # neither the earlier answer nor a fragment of the broken one


def test_an_unclosed_final_object_gives_no_answer() -> None:
    text = '{"answer": 25}\nActually: {"answer": 30'
    assert parse_json(text) == text


def test_a_fenced_array_is_read_whole() -> None:
    assert parse_json('```json\n[{"answer":1},{"answer":2}]\n```') == [{"answer": 1}, {"answer": 2}]


def test_escaped_quotes_and_braces_inside_strings() -> None:
    text = 'Here: {"answer": "a \\"quoted\\" } brace", "n": 1} done'
    assert parse_json(text) == {"answer": 'a "quoted" } brace', "n": 1}


def test_prose_brackets_and_apostrophes_are_ignored() -> None:
    text = 'It\'s [roughly] this: {"answer": 5} (see [note]).'
    assert parse_json(text) == {"answer": 5}


def test_the_envelope_survives_a_trailing_quoted_action() -> None:
    text = (
        '{"candidates":[{"kind":"FINISH","args":{"answer":25}}]}\n'
        'The FINISH action is {"kind":"FINISH","args":{"answer":25}}.'
    )
    parsed = parse_json(text, envelope="candidates")
    assert isinstance(parsed, dict) and parsed["candidates"][0]["args"] == {"answer": 25}
    assert parse_json('{"kind": "FINISH"}', envelope="candidates") == '{"kind": "FINISH"}'


def test_a_trailing_example_object_is_still_ambiguous() -> None:
    # Known limit: no parser can tell a trailing example from a correction. The prompts now ask
    # for the corrected object last and no example objects; this pins the documented behaviour.
    assert parse_json('{"answer":25}\nExample only: {"answer":0}') == {"answer": 0}


# Second round of adversarial cases (Sol, R7).


def test_a_mismatched_bracket_never_exposes_a_nested_fragment() -> None:
    text = '{"answer":25} Correction: {"x": ], "nested":{"answer":0}}'
    assert parse_json(text) == text


def test_an_unclosed_final_array_gives_no_answer() -> None:
    text = '{"answer": 25}\nRevised: [{"answer": 30}'
    assert parse_json(text) == text


def test_a_broken_correction_blocks_an_earlier_envelope() -> None:
    good = '{"candidates":[{"kind":"FINISH","args":{"answer":25}}]}'
    for tail in ('\nCorrection: {"candidates":[broken]}', '\nCorrection: {"candidates":['):
        assert parse_json(good + tail, envelope="candidates") == good + tail


def test_prose_braces_and_brackets_do_not_cancel_an_answer() -> None:
    assert parse_json('{"answer":0.5} because \\frac{1}{2}') == {"answer": 0.5}
    assert parse_json('{"answer":5} (where {n} denotes the count)') == {"answer": 5}
    assert parse_json('{"answer":5} (see [1])') == {"answer": 5}


def test_a_huge_integer_is_unreadable_not_a_crash() -> None:
    text = '{"answer": ' + "9" * 5000 + "}"
    assert parse_json(text) == text
    later = '{"answer": 3}\n{"answer": ' + "9" * 5000 + "}"
    assert parse_json(later) == later  # the broken last object blocks the earlier answer


def test_protocol_replies_keep_their_envelope_over_a_quoted_answer() -> None:
    # Bee's assessment would have read {"answer": 25} and counted a rejection (Sol, R7).
    text = '{"accept": true, "confidence": 0.9} Accepted: {"answer": 25}'
    assert parse_json(text, envelope="accept") == {"accept": True, "confidence": 0.9}


def test_a_stray_closer_inside_a_broken_correction_ends_the_scan() -> None:
    # Sol, R8: the mismatched "}" once ended recovery early and exposed {"answer":0}.
    text = '{"answer":25}\nCorrection: {"answer":30,"steps":[},"final":{"answer":0}}'
    assert parse_json(text) == text
    for key in ("candidates", "advice", "nodes", "accept", "decision"):
        broken = '{"' + key + '": 1}\nFix: {"x":[},"y":{"' + key + '": 2}}'
        assert parse_json(broken, envelope=key) == broken, key
