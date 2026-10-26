import pytest

from bioalloc.actions import Action
from bioalloc.evaluation.gsm8k_data import (
    GSM8KItem,
    grade,
    group_id,
    public_task,
    reference_from_solution,
)
from bioalloc.tasks.common import CalcError, calculate, normalize_number
from bioalloc.tasks.gsm8k import CONTRACT, GSM8K


@pytest.mark.parametrize(
    ("expr", "value"),
    [
        ("2 + 3 * 4", 14),
        ("(12 + 3) * 4 / 5", 12),
        ("7 / 2", 3.5),
        ("2^3", 8),
        ("1,200 + 5", 1205),
        ("-3 + abs(-4)", 1),
        ("round(2.567, 2)", 2.57),
        ("max(3, 9, 4)", 9),
        ("sqrt(16)", 4),
    ],
)
def test_calculator(expr: str, value: float) -> None:
    assert calculate(expr) == pytest.approx(value)


@pytest.mark.parametrize(
    "expr",
    [
        "__import__('os')",
        "open('x')",
        "2 ** 1000",
        "1 / 0",
        "a + 1",
        "[1, 2]",
        "9" * 300,
        "1e300 * 1e300",
    ],
)
def test_calculator_rejects_unsafe_or_invalid_input(expr: str) -> None:
    with pytest.raises(CalcError):
        calculate(expr)


@pytest.mark.parametrize(
    ("raw", "norm"),
    [
        (18, "18"),
        (3.0, "3"),
        ("1,200.50", "1200.5"),
        ("$7", "7"),
        ("12%", "12"),
        (" -4 ", "-4"),
        ("0.25", "0.25"),
        (".5", "0.5"),
        ("72 apples", "72"),
        ("3 and 4", None),
        ("", None),
        (None, None),
        (True, None),
        ([3], None),
    ],
)
def test_normalize_number(raw: object, norm: str | None) -> None:
    assert normalize_number(raw) == norm  # type: ignore[arg-type]


def test_reference_extraction_and_grading() -> None:
    sol = "Natalia sold 48/2 = <<48/2=24>>24 clips in May.\n#### 1,072"
    assert reference_from_solution(sol) == "1072"
    item = GSM8KItem(public_task("g1", "q?"), group_id("q?"), "train", "1072")
    assert grade(item, {"answer": "1072.0"}).correct
    assert grade(item, {"answer": 1072}).correct
    assert not grade(item, {"answer": 1071}).correct
    assert not grade(item, "1072").correct
    with pytest.raises(ValueError):
        reference_from_solution("no marker here")


def test_public_task_carries_no_answer() -> None:
    task = public_task("g1", "How many?")
    assert set(task.inputs) == {"question"} and task.contract == CONTRACT
    assert "calculate" in task.allowed_tools
    assert group_id("How  many?") == group_id("how many?")


def test_family_checks_normalizer_and_actions() -> None:
    fam = GSM8K()
    node = public_task("g1", "q").as_single_node().nodes[0]
    results = [c.evaluate(node, {"answer": "7 apples"}, {}) for c in fam.checks(node)]
    assert all(r.verdict == "pass" for r in results) and all(
        r.evidence == "format" for r in results
    )
    assert (
        fam.normalize(node, {"answer": "1,000"})
        == fam.normalize(node, {"answer": 1000.0})
        == "1000"
    )
    assert fam.normalize(node, {"answer": "seven"}) is None
    space = fam.action_space(node)
    s = space.start(node, {})
    s, obs = space.apply(s, Action(kind="NOTE", args={"name": "eggs", "value": 16}))
    s, obs = space.apply(
        s, Action(kind="CALCULATE", args={"name": "left", "expression": "eggs - 3 - 4"})
    )
    assert obs == "left = 9"
    assert space.legal(s, Action(kind="CALCULATE", args={"name": "x", "expression": "eggs / 0"}))
    assert space.finish(s, Action(kind="FINISH", args={"answer": "left"})) == {"answer": 9}
    assert "calculate" in fam.tools()


@pytest.mark.parametrize(
    "expr", ["sqrt(-1)", "round(2, 2.5)", "(-1)**0.5", "min()", "abs(1, 2)", "10.0 ** 10 * 1e300"]
)
def test_calculator_turns_every_arithmetic_failure_into_calc_error(expr: str) -> None:
    with pytest.raises(CalcError):
        calculate(expr)
