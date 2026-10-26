import pytest

from bioalloc.actions import Action
from bioalloc.evaluation.planbench_data import PlanItem, grade, public_task, structure_group
from bioalloc.tasks.planbench import PlanBench
from bioalloc.tasks.strips import PDDLError, parse_domain, parse_problem, simulate

DOMAIN = """(define (domain blocksworld-4ops) (:requirements :strips)
(:predicates (clear ?x) (ontable ?x) (handempty) (holding ?x) (on ?x ?y))
(:action pick-up :parameters (?ob)
  :precondition (and (clear ?ob) (ontable ?ob) (handempty))
  :effect (and (holding ?ob) (not (clear ?ob)) (not (ontable ?ob)) (not (handempty))))
(:action put-down :parameters (?ob) :precondition (holding ?ob)
  :effect (and (clear ?ob) (handempty) (ontable ?ob) (not (holding ?ob))))
(:action stack :parameters (?ob ?underob) :precondition (and (clear ?underob) (holding ?ob))
  :effect (and (handempty) (clear ?ob) (on ?ob ?underob) (not (clear ?underob)) (not (holding ?ob))))
(:action unstack :parameters (?ob ?underob)
  :precondition (and (on ?ob ?underob) (clear ?ob) (handempty))
  :effect (and (holding ?ob) (clear ?underob) (not (on ?ob ?underob)) (not (clear ?ob)) (not (handempty)))))"""


def problem(objs: str, goal: str) -> str:
    init = " ".join(f"(ontable {o}) (clear {o})" for o in objs.split())
    return f"(define (problem p) (:domain blocksworld-4ops) (:objects {objs}) (:init (handempty) {init}) (:goal (and {goal})))"


PROB = problem("a b c", "(on a b) (on b c)")
GOOD = ["(pick-up b)", "(stack b c)", "(pick-up a)", "(stack a b)"]


def test_simulator_accepts_a_valid_plan_and_explains_failures() -> None:
    d, p = parse_domain(DOMAIN), parse_problem(PROB)
    assert simulate(d, p, GOOD).goal_reached
    assert simulate(d, p, ["PICK-UP B", "stack b c", "(pick-up a)", "(stack a b)"]).goal_reached
    bad = simulate(d, p, ["(pick-up b)", "(pick-up a)"])
    assert not bad.valid and "precondition not met" in str(bad.error) and bad.steps_applied == 1
    short = simulate(d, p, GOOD[:2])
    assert short.valid and not short.goal_reached
    assert "unknown action" in str(simulate(d, p, ["(fly b)"]).error)
    assert "unknown objects" in str(simulate(d, p, ["(pick-up z)"]).error)
    assert "takes 1 arguments" in str(simulate(d, p, ["(pick-up a b)"]).error)


def test_unsupported_pddl_is_refused() -> None:
    with pytest.raises(PDDLError):
        parse_domain(
            DOMAIN.replace(
                "(and (clear ?ob) (ontable ?ob) (handempty))", "(or (clear ?ob) (ontable ?ob))"
            )
        )


def test_plan_check_gives_goal_evidence_only_for_a_solving_plan() -> None:
    node = public_task("p", DOMAIN, PROB).as_single_node().nodes[0]
    fam = PlanBench()

    def verdicts(plan: object) -> list[tuple[str, str]]:
        return [
            (r.verdict, r.evidence) for r in (c.evaluate(node, plan, {}) for c in fam.checks(node))
        ]

    assert verdicts({"plan": GOOD}) == [("pass", "format"), ("pass", "goal")]
    assert verdicts({"plan": GOOD[:2]})[1] == ("fail", "goal")
    assert verdicts({"plan": ["(stack a b)"]})[1] == ("fail", "operational")
    assert verdicts({"plan": "pick up"})[1] == ("fail", "format")


def test_operator_action_space() -> None:
    node = public_task("p", DOMAIN, PROB).as_single_node().nodes[0]
    space = PlanBench().action_space(node)
    assert space is not None and space.kinds == (
        "PICK-UP",
        "PUT-DOWN",
        "STACK",
        "UNSTACK",
        "FINISH",
    )
    s = space.start(node, {})
    assert space.legal(s, Action(kind="STACK", args={"objects": ["a", "b"]}))
    for kind, objs in [
        ("PICK-UP", ["b"]),
        ("STACK", ["b", "c"]),
        ("PICK-UP", ["a"]),
        ("STACK", ["a", "b"]),
    ]:
        a = Action(kind=kind, args={"objects": objs})
        assert space.legal(s, a) is None
        s, _ = space.apply(s, a)
    out = space.finish(s, Action(kind="FINISH"))
    assert out == {"plan": GOOD}


def test_grading_and_isomorphic_grouping() -> None:
    item = PlanItem(public_task("p", DOMAIN, PROB), "g", "blocksworld", 3)
    assert grade(item, {"plan": GOOD}).correct
    assert not grade(item, {"plan": GOOD[:3]}).correct
    renamed = parse_problem(problem("x y z", "(on x y) (on y z)"))
    other = parse_problem(problem("a b c", "(on a b)"))
    g = structure_group("blocksworld", parse_problem(PROB))
    assert g == structure_group("blocksworld", renamed)
    assert g != structure_group("blocksworld", other)
    assert (
        PlanBench().normalize(item.task.as_single_node().nodes[0], {"plan": ["(PICK-UP  b)"]})
        == '["pick-up b"]'
    )
