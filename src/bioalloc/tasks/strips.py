"""A minimal STRIPS interpreter for the PlanBench domains (Blocksworld, Logistics).

Parses the subset of PDDL these domains use: untyped or typed objects, actions whose preconditions
are conjunctions of positive atoms and whose effects are conjunctions of atoms and negated atoms,
and conjunctive positive goals. It is used as the public plan check (legal actions, goal reached)
and, until the pinned VAL binary is available, as the grader; it is cross-checked against VAL on
fixtures when VAL is built (decision D16). Anything outside the subset raises PDDLError rather
than being silently misread.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

Atom = tuple[str, ...]


class PDDLError(ValueError):
    pass


def _tokens(text: str) -> list[str]:
    text = re.sub(r";[^\n]*", "", text).lower()
    return re.findall(r"\(|\)|[^\s()]+", text)


def parse_sexpr(text: str) -> list[object]:
    tokens = _tokens(text)
    pos = 0

    def read() -> object:
        nonlocal pos
        tok = tokens[pos]
        pos += 1
        if tok == "(":
            items: list[object] = []
            while tokens[pos] != ")":
                items.append(read())
            pos += 1
            return items
        if tok == ")":
            raise PDDLError("unexpected ')'")
        return tok

    try:
        tree = read()
    except IndexError as exc:
        raise PDDLError("unbalanced parentheses") from exc
    if not isinstance(tree, list):
        raise PDDLError("expected a list")
    return tree


def _strip_types(items: list[object]) -> list[str]:
    """['?a', '?b', '-', 'truck', '?c'] -> ['?a', '?b', '?c']."""
    out: list[str] = []
    skip = False
    for x in items:
        if skip:
            skip = False
            continue
        if x == "-":
            skip = True
            continue
        if not isinstance(x, str):
            raise PDDLError(f"unexpected parameter list item {x!r}")
        out.append(x)
    return out


def _conj(expr: object) -> tuple[list[Atom], list[Atom]]:
    """Positive and negative atoms of an atom or (and ...) conjunction."""
    if not isinstance(expr, list) or not expr:
        return [], []
    if expr[0] == "and":
        pos: list[Atom] = []
        neg: list[Atom] = []
        for e in expr[1:]:
            p, n = _conj(e)
            pos += p
            neg += n
        return pos, neg
    if expr[0] == "not":
        p, n = _conj(expr[1])
        if n or len(p) != 1:
            raise PDDLError("only negated atoms are supported")
        return [], p
    if not all(isinstance(t, str) for t in expr) or expr[0] in ("or", "forall", "exists", "when"):
        raise PDDLError(f"unsupported expression {expr!r}")
    return [tuple(expr)], []


@dataclass(frozen=True)
class Operator:
    name: str
    params: tuple[str, ...]
    pre: tuple[Atom, ...]
    add: tuple[Atom, ...]
    delete: tuple[Atom, ...]


@dataclass(frozen=True)
class Domain:
    name: str
    operators: dict[str, Operator]


@dataclass(frozen=True)
class Problem:
    name: str
    objects: tuple[str, ...]
    init: frozenset[Atom]
    goal: tuple[Atom, ...]


def _section(tree: list[object], key: str) -> list[object] | None:
    for item in tree:
        if isinstance(item, list) and item and item[0] == key:
            return item
    return None


def parse_domain(text: str) -> Domain:
    tree = parse_sexpr(text)
    name_part = _section(tree, "domain")
    ops: dict[str, Operator] = {}
    for item in tree:
        if not (isinstance(item, list) and item and item[0] == ":action"):
            continue
        name = str(item[1])
        fields = {str(item[i]): item[i + 1] for i in range(2, len(item) - 1, 2)}
        params = fields.get(":parameters", [])
        pre_pos, pre_neg = _conj(fields.get(":precondition", []))
        if pre_neg:
            raise PDDLError("negative preconditions are not supported")
        add, delete = _conj(fields.get(":effect", []))
        ops[name] = Operator(
            name,
            tuple(_strip_types(params if isinstance(params, list) else [])),
            tuple(pre_pos),
            tuple(add),
            tuple(delete),
        )
    if not ops:
        raise PDDLError("no actions found")
    return Domain(str(name_part[1]) if name_part else "", ops)


def parse_problem(text: str) -> Problem:
    tree = parse_sexpr(text)
    name_part = _section(tree, "problem")
    objects = _section(tree, ":objects") or [":objects"]
    init = _section(tree, ":init") or [":init"]
    goal = _section(tree, ":goal") or [":goal", []]
    init_pos, init_neg = _conj(["and", *init[1:]])
    goal_pos, goal_neg = _conj(goal[1])
    if init_neg or goal_neg:
        raise PDDLError("negative literals in init or goal are not supported")
    return Problem(
        str(name_part[1]) if name_part else "",
        tuple(_strip_types(objects[1:])),
        frozenset(init_pos),
        tuple(goal_pos),
    )


def parse_step(step: str) -> Atom:
    """'(pick-up b)', 'pick-up b' or 'PICK-UP B' -> ('pick-up', 'b')."""
    words = step.replace("(", " ").replace(")", " ").lower().split()
    if not words:
        raise PDDLError("empty plan step")
    return tuple(words)


@dataclass(frozen=True)
class Simulation:
    valid: bool  # every step was a known, applicable action
    goal_reached: bool
    steps_applied: int
    error: str | None
    state: frozenset[Atom]


def ground(op: Operator, args: tuple[str, ...]) -> dict[str, str]:
    if len(args) != len(op.params):
        raise PDDLError(f"{op.name} takes {len(op.params)} arguments, got {len(args)}")
    return dict(zip(op.params, args, strict=True))


def _bind(atoms: tuple[Atom, ...], binding: dict[str, str]) -> set[Atom]:
    return {tuple(binding.get(t, t) for t in a) for a in atoms}


def applicable(domain: Domain, problem: Problem, state: frozenset[Atom], step: Atom) -> str | None:
    op = domain.operators.get(step[0])
    if op is None:
        return f"unknown action {step[0]!r}"
    unknown = [a for a in step[1:] if a not in problem.objects]
    if unknown:
        return f"unknown objects {unknown}"
    try:
        binding = ground(op, step[1:])
    except PDDLError as exc:
        return str(exc)
    missing = sorted(_bind(op.pre, binding) - state)
    if missing:
        return f"precondition not met: {' '.join('(' + ' '.join(a) + ')' for a in missing)}"
    return None


def apply(domain: Domain, state: frozenset[Atom], step: Atom) -> frozenset[Atom]:
    op = domain.operators[step[0]]
    binding = ground(op, step[1:])
    return frozenset((state - _bind(op.delete, binding)) | _bind(op.add, binding))


def simulate(domain: Domain, problem: Problem, plan: list[str]) -> Simulation:
    state = problem.init
    for i, raw in enumerate(plan):
        try:
            step = parse_step(raw)
        except PDDLError as exc:
            return Simulation(False, False, i, f"step {i + 1}: {exc}", state)
        problem_with = applicable(domain, problem, state, step)
        if problem_with is not None:
            return Simulation(False, False, i, f"step {i + 1} {raw!r}: {problem_with}", state)
        state = apply(domain, state, step)
    reached = all(g in state for g in problem.goal)
    error = None if reached else "the plan ends before the goal is reached"
    return Simulation(True, reached, len(plan), error, state)
