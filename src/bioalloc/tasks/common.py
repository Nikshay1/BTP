"""Helpers shared by task families: a safe calculator and number normalization."""

from __future__ import annotations

import ast
import math
import operator
import re
from collections.abc import Callable
from decimal import Decimal, InvalidOperation
from typing import Any

from pydantic import JsonValue

from bioalloc.tools import Tool, ToolSpec

_BIN = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_FUNCS: dict[str, Callable[..., Any]] = {
    "abs": abs,
    "round": round,
    "min": min,
    "max": max,
    "sqrt": math.sqrt,
}
MAX_EXPR_CHARS = 200
_THOUSANDS = re.compile(r"(?<=\d),(?=\d{3}(?!\d))")
MAX_MAGNITUDE = 1e15


class CalcError(ValueError):
    pass


def calculate(expression: str) -> float | int:
    """Evaluate an arithmetic expression safely: numbers, + - * / // % **, parentheses and a few
    functions (abs, round, min, max, sqrt). Raises CalcError for anything else. A comma between
    digits followed by exactly three digits is a thousands separator ('1,200'); function arguments
    are separated by ', '."""
    if len(expression) > MAX_EXPR_CHARS:
        raise CalcError("expression too long")
    try:
        cleaned = _THOUSANDS.sub("", expression).replace("^", "**")
        tree = ast.parse(cleaned, mode="eval")
    except SyntaxError as exc:
        raise CalcError(f"not an arithmetic expression: {exc.msg}") from exc

    def ev(node: ast.AST) -> Any:
        if isinstance(node, ast.Expression):
            return ev(node.body)
        if (
            isinstance(node, ast.Constant)
            and isinstance(node.value, int | float)
            and not isinstance(node.value, bool)
        ):
            return node.value
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub | ast.UAdd):
            v = ev(node.operand)
            return -v if isinstance(node.op, ast.USub) else v
        if isinstance(node, ast.BinOp) and type(node.op) in _BIN:
            left, right = ev(node.left), ev(node.right)
            if isinstance(node.op, ast.Pow) and abs(right) > 10:
                raise CalcError("exponent too large")
            try:
                v = _BIN[type(node.op)](left, right)
            except ZeroDivisionError as exc:
                raise CalcError("division by zero") from exc
            if abs(v) > MAX_MAGNITUDE:
                raise CalcError("result too large")
            return v
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id in _FUNCS
            and not node.keywords
        ):
            return _FUNCS[node.func.id](*(ev(a) for a in node.args))
        raise CalcError(f"unsupported element: {type(node).__name__}")

    try:
        value = ev(tree)
    except (ValueError, TypeError, OverflowError, RecursionError) as exc:
        # e.g. sqrt(-1), round(2, 2.5), huge intermediate results
        raise CalcError(f"{type(exc).__name__}: {exc}") from exc
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise CalcError(f"the result is not a real number ({type(value).__name__})")
    if isinstance(value, float) and not math.isfinite(value):
        raise CalcError("the result is not finite")
    if abs(value) > MAX_MAGNITUDE:
        raise CalcError("result too large")
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return value


CALCULATOR = Tool(
    ToolSpec(
        name="calculate",
        description="Evaluate an arithmetic expression, e.g. '(12 + 3) * 4 / 5'.",
        parameters={
            "type": "object",
            "properties": {"expression": {"type": "string"}},
            "required": ["expression"],
            "additionalProperties": False,
        },
    ),
    lambda args: calculate(str(args["expression"])),
)

_NUM = re.compile(r"-?\d[\d,]*(?:\.\d+)?|-?\.\d+")


def normalize_number(value: JsonValue) -> str | None:
    """Canonical text of a number: '1,200.50' -> '1200.5', '$7' -> '7', 3.0 -> '3'. None if the
    value holds no single number. Uses no information about the expected answer."""
    if isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        text = repr(value)
    elif isinstance(value, str):
        found = _NUM.findall(value.replace("$", "").replace("%", ""))
        if len(found) != 1:
            return None
        text = found[0]
    else:
        return None
    try:
        d = Decimal(text.replace(",", ""))
    except InvalidOperation:
        return None
    if not d.is_finite():
        return None
    d = d.normalize()
    if d == d.to_integral_value():
        return str(int(d))
    return format(d, "f")
