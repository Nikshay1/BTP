"""Runtime checks: what a system may use to accept or reject a result while it runs.

Only checks that would exist in deployment belong here: output schema and type, tool status,
constraint checkers that are a public part of the task, tests supplied with the task, agreement
between samples. Reference answers never reach a runtime check (research plan, Section 4.4).
A check that cannot decide returns ``unknown``, which is kept distinct from ``pass``.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Protocol

import jsonschema
from pydantic import JsonValue

from bioalloc.schemas import CheckResult, Subtask


class RuntimeCheck(Protocol):
    name: str

    def evaluate(
        self, subtask: Subtask, output: JsonValue, upstream: Mapping[str, JsonValue]
    ) -> CheckResult: ...


class SchemaCheck:
    """Output must match the subtask's JSON Schema; unknown if the subtask declares none."""

    name = "schema"

    def evaluate(
        self, subtask: Subtask, output: JsonValue, upstream: Mapping[str, JsonValue]
    ) -> CheckResult:
        if subtask.output_schema is None:
            return CheckResult(check=self.name, verdict="unknown", reason="no schema declared")
        try:
            jsonschema.validate(output, subtask.output_schema)
        except jsonschema.ValidationError as exc:
            return CheckResult(check=self.name, verdict="fail", reason=exc.message)
        return CheckResult(check=self.name, verdict="pass")


def combine(results: list[CheckResult]) -> str:
    """Overall verdict: fail if any check fails, pass if at least one passes, else unknown."""
    verdicts = {r.verdict for r in results}
    if "fail" in verdicts:
        return "fail"
    if "pass" in verdicts:
        return "pass"
    return "unknown"
