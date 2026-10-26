"""Data contracts shared by the coordinator, executors, checks and the evaluation harness.

Everything here is serializable and immutable. Reference answers have no field anywhere in this
module: they live only in ``bioalloc.evaluation`` so that no system under test can read them.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue, NonNegativeFloat, NonNegativeInt


class Wire(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class Subtask(Wire):
    id: str = Field(min_length=1)
    instruction: str
    inputs: dict[str, JsonValue] = {}
    depends_on: tuple[str, ...] = ()
    allowed_tools: tuple[str, ...] = ()
    # JSON Schema the output must satisfy; None means free text.
    output_schema: dict[str, JsonValue] | None = None
    # Features observable before execution, used by the selector.
    features: dict[str, float] = {}
    # Agent that wrote the instruction (e.g. a model decomposer), or None for task text. Reading
    # model-written instructions counts as communication (decision D6).
    written_by: str | None = None
    # Which public output contract (and so which runtime checks) applies. For a model
    # decomposition only the final node carries the task's contract (decision D17).
    contract: str | None = None


class Budget(Wire):
    """Hard limits for one task or one dispatch. Tokens count prompt plus completion."""

    max_tokens: NonNegativeInt
    max_model_calls: NonNegativeInt
    max_elapsed_s: NonNegativeFloat | None = None
    max_tool_calls: NonNegativeInt | None = None


class Tokens(Wire):
    prompt: NonNegativeInt = 0
    completion: NonNegativeInt = 0

    @property
    def total(self) -> int:
        return self.prompt + self.completion

    def __add__(self, other: Tokens) -> Tokens:
        return Tokens(
            prompt=self.prompt + other.prompt, completion=self.completion + other.completion
        )


Actor = Literal["coordinator", "worker"]


class Cost(Wire):
    """Cost of a run or part of one.

    Two views of the same tokens, each adding up to the total:

    * actor view: ``coordinator`` + ``worker``;
    * purpose view: local work + ``communication``, where communication is the prompt tokens a
      model call spent reading content written by another agent. It is a subset of the actor
      totals and is never added to them.
    """

    coordinator: Tokens = Tokens()
    worker: Tokens = Tokens()
    communication: NonNegativeInt = 0
    reasoning: NonNegativeInt = 0  # subset of completion tokens, when the provider reports it
    model_calls: NonNegativeInt = 0
    tool_calls: NonNegativeInt = 0
    elapsed_s: NonNegativeFloat = 0.0  # wall time, not the sum of call durations
    usd: NonNegativeFloat = 0.0
    # False if any call's usage was unknown (e.g. a timeout) and a reservation was charged instead.
    usage_complete: bool = True
    # True if reported usage pushed spending past the token limit (reservations are estimates).
    overrun: bool = False

    @property
    def tokens(self) -> Tokens:
        return self.coordinator + self.worker

    @property
    def local(self) -> int:
        return self.tokens.total - self.communication


Verdict = Literal["pass", "fail", "unknown"]

# How much a passing check says about the result (decision D15):
# format: schema or type only; operational: tool status, legal actions, consistency;
# agreement: independent samples agree; goal: a public checker verifies the subtask's goal.
Evidence = Literal["format", "operational", "agreement", "goal"]


class CheckResult(Wire):
    check: str
    verdict: Verdict
    reason: str = ""
    evidence: Evidence = "format"


ExecutorStatus = Literal["completed", "budget_exhausted", "timeout", "error"]


class ExecutorResult(Wire):
    executor: str
    subtask_id: str
    output: JsonValue
    # The executor's own estimate in [0, 1]; not assumed to be calibrated.
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    cost: Cost
    status: ExecutorStatus
    # Root span of this execution in the run trace; the full trace is read from there.
    span_id: str
