"""Ant: reinforced search over typed action choices (research plan, Section 4.2; decision D10).

The executor builds solutions as paths of typed actions from the family's action space. At each
step one model call proposes and ranks candidate actions; the illegal or unknown ones are dropped,
the rank gives a prior eta = 1 / rank, and an action is sampled with probability proportional to
tau[kind]^alpha * eta^beta from the seeded RNG. FINISH ends the path with a candidate output, which
is run through the public checks; the path's reward is 1 if a goal-level check verifies it (D15),
else 0. After each finished or abandoned path, every weight evaporates,
tau <- max(tau_min, (1 - rho) * tau), and each distinct action kind on the path receives a deposit
delta * reward (once per kind, so longer paths earn nothing extra).

Weights live in a table private to the task run and the subtask, initialized from a frozen
snapshot (uniform in M2), shared by retries of the same subtask, and dropped when the task ends.

The search stops at the first goal-verified path (D27), after ``max_paths`` paths, or once
``min_finished`` paths have finished and the best candidate has not changed for ``patience``
paths, or when the budget runs out. The best
candidate ranks goal-verified first, then not rejected by any check, then fewest tokens.

Controls (Table 3): ``no_deposit`` (delta = 0), ``no_evaporation`` (rho = 0) and ``uniform``
(alpha = beta = 0, still calling the proposer so that only the sampling bias is removed).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from typing import Any

from pydantic import JsonValue, ValidationError

from bioalloc import prompts
from bioalloc.actions import FINISH, Action, ActionSpace
from bioalloc.budget import BudgetExceeded
from bioalloc.checks import goal_verified
from bioalloc.executors.base import ExecutionContext, ExecutorOutcome, parse_json, subtask_messages
from bioalloc.llm.base import LLMRequest, Message
from bioalloc.schemas import Subtask


def normalize_candidate(item: Any) -> Any:
    """A candidate written flat ({"kind": K, <args>...}) means {"kind": K, "args": {<args>}}.
    Only that unambiguous form is rewritten; anything else is left for validation to reject."""
    if isinstance(item, dict) and "kind" in item and "args" not in item:
        return {"kind": item["kind"], "args": {k: v for k, v in item.items() if k != "kind"}}
    return item


@dataclass(frozen=True)
class AntParams:
    alpha: float = 1.0
    beta: float = 2.0
    rho: float = 0.1
    delta: float = 1.0
    tau0: float = 1.0
    tau_min: float = 0.01
    max_paths: int = 8
    # Long enough for the longest plan in the task pool plus FINISH; the task budget binds first.
    max_choices: int = 30
    min_finished: int = 2
    patience: int = 3
    # D27: stop at the first goal-verified path (its weights are still updated). Off restores the
    # D10 search, which keeps looking for a cheaper verified path.
    stop_on_certificate: bool = True


@dataclass(frozen=True)
class _Candidate:
    output: JsonValue
    verified: bool
    rejected: bool
    tokens: int

    def key(self) -> tuple[bool, bool, int]:
        return (not self.verified, self.rejected, self.tokens)


@dataclass
class Ant:
    model: str
    params: AntParams = field(default_factory=AntParams)
    # Frozen initial weights per action kind; missing kinds start at params.tau0 (uniform in M2).
    snapshot: dict[str, float] = field(default_factory=dict)
    max_completion_tokens: int = 1024
    temperature: float = 0.6
    reasoning_effort: str | None = None
    name: str = "ant"

    def weights(
        self, ctx: ExecutionContext, subtask: Subtask, space: ActionSpace
    ) -> dict[str, float]:
        key = f"ant:{self.name}:{subtask.id}"
        if key not in ctx.task_state:
            ctx.task_state[key] = {k: self.snapshot.get(k, self.params.tau0) for k in space.kinds}
        table: dict[str, float] = ctx.task_state[key]
        return table

    async def execute(self, subtask: Subtask, ctx: ExecutionContext) -> ExecutorOutcome:
        space = ctx.actions
        if space is None:
            return ExecutorOutcome(output=None, incomplete="no action space for this subtask")
        tau = self.weights(ctx, subtask, space)
        best: _Candidate | None = None
        finished = unchanged = 0
        stop_reason = "path_limit"
        for path in range(self.params.max_paths):
            try:
                candidate, kinds_used = await self._walk(subtask, ctx, space, tau, path)
            except BudgetExceeded:
                stop_reason = "budget"
                break
            reward = 1.0 if candidate is not None and candidate.verified else 0.0
            self._update(tau, kinds_used, reward)
            ctx.emit(
                "ant_path",
                {
                    "path": path,
                    "finished": candidate is not None,
                    "reward": reward,
                    "kinds": sorted(kinds_used),
                    "tau": dict(tau),
                },
            )
            if candidate is not None:
                finished += 1
                if best is None or candidate.key() < best.key():
                    best, unchanged = candidate, 0
                else:
                    unchanged += 1
                if self.params.stop_on_certificate and candidate.verified:
                    stop_reason = "certificate"
                    break
            if finished >= self.params.min_finished and unchanged >= self.params.patience:
                stop_reason = "stagnation"
                break
        ctx.emit("executor_stop", {"reason": stop_reason, "paths_finished": finished})
        if best is None:
            if stop_reason == "budget":
                raise BudgetExceeded("budget ran out before any path finished")
            return ExecutorOutcome(output=None, incomplete="no path reached FINISH")
        return ExecutorOutcome(output=best.output)

    def _update(self, tau: dict[str, float], kinds_used: set[str], reward: float) -> None:
        p = self.params
        for k in tau:
            tau[k] = max(p.tau_min, (1 - p.rho) * tau[k])
        for k in kinds_used:
            tau[k] += p.delta * reward

    async def _walk(
        self,
        subtask: Subtask,
        ctx: ExecutionContext,
        space: ActionSpace,
        tau: dict[str, float],
        path: int,
    ) -> tuple[_Candidate | None, set[str]]:
        plain = {k: v.output for k, v in ctx.upstream.items()}
        state: Any = space.start(subtask, plain)
        history: list[str] = []
        kinds_used: set[str] = set()
        before = ctx.llm.account.snapshot().tokens.total
        for step in range(self.params.max_choices):
            proposals = await self._propose(subtask, ctx, space, state, history, path, step)
            if not proposals:
                return None, kinds_used
            action = self._sample(proposals, tau, ctx)
            kinds_used.add(action.kind)
            if action.kind == FINISH:
                output = space.finish(state, action)
                verdict, results = ctx.check(subtask, output)
                tokens = ctx.llm.account.snapshot().tokens.total - before
                return _Candidate(
                    output, goal_verified(results), verdict == "fail", tokens
                ), kinds_used
            state, observation = await self._act(ctx, space, state, action, path, step)
            history.append(f"{action.kind} {json.dumps(action.args)} -> {observation}")
        return None, kinds_used

    async def _act(
        self,
        ctx: ExecutionContext,
        space: ActionSpace,
        state: Any,
        action: Action,
        path: int,
        step: int,
    ) -> tuple[Any, str]:
        """Apply a chosen action. Tool-backed actions go through the metered tool service, with
        the same limits, counting and tracing as for every other executor."""
        call = space.tool_call(state, action, f"{ctx.agent}/path{path}/step{step}")
        if call is None:
            return space.apply(state, action)
        if ctx.tools is None:
            return state, f"{action.kind} needs a tool this subtask does not allow"
        result = await ctx.tools.call(call)  # BudgetExceeded at the tool limit ends the search
        return space.observe(state, action, result)

    async def _propose(
        self,
        subtask: Subtask,
        ctx: ExecutionContext,
        space: ActionSpace,
        state: Any,
        history: list[str],
        path: int,
        step: int,
    ) -> list[Action]:
        # The shared subtask messages (instruction, inputs, upstream results, retry feedback, each
        # with its provenance), then the action-specific state.
        messages = subtask_messages(prompts.load("propose_actions"), subtask, ctx)
        steps = "\n".join(history) if history else "(none)"
        messages.append(
            Message(
                role="user",
                content=f"Allowed actions and their arguments:\n{space.describe_actions()}\n\n"
                f"State: {space.describe(state)}\n\nSteps so far on this attempt:\n{steps}",
            )
        )
        request = LLMRequest(
            model=self.model,
            messages=tuple(messages),
            max_completion_tokens=self.max_completion_tokens,
            temperature=self.temperature,
            reasoning_effort=self.reasoning_effort,  # type: ignore[arg-type]
            seed=ctx.seed + 1000 * path + step,
            sample_id=f"{ctx.agent}/path{path}/step{step}",
        )
        reply = await ctx.llm.complete(request)
        incomplete = incompleteness_of(reply.finish_reason)
        if incomplete is not None:
            ctx.emit("ant_rejected", {"path": path, "step": step, "reasons": [incomplete]})
            return []
        parsed = parse_json(reply.content, envelope="candidates")
        raw = parsed.get("candidates") if isinstance(parsed, dict) else None
        valid: list[Action] = []
        rejected: list[str] = []
        if not isinstance(raw, list):
            rejected.append("no candidates list")
        elif not raw:
            rejected.append("empty candidates list")
        for item in raw if isinstance(raw, list) else []:
            try:
                action = Action.model_validate(normalize_candidate(item))
            except ValidationError:
                rejected.append(f"malformed: {json.dumps(item)[:120]}")
                continue
            if action.kind not in space.kinds:
                rejected.append(f"unknown kind {action.kind}")
                continue
            if action in valid:
                continue
            why = space.legal(state, action)
            if why is None:
                valid.append(action)
            else:
                rejected.append(f"illegal {action.kind}: {why}")
        if rejected:
            ctx.emit("ant_rejected", {"path": path, "step": step, "reasons": rejected})
        return valid

    def choice_weights(self, proposals: list[Action], tau: dict[str, float]) -> list[float]:
        """Unnormalized sampling weights tau[kind]^alpha * (1/rank)^beta, best-ranked first."""
        p = self.params
        return [
            (tau[a.kind] ** p.alpha) * ((1.0 / rank) ** p.beta)
            for rank, a in enumerate(proposals, start=1)
        ]

    def _sample(
        self, proposals: list[Action], tau: dict[str, float], ctx: ExecutionContext
    ) -> Action:
        return ctx.rng.choices(proposals, weights=self.choice_weights(proposals, tau), k=1)[0]


def incompleteness_of(finish_reason: str) -> str | None:
    return "cut off" if finish_reason in ("length", "content_filter") else None


def ant_controls(model: str, **kw: Any) -> dict[str, Ant]:
    """The full ant and its three Table 3 controls, keyed by executor name."""
    base = AntParams()
    variants = {
        "ant": base,
        "ant_no_deposit": replace(base, delta=0.0),
        "ant_no_evaporation": replace(base, rho=0.0),
        "ant_uniform": replace(base, alpha=0.0, beta=0.0),
    }
    return {name: Ant(model=model, params=p, name=name, **kw) for name, p in variants.items()}
