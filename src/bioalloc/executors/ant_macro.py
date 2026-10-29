"""Macro ant: reinforced choice between proposed plan segments, with milestone rewards (D30).

The step-per-call ant (D10) cannot finish long plans at a realistic budget (R13: 14-45 actions,
5-7 calls). This variant, designed by GPT-6.1 Sol (R14) as an explicit amendment to D10, works
on planning problems with a public goal:

* each decision is one model call proposing up to two ranked segments of at most
  ``max_segment`` ground actions from the current state;
* every segment is simulated; a segment with an inapplicable step is dropped whole, and the
  reasons are shown at the next decision (the same feedback reaches every control);
* among the legal segments one is sampled with probability tau[feature]^alpha * (1/rank)^beta from
  the seeded RNG, where the feature is the segment's shape: its action kinds in order of first
  use (for example ``drive-truck>load-truck>unload-truck``), a fixed public vocabulary;
* after every decision all weights evaporate, tau <- max(tau_min, (1 - rho) * tau); the chosen
  segment's feature receives delta if the segment is a milestone (it makes at least one goal atom
  true that was not true before; mere legality earns nothing);
* the search stops on the goal certificate (the plan reaches the goal), after ``max_decisions``
  decisions, or when the budget runs out; the answer is the plan so far.

Controls (Table 3, same proposer, simulation, feedback, budgets and stopping): ``no_deposit``
(delta 0), ``no_evaporation`` (rho 0), ``uniform`` (alpha = beta = 0) and ``greedy``, the ordinary
macro planner that always applies the best-ranked legal segment. Weights are task-local and start
uniform; ``ant_macro_choices`` in the trace records, at every decision, each candidate's feature,
weight and probability, so it can be checked that deposits change later choices.
"""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, field
from typing import Any

from pydantic import JsonValue

from bioalloc import prompts
from bioalloc.budget import BudgetExceeded
from bioalloc.executors.base import ExecutionContext, ExecutorOutcome, parse_json, subtask_messages
from bioalloc.llm.base import LLMError, LLMRequest, Message
from bioalloc.schemas import Subtask
from bioalloc.tasks.planbench import models
from bioalloc.tasks.strips import PDDLError, applicable, apply, parse_step


@dataclass(frozen=True)
class MacroParams:
    alpha: float = 1.0
    beta: float = 2.0
    rho: float = 0.1
    delta: float = 1.0
    tau0: float = 1.0
    tau_min: float = 0.01
    max_decisions: int = 5
    max_segment: int = 12
    candidates: int = 2
    greedy: bool = False


def feature(segment: list[tuple[str, ...]]) -> str:
    seen: list[str] = []
    for step in segment:
        if step[0] not in seen:
            seen.append(step[0])
    return ">".join(seen)


@dataclass
class MacroAnt:
    model: str
    params: MacroParams = field(default_factory=MacroParams)
    max_completion_tokens: int = 2048
    temperature: float = 0.6
    reasoning_effort: str | None = None
    name: str = "ant_macro"

    async def execute(self, subtask: Subtask, ctx: ExecutionContext) -> ExecutorOutcome:
        try:
            domain, problem = models(subtask)
        except (KeyError, PDDLError) as exc:
            return ExecutorOutcome(output=None, incomplete=f"not a planning problem: {exc}")
        p = self.params
        facts = problem.init
        plan: list[str] = []
        tau: dict[str, float] = {}
        feedback = ""
        reason = "decision_limit"
        for decision in range(p.max_decisions):
            try:
                proposals = await self._propose(
                    subtask, ctx, facts, plan, problem.goal, feedback, decision
                )
            except BudgetExceeded:
                reason = "budget"
                break
            except LLMError:
                feedback = "The last proposal failed; propose again."
                continue
            legal, rejected = [], []
            for rank, raw in enumerate(proposals[: p.candidates], start=1):
                result = self._simulate(domain, problem, facts, raw)
                if isinstance(result, str):
                    rejected.append(f"segment {rank}: {result}")
                else:
                    legal.append((rank, raw, result))
            feedback = (
                "Your proposed segments could not be applied: " + "; ".join(rejected)
                if rejected and not legal
                else ""
            )
            if not legal:
                ctx.emit("ant_macro_choices", {"decision": decision, "rejected": rejected})
                self._evaporate(tau)
                continue
            for _, _, (steps, _) in legal:
                tau.setdefault(feature(steps), p.tau0)
            weights = [
                (tau[feature(steps)] ** p.alpha) * ((1.0 / rank) ** p.beta)
                for rank, _, (steps, _) in legal
            ]
            index = 0 if p.greedy else ctx.rng.choices(range(len(legal)), weights=weights)[0]
            rank, raw, (steps, after) = legal[index]
            newly = [g for g in problem.goal if g in after and g not in facts]
            total = sum(weights)
            ctx.emit(
                "ant_macro_choices",
                {
                    "decision": decision,
                    "rejected": rejected,
                    "candidates": [
                        {"rank": r, "feature": feature(s), "weight": w, "p": w / total}
                        for (r, _, (s, _)), w in zip(legal, weights, strict=True)
                    ],
                    "chosen": rank,
                    "milestone": len(newly),
                },
            )
            self._evaporate(tau)
            if newly:
                tau[feature(steps)] += p.delta
            facts = after
            plan += ["(" + " ".join(s) + ")" for s in steps]
            if all(g in facts for g in problem.goal):
                reason = "certificate"
                break
        ctx.emit("executor_stop", {"reason": reason, "plan_length": len(plan)})
        if reason == "budget" and not plan:
            raise BudgetExceeded("budget ran out before any segment was applied")
        return ExecutorOutcome(output={"plan": list[JsonValue](plan)})

    def _evaporate(self, tau: dict[str, float]) -> None:
        for k in tau:
            tau[k] = max(self.params.tau_min, (1 - self.params.rho) * tau[k])

    def _simulate(self, domain: Any, problem: Any, facts: Any, raw: Any) -> Any:
        """(steps, facts after) for a fully applicable segment, else the reason it is not."""
        if not isinstance(raw, list) or not raw or not all(isinstance(s, str) for s in raw):
            return "not a list of action strings"
        if len(raw) > self.params.max_segment:
            return f"longer than {self.params.max_segment} actions"
        steps = []
        for i, text in enumerate(raw, start=1):
            try:
                step = parse_step(text)
            except PDDLError as exc:
                return f"step {i}: {exc}"
            why = applicable(domain, problem, facts, step)
            if why is not None:
                return f"step {i} {text}: {why}"
            facts = apply(domain, facts, step)
            steps.append(step)
        return steps, facts

    async def _propose(
        self,
        subtask: Subtask,
        ctx: ExecutionContext,
        facts: Any,
        plan: list[str],
        goal: Any,
        feedback: str,
        decision: int,
    ) -> list[Any]:
        messages = subtask_messages(prompts.load("propose_segments"), subtask, ctx)
        reached = [" ".join(g) for g in goal if g in facts]
        state = (
            f"True facts: {json.dumps(sorted(' '.join(a) for a in facts))}.\n"
            f"Plan so far: {json.dumps(plan)}.\nGoal atoms reached: {json.dumps(reached)}."
        )
        if feedback:
            state += f"\n{feedback}"
        messages.append(Message(role="user", content=state))
        request = LLMRequest(
            model=self.model,
            messages=tuple(messages),
            max_completion_tokens=self.max_completion_tokens,
            temperature=self.temperature,
            reasoning_effort=self.reasoning_effort,  # type: ignore[arg-type]
            seed=ctx.seed + decision,
            sample_id=f"{ctx.agent}/decision{decision}",
        )
        client = dataclasses.replace(ctx.llm, agent=ctx.agent)
        reply = await client.complete(request)
        if reply.finish_reason in ("length", "content_filter"):
            return []
        parsed = parse_json(reply.content, envelope="segments")
        segments = parsed.get("segments") if isinstance(parsed, dict) else None
        return segments if isinstance(segments, list) else []


def macro_controls(model: str, **kw: Any) -> dict[str, MacroAnt]:
    base = MacroParams()
    variants = {
        "ant_macro": base,
        "ant_macro_no_deposit": dataclasses.replace(base, delta=0.0),
        "ant_macro_no_evaporation": dataclasses.replace(base, rho=0.0),
        "ant_macro_uniform": dataclasses.replace(base, alpha=0.0, beta=0.0),
        "macro_greedy": dataclasses.replace(base, greedy=True),
    }
    return {name: MacroAnt(model=model, params=p, name=name, **kw) for name, p in variants.items()}
