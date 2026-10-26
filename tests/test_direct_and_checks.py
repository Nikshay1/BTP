import asyncio
import json

from bioalloc.budget import Account, MeteredClient
from bioalloc.checks import SchemaCheck, combine
from bioalloc.evaluation.toy_data import grade, make_instance, optimal_legs
from bioalloc.executors.base import ExecutionContext, Upstream, parse_json
from bioalloc.executors.direct import Direct
from bioalloc.llm.fake import FakeTransport
from bioalloc.schemas import Budget, CheckResult
from bioalloc.tasks.toy import RouteCheck, ToyRoutes

INSTANCE = make_instance(seed=3)
LEG1, LEG2 = optimal_legs(INSTANCE)


def ctx(fake: FakeTransport, upstream: dict[str, Upstream] | None = None) -> ExecutionContext:
    account = Account(Budget(max_tokens=20_000, max_model_calls=5))
    llm = MeteredClient(fake, account, actor="worker", agent="direct:leg2:0")
    return ExecutionContext(llm=llm, span_id="s1", upstream=upstream or {})


def test_direct_makes_exactly_one_call_and_passes_upstream_as_communication() -> None:
    fake = FakeTransport().always(f'```json\n{{"route": {json.dumps(LEG2)}}}\n```')
    up = {"leg1": Upstream(subtask_id="leg1", agent="direct:leg1:0", output={"route": LEG1})}
    c = ctx(fake, up)
    leg2 = INSTANCE.graph.node("leg2")
    outcome = asyncio.run(Direct(model="fake").execute(leg2, c))
    assert outcome.output == {"route": LEG2}
    assert len(fake.requests) == 1
    sent = fake.requests[0].messages
    assert sent[-1].source == "direct:leg1:0" and json.dumps(LEG1) in sent[-1].content
    assert c.llm.account.snapshot().communication > 0


def test_parse_json_tolerates_prose_and_keeps_raw_text_on_failure() -> None:
    assert parse_json('Sure! {"route": ["A"]} hope this helps') == {"route": ["A"]}
    assert parse_json("no json here") == "no json here"


def test_schema_check() -> None:
    leg1 = INSTANCE.graph.node("leg1")
    check = SchemaCheck()
    assert check.evaluate(leg1, {"route": LEG1}, {}).verdict == "pass"
    assert check.evaluate(leg1, "not json", {}).verdict == "fail"
    assert check.evaluate(leg1, {"route": []}, {}).verdict == "fail"


def test_route_check_uses_only_public_information() -> None:
    leg1, leg2 = INSTANCE.graph.node("leg1"), INSTANCE.graph.node("leg2")
    check = RouteCheck()
    assert check.evaluate(leg1, {"route": LEG1}, {}).verdict == "pass"
    assert check.evaluate(leg1, {"route": LEG1[:-1]}, {}).verdict == "fail"
    bogus = [LEG1[0], "Z", LEG1[-1]]
    assert check.evaluate(leg1, {"route": bogus}, {}).verdict == "fail"
    # leg2 starts wherever leg1 ended, known only from upstream output.
    assert check.evaluate(leg2, {"route": LEG2}, {"leg1": {"route": LEG1}}).verdict == "pass"
    assert check.evaluate(leg2, {"route": LEG2}, {}).verdict == "unknown"


def test_valid_but_suboptimal_route_passes_checks_but_fails_grading() -> None:
    # Seed 0: the cheapest leg 1 is A-D-G; A-E-G is valid but costs more.
    inst = make_instance(seed=0)
    leg1_best, leg2_best = optimal_legs(inst)
    assert leg1_best == ["A", "D", "G"]
    detour = ["A", "E", "G"]
    family = ToyRoutes()
    leg1 = inst.graph.node("leg1")
    assert RouteCheck().evaluate(leg1, {"route": detour}, {}).verdict == "pass"
    best = family.aggregate(
        inst.graph, {"leg1": {"route": leg1_best}, "leg2": {"route": leg2_best}}
    )
    worse = family.aggregate(inst.graph, {"leg1": {"route": detour}, "leg2": {"route": leg2_best}})
    assert grade(inst, best).correct
    g = grade(inst, worse)
    assert not g.correct and "not cheapest" in g.detail


def test_combine_verdicts() -> None:
    def r(v: str) -> CheckResult:
        return CheckResult(check="c", verdict=v)  # type: ignore[arg-type]

    assert combine([r("pass"), r("unknown")]) == "pass"
    assert combine([r("pass"), r("fail")]) == "fail"
    assert combine([r("unknown")]) == "unknown"
