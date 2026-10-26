import asyncio
import json
import time
from pathlib import Path

import pytest

from bioalloc.budget import Account, BudgetExceeded, MeteredClient, reserve_prompt_tokens
from bioalloc.coordinator import Coordinator
from bioalloc.executors.base import ExecutionContext
from bioalloc.executors.direct import Direct
from bioalloc.graph import TaskGraph
from bioalloc.llm.base import LLMReply, LLMRequest, Message, ToolCall, Usage
from bioalloc.llm.fake import FakeTransport
from bioalloc.policies import Constant
from bioalloc.schemas import Budget, Subtask
from bioalloc.tools import Tool, ToolService, ToolSpec
from bioalloc.trace import Tracer, read_trace, tokens_from_trace

ADD = Tool(
    ToolSpec(
        name="add",
        description="Add two integers.",
        parameters={
            "type": "object",
            "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
            "required": ["a", "b"],
            "additionalProperties": False,
        },
    ),
    lambda args: args["a"] + args["b"],
)


def boom(args):  # type: ignore[no-untyped-def]
    raise ValueError("bad input")


def slow(args):  # type: ignore[no-untyped-def]
    time.sleep(0.5)
    return 1


ANY = {"type": "object"}
REGISTRY = {
    "add": ADD,
    "boom": Tool(ToolSpec(name="boom", description="", parameters=ANY), boom),
    "slow": Tool(ToolSpec(name="slow", description="", parameters=ANY), slow),
    "big": Tool(ToolSpec(name="big", description="", parameters=ANY), lambda a: "x" * 10_000),
    "secret": ADD,
}


def service(budget: Budget | None = None) -> tuple[ToolService, list[dict]]:  # type: ignore[type-arg]
    events: list[dict] = []  # type: ignore[type-arg]
    account = Account(budget or Budget(max_tokens=100, max_model_calls=1))
    svc = ToolService(
        REGISTRY,
        ("add", "boom", "slow", "big"),
        account,
        emit=lambda k, d: events.append(d),
        timeout_s=0.1,
    )
    return svc, events


def call(svc: ToolService, name: str, args: str) -> tuple[bool, str | None, object]:
    r = asyncio.run(svc.call(ToolCall(id="c1", name=name, arguments=args)))
    return r.ok, r.error, r.output


def test_tool_service_runs_valid_calls_and_reports_failures_as_observations() -> None:
    svc, events = service()
    assert call(svc, "add", '{"a": 2, "b": 3}') == (True, None, 5)
    assert "not available" in str(call(svc, "secret", '{"a": 1, "b": 1}')[1])
    assert "valid JSON" in str(call(svc, "add", "{oops")[1])
    assert "invalid arguments" in str(call(svc, "add", '{"a": "two", "b": 3}')[1])
    assert "ValueError: bad input" in str(call(svc, "boom", "{}")[1])
    assert "timed out" in str(call(svc, "slow", "{}")[1])
    assert "too large" in str(call(svc, "big", "{}")[1])
    # Every attempt counts, failures included, and every attempt is traced.
    assert svc.account.snapshot().tool_calls == 7 and len(events) == 7


def test_tool_call_limit_stops_execution() -> None:
    svc, events = service(Budget(max_tokens=100, max_model_calls=1, max_tool_calls=1))
    call(svc, "add", '{"a": 1, "b": 1}')
    with pytest.raises(BudgetExceeded, match="tool call limit"):
        call(svc, "add", '{"a": 1, "b": 1}')
    assert svc.account.snapshot().tool_calls == 1 and len(events) == 1


def test_reservation_counts_tool_calls_in_messages() -> None:
    plain = Message(role="assistant", content="")
    with_call = plain.model_copy(
        update={"tool_calls": (ToolCall(id="x", name="add", arguments='{"a": 1, "b": 2}' * 20),)}
    )

    def r(m: Message) -> LLMRequest:
        return LLMRequest(model="m", messages=(m,), max_completion_tokens=1)

    assert reserve_prompt_tokens(r(with_call)) > reserve_prompt_tokens(r(plain)) + 100


def reply(content: str = "", calls: tuple[ToolCall, ...] = ()) -> LLMReply:
    return LLMReply(
        content=content,
        tool_calls=calls,
        model="m",
        usage=Usage(prompt_tokens=10, completion_tokens=5),
        finish_reason="tool_calls" if calls else "stop",
    )


def test_direct_runs_one_tool_episode_and_reads_the_result_back(tmp_path: Path) -> None:
    graph = TaskGraph(
        id="t",
        nodes=(Subtask(id="a", instruction="What is 2 + 3?", allowed_tools=("add",)),),
        final_nodes=("a",),
    )
    fake = FakeTransport().always(
        reply(calls=(ToolCall(id="call_1", name="add", arguments='{"a": 2, "b": 3}'),)),
        reply('{"answer": 5}'),
    )
    llm = MeteredClient(
        fake, Account(Budget(max_tokens=50_000, max_model_calls=10)), "coordinator", "coordinator"
    )

    class Echo:
        name = "echo"

        def checks(self, s):  # type: ignore[no-untyped-def]
            return []

        def aggregate(self, g, outputs):  # type: ignore[no-untyped-def]
            return dict(outputs)

    coord = Coordinator({"direct": Direct(model="m")}, Constant("direct"), tools=REGISTRY)
    tracer = Tracer(tmp_path, "r")
    result = asyncio.run(coord.run(graph, Echo(), llm, tracer))
    tracer.close()
    assert result.answer == {"a": {"answer": 5}}
    first, second = fake.requests
    assert [t["function"]["name"] for t in first.tools] == ["add"]  # type: ignore[index]
    assistant, tool = second.messages[-2:]
    assert assistant.tool_calls[0].id == "call_1" and tool.tool_call_id == "call_1"
    assert json.loads(tool.content) == 5 and tool.source is None
    assert result.cost.tool_calls == 1 and result.cost.model_calls == 2
    totals = tokens_from_trace(list(read_trace(tmp_path)))
    assert totals["tool_calls"] == 1 and totals["model_calls"] == 2


def test_tool_episode_is_bounded() -> None:
    loop_call = (ToolCall(id="c", name="add", arguments='{"a": 1, "b": 1}'),)
    fake = FakeTransport().always(*[reply(calls=loop_call) for _ in range(10)])
    account = Account(Budget(max_tokens=50_000, max_model_calls=10))
    llm = MeteredClient(fake, account, "worker", "w")
    ctx = ExecutionContext(llm=llm, span_id="s", tools=ToolService(REGISTRY, ("add",), account))
    asyncio.run(Direct(model="m", max_completions=3).execute(Subtask(id="a", instruction="x"), ctx))
    assert len(fake.requests) == 3 and account.snapshot().tool_calls == 2


def test_direct_without_tools_makes_exactly_one_call() -> None:
    fake = FakeTransport().always(reply(calls=(ToolCall(id="c", name="add", arguments="{}"),)))
    llm = MeteredClient(fake, Account(Budget(max_tokens=50_000, max_model_calls=10)), "worker", "w")
    ctx = ExecutionContext(llm=llm, span_id="s")
    asyncio.run(Direct(model="m").execute(Subtask(id="a", instruction="x"), ctx))
    assert len(fake.requests) == 1 and fake.requests[0].tools == ()


def test_truncated_or_unfinished_output_is_rejected_even_if_it_parses(tmp_path: Path) -> None:
    graph = TaskGraph(id="t", nodes=(Subtask(id="a", instruction="x"),), final_nodes=("a",))
    cut = LLMReply(
        content='{"answer": 4}',
        model="m",
        finish_reason="length",
        usage=Usage(prompt_tokens=10, completion_tokens=5),
    )
    fake = FakeTransport().always(cut, reply('{"answer": 5}'))
    llm = MeteredClient(
        fake, Account(Budget(max_tokens=50_000, max_model_calls=10)), "coordinator", "coordinator"
    )

    class Lenient:  # no checks at all: everything would otherwise be "unknown" and accepted
        name = "lenient"

        def checks(self, s):  # type: ignore[no-untyped-def]
            return []

        def aggregate(self, g, outputs):  # type: ignore[no-untyped-def]
            return dict(outputs)

    result = asyncio.run(
        Coordinator({"direct": Direct(model="m")}, Constant("direct")).run(
            graph, Lenient(), llm, Tracer(tmp_path, "r")
        )
    )
    first = result.attempts[0]
    assert not first.accepted and first.checks[-1].check == "generation"
    assert result.answer == {"a": {"answer": 5}}


def test_non_json_tool_output_is_an_observation_not_a_crash() -> None:
    registry = {
        "odd": Tool(ToolSpec(name="odd", description="", parameters=ANY), lambda a: complex(1, 1)),
        "nan": Tool(ToolSpec(name="nan", description="", parameters=ANY), lambda a: float("nan")),
    }
    svc = ToolService(registry, ("odd", "nan"), Account(Budget(max_tokens=1, max_model_calls=1)))
    for name in ("odd", "nan"):
        r = asyncio.run(svc.call(ToolCall(id="x", name=name, arguments="{}")))
        assert not r.ok and "not JSON" in str(r.error)
