"""Tests for the BaseAgent runtime and tool loop (Issue #13).

A scripted LLM client returns queued actions, so the tests exercise prompting,
tool execution, AHP validation and retry behaviour without any network or GPU.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from ahp.interfaces import Agent
from ahp.models import (
    AHPErrorCode,
    DoneMessage,
    DoneStatus,
    HandoffMessage,
    NeedInfoMessage,
)
from ahp.router import InProcessRouter, RouterConfig
from engine.client import LLMResponseError
from engine.interfaces import LLMClient
from engine.models import LLMResult
from harness.agent import AgentReplyError, BaseAgent, action_schema
from harness.tools import SQLiteMockDB, Tool, sqlite_tools


class ScriptedLLM:
    """Queue-backed fake ``LLMClient`` recording every prompt it receives."""

    def __init__(self, *responses: dict[str, Any] | str | None | Exception) -> None:
        self.responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    async def chat(
        self,
        messages: list[dict[str, Any]],
        schema: dict[str, Any] | None = None,
        temperature: float = 0.0,
    ) -> LLMResult:
        self.calls.append(
            {
                "messages": [dict(message) for message in messages],
                "schema": schema,
                "temperature": temperature,
            }
        )
        if not self.responses:
            raise AssertionError("scripted LLM ran out of responses")
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        if isinstance(response, dict):
            return LLMResult(text=json.dumps(response), structured_data=response)
        if response is None:
            return LLMResult(text="", structured_data=None)
        return LLMResult(text=response, structured_data=None)

    async def aclose(self) -> None:
        return None


def tool_call(name: str, **arguments: Any) -> dict[str, Any]:
    return {"action": "tool_call", "tool_call": {"name": name, "arguments": arguments}}


def reply(payload: dict[str, Any]) -> dict[str, Any]:
    return {"action": "reply", "reply": payload}


def handoff_reply(**overrides: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "type": "HANDOFF",
        "to_agent": "billing",
        "goal": "Decide whether the duplicate charge should be refunded",
        "summary": "Verified the customer account and billing history.",
        "facts": {"customer_id": "CUST-001"},
    }
    payload.update(overrides)
    return reply(payload)


def make_handoff(**overrides: Any) -> HandoffMessage:
    payload: dict[str, Any] = {
        "conversation_id": "conv-1",
        "from_agent": "orchestrator",
        "to_agent": "triage",
        "goal": "Resolve the customer's billing complaint",
        "summary": "Customer reports a duplicate Pro charge.",
        "facts": {"customer_id": "CUST-001"},
    }
    payload.update(overrides)
    return HandoffMessage.model_validate(payload)


def make_agent(
    *responses: dict[str, Any] | str | None | Exception,
    name: str = "triage",
    tools: Any = (),
    **kwargs: Any,
) -> tuple[BaseAgent, ScriptedLLM]:
    llm = ScriptedLLM(*responses)
    agent = BaseAgent(name, "You resolve support tickets accurately.", llm, tools=tools, **kwargs)
    return agent, llm


@pytest.fixture
def db() -> SQLiteMockDB:
    database = SQLiteMockDB()
    try:
        yield database
    finally:
        database.close()


# ---------------------------------------------------------------------------
# Tool calls and AHP drafting
# ---------------------------------------------------------------------------


async def test_agent_executes_tool_and_drafts_valid_handoff(db: SQLiteMockDB) -> None:
    agent, llm = make_agent(
        tool_call("query_customer", customer_id="CUST-001"),
        handoff_reply(
            facts={
                "customer_id": "CUST-001",
                "customer_name": "Ava Thompson",
                "plan": "pro",
            }
        ),
        tools=sqlite_tools(db),
    )

    result = await agent.handle_handoff(make_handoff())

    assert isinstance(result, HandoffMessage)
    assert result.from_agent == "triage"
    assert result.to_agent == "billing"
    assert result.conversation_id == "conv-1"
    assert result.facts == {
        "customer_id": "CUST-001",
        "customer_name": "Ava Thompson",
        "plan": "pro",
    }

    assert len(llm.calls) == 2
    assert llm.calls[0]["schema"] == agent.response_schema
    assert "Available tools" in llm.calls[0]["messages"][0]["content"]

    tool_context = llm.calls[1]["messages"][-1]["content"]
    assert "Tool result" in tool_context
    assert "Ava Thompson" in tool_context
    assert [message["role"] for message in llm.calls[1]["messages"]] == [
        "system",
        "user",
        "assistant",
        "user",
    ]

    assert len(agent.tool_calls) == 1
    record = agent.tool_calls[0]
    assert record.tool == "query_customer"
    assert record.ok is True
    assert record.arguments == {"customer_id": "CUST-001"}
    assert record.result["name"] == "Ava Thompson"


async def test_agent_integrates_charge_lookup_into_context(db: SQLiteMockDB) -> None:
    agent, llm = make_agent(
        tool_call("query_charges", customer_id="CUST-001"),
        handoff_reply(facts={"customer_id": "CUST-001", "total_amount": 147.0}),
        tools=sqlite_tools(db),
    )

    await agent.handle_handoff(make_handoff())

    tool_context = llm.calls[1]["messages"][-1]["content"]
    assert '"total_amount": 147.0' in tool_context
    assert "CHG-1001" in tool_context
    assert "CHG-1002" in tool_context
    assert agent.tool_calls[0].result["suspected_duplicate_charge_ids"] == [
        "CHG-1001",
        "CHG-1002",
    ]


async def test_text_only_json_replies_are_accepted() -> None:
    agent, llm = make_agent(json.dumps(handoff_reply()))

    result = await agent.handle_handoff(make_handoff())

    assert isinstance(result, HandoffMessage)
    assert len(llm.calls) == 1


async def test_agent_handles_need_info_and_answers_with_updated_handoff() -> None:
    incoming = NeedInfoMessage(
        conversation_id="conv-2",
        from_agent="billing",
        to_agent="triage",
        missing_fields=["customer_id"],
        query="Which customer should I bill?",
    )
    agent, llm = make_agent(
        handoff_reply(facts={"customer_id": "CUST-003"}),
    )

    result = await agent.handle_handoff(incoming)

    assert isinstance(result, HandoffMessage)
    assert result.conversation_id == "conv-2"
    assert result.from_agent == "triage"
    assert result.to_agent == "billing"
    assert result.facts == {"customer_id": "CUST-003"}
    assert "missing_fields" in llm.calls[0]["messages"][1]["content"]


async def test_done_reply_defaults_to_the_sender() -> None:
    agent, _ = make_agent(
        reply(
            {
                "type": "DONE",
                "final_result": {"decision": "refund", "amount": 49.0},
                "summary": "Refunded the duplicate charge.",
            }
        ),
        name="billing",
    )

    result = await agent.handle_handoff(make_handoff(to_agent="billing"))

    assert isinstance(result, DoneMessage)
    assert result.from_agent == "billing"
    assert result.to_agent == "orchestrator"
    assert result.status is DoneStatus.RESOLVED
    assert result.final_result == {"decision": "refund", "amount": 49.0}


async def test_need_info_reply_defaults_to_the_sender() -> None:
    agent, _ = make_agent(
        reply(
            {
                "type": "NEED_INFO",
                "missing_fields": ["customer_id"],
                "query": "Which customer reported the issue?",
            }
        )
    )

    result = await agent.handle_handoff(make_handoff())

    assert isinstance(result, NeedInfoMessage)
    assert result.from_agent == "triage"
    assert result.to_agent == "orchestrator"
    assert result.missing_fields == ["customer_id"]


# ---------------------------------------------------------------------------
# Re-prompting and protocol errors
# ---------------------------------------------------------------------------


async def test_malformed_output_is_reprompted_then_accepted() -> None:
    agent, llm = make_agent("not json", handoff_reply())

    result = await agent.handle_handoff(make_handoff())

    assert isinstance(result, HandoffMessage)
    assert len(llm.calls) == 2
    correction = llm.calls[1]["messages"][-1]["content"]
    assert "rejected" in correction
    assert "exactly one JSON object" in correction


async def test_schema_violation_is_reprompted_then_accepted() -> None:
    agent, llm = make_agent(reply({"type": "ACK"}), handoff_reply())

    result = await agent.handle_handoff(make_handoff())

    assert isinstance(result, HandoffMessage)
    assert len(llm.calls) == 2
    correction = llm.calls[1]["messages"][-1]["content"]
    assert "rejected" in correction
    assert "ACK" in correction


async def test_client_validation_error_is_reprompted() -> None:
    agent, llm = make_agent(
        LLMResponseError("model output is not valid JSON: Expecting value"),
        handoff_reply(),
    )

    result = await agent.handle_handoff(make_handoff())

    assert isinstance(result, HandoffMessage)
    assert len(llm.calls) == 2
    correction = llm.calls[1]["messages"][-1]["content"]
    assert "rejected" in correction
    assert "not valid JSON" in correction


async def test_client_validation_errors_exhaust_the_retry_budget() -> None:
    agent, llm = make_agent(
        *(LLMResponseError("invalid structured output") for _ in range(3)),
    )

    with pytest.raises(AgentReplyError, match="after 3 attempt"):
        await agent.handle_handoff(make_handoff())

    assert len(llm.calls) == 3


async def test_three_invalid_outputs_raise_a_protocol_error() -> None:
    agent, llm = make_agent(
        "not json",
        reply({"type": "ACK"}),
        reply({"type": "HANDOFF"}),  # parses, but misses required AHP fields
    )

    with pytest.raises(AgentReplyError, match="after 3 attempt") as captured:
        await agent.handle_handoff(make_handoff())

    assert len(llm.calls) == 3
    assert captured.value.code == AHPErrorCode.AGENT_ERROR


async def test_unknown_tool_is_reprompted_then_accepted(db: SQLiteMockDB) -> None:
    agent, llm = make_agent(
        tool_call("refund_customer", customer_id="CUST-001"),
        handoff_reply(),
        tools=sqlite_tools(db),
    )

    result = await agent.handle_handoff(make_handoff())

    assert isinstance(result, HandoffMessage)
    assert len(llm.calls) == 2
    correction = llm.calls[1]["messages"][-1]["content"]
    assert "unknown tool" in correction
    assert "refund_customer" in correction
    assert agent.tool_calls == []


async def test_missing_tool_arguments_are_reprompted(db: SQLiteMockDB) -> None:
    agent, llm = make_agent(
        tool_call("query_customer"),
        handoff_reply(),
        tools=sqlite_tools(db),
    )

    result = await agent.handle_handoff(make_handoff())

    assert isinstance(result, HandoffMessage)
    assert len(llm.calls) == 2
    correction = llm.calls[1]["messages"][-1]["content"]
    assert "customer_id" in correction
    assert "required" in correction.lower()


async def test_tool_exceptions_are_returned_to_the_model() -> None:
    def explode(**_: Any) -> Any:
        raise RuntimeError("kaboom")

    failing_tool = Tool(
        name="explode",
        description="Always fails.",
        handler=explode,
        parameters={"type": "object", "properties": {}, "additionalProperties": False},
    )
    agent, llm = make_agent(
        tool_call("explode"),
        handoff_reply(),
        tools=[failing_tool],
    )

    result = await agent.handle_handoff(make_handoff())

    assert isinstance(result, HandoffMessage)
    assert len(llm.calls) == 2
    record = agent.tool_calls[0]
    assert record.ok is False
    assert record.error is not None
    assert "kaboom" in record.error
    assert "kaboom" in llm.calls[1]["messages"][-1]["content"]


async def test_tool_iteration_limit_raises(db: SQLiteMockDB) -> None:
    agent, llm = make_agent(
        tool_call("query_customer", customer_id="CUST-001"),
        tool_call("query_customer", customer_id="CUST-002"),
        tool_call("query_customer", customer_id="CUST-003"),
        tools=sqlite_tools(db),
        max_tool_iterations=2,
    )

    with pytest.raises(AgentReplyError, match="tool-call limit"):
        await agent.handle_handoff(make_handoff())

    assert len(llm.calls) == 3
    assert len(agent.tool_calls) == 2


# ---------------------------------------------------------------------------
# Construction and protocol conformance
# ---------------------------------------------------------------------------


async def test_agent_satisfies_the_agent_protocol() -> None:
    agent, _ = make_agent(handoff_reply())

    assert isinstance(agent, Agent)
    assert isinstance(agent.llm, LLMClient)


def test_constructor_validates_its_arguments() -> None:
    llm = ScriptedLLM()

    with pytest.raises(ValueError):
        BaseAgent("", "prompt", llm)
    with pytest.raises(ValueError):
        BaseAgent("triage", "   ", llm)
    with pytest.raises(ValueError):
        BaseAgent("triage", "prompt", llm, max_retries=-1)
    with pytest.raises(ValueError):
        BaseAgent("triage", "prompt", llm, max_tool_iterations=0)
    with pytest.raises(ValueError):
        BaseAgent("triage", "prompt", llm, temperature=-0.1)


def test_custom_response_schema_is_deep_copied() -> None:
    schema = action_schema()
    agent, _ = make_agent(handoff_reply(), response_schema=schema)

    schema["oneOf"].clear()

    assert agent.response_schema["oneOf"]


# ---------------------------------------------------------------------------
# Router integration
# ---------------------------------------------------------------------------


async def test_router_connects_two_base_agents_end_to_end(db: SQLiteMockDB) -> None:
    triage, triage_llm = make_agent(
        tool_call("query_customer", customer_id="CUST-001"),
        handoff_reply(
            facts={"customer_id": "CUST-001", "customer_name": "Ava Thompson"},
        ),
        tools=sqlite_tools(db),
    )
    billing, billing_llm = make_agent(
        reply(
            {
                "type": "DONE",
                "final_result": {
                    "decision": "refund",
                    "amount": 49.0,
                    "duplicate_charge_ids": ["CHG-1001", "CHG-1002"],
                },
                "summary": "Refunded one duplicate Pro charge.",
            }
        ),
        name="billing",
    )

    router = InProcessRouter(RouterConfig(ack_timeout_s=1.0), run_id="run-13")
    router.register_agent("triage", triage.handle_handoff)
    router.register_agent("billing", billing.handle_handoff)

    try:
        result = await router.run_workflow(
            make_handoff(conversation_id="conv-router"),
            timeout_s=5.0,
        )
    finally:
        await router.stop()

    assert isinstance(result, DoneMessage)
    assert result.from_agent == "billing"
    assert result.conversation_id == "conv-router"
    assert result.final_result["decision"] == "refund"
    assert triage.tool_calls[0].result["name"] == "Ava Thompson"
    assert billing.tool_calls == []
    assert len(triage_llm.calls) == 2
    assert len(billing_llm.calls) == 1
