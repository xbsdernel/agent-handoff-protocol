"""Tests for the support triage chain (Issue #14).

A prompt-driven fake ``LLMClient`` plays each role, so the tests exercise the
real router, agent tool loops, fact accumulation and schema validation without
network, GPU or a real engine.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from ahp.models import DoneMessage, DoneStatus, NeedInfoMessage
from engine.models import LLMResult
from harness.tools import SQLiteMockDB
from harness.workflows.schemas import (
    SUMMARY_MAX_CHARS,
    billing_reply_schema,
    reply_reply_schema,
    triage_reply_schema,
)
from harness.workflows.support_chain import (
    BILLING_PROMPT,
    REPLY_PROMPT,
    TRIAGE_PROMPT,
    SupportChain,
    SupportTicket,
    assess_refund,
    billing_tools,
    load_support_tickets,
)

# ---------------------------------------------------------------------------
# Fake role client
# ---------------------------------------------------------------------------


def _tool_call_action(name: str, **arguments: Any) -> dict[str, Any]:
    return {"action": "tool_call", "tool_call": {"name": name, "arguments": arguments}}


def _reply_action(payload: dict[str, Any]) -> dict[str, Any]:
    return {"action": "reply", "reply": payload}


def _incoming_payload(messages: list[dict[str, Any]]) -> dict[str, Any]:
    for message in messages:
        content = str(message.get("content", ""))
        if message.get("role") == "user" and content.startswith(
            "Process this incoming AHP message"
        ):
            return json.loads(content.split("\n\n", 1)[1])
    raise AssertionError("incoming AHP message not found in prompt")


def _ticket_record(messages: list[dict[str, Any]]) -> dict[str, Any] | None:
    for message in messages:
        content = str(message.get("content", ""))
        if message.get("role") == "user" and content.startswith("Authoritative ticket record"):
            return json.loads(content.split("\n", 1)[1])
    return None


def _last_tool_result(messages: list[dict[str, Any]]) -> dict[str, Any] | None:
    for message in reversed(messages):
        content = str(message.get("content", ""))
        if message.get("role") == "user" and content.startswith("Tool result:"):
            return json.loads(content.split("\n", 1)[1])
    return None


def _infer_issue_type(text: str) -> str:
    lowered = text.lower()
    if any(word in lowered for word in ("duplicate", "twice", "double")):
        return "duplicate_charge"
    if "failed" in lowered:
        return "failed_payment"
    if "refund" in lowered or "money back" in lowered:
        return "refund_request"
    if any(word in lowered for word in ("question", "explain", "confirm", "check", "dispute")):
        return "billing_question"
    return "other"


class ChainLLM:
    """Prompt-driven fake that plays one support-chain role."""

    def __init__(
        self,
        role: str,
        *,
        omit_customer_id_first: bool = False,
        ask_customer_first: bool = False,
    ) -> None:
        if role not in {"triage", "billing", "reply"}:
            raise ValueError(role)
        self.role = role
        self.omit_customer_id_first = omit_customer_id_first
        self.ask_customer_first = ask_customer_first
        self.calls: list[dict[str, Any]] = []
        self._triage_calls = 0

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
        if self.role == "triage":
            payload = self._triage(messages)
        elif self.role == "billing":
            payload = self._billing(messages)
        else:
            payload = self._reply(messages)
        return LLMResult(text=json.dumps(payload), structured_data=payload)

    async def aclose(self) -> None:
        return None

    def _triage(self, messages: list[dict[str, Any]]) -> dict[str, Any]:
        self._triage_calls += 1
        incoming = _incoming_payload(messages)
        ticket = _ticket_record(messages) or {}
        incoming_facts = incoming.get("facts") or {}

        if self.ask_customer_first and self._triage_calls == 1:
            return _reply_action(
                {
                    "type": "NEED_INFO",
                    "to_agent": "customer",
                    "missing_fields": ["customer_id"],
                    "query": "Please provide your customer ID so I can locate the account.",
                }
            )

        customer_id = incoming_facts.get("customer_id") or ticket.get("customer_id")
        if not customer_id:
            return _reply_action(
                {
                    "type": "NEED_INFO",
                    "to_agent": "customer",
                    "missing_fields": ["customer_id"],
                    "query": "Please provide your customer ID so I can locate the account.",
                }
            )

        ticket_text = ticket.get("input_text") or incoming_facts.get("ticket_text") or ""
        issue_type = _infer_issue_type(str(ticket_text))
        facts: dict[str, Any] = {"issue_type": issue_type}
        if not (self.omit_customer_id_first and self._triage_calls == 1):
            facts["customer_id"] = customer_id
        for key, value in (
            ("customer_name", ticket.get("customer_name") or incoming_facts.get("customer_name")),
            ("plan", ticket.get("plan") or incoming_facts.get("plan")),
            ("ticket_text", ticket_text),
        ):
            if value is not None:
                facts[key] = value

        return _reply_action(
            {
                "type": "HANDOFF",
                "to_agent": "billing_agent",
                "goal": f"Apply the refund policy for {customer_id}",
                "summary": f"Verified {customer_id}; issue is {issue_type}.",
                "facts": facts,
            }
        )

    def _billing(self, messages: list[dict[str, Any]]) -> dict[str, Any]:
        incoming = _incoming_payload(messages)
        incoming_facts = incoming.get("facts") or {}
        customer_id = incoming_facts.get("customer_id")
        tool_result = _last_tool_result(messages)

        if tool_result is not None:
            result = tool_result.get("result") or {}
            if tool_result.get("tool") == "query_charges":
                return _tool_call_action("assess_refund", customer_id=customer_id)
            if tool_result.get("tool") == "assess_refund":
                return _reply_action(
                    {
                        "type": "HANDOFF",
                        "to_agent": "reply_agent",
                        "goal": "Draft the customer reply",
                        "summary": (
                            f"Decision: {result['decision']} (${result['refund_amount']:.2f})."
                        ),
                        "facts": {
                            "customer_id": result["customer_id"],
                            "decision": result["decision"],
                            "refund_amount": result["refund_amount"],
                            "duplicate_charge_ids": result["duplicate_charge_ids"],
                            "reason": result["reason"],
                        },
                    }
                )
            raise AssertionError(f"unexpected tool result: {tool_result!r}")

        if not customer_id:
            return _reply_action(
                {
                    "type": "NEED_INFO",
                    "to_agent": "triage_agent",
                    "missing_fields": ["customer_id"],
                    "query": "Which customer should I inspect? Please send the customer_id.",
                }
            )

        return _tool_call_action("query_charges", customer_id=customer_id)

    def _reply(self, messages: list[dict[str, Any]]) -> dict[str, Any]:
        incoming = _incoming_payload(messages)
        facts = incoming.get("facts") or {}
        decision = str(facts["decision"])
        amount = float(facts["refund_amount"])
        if decision == "refund":
            customer_message = (
                f"We found the duplicate charge and refunded ${amount:.2f} "
                "to your original payment method."
            )
        elif decision == "no_refund":
            customer_message = (
                "We reviewed your billing history and found no duplicate or "
                "eligible charge to refund."
            )
        else:
            customer_message = (
                "We escalated your case to a human billing specialist, who will follow up shortly."
            )
        return _reply_action(
            {
                "type": "DONE",
                "to_agent": "customer",
                "final_result": {
                    "decision": decision,
                    "refund_amount": amount,
                    "customer_message": customer_message,
                    "customer_id": facts["customer_id"],
                    "duplicate_charge_ids": facts.get("duplicate_charge_ids", []),
                    "reason": facts["reason"],
                },
                "summary": f"Resolved {facts['customer_id']}: {decision} (${amount:.2f}).",
                "status": "escalated_unresolved" if decision == "escalate" else "resolved",
            }
        )


# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def db() -> SQLiteMockDB:
    database = SQLiteMockDB()
    try:
        yield database
    finally:
        database.close()


def _ticket(task_id: str) -> SupportTicket:
    return next(ticket for ticket in load_support_tickets() if ticket.task_id == task_id)


def _chain(
    db: SQLiteMockDB,
    *,
    triage: ChainLLM | None = None,
    billing: ChainLLM | None = None,
    reply: ChainLLM | None = None,
) -> tuple[SupportChain, ChainLLM, ChainLLM, ChainLLM]:
    triage_llm = triage or ChainLLM("triage")
    billing_llm = billing or ChainLLM("billing")
    reply_llm = reply or ChainLLM("reply")
    return (
        SupportChain(triage_llm, billing_llm, reply_llm, db=db),
        triage_llm,
        billing_llm,
        reply_llm,
    )


# ---------------------------------------------------------------------------
# Normal happy path
# ---------------------------------------------------------------------------


async def test_normal_ticket_runs_three_handoffs_and_a_valid_done(
    db: SQLiteMockDB,
) -> None:
    ticket = _ticket("SUP-001")
    chain, triage_llm, _, _ = _chain(db)

    result = await chain.run_ticket(ticket)

    assert isinstance(result.done, DoneMessage)
    assert result.done.status is DoneStatus.RESOLVED
    assert result.done.to_agent == "customer"
    assert result.decision == "refund"
    assert result.refund_amount == pytest.approx(49.0)
    assert result.matches_ground_truth
    assert [handoff.from_agent for handoff in result.handoffs] == [
        "customer",
        "triage_agent",
        "billing_agent",
    ]
    assert len(result.agent_handoffs) == 2
    assert result.need_info == ()
    assert [handoff.to_agent for handoff in result.handoffs] == [
        "triage_agent",
        "billing_agent",
        "reply_agent",
    ]

    # Structured output is requested with the role schema embedded.
    assert triage_llm.calls[0]["schema"] is not None
    embedded = triage_llm.calls[0]["schema"]["oneOf"][1]["properties"]["reply"]
    assert embedded == triage_reply_schema(SUMMARY_MAX_CHARS)


async def test_billing_calls_the_tools_and_reads_the_results(
    db: SQLiteMockDB,
) -> None:
    ticket = _ticket("SUP-001")
    chain, _, billing_llm, _ = _chain(db)

    result = await chain.run_ticket(ticket)

    records = result.tool_calls["billing_agent"]
    assert [record.tool for record in records] == ["query_charges", "assess_refund"]
    assert all(record.ok for record in records)

    query_record, assess_record = records
    assert query_record.result["suspected_duplicate_charge_ids"] == ["CHG-1001", "CHG-1002"]
    assert assess_record.result["decision"] == "refund"
    assert assess_record.result["refund_amount"] == 49.0
    assert assess_record.result["reason"] == "duplicate_charge"

    # The query result is fed back into the next billing prompt.
    second_prompt = billing_llm.calls[1]["messages"]
    assert any(
        message["role"] == "user"
        and message["content"].startswith("Tool result:")
        and "suspected_duplicate_charge_ids" in message["content"]
        for message in second_prompt
    )

    # The policy result is fed back into the final billing prompt.
    third_prompt = billing_llm.calls[2]["messages"]
    assert any(
        message["role"] == "user"
        and message["content"].startswith("Tool result:")
        and "assess_refund" in message["content"]
        and "duplicate_charge" in message["content"]
        for message in third_prompt
    )


async def test_facts_accumulate_and_summaries_stay_tight(
    db: SQLiteMockDB,
) -> None:
    ticket = _ticket("SUP-001")
    chain, _, _, _ = _chain(db)

    result = await chain.run_ticket(ticket)

    billing_handoff = next(
        handoff for handoff in result.handoffs if handoff.from_agent == "billing_agent"
    )
    assert billing_handoff.facts == {
        "task_id": ticket.task_id,
        "ticket_text": ticket.input_text,
        "customer_id": "CUST-001",
        "customer_name": "Ava Thompson",
        "plan": "pro",
        "issue_type": "duplicate_charge",
        "decision": "refund",
        "refund_amount": 49.0,
        "duplicate_charge_ids": ["CHG-1001", "CHG-1002"],
        "reason": "duplicate_charge",
    }

    assert set(result.summary_lengths) == {
        "customer->triage_agent",
        "triage_agent->billing_agent",
        "billing_agent->reply_agent",
        "reply_agent->customer",
    }
    assert all(length <= SUMMARY_MAX_CHARS for length in result.summary_lengths.values())


async def test_long_ticket_text_is_truncated_in_the_initial_summary(
    db: SQLiteMockDB,
) -> None:
    ticket = SupportTicket(
        task_id="SUP-LONG",
        input_text="x" * 1000,
        customer_id="CUST-002",
        expected_decision="no_refund",
        expected_refund_amount=0.0,
    )
    chain, _, _, _ = _chain(db)

    result = await chain.run_ticket(ticket)

    initial = result.handoffs[0]
    assert len(initial.summary) == SUMMARY_MAX_CHARS
    assert initial.summary.endswith("…")
    assert result.matches_ground_truth


# ---------------------------------------------------------------------------
# Missing customer details
# ---------------------------------------------------------------------------


async def test_billing_asks_triage_for_a_missing_customer_id(
    db: SQLiteMockDB,
) -> None:
    ticket = _ticket("SUP-041")
    triage_llm = ChainLLM("triage", omit_customer_id_first=True)
    chain, _, _, _ = _chain(db, triage=triage_llm)

    result = await chain.run_ticket(ticket)

    assert len(result.need_info) == 1
    need_info = result.need_info[0]
    assert isinstance(need_info, NeedInfoMessage)
    assert need_info.from_agent == "billing_agent"
    assert need_info.to_agent == "triage_agent"
    assert need_info.missing_fields == ["customer_id"]
    assert len(triage_llm.calls) == 2

    triage_handoffs = [
        handoff for handoff in result.handoffs if handoff.from_agent == "triage_agent"
    ]
    assert len(triage_handoffs) == 2
    assert "customer_id" not in triage_handoffs[0].facts
    assert triage_handoffs[1].facts["customer_id"] == "CUST-001"
    assert result.matches_ground_truth


async def test_triage_can_ask_the_customer_for_missing_details(
    db: SQLiteMockDB,
) -> None:
    ticket = _ticket("SUP-046")
    triage_llm = ChainLLM("triage", ask_customer_first=True)
    chain, _, _, _ = _chain(db, triage=triage_llm)

    result = await chain.run_ticket(ticket)

    assert len(result.need_info) == 1
    need_info = result.need_info[0]
    assert need_info.from_agent == "triage_agent"
    assert need_info.to_agent == "customer"
    assert need_info.missing_fields == ["customer_id"]
    assert len(triage_llm.calls) == 2

    customer_handoffs = [handoff for handoff in result.handoffs if handoff.from_agent == "customer"]
    assert len(customer_handoffs) == 2
    assert customer_handoffs[1].facts["customer_id"] == "CUST-002"
    assert result.matches_ground_truth


# ---------------------------------------------------------------------------
# Escalation
# ---------------------------------------------------------------------------


async def test_failed_payment_escalates_with_a_valid_done(
    db: SQLiteMockDB,
) -> None:
    ticket = _ticket("SUP-034")
    chain, _, _, _ = _chain(db)

    result = await chain.run_ticket(ticket)

    assert result.done.status is DoneStatus.ESCALATED_UNRESOLVED
    assert result.decision == "escalate"
    assert result.refund_amount == 0.0
    assert result.matches_ground_truth

    assess_record = next(
        record for record in result.tool_calls["billing_agent"] if record.tool == "assess_refund"
    )
    assert assess_record.result["reason"] == "failed_payment"
    assert result.done.final_result["reason"] == "failed_payment"
    assert "escalated" in result.customer_message.lower()


async def test_unknown_customer_escalates(db: SQLiteMockDB) -> None:
    ticket = _ticket("SUP-047")
    chain, _, _, _ = _chain(db)

    result = await chain.run_ticket(ticket)

    assert result.done.status is DoneStatus.ESCALATED_UNRESOLVED
    assert result.decision == "escalate"
    assert result.refund_amount == 0.0
    assert result.matches_ground_truth
    assert result.done.final_result["reason"] == "customer_not_found"


# ---------------------------------------------------------------------------
# Whole dataset
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "ticket",
    load_support_tickets(),
    ids=lambda ticket: ticket.task_id,
)
async def test_every_dataset_ticket_matches_ground_truth(ticket: SupportTicket) -> None:
    db = SQLiteMockDB()
    try:
        chain, _, _, _ = _chain(db)
        result = await chain.run_ticket(ticket)

        assert result.matches_ground_truth, (
            ticket.task_id,
            result.decision,
            result.refund_amount,
            ticket.expected_decision,
            ticket.expected_refund_amount,
        )
        expected_status = (
            DoneStatus.ESCALATED_UNRESOLVED
            if ticket.expected_decision == "escalate"
            else DoneStatus.RESOLVED
        )
        assert result.done.status is expected_status
        assert len(result.handoffs) == 3
        assert len(result.agent_handoffs) == 2
        assert all(length <= SUMMARY_MAX_CHARS for length in result.summary_lengths.values())
        assert result.done.summary
    finally:
        db.close()


# ---------------------------------------------------------------------------
# Prompts, schemas and tools
# ---------------------------------------------------------------------------


def test_role_prompts_load() -> None:
    assert "TRIAGE AGENT" in TRIAGE_PROMPT
    assert "BILLING AGENT" in BILLING_PROMPT
    assert "REPLY AGENT" in REPLY_PROMPT
    assert "billing_agent" in TRIAGE_PROMPT
    assert "assess_refund" in BILLING_PROMPT
    assert "DONE" in REPLY_PROMPT


def test_role_schemas_cap_summaries() -> None:
    for factory in (triage_reply_schema, billing_reply_schema, reply_reply_schema):
        schema = factory(120)
        branches = schema.get("oneOf", [schema])
        for branch in branches:
            summary = branch.get("properties", {}).get("summary")
            if summary is not None:
                assert summary["maxLength"] == 120


def test_billing_tools_include_sqlite_lookups_and_the_policy(db: SQLiteMockDB) -> None:
    assert [tool.name for tool in billing_tools(db)] == [
        "query_customer",
        "query_charges",
        "query_inventory",
        "assess_refund",
    ]


def test_assess_refund_is_deterministic(db: SQLiteMockDB) -> None:
    assert assess_refund(db, "CUST-001").to_dict() == {
        "decision": "refund",
        "refund_amount": 49.0,
        "customer_id": "CUST-001",
        "duplicate_charge_ids": ["CHG-1001", "CHG-1002"],
        "reason": "duplicate_charge",
    }
    assert assess_refund(db, "CUST-002").decision == "no_refund"
    assert assess_refund(db, "CUST-006").decision == "escalate"
    assert assess_refund(db, "CUST-999").reason == "customer_not_found"


def test_chain_rejects_invalid_configuration() -> None:
    llm = ChainLLM("triage")
    with pytest.raises(ValueError):
        SupportChain(llm, llm, llm, summary_max_chars=0)
    with pytest.raises(ValueError):
        SupportChain(llm, llm, llm, timeout_s=0)
    with pytest.raises(ValueError):
        SupportChain(llm, llm, llm, max_hops=0)
    with pytest.raises(ValueError):
        SupportChain(llm, llm, llm, ack_timeout_s=0)
