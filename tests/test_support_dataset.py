"""Tests for the frozen support-ticket dataset (Issue #14).

The dataset is ground truth for the triage chain, so these tests pin its size,
class balance, policy labels and the deliberate missing-customer-ID cases.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pytest
from pydantic import ValidationError

from harness.tools import SQLiteMockDB
from harness.workflows.support_chain import (
    SupportTicket,
    assess_refund,
    load_support_tickets,
)

REQUIRED_FACT_KEYS = {
    "customer_id",
    "customer_name",
    "email",
    "plan",
    "issue_type",
    "expected_charge_ids",
    "policy_reason",
}


@pytest.fixture(scope="module")
def tickets() -> list[SupportTicket]:
    return load_support_tickets()


@pytest.fixture(scope="module")
def db() -> SQLiteMockDB:
    database = SQLiteMockDB()
    try:
        yield database
    finally:
        database.close()


# ---------------------------------------------------------------------------
# Shape and class balance
# ---------------------------------------------------------------------------


def test_dataset_has_50_unique_ordered_tickets(tickets: list[SupportTicket]) -> None:
    assert len(tickets) == 50
    assert len({ticket.task_id for ticket in tickets}) == 50
    assert [ticket.task_id for ticket in tickets] == [f"SUP-{index:03d}" for index in range(1, 51)]


def test_dataset_covers_every_decision(tickets: list[SupportTicket]) -> None:
    assert Counter(ticket.expected_decision for ticket in tickets) == {
        "refund": 12,
        "no_refund": 27,
        "escalate": 11,
    }


def test_refund_cases_match_the_seeded_duplicate_charges(tickets: list[SupportTicket]) -> None:
    refunds = Counter(
        (ticket.customer_id, ticket.expected_refund_amount)
        for ticket in tickets
        if ticket.expected_decision == "refund"
    )
    assert refunds == {("CUST-001", 49.0): 7, ("CUST-005", 19.0): 5}


def test_escalation_cases_cover_failed_and_unknown_customers(
    tickets: list[SupportTicket],
) -> None:
    escalations = Counter(
        ticket.customer_id for ticket in tickets if ticket.expected_decision == "escalate"
    )
    assert escalations == {"CUST-006": 6, "CUST-999": 5}


def test_missing_customer_id_tickets_are_deliberate(tickets: list[SupportTicket]) -> None:
    missing = [ticket for ticket in tickets if ticket.requires_customer_lookup]
    assert [ticket.task_id for ticket in missing] == [
        "SUP-041",
        "SUP-042",
        "SUP-043",
        "SUP-044",
        "SUP-045",
        "SUP-046",
    ]
    for ticket in missing:
        assert ticket.customer_id is not None
        assert "CUST-" not in ticket.input_text
        assert ticket.ground_truth_facts["customer_id"] == ticket.customer_id


def test_normal_tickets_name_the_customer_in_the_input_text(
    tickets: list[SupportTicket],
) -> None:
    for ticket in tickets:
        if ticket.requires_customer_lookup:
            continue
        assert ticket.customer_id is not None
        assert ticket.customer_id in ticket.input_text


# ---------------------------------------------------------------------------
# Ground truth consistency
# ---------------------------------------------------------------------------


def test_ground_truth_matches_the_refund_policy(
    tickets: list[SupportTicket], db: SQLiteMockDB
) -> None:
    for ticket in tickets:
        assert ticket.customer_id is not None
        decision = assess_refund(db, ticket.customer_id)
        assert decision.decision == ticket.expected_decision, ticket.task_id
        assert decision.refund_amount == pytest.approx(ticket.expected_refund_amount), (
            ticket.task_id
        )
        assert ticket.ground_truth_facts["policy_reason"] == decision.reason, ticket.task_id


def test_expected_charge_ids_match_the_seeded_history(
    tickets: list[SupportTicket], db: SQLiteMockDB
) -> None:
    for ticket in tickets:
        assert ticket.customer_id is not None
        expected = ticket.ground_truth_facts["expected_charge_ids"]
        assert isinstance(expected, list)
        if ticket.customer_id == "CUST-999":
            assert expected == []
            continue

        charges = db.query_charges(ticket.customer_id)
        assert charges["found"] is True
        if ticket.expected_decision == "refund":
            assert expected == charges["suspected_duplicate_charge_ids"]
        else:
            assert set(expected) == {charge["charge_id"] for charge in charges["charges"]}


def test_ground_truth_facts_have_every_required_key(tickets: list[SupportTicket]) -> None:
    for ticket in tickets:
        assert ticket.ground_truth_facts.keys() >= REQUIRED_FACT_KEYS, ticket.task_id
        assert ticket.ground_truth_facts["issue_type"] in {
            "duplicate_charge",
            "billing_question",
            "failed_payment",
            "refund_request",
            "other",
        }


def test_prompt_record_excludes_the_answer(tickets: list[SupportTicket]) -> None:
    ticket = tickets[0]
    record = ticket.prompt_record()
    assert record["task_id"] == ticket.task_id
    assert record["input_text"] == ticket.input_text
    assert record["customer_id"] == ticket.customer_id
    assert "expected_decision" not in record
    assert "expected_refund_amount" not in record
    assert "expected_charge_ids" not in record
    assert "policy_reason" not in record


# ---------------------------------------------------------------------------
# Model and loader validation
# ---------------------------------------------------------------------------


def test_refund_tickets_require_a_positive_amount() -> None:
    with pytest.raises(ValidationError):
        SupportTicket(
            task_id="SUP-X",
            input_text="refund please",
            customer_id="CUST-001",
            expected_decision="refund",
            expected_refund_amount=0.0,
        )


def test_non_refund_tickets_must_not_promise_a_refund() -> None:
    with pytest.raises(ValidationError):
        SupportTicket(
            task_id="SUP-X",
            input_text="question",
            customer_id="CUST-002",
            expected_decision="no_refund",
            expected_refund_amount=19.0,
        )


def test_ground_truth_customer_id_must_match_the_ticket() -> None:
    with pytest.raises(ValidationError):
        SupportTicket(
            task_id="SUP-X",
            input_text="question",
            customer_id="CUST-002",
            expected_decision="no_refund",
            expected_refund_amount=0.0,
            ground_truth_facts={"customer_id": "CUST-003"},
        )


def test_loader_rejects_duplicate_task_ids(tmp_path: Path, tickets: list[SupportTicket]) -> None:
    path = tmp_path / "tickets.jsonl"
    row = tickets[0].model_dump(mode="json")
    path.write_text(
        json.dumps(row) + "\n" + json.dumps(row) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="duplicate task_id"):
        load_support_tickets(path)


def test_loader_reports_the_bad_line(tmp_path: Path) -> None:
    path = tmp_path / "tickets.jsonl"
    path.write_text(
        json.dumps(
            {
                "task_id": "SUP-X",
                "input_text": "hello",
                "expected_decision": "maybe",
                "expected_refund_amount": 0,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="line 1"):
        load_support_tickets(path)


def test_loader_skips_blank_lines(tmp_path: Path, tickets: list[SupportTicket]) -> None:
    path = tmp_path / "tickets.jsonl"
    path.write_text(
        "\n" + json.dumps(tickets[0].model_dump(mode="json")) + "\n\n",
        encoding="utf-8",
    )

    loaded = load_support_tickets(path)
    assert [ticket.task_id for ticket in loaded] == [tickets[0].task_id]
