"""Tests for the AHP message models (Issue #3)."""

from __future__ import annotations

import json
from uuid import uuid4

import pytest
from pydantic import ValidationError

from ahp.models import (
    AckMessage,
    AckStatus,
    AHPErrorCode,
    AHPMessageAdapter,
    DoneMessage,
    DoneStatus,
    HandoffMessage,
    MessageType,
    NeedInfoMessage,
    OnFailurePolicy,
    dump_ahp_message,
    parse_ahp_message,
)


def handoff(**overrides: object) -> HandoffMessage:
    payload: dict[str, object] = {
        "conversation_id": "conv-1",
        "from_agent": "triage_agent",
        "to_agent": "billing_agent",
        "goal": "Decide whether to refund a duplicate charge",
        "summary": "Customer was charged twice for the Pro plan.",
    }
    payload.update(overrides)
    return HandoffMessage.model_validate(payload)


def ack(**overrides: object) -> AckMessage:
    payload: dict[str, object] = {
        "conversation_id": "conv-1",
        "from_agent": "billing_agent",
        "to_agent": "triage_agent",
        "acknowledged_message_id": uuid4(),
    }
    payload.update(overrides)
    return AckMessage.model_validate(payload)


def need_info(**overrides: object) -> NeedInfoMessage:
    payload: dict[str, object] = {
        "conversation_id": "conv-1",
        "from_agent": "billing_agent",
        "to_agent": "triage_agent",
        "missing_fields": ["customer_id"],
        "query": "Which customer should I look up?",
    }
    payload.update(overrides)
    return NeedInfoMessage.model_validate(payload)


def done(**overrides: object) -> DoneMessage:
    payload: dict[str, object] = {
        "conversation_id": "conv-1",
        "from_agent": "reply_agent",
        "to_agent": "orchestrator",
        "final_result": {"decision": "refund", "amount": 20},
        "summary": "Refunded the duplicate charge.",
    }
    payload.update(overrides)
    return DoneMessage.model_validate(payload)


def test_handoff_defaults() -> None:
    message = handoff()

    assert message.type is MessageType.HANDOFF
    assert message.conversation_id == "conv-1"
    assert message.hop_count == 0
    assert message.facts == {}
    assert message.done_so_far == []
    assert message.expected_output == {}
    assert message.constraints == {}
    assert message.on_failure is OnFailurePolicy.ESCALATE_TO_HUMAN
    assert message.timestamp_ns > 0


def test_ack_defaults_and_status() -> None:
    message = ack()
    assert message.type is MessageType.ACK
    assert message.status is AckStatus.ACCEPTED

    busy = ack(status="busy")
    assert busy.status is AckStatus.BUSY


def test_need_info_and_done() -> None:
    info = need_info()
    assert info.type is MessageType.NEED_INFO
    assert info.missing_fields == ["customer_id"]

    result = done()
    assert result.type is MessageType.DONE
    assert result.status is DoneStatus.RESOLVED
    assert result.execution_metrics == {}

    escalated = done(status="escalated_unresolved")
    assert escalated.status is DoneStatus.ESCALATED_UNRESOLVED


def test_messages_are_frozen() -> None:
    message = handoff()

    with pytest.raises(ValidationError):
        message.goal = "changed"  # type: ignore[misc]


@pytest.mark.parametrize(
    ("payload", "expected_code"),
    [
        ({}, "union_tag_not_found"),
        ({"type": "BOGUS"}, "union_tag_invalid"),
        (
            {"type": "HANDOFF", "conversation_id": "c", "from_agent": "a", "to_agent": "b"},
            "missing",
        ),
        (
            {
                "type": "HANDOFF",
                "conversation_id": "c",
                "from_agent": "a",
                "to_agent": "b",
                "goal": "g",
                "summary": "s",
                "hop_count": -1,
            },
            "greater_than_equal",
        ),
        (
            {
                "type": "HANDOFF",
                "conversation_id": "c",
                "from_agent": "a",
                "to_agent": "b",
                "goal": "g",
                "summary": "s",
                "surprise": True,
            },
            "extra_forbidden",
        ),
        ({"type": "ACK", "conversation_id": "c", "from_agent": "a", "to_agent": "b"}, "missing"),
        (
            {
                "type": "ACK",
                "conversation_id": "c",
                "from_agent": "a",
                "to_agent": "b",
                "acknowledged_message_id": str(uuid4()),
                "status": "maybe",
            },
            "enum",
        ),
        (
            {
                "type": "NEED_INFO",
                "conversation_id": "c",
                "from_agent": "a",
                "to_agent": "b",
                "missing_fields": [],
                "query": "what?",
            },
            "too_short",
        ),
        ({"type": "DONE", "conversation_id": "c", "from_agent": "a", "to_agent": "b"}, "missing"),
        (
            {
                "type": "HANDOFF",
                "conversation_id": "c",
                "from_agent": "a",
                "to_agent": "b",
                "goal": "g",
                "summary": "s",
                "message_id": "not-a-uuid",
            },
            "uuid_parsing",
        ),
    ],
)
def test_malformed_messages_are_rejected(payload: dict[str, object], expected_code: str) -> None:
    with pytest.raises(ValidationError) as excinfo:
        AHPMessageAdapter.validate_python(payload)

    codes = {error["type"] for error in excinfo.value.errors()}
    assert expected_code in codes


@pytest.mark.parametrize(
    ("document", "message_type"),
    [
        (handoff(), HandoffMessage),
        (ack(), AckMessage),
        (need_info(), NeedInfoMessage),
        (done(), DoneMessage),
    ],
)
def test_discriminated_union_parses_json(document: object, message_type: type) -> None:
    raw = document.model_dump_json()  # type: ignore[union-attr]

    parsed = parse_ahp_message(raw)

    assert isinstance(parsed, message_type)
    assert parsed == document


def test_union_accepts_plain_mapping() -> None:
    payload = json.loads(handoff().model_dump_json())

    parsed = parse_ahp_message(payload)

    assert isinstance(parsed, HandoffMessage)


def test_round_trip_preserves_identity_and_hop_count() -> None:
    message = handoff(hop_count=3, facts={"customer_id": "C-1", "amount": 20})

    restored = parse_ahp_message(dump_ahp_message(message))

    assert restored == message
    assert restored.message_id == message.message_id
    assert restored.hop_count == 3


def test_hop_increment_with_model_copy() -> None:
    message = handoff()

    forwarded = message.model_copy(update={"hop_count": message.hop_count + 1})

    assert forwarded.hop_count == 1
    assert message.hop_count == 0
    assert forwarded.message_id == message.message_id


def test_json_schema_uses_type_discriminator() -> None:
    schema = AHPMessageAdapter.json_schema()

    assert "oneOf" in schema or "anyOf" in schema
    assert schema["discriminator"]["propertyName"] == "type"


def test_error_codes_are_stable_strings() -> None:
    assert AHPErrorCode.ACK_TIMEOUT == "ACK_TIMEOUT"
    assert AHPErrorCode.HOP_LIMIT_EXCEEDED == "HOP_LIMIT_EXCEEDED"
    assert AHPErrorCode.SCHEMA_VIOLATION.value == "SCHEMA_VIOLATION"
