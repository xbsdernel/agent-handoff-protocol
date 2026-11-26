"""JSON Schemas for the three support-chain agent replies.

Each schema is the ``reply`` payload of a :class:`harness.agent.BaseAgent`
action. ``BaseAgent`` embeds it in the structured-output action schema, shows it
in the system prompt and validates it again before building an AHP message.

Summaries are capped by ``summary_max_chars`` so context stays tight as facts
accumulate across the chain.
"""

from __future__ import annotations

from typing import Any, Final

SUMMARY_MAX_CHARS: Final[int] = 600

DECISIONS: Final[tuple[str, ...]] = ("refund", "no_refund", "escalate")
ISSUE_TYPES: Final[tuple[str, ...]] = (
    "duplicate_charge",
    "billing_question",
    "failed_payment",
    "refund_request",
    "other",
)


def _summary_schema(summary_max_chars: int) -> dict[str, Any]:
    return {"type": "string", "minLength": 1, "maxLength": summary_max_chars}


def _need_info_branch(to_agent: str) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "type": {"const": "NEED_INFO"},
            "to_agent": {"const": to_agent},
            "missing_fields": {
                "type": "array",
                "items": {"type": "string", "minLength": 1},
                "minItems": 1,
            },
            "query": {"type": "string", "minLength": 1},
        },
        "required": ["type", "to_agent", "missing_fields", "query"],
    }


def triage_reply_schema(summary_max_chars: int = SUMMARY_MAX_CHARS) -> dict[str, Any]:
    """Triage either asks the customer for details or hands off to billing."""

    return {
        "oneOf": [
            {
                "type": "object",
                "properties": {
                    "type": {"const": "HANDOFF"},
                    "to_agent": {"const": "billing_agent"},
                    "goal": {"type": "string", "minLength": 1},
                    "summary": _summary_schema(summary_max_chars),
                    "facts": {
                        "type": "object",
                        "properties": {
                            "customer_id": {"type": "string", "minLength": 1},
                            "customer_name": {"type": "string"},
                            "plan": {"type": "string"},
                            "issue_type": {"enum": list(ISSUE_TYPES)},
                            "ticket_text": {"type": "string"},
                        },
                        # customer_id is deliberately optional: billing is the
                        # agent that asks triage to complete missing details.
                        "required": ["issue_type"],
                    },
                },
                "required": ["type", "to_agent", "goal", "summary", "facts"],
            },
            _need_info_branch("customer"),
        ]
    }


def billing_reply_schema(summary_max_chars: int = SUMMARY_MAX_CHARS) -> dict[str, Any]:
    """Billing either asks triage for details or hands a decision to reply."""

    return {
        "oneOf": [
            {
                "type": "object",
                "properties": {
                    "type": {"const": "HANDOFF"},
                    "to_agent": {"const": "reply_agent"},
                    "goal": {"type": "string", "minLength": 1},
                    "summary": _summary_schema(summary_max_chars),
                    "facts": {
                        "type": "object",
                        "properties": {
                            "customer_id": {"type": "string", "minLength": 1},
                            "decision": {"enum": list(DECISIONS)},
                            "refund_amount": {"type": "number", "minimum": 0},
                            "duplicate_charge_ids": {
                                "type": "array",
                                "items": {"type": "string", "minLength": 1},
                            },
                            "reason": {"type": "string", "minLength": 1},
                        },
                        "required": [
                            "customer_id",
                            "decision",
                            "refund_amount",
                            "duplicate_charge_ids",
                            "reason",
                        ],
                    },
                },
                "required": ["type", "to_agent", "goal", "summary", "facts"],
            },
            _need_info_branch("triage_agent"),
        ]
    }


def reply_reply_schema(summary_max_chars: int = SUMMARY_MAX_CHARS) -> dict[str, Any]:
    """Reply turns the accumulated facts into a customer-facing DONE message."""

    return {
        "type": "object",
        "properties": {
            "type": {"const": "DONE"},
            "to_agent": {"const": "customer"},
            "final_result": {
                "type": "object",
                "properties": {
                    "decision": {"enum": list(DECISIONS)},
                    "refund_amount": {"type": "number", "minimum": 0},
                    "customer_message": {"type": "string", "minLength": 1},
                    "customer_id": {"type": "string", "minLength": 1},
                    "duplicate_charge_ids": {
                        "type": "array",
                        "items": {"type": "string", "minLength": 1},
                    },
                    "reason": {"type": "string", "minLength": 1},
                },
                "required": [
                    "decision",
                    "refund_amount",
                    "customer_message",
                    "customer_id",
                    "duplicate_charge_ids",
                    "reason",
                ],
            },
            "summary": _summary_schema(summary_max_chars),
            "status": {"enum": ["resolved", "escalated_unresolved"]},
            "execution_metrics": {"type": "object"},
        },
        "required": ["type", "to_agent", "final_result", "summary", "status"],
    }


__all__ = [
    "DECISIONS",
    "ISSUE_TYPES",
    "SUMMARY_MAX_CHARS",
    "billing_reply_schema",
    "reply_reply_schema",
    "triage_reply_schema",
]
