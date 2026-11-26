"""JSON Schemas for the research fork/join workflow.

The lead returns a two-scope plan, each topic researcher returns a grounded
handoff to the reviewer, and the reviewer returns a joined ``DONE`` message.
Every summary is capped so accumulated context stays tight.
"""

from __future__ import annotations

from typing import Any, Final

SUMMARY_MAX_CHARS: Final[int] = 600
SCOPE_ID_PATTERN: Final[str] = r"^[a-z0-9][a-z0-9_-]*$"


def _summary_schema(summary_max_chars: int) -> dict[str, Any]:
    return {"type": "string", "minLength": 1, "maxLength": summary_max_chars}


def _string_array(min_items: int = 0) -> dict[str, Any]:
    return {
        "type": "array",
        "items": {"type": "string", "minLength": 1},
        "minItems": min_items,
    }


def lead_plan_schema() -> dict[str, Any]:
    """Schema for the lead's two-scope research plan."""

    return {
        "type": "object",
        "properties": {
            "scopes": {
                "type": "array",
                "minItems": 2,
                "maxItems": 2,
                "items": {
                    "type": "object",
                    "properties": {
                        "scope_id": {"type": "string", "pattern": SCOPE_ID_PATTERN},
                        "title": {"type": "string", "minLength": 1},
                        "question": {"type": "string", "minLength": 1},
                        "focus": {"type": "string"},
                    },
                    "required": ["scope_id", "title", "question"],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["scopes"],
        "additionalProperties": False,
    }


def topic_reply_schema(summary_max_chars: int = SUMMARY_MAX_CHARS) -> dict[str, Any]:
    """Schema for one topic researcher's handoff to the reviewer."""

    return {
        "type": "object",
        "properties": {
            "type": {"const": "HANDOFF"},
            "to_agent": {"const": "reviewer_agent"},
            "goal": {"type": "string", "minLength": 1},
            "summary": _summary_schema(summary_max_chars),
            "facts": {
                "type": "object",
                "properties": {
                    "scope_id": {"type": "string", "minLength": 1},
                    "title": {"type": "string", "minLength": 1},
                    "findings": {"type": "string", "minLength": 1},
                    "key_facts": _string_array(1),
                    "sources": _string_array(0),
                    "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                },
                "required": [
                    "scope_id",
                    "title",
                    "findings",
                    "key_facts",
                    "sources",
                    "confidence",
                ],
            },
        },
        "required": ["type", "to_agent", "goal", "summary", "facts"],
    }


def reviewer_reply_schema(summary_max_chars: int = SUMMARY_MAX_CHARS) -> dict[str, Any]:
    """Schema for the reviewer's joined ``DONE`` message."""

    return {
        "type": "object",
        "properties": {
            "type": {"const": "DONE"},
            "to_agent": {"const": "user"},
            "final_result": {
                "type": "object",
                "properties": {
                    "brief": {"type": "string", "minLength": 1},
                    "key_facts": _string_array(1),
                    "discrepancies": _string_array(0),
                    "sources": _string_array(0),
                    "scope_summaries": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "scope_id": {"type": "string", "minLength": 1},
                                "title": {"type": "string", "minLength": 1},
                                "findings": {"type": "string", "minLength": 1},
                            },
                            "required": ["scope_id", "title", "findings"],
                        },
                    },
                },
                "required": [
                    "brief",
                    "key_facts",
                    "discrepancies",
                    "sources",
                    "scope_summaries",
                ],
            },
            "summary": _summary_schema(summary_max_chars),
            "status": {"enum": ["resolved", "escalated_unresolved"]},
        },
        "required": ["type", "to_agent", "final_result", "summary", "status"],
    }


__all__ = [
    "SCOPE_ID_PATTERN",
    "SUMMARY_MAX_CHARS",
    "lead_plan_schema",
    "reviewer_reply_schema",
    "topic_reply_schema",
]
