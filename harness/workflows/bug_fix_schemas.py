"""JSON Schemas for the bug-triage/fix/test loop.

Each schema is the ``reply`` payload of a :class:`harness.agent.BaseAgent`
action. Triage hands a reproduction specification to the developer, the
developer hands a patch to the tester, and the tester either bounces the patch
back with failure logs or closes the workflow with ``DONE``.
"""

from __future__ import annotations

from typing import Any, Final

SUMMARY_MAX_CHARS: Final[int] = 600
MAX_REPAIR_CYCLES: Final[int] = 3


def _summary_schema(summary_max_chars: int) -> dict[str, Any]:
    return {"type": "string", "minLength": 1, "maxLength": summary_max_chars}


def _string_array(min_items: int = 0) -> dict[str, Any]:
    return {
        "type": "array",
        "items": {"type": "string", "minLength": 1},
        "minItems": min_items,
    }


def _patch_schema() -> dict[str, Any]:
    return {
        "type": "array",
        "minItems": 1,
        "items": {
            "type": "object",
            "properties": {
                "find": {"type": "string", "minLength": 1},
                "replace": {"type": "string", "minLength": 1},
            },
            "required": ["find", "replace"],
            "additionalProperties": False,
        },
    }


def triage_reply_schema(summary_max_chars: int = SUMMARY_MAX_CHARS) -> dict[str, Any]:
    """Triage always hands a reproduction specification to the developer."""

    return {
        "type": "object",
        "properties": {
            "type": {"const": "HANDOFF"},
            "to_agent": {"const": "developer_agent"},
            "goal": {"type": "string", "minLength": 1},
            "summary": _summary_schema(summary_max_chars),
            "facts": {
                "type": "object",
                "properties": {
                    "bug_id": {"type": "string", "minLength": 1},
                    "file_path": {"type": "string", "minLength": 1},
                    "reproduction_steps": _string_array(1),
                    "suspected_cause": {"type": "string", "minLength": 1},
                    "failing_tests": _string_array(1),
                },
                "required": [
                    "bug_id",
                    "file_path",
                    "reproduction_steps",
                    "suspected_cause",
                    "failing_tests",
                ],
            },
        },
        "required": ["type", "to_agent", "goal", "summary", "facts"],
    }


def developer_reply_schema(summary_max_chars: int = SUMMARY_MAX_CHARS) -> dict[str, Any]:
    """The developer hands a concrete patch to the tester."""

    return {
        "type": "object",
        "properties": {
            "type": {"const": "HANDOFF"},
            "to_agent": {"const": "tester_agent"},
            "goal": {"type": "string", "minLength": 1},
            "summary": _summary_schema(summary_max_chars),
            "facts": {
                "type": "object",
                "properties": {
                    "bug_id": {"type": "string", "minLength": 1},
                    "file_path": {"type": "string", "minLength": 1},
                    "attempt": {"type": "integer", "minimum": 1},
                    "rationale": {"type": "string", "minLength": 1},
                    "patch": _patch_schema(),
                },
                "required": [
                    "bug_id",
                    "file_path",
                    "attempt",
                    "rationale",
                    "patch",
                ],
            },
        },
        "required": ["type", "to_agent", "goal", "summary", "facts"],
    }


def tester_reply_schema(summary_max_chars: int = SUMMARY_MAX_CHARS) -> dict[str, Any]:
    """The tester either bounces the patch back or closes the workflow."""

    return {
        "oneOf": [
            {
                "type": "object",
                "properties": {
                    "type": {"const": "HANDOFF"},
                    "to_agent": {"const": "developer_agent"},
                    "goal": {"type": "string", "minLength": 1},
                    "summary": _summary_schema(summary_max_chars),
                    "facts": {
                        "type": "object",
                        "properties": {
                            "bug_id": {"type": "string", "minLength": 1},
                            "attempt": {"type": "integer", "minimum": 1},
                            "failure_log": {"type": "string", "minLength": 1},
                            "failing_tests": _string_array(1),
                            "suggested_next_step": {
                                "type": "string",
                                "minLength": 1,
                            },
                        },
                        "required": [
                            "bug_id",
                            "attempt",
                            "failure_log",
                            "failing_tests",
                            "suggested_next_step",
                        ],
                    },
                },
                "required": ["type", "to_agent", "goal", "summary", "facts"],
            },
            {
                "type": "object",
                "properties": {
                    "type": {"const": "DONE"},
                    "to_agent": {"const": "user"},
                    "final_result": {
                        "type": "object",
                        "properties": {
                            "bug_id": {"type": "string", "minLength": 1},
                            "attempts": {"type": "integer", "minimum": 1},
                            "patch": _patch_schema(),
                            "tests_passed": {"type": "integer", "minimum": 0},
                            "total_tests": {"type": "integer", "minimum": 1},
                            "failure_log": {"type": "string"},
                        },
                        "required": [
                            "bug_id",
                            "attempts",
                            "patch",
                            "tests_passed",
                            "total_tests",
                        ],
                    },
                    "summary": _summary_schema(summary_max_chars),
                    "status": {"enum": ["resolved", "escalated_unresolved"]},
                    "execution_metrics": {"type": "object"},
                },
                "required": ["type", "to_agent", "final_result", "summary", "status"],
            },
        ]
    }


__all__ = [
    "MAX_REPAIR_CYCLES",
    "SUMMARY_MAX_CHARS",
    "developer_reply_schema",
    "tester_reply_schema",
    "triage_reply_schema",
]
