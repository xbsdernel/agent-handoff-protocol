"""Strongly-typed telemetry events shared by every harness layer.

The base :class:`~telemetry.interfaces.Event` envelope carries the identity and
ordering fields (``run_id``, ``event_id``, ``timestamp_ns``). This module adds
the concrete event families used by the router, the engine client, workflow
orchestrators and the run-directory writer.

Events are deliberately strict (``extra="forbid"``): a producer that adds an
unexpected field fails at construction time instead of silently writing an
unqueryable row. :func:`parse_event` re-validates a persisted JSON object back
into the narrowest matching model, which is how the test suite and downstream
scorers guarantee that ``events.jsonl`` stays schema-clean.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal

from pydantic import Field

from telemetry.interfaces import Event

#: Event types accepted by :class:`WorkflowEvent`.
WORKFLOW_EVENT_TYPES: tuple[str, ...] = ("started", "completed", "failed")

#: Event types accepted by :class:`SystemEvent`.
SYSTEM_EVENT_TYPES: tuple[str, ...] = (
    "system.error",
    "system.retry",
    "system.escalation",
    "agent.error",
)

Severity = Literal["info", "warning", "error", "critical"]


class ProtocolEvent(Event):
    """Router and fault-injection lifecycle event.

    This is the shared replacement for the router-local ``RouterEvent``. Every
    field is optional because the same envelope is used for transport events,
    state-machine transitions and fault decisions.
    """

    event_type: str
    message_id: str | None = None
    acknowledged_message_id: str | None = None
    message_type: str | None = None
    from_agent: str | None = None
    to_agent: str | None = None
    hop_count: int | None = Field(default=None, ge=0)
    attempt: int | None = Field(default=None, ge=1)
    error_code: str | None = None
    state: str | None = None
    detail: str | None = None
    valid_schema: bool = True


class HandoffEvent(ProtocolEvent):
    """A protocol handoff was sent, received, retried or rejected."""

    event_type: str = "handoff"
    from_agent: str
    to_agent: str
    message_type: str
    hop_count: int = Field(default=0, ge=0)


class WorkflowEvent(Event):
    """High-level workflow lifecycle event."""

    event_type: Literal["started", "completed", "failed"]
    workflow_id: str
    detail: str = ""


class LLMCallEvent(Event):
    """One completion request, with the metrics needed for latency analysis."""

    event_type: Literal["llm_call"] = "llm_call"
    engine_name: str = "unknown"
    model_name: str = "unknown"
    ttft_ms: float = Field(default=0.0, ge=0)
    latency_ms: float = Field(default=0.0, ge=0)
    tokens_per_second: float = Field(default=0.0, ge=0)
    prompt_tokens: int = Field(default=0, ge=0)
    completion_tokens: int = Field(default=0, ge=0)
    total_tokens: int = Field(default=0, ge=0)
    finish_reason: str | None = None
    success: bool = True
    error: str | None = None


class SystemEvent(ProtocolEvent):
    """Errors, retries and escalations raised outside a normal handoff."""

    event_type: str = "system.error"
    severity: Severity = "error"


def parse_event(data: Event | str | bytes | Mapping[str, Any]) -> Event:
    """Validate one persisted event into its narrowest matching model.

    ``event_type`` selects the model. Unknown families fall back to the
    permissive :class:`ProtocolEvent`, which still rejects unknown fields.
    """

    if isinstance(data, Event):
        return data
    if isinstance(data, bytes):
        data = data.decode("utf-8")
    if isinstance(data, str):
        payload = json.loads(data)
    elif isinstance(data, Mapping):
        payload = dict(data)
    else:
        raise TypeError(f"event must be JSON, a mapping or Event, got {type(data).__name__}")
    if not isinstance(payload, Mapping):
        raise ValueError(f"event must be a JSON object, got {type(payload).__name__}")

    event_type = str(payload.get("event_type", ""))
    if event_type == "llm_call":
        model: type[Event] = LLMCallEvent
    elif event_type in WORKFLOW_EVENT_TYPES:
        model = WorkflowEvent
    elif event_type.startswith("handoff.") or event_type == "handoff":
        model = HandoffEvent
    elif event_type in SYSTEM_EVENT_TYPES or event_type.startswith("system."):
        model = SystemEvent
    else:
        model = ProtocolEvent
    return model.model_validate(payload)


def load_events(path: str | Path) -> list[Event]:
    """Read a JSONL event file and re-validate every row."""

    events: list[Event] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                events.append(parse_event(stripped))
            except ValueError as exc:
                raise ValueError(f"invalid telemetry event on line {line_number}: {exc}") from exc
    return events


__all__ = [
    "SYSTEM_EVENT_TYPES",
    "WORKFLOW_EVENT_TYPES",
    "HandoffEvent",
    "LLMCallEvent",
    "ProtocolEvent",
    "Severity",
    "SystemEvent",
    "WorkflowEvent",
    "load_events",
    "parse_event",
]
