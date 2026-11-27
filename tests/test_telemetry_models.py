"""Tests for the shared telemetry event models (Issue #17)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from telemetry.models import (
    HandoffEvent,
    LLMCallEvent,
    ProtocolEvent,
    SystemEvent,
    WorkflowEvent,
    load_events,
    parse_event,
)
from telemetry.run_context import collect_environment, default_run_id


def test_model_inherits_from_event() -> None:
    """Every model must inherit from the base Event."""
    from telemetry.models import Event
    for model in (ProtocolEvent, HandoffEvent, WorkflowEvent, LLMCallEvent, SystemEvent):
        assert issubclass(model, Event)


def test_protocol_event_with_run_id() -> None:
    """ProtocolEvent must require event_type and allow optional fields, with run_id."""
    event = ProtocolEvent(
        event_type="handoff",
        run_id="run-1",
        from_agent="a",
        to_agent="b",
        message_type="HANDOFF",
    )
    assert event.event_type == "handoff"
    assert event.from_agent == "a"
    assert event.to_agent == "b"
    assert event.message_type == "HANDOFF"
    assert event.hop_count is None
    assert event.attempt is None


def test_protocol_event_extra_forbidden() -> None:
    """ProtocolEvent must reject extra fields."""
    with pytest.raises(Exception):  # pydantic ValidationError
        ProtocolEvent(  # type: ignore[arg-expr]
            event_type="test",
            run_id="run-1",
            unexpected_field="should_fail",
        )


def test_llm_call_event_with_run_id() -> None:
    """LLMCallEvent must have correct default values, with run_id."""
    event = LLMCallEvent(event_type="llm_call", run_id="run-1")
    assert event.engine_name == "unknown"
    assert event.model_name == "unknown"
    assert event.ttft_ms == 0.0
    assert event.latency_ms == 0.0
    assert event.tokens_per_second == 0.0
    assert event.prompt_tokens == 0
    assert event.completion_tokens == 0
    assert event.total_tokens == 0


def test_llm_call_event_roundtrip() -> None:
    """LLMCallEvent must survive model_validate_json round-trip."""
    payload = {
        "event_type": "llm_call",
        "event_id": "test-1",
        "run_id": "run-1",
        "workflow_id": "wf-1",
        "engine_name": "mock",
        "model_name": "gpt-4",
        "ttft_ms": 100.5,
        "latency_ms": 200.0,
        "tokens_per_second": 50.0,
        "prompt_tokens": 10,
        "completion_tokens": 20,
        "total_tokens": 30,
        "finish_reason": "stop",
        "success": True,
        "error": None,
        "timestamp_ns": 123456789,
    }
    event = LLMCallEvent.model_validate(payload)
    data = event.model_dump()
    assert data["engine_name"] == "mock"
    assert data["model_name"] == "gpt-4"


def test_workflow_event_with_run_id() -> None:
    """WorkflowEvent event_type must be one of started/completed/failed, with run_id."""
    e = WorkflowEvent(event_type="started", run_id="run-1", workflow_id="wf-1")
    assert e.event_type == "started"
    with pytest.raises(Exception):  # ValidationError
        WorkflowEvent(event_type="unknown", run_id="run-1", workflow_id="wf-1")  # type: ignore[arg-expr]


def test_system_event_with_run_id() -> None:
    """SystemEvent severity defaults to 'error', with run_id."""
    e = SystemEvent(event_type="agent.error", run_id="run-1")
    assert e.severity == "error"


def test_parse_event_llm_call() -> None:
    """parse_event routes llm_call → LLMCallEvent."""
    data = {"event_type": "llm_call", "event_id": "x", "run_id": "r"}
    event = parse_event(data)
    assert isinstance(event, LLMCallEvent)


def test_parse_event_started_completed_failed() -> None:
    """parse_event routes started/completed/failed → WorkflowEvent."""
    for et in ("started", "completed", "failed"):
        data = {"event_type": et, "event_id": "x", "run_id": "r", "workflow_id": "wf"}
        event = parse_event(data)
        assert isinstance(event, WorkflowEvent)


def test_parse_event_handoff_prefix() -> None:
    """parse_event routes handoff.* → HandoffEvent."""
    for et in ("handoff.sent", "handoff.received", "handoff"):
        data = {"event_type": et, "event_id": "x", "run_id": "r", "from_agent": "a", "to_agent": "b", "message_type": "HANDOFF"}
        event = parse_event(data)
        assert isinstance(event, HandoffEvent)


def test_parse_event_system_error() -> None:
    """parse_event routes system.* → SystemEvent."""
    for et in ("system.error", "system.retry", "agent.error"):
        data = {"event_type": et, "event_id": "x", "run_id": "r", "severity": "warning"}
        event = parse_event(data)
        assert isinstance(event, SystemEvent)


def test_parse_event_fallback_protocol() -> None:
    """parse_event falls back to ProtocolEvent for unknown types."""
    data = {"event_type": "unknown.type", "event_id": "x", "run_id": "r", "detail": "something"}
    event = parse_event(data)
    assert isinstance(event, ProtocolEvent)


def test_parse_event_invalid_jsonl() -> None:
    """parse_event must raise on non-object payload."""
    try:
        parse_event("not json")
        assert False, "Should have raised"
    except (ValueError, TypeError):
        pass  # expected


def test_parse_event_bytes_decodes() -> None:
    """parse_event must decode bytes payload."""
    data = b'{"event_type": "llm_call", "event_id": "x", "run_id": "r"}'
    event = parse_event(data)
    assert isinstance(event, LLMCallEvent)


def test_load_events_revalidates_jsonl() -> None:
    """load_events must re-validate every row in a JSONL file."""
    line1 = '{"event_type": "llm_call", "event_id": "a", "run_id": "r1"}'
    line2 = '{"event_type": "started", "event_id": "b", "run_id": "r1", "workflow_id": "wf"}'
    path = Path("tests/fixtures/events_test.jsonl")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(line1 + "\n" + line2 + "\n", encoding="utf-8")
    events = load_events(path)
    assert len(events) == 2
    assert isinstance(events[0], LLMCallEvent)
    assert isinstance(events[1], WorkflowEvent)
    # Cleanup
    path.unlink(missing_ok=True)


def test_collect_environment_has_all_keys() -> None:
    """collect_environment must return all expected keys."""
    env = collect_environment()
    expected = {
        "created_at",
        "python_version",
        "python_implementation",
        "platform",
        "system",
        "release",
        "machine",
        "processor",
        "cpu_count",
        "hostname",
        "git_commit",
        "git_branch",
        "git_dirty",
        "gpu_model",
        "gpu_driver",
        "docker_image",
    }
    assert set(env.keys()) == expected


def test_default_run_id_is_non_empty() -> None:
    """default_run_id must produce a non-empty sortable string."""
    rid = default_run_id()
    assert isinstance(rid, str)
    assert len(rid) > 0
    assert "T" in rid  # YYYYMMDDTHHMMSS format


def test_model_dump_json_roundtrip() -> None:
    """model_dump_json must produce valid JSON for all models."""
    from telemetry.models import Event
    models = [
        ProtocolEvent(event_type="test", run_id="r", from_agent="a", to_agent="b", message_type="HANDOFF"),
        HandoffEvent(event_type="handoff", run_id="r", from_agent="a", to_agent="b", message_type="HANDOFF", hop_count=1),
        WorkflowEvent(event_type="started", run_id="r", workflow_id="wf"),
        LLMCallEvent(event_type="llm_call", run_id="r"),
        SystemEvent(event_type="agent.error", run_id="r"),
    ]
    for m in models:
        json_str = m.model_dump_json()
        parsed = json.loads(json_str)
        assert parsed["event_type"] == m.event_type