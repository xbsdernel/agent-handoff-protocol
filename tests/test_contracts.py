"""Day-1 contract smoke tests.

These tests guard the interfaces both team members code against before the
concrete implementations land. They must stay runnable without a GPU or any
network access.
"""

from __future__ import annotations

import inspect
from typing import Any

import pytest
from pydantic import ValidationError

import ahp
import bench
import engine
import harness
import telemetry
from ahp.interfaces import Agent
from engine.interfaces import LLMClient, LLMResult
from telemetry.interfaces import Event, EventSink


class FakeAgent:
    """Minimal object satisfying the Agent protocol."""

    name = "fake_agent"

    async def handle_handoff(self, message: Any) -> Any:
        return message


class InMemorySink(EventSink):
    """Minimal sink used to prove the EventSink contract is usable."""

    def __init__(self) -> None:
        self.events: list[Event] = []
        self.flushes = 0

    async def emit(self, event: Event) -> None:
        self.events.append(event)

    async def flush(self) -> None:
        self.flushes += 1


class FakeLLMClient:
    """Minimal object satisfying the LLMClient protocol."""

    async def chat(
        self,
        messages: list[dict[str, Any]],
        schema: dict[str, Any] | None = None,
        temperature: float = 0.0,
    ) -> LLMResult:
        return LLMResult(text="ok")

    async def aclose(self) -> None:
        return None


def test_top_level_packages_import() -> None:
    for module in (ahp, bench, engine, harness, telemetry):
        assert module.__doc__


def test_agent_protocol_is_runtime_checkable() -> None:
    assert isinstance(FakeAgent(), Agent)
    assert inspect.iscoroutinefunction(FakeAgent.handle_handoff)


def test_llm_result_defaults() -> None:
    result = LLMResult(text="hello")

    assert result.text == "hello"
    assert result.structured_data is None
    assert result.ttft_ms == 0.0
    assert result.total_latency_ms == 0.0
    assert result.tokens_per_second == 0.0
    assert result.prompt_tokens == 0
    assert result.completion_tokens == 0
    assert result.total_tokens == 0
    assert result.engine_name == "unknown"
    assert result.model_name == "unknown"


def test_llm_result_is_immutable() -> None:
    result = LLMResult(text="hello", total_tokens=3)

    with pytest.raises(AttributeError):
        result.text = "changed"  # type: ignore[misc]


def test_llm_client_protocol_is_runtime_checkable() -> None:
    assert isinstance(FakeLLMClient(), LLMClient)


def test_event_round_trips_through_json() -> None:
    event = Event(run_id="run-1", workflow_id="wf-1", event_type="workflow.started")

    restored = Event.model_validate_json(event.model_dump_json())

    assert restored == event
    assert restored.schema_version == "1.0"
    assert restored.timestamp_ns > 0


def test_event_rejects_unknown_fields() -> None:
    with pytest.raises(ValidationError):
        Event(run_id="run-1", event_type="workflow.started", surprise="nope")


def test_event_sink_cannot_be_instantiated_directly() -> None:
    with pytest.raises(TypeError):
        EventSink()  # type: ignore[abstract]


async def test_event_sink_contract() -> None:
    sink = InMemorySink()
    event = Event(run_id="run-1", event_type="llm_call.finished")

    await sink.emit(event)
    await sink.aclose()

    assert sink.events == [event]
    assert sink.flushes == 1
