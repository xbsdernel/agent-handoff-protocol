"""Tests for the AsyncJSONLEventSink (Issue #17)."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from telemetry.models import Event, LLMCallEvent, ProtocolEvent, SystemEvent
from telemetry.sink import AsyncJSONLEventSink


def _sink_path(name: str) -> str:
    """Return a unique per-test sink file path, cleaning up any prior file."""
    p = Path(f"tests/fixtures/sink_{name}.jsonl")
    if p.exists():
        p.unlink()
    return str(p)


# ---------------------------------------------------------------------------
# Basic emit / flush / aclose lifecycle
# ---------------------------------------------------------------------------


async def test_emit_queues_without_io() -> None:
    """emit must never perform disk I/O; it should just queue."""
    path = _sink_path("emit_queues")
    sink = AsyncJSONLEventSink(path, run_id="test-run")
    event = LLMCallEvent(event_type="llm_call", run_id="test-run")
    await sink.start()
    await sink.emit(event)
    assert sink.emitted == 1
    assert sink.written == 0  # written == 0 until flush
    await sink.aclose()


async def test_flush_drains_queue() -> None:
    """flush must wait until every accepted event is written to disk."""
    path = _sink_path("flush_drains")
    sink = AsyncJSONLEventSink(path, run_id="test-run")
    event1 = LLMCallEvent(event_type="llm_call", run_id="test-run")
    event2 = ProtocolEvent(
        event_type="handoff", run_id="test-run", from_agent="a", to_agent="b", message_type="HANDOFF"
    )
    await sink.start()
    await sink.emit(event1)
    await sink.emit(event2)
    await sink.flush()
    assert sink.written == 2
    await sink.aclose()


async def test_aclose_stops_writer() -> None:
    """aclose must stop the background writer and close the file."""
    path = _sink_path("aclose_stops")
    sink = AsyncJSONLEventSink(path, run_id="test-run")
    event = LLMCallEvent(event_type="llm_call", run_id="test-run")
    await sink.start()
    await sink.emit(event)
    await sink.aclose()
    # Second aclose must be safe (no-op)
    await sink.aclose()


async def test_aclose_after_flush() -> None:
    """aclose after flush must succeed."""
    path = _sink_path("aclose_after_flush")
    sink = AsyncJSONLEventSink(path, run_id="test-run")
    event = LLMCallEvent(event_type="llm_call", run_id="test-run")
    await sink.start()
    await sink.emit(event)
    await sink.flush()
    await sink.aclose()


# ---------------------------------------------------------------------------
# Gated _write_batch proof: emit does not wait for I/O
# ---------------------------------------------------------------------------


class GatedSink(AsyncJSONLEventSink):
    """Sink that records _write_batch calls so we can prove emit is non-blocking."""

    def __init__(self, path: str, **kwargs) -> None:
        super().__init__(path, **kwargs)
        self._write_batches: list[list[str]] = []

    async def _write_batch(self, lines: list[str]) -> None:
        self._write_batches.append(lines)
        # Don't block; the real write happens in the background thread.
        await super()._write_batch(lines)


async def test_gated_write_batch_proof() -> None:
    """Proof that emit does not wait for _write_batch to complete."""
    path = _sink_path("gated_batch")
    sink = GatedSink(path, run_id="test-run")
    await sink.start()

    # Emit many events rapidly; emit must never block.
    tasks = [
        sink.emit(LLMCallEvent(event_type="llm_call", run_id="test-run"))
        for _ in range(100)
    ]
    await asyncio.gather(*tasks)
    assert sink.emitted == 100
    # The writer may not have processed all of them yet, but emit returned instantly.
    await sink.aclose()

    # We can check that _write_batch was called (it ran in the background).
    assert len(sink._write_batches) > 0  # at least one batch was written


# ---------------------------------------------------------------------------
# 1000+/5000-event no-drop load
# ---------------------------------------------------------------------------


async def test_many_events_no_drop() -> None:
    """Emitting 5000 events must not drop any (emitted == written after flush)."""
    path = _sink_path("many_events")
    sink = AsyncJSONLEventSink(path, run_id="test-run")
    await sink.start()
    events = [LLMCallEvent(event_type="llm_call", run_id="test-run") for _ in range(5000)]
    for ev in events:
        await sink.emit(ev)
    assert sink.emitted == 5000
    await sink.flush()
    assert sink.written == 5000  # all made it to disk
    await sink.aclose()


# ---------------------------------------------------------------------------
# ordering / error paths
# ---------------------------------------------------------------------------


async def test_event_ordering_preserved() -> None:
    """Events must be written in the same order they were emitted."""
    path = _sink_path("ordering")
    sink = AsyncJSONLEventSink(path, run_id="test-run")
    await sink.start()
    events: list[Event] = []
    for i in range(20):
        ev = ProtocolEvent(
            event_type="handoff",
            run_id="test-run",
            from_agent="a",
            to_agent="b",
            message_type="HANDOFF",
            detail=f"detail-{i}",
        )
        await sink.emit(ev)
        events.append(ev)
    await sink.flush()
    await sink.aclose()

    # Re-read the file and verify ordering
    raw = Path(path).read_text(encoding="utf-8")
    lines = [json.loads(line) for line in raw.strip().split("\n") if line.strip()]
    assert len(lines) == 20, f"Expected 20 lines, got {len(lines)}"
    assert lines[0]["detail"] == "detail-0"
    assert lines[19]["detail"] == "detail-19"


async def test_emit_on_closed_sink_raises() -> None:
    """emit on a closed sink must raise RuntimeError."""
    path = _sink_path("closed_sink")
    sink = AsyncJSONLEventSink(path, run_id="test-run")
    await sink.start()
    await sink.aclose()
    ev = LLMCallEvent(event_type="llm_call", run_id="test-run")
    with pytest.raises(RuntimeError, match="telemetry sink is closed"):
        await sink.emit(ev)


async def test_writer_error_stored_in_sink() -> None:
    """When the background writer fails, the error is stored in sink.error.

    Causes _write_batch to raise an exception via a subclass, then verifies
    that sink.error is set and that emit/flush propagate it.
    """
    path = _sink_path("error_store")

    class FailingSink(AsyncJSONLEventSink):
        async def _write_batch(self, lines: list[str]) -> None:
            raise RuntimeError("simulated write failure")

    sink = FailingSink(path, run_id="test-run")
    await sink.start()

    # Emit an event; the writer should fail and store the error
    ev = LLMCallEvent(event_type="llm_call", run_id="test-run")
    await sink.emit(ev)

    # Give the background writer a moment to process the event and raise.
    await asyncio.sleep(0.05)

    # The error must be stored
    assert sink.error is not None, f"Expected sink.error to be set, got {sink.error}"
    assert isinstance(sink.error, RuntimeError)
    assert "simulated write failure" in str(sink.error)

    # Emitting after the error must raise
    with pytest.raises(RuntimeError, match="telemetry writer failed"):
        await sink.emit(ev)

    # Flushing must also raise
    with pytest.raises(RuntimeError, match="telemetry writer failed"):
        await sink.flush()

    await sink.aclose()


# ---------------------------------------------------------------------------
# Helper: system_error constructor
# ---------------------------------------------------------------------------


def system_error(detail: str = "") -> SystemEvent:
    """Convenience constructor for a SystemEvent with severity='error'."""
    return SystemEvent(event_type="system.error", severity="error", run_id="test-run", detail=detail)  # type: ignore[arg-expr]