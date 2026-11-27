"""Telemetry contracts shared by every layer of the harness.

Concrete event subclasses (``WorkflowEvent``, ``HandoffEvent``, ...) live in
``telemetry.models``, and the buffered JSONL sink lives in ``telemetry.sink``.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field


class Event(BaseModel):
    """Base envelope carried by every telemetry event.

    ``run_id`` ties an event to a single experiment run; ``timestamp_ns`` uses
    ``time.time_ns()`` so events across processes can be ordered. Subclasses
    narrow ``event_type`` and add their own payload fields.
    """

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1.0"] = "1.0"
    event_id: str = Field(default_factory=lambda: str(uuid4()))
    run_id: str
    workflow_id: str | None = None
    event_type: str
    timestamp_ns: int = Field(default_factory=time.time_ns)


class EventSink(ABC):
    """Destination for telemetry events.

    Implementations must be non-blocking from the caller's point of view: the
    harness emits events on the hot path, so buffering and I/O happen inside
    the sink.
    """

    @abstractmethod
    async def emit(self, event: Event) -> None:
        """Accept ``event`` for later persistence."""

    @abstractmethod
    async def flush(self) -> None:
        """Persist everything accepted so far."""

    async def aclose(self) -> None:
        """Flush and release resources. Safe to call more than once."""
        await self.flush()


__all__ = ["Event", "EventSink"]
