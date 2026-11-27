"""Non-blocking JSONL event sink.

``emit`` never performs disk I/O: it appends the event to an ``asyncio.Queue``
and returns. A single background writer task drains the queue in batches and
hands each batch to a worker thread, so file writes cannot stall the event
loop even under thousands of events per second.

``flush`` waits until every accepted event has been written and the file handle
has been flushed. The queue is unbounded by default, which is what guarantees
that high load never drops events; pass ``max_queue_size`` to opt into
cooperative backpressure instead.
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from types import TracebackType
from typing import Any, Final

from telemetry.interfaces import Event, EventSink

_STOP: Final[object] = object()


class AsyncJSONLEventSink(EventSink):
    """Buffered, asyncio-native JSONL sink.

    Parameters
    ----------
    path:
        Destination ``events.jsonl``-style file. Parent directories are created
        on :meth:`start`.
    run_id:
        Run identifier recorded by the sink for diagnostics. Events keep their
        own ``run_id``; this value is not used to rewrite them.
    max_queue_size:
        ``0`` (default) means unbounded. A positive value applies cooperative
        backpressure from :meth:`emit`.
    fsync:
        Call ``os.fsync`` after each :meth:`flush`. Slower, but useful when a
        crash must not lose the tail of the log.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        run_id: str = "run",
        max_queue_size: int = 0,
        fsync: bool = False,
    ) -> None:
        if max_queue_size < 0:
            raise ValueError("max_queue_size must be >= 0")
        self.path = Path(path)
        self.run_id = run_id
        self.fsync = fsync
        self._queue: asyncio.Queue[Event | object] = asyncio.Queue(maxsize=max_queue_size)
        self._handle: Any | None = None
        self._writer: asyncio.Task[None] | None = None
        self._started = False
        self._closed = False
        self._error: BaseException | None = None
        self._emitted = 0
        self._written = 0

    # -- lifecycle -----------------------------------------------------------

    async def start(self) -> None:
        """Open the file and launch the writer task. Safe to call twice."""

        if self._closed:
            raise RuntimeError("telemetry sink is closed")
        if self._started:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = await asyncio.to_thread(
            self.path.open,
            "a",
            encoding="utf-8",
            newline="\n",
        )
        self._started = True
        self._writer = asyncio.create_task(
            self._run_writer(),
            name=f"telemetry-sink-{self.path.name}",
        )

    async def emit(self, event: Event) -> None:
        """Queue ``event`` without touching the filesystem."""

        if self._closed:
            raise RuntimeError("telemetry sink is closed")
        if self._error is not None:
            raise RuntimeError("telemetry writer failed") from self._error
        if not isinstance(event, Event):
            raise TypeError(f"event must be an Event, got {type(event).__name__}")
        await self.start()
        self._emitted += 1
        await self._queue.put(event)

    async def flush(self) -> None:
        """Wait until every accepted event is written and flushed to disk."""

        if not self._started:
            return
        await self._queue.join()
        await self._flush_file()
        if self._error is not None:
            raise RuntimeError("telemetry writer failed") from self._error

    async def aclose(self) -> None:
        """Stop the writer and close the file. Safe to call more than once."""

        if self._closed:
            return
        self._closed = True
        if not self._started:
            return

        await self._queue.put(_STOP)
        if self._writer is not None:
            await self._writer
        await self._flush_file()
        if self._handle is not None:
            handle = self._handle
            self._handle = None
            await asyncio.to_thread(handle.close)
        self._started = False

    async def __aenter__(self) -> AsyncJSONLEventSink:
        await self.start()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        await self.aclose()

    # -- writer --------------------------------------------------------------

    async def _run_writer(self) -> None:
        try:
            while True:
                item = await self._queue.get()
                if item is _STOP:
                    self._queue.task_done()
                    return

                batch: list[Event] = [item]  # type: ignore[list-item]
                stop = False
                while True:
                    try:
                        following = self._queue.get_nowait()
                    except asyncio.QueueEmpty:
                        break
                    if following is _STOP:
                        self._queue.task_done()
                        stop = True
                        break
                    batch.append(following)  # type: ignore[arg-type]

                try:
                    lines = [event.model_dump_json() for event in batch]
                    await self._write_batch(lines)
                    self._written += len(batch)
                finally:
                    for _ in batch:
                        self._queue.task_done()

                if stop:
                    return
        except asyncio.CancelledError:
            raise
        except BaseException as exc:  # noqa: BLE001 - surfaced through flush()
            self._error = exc
            # Keep draining so a concurrent flush()/aclose() cannot deadlock.
            while True:
                item = await self._queue.get()
                self._queue.task_done()
                if item is _STOP:
                    return

    async def _write_batch(self, lines: list[str]) -> None:
        """Persist one batch off the event loop.

        Subclasses may override this to gate or redirect writes in tests.
        """

        await asyncio.to_thread(self._write_lines, lines)

    def _write_lines(self, lines: list[str]) -> None:
        if self._handle is None:
            raise RuntimeError("telemetry sink has no open file")
        self._handle.write("\n".join(lines) + "\n")

    async def _flush_file(self) -> None:
        if self._handle is None:
            return
        handle = self._handle
        await asyncio.to_thread(handle.flush)
        if self.fsync:
            await asyncio.to_thread(os.fsync, handle.fileno())

    # -- observability -------------------------------------------------------

    @property
    def emitted(self) -> int:
        """Events accepted by :meth:`emit`."""

        return self._emitted

    @property
    def written(self) -> int:
        """Events successfully persisted."""

        return self._written

    @property
    def pending(self) -> int:
        """Events still waiting in the queue."""

        return self._queue.qsize()

    @property
    def error(self) -> BaseException | None:
        """The writer exception, if the background task failed."""

        return self._error

    @property
    def stats(self) -> dict[str, int | bool]:
        return {
            "emitted": self._emitted,
            "written": self._written,
            "pending": self.pending,
            "closed": self._closed,
        }


__all__ = ["AsyncJSONLEventSink"]
