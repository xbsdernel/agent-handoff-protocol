"""Async engine Prometheus metrics scraper (Issue #18).

Periodically polls the engine's Prometheus /metrics endpoint at 1 Hz during
active runs, parses the Prometheus text format, and appends time-series
points to ``engine_metrics.jsonl``.

Depends on Issue #17 (shared telemetry models & async JSONL sink).
"""

from __future__ import annotations

import asyncio
import contextlib
from typing import Any

from telemetry.sink import AsyncJSONLEventSink

# ---------------------------------------------------------------------------
# Prometheus text-format parsing
# ---------------------------------------------------------------------------

_PROMETHEUS_GAUGES: tuple[str, ...] = (
    "vllm:num_requests_waiting",
    "vllm:num_requests_running",
    "vllm:gpu_cache_usage_factor",
    "sglang:prefix_cache_hit_rate",
)


def _parse_prometheus_text(text: str) -> dict[str, float]:
    """Parse a Prometheus text-format dump into a dict of gauge values.

    Returns only the metrics listed in ``_PROMETHEUS_GAUGES``.
    """
    result: dict[str, float] = {}
    with contextlib.suppress(ValueError):
        for line in text.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            # Prometheus format: metric{labels} value [timestamp]
            # We strip labels and just keep the bare metric name + value
            parts = line.split()
            if len(parts) < 2:
                continue
            metric_name = parts[0].rstrip("{").rstrip("}")
            if metric_name in _PROMETHEUS_GAUGES:
                result[metric_name] = float(parts[1])
    return result


# ---------------------------------------------------------------------------
# EngineScraper
# ---------------------------------------------------------------------------


class EngineScraper:
    """Background scraper that polls an engine's Prometheus /metrics endpoint.

    Parameters
    ----------
    sink:
        AsyncJSONLEventSink wired to the run's ``engine_metrics.jsonl``.
    url:
        Prometheus ``/metrics`` endpoint base URL (e.g. ``http://127.0.0.1:8000/v1/metrics``).
    concurrency:
        Identifier for the concurrent load (used as a label in the event).
    """

    def __init__(
        self,
        sink: AsyncJSONLEventSink,
        url: str,
        concurrency: str = "unknown",
    ) -> None:
        self.sink = sink
        self.url = url
        self.concurrency = concurrency
        self._task: asyncio.Task[None] | None = None
        self._running = False

    # -- lifecycle -----------------------------------------------------------

    async def start(self) -> None:
        """Launch the periodic polling task."""
        if self._running:
            return
        self._running = True
        self._task = asyncio.create_task(self._run_loop(), name="engine-scraper")

    async def stop(self) -> None:
        """Signal the task to stop and wait for it to exit."""
        self._running = False
        if self._task is not None:
            await self._task
            self._task = None
        self._running = False

    # -- internals -----------------------------------------------------------

    async def _run_loop(self) -> None:
        """Poll ``/metrics`` every second while ``_running`` is True."""
        while self._running:
            try:
                with contextlib.suppress(Exception):
                    await self._scrape_one()
            except Exception:  # noqa: BLE001 - scraper must never break the run
                pass
            await asyncio.sleep(1.0)

    async def _scrape_one(self) -> None:
        """Fetch ``/metrics``, parse, and emit one event."""
        try:
            import urllib.request

            req = urllib.request.Request(self.url)
            # Add a simple user-agent so simple servers don't 403 us
            req.add_header("User-Agent", "ahp-telemeta-1.0")
            with urllib.request.urlopen(req, timeout=5) as resp:
                body = resp.read().decode("utf-8", errors="replace")
        except Exception:  # noqa: BLE001 - network failures must not crash the run
            return

        gauges = _parse_prometheus_text(body)
        if not gauges:
            return

        # Build a compact event dict; we reuse the shared sink to write JSONL.
        event_dict: dict[str, Any] = {
            "concurrency": self.concurrency,
            "timestamp_ns": __import__("time").time_ns(),
            **gauges,
        }

        # Emit via the shared sink; the sink expects an ``Event`` instance,
        # so we build a minimal ProtocolEvent with the gathered gauges.
        from telemetry.models import ProtocolEvent

        event = ProtocolEvent(
            event_type="engine.metrics",
            run_id=self.sink.run_id,
            **event_dict,
        )
        await self.sink.emit(event)


# ---------------------------------------------------------------------------
# Convenience helper: launch scraper from a RunContext
# ---------------------------------------------------------------------------

async def launch_engine_scraper(
    ctx: Any,
    *,
    url: str = "http://127.0.0.1:8000/v1/metrics",
    concurrency: str = "run-1",
) -> tuple:
    """Start an ``EngineScraper`` tied to ``ctx``'s metric sink.

    Returns ``(scraper, sink)`` so the caller can ``await scraper.stop()``
    when the experiment ends.
    """
    sink = ctx.event_sink()  # wired to the run's event sink
    scraper = EngineScraper(sink=sink, url=url, concurrency=concurrency)
    await scraper.start()
    return scraper, sink