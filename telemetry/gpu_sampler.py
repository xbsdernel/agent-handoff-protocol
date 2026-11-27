"""Async NVML GPU sampler (Issue #18).

Background task that samples ``pynvml`` at 2 Hz during active runs and
appends time-series points to ``gpu_metrics.jsonl``.

If ``pynvml`` is unavailable (CPU-only host) or cannot be initialised, the
sampler gracefully disables itself and logs a diagnostic event — it never
throws an exception that would break the experiment run.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from typing import Any

try:
    import pynvml

    _PNVML_AVAILABLE = True
except ImportError:  # pragma: no cover
    _PNVML_AVAILABLE = False  # type: ignore[assignment]

from telemetry.models import ProtocolEvent
from telemetry.sink import AsyncJSONLEventSink

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helper: initialise pynvml once (lazy, process-global)
# ---------------------------------------------------------------------------

def _init_pynvml() -> bool:
    """Initialise pynvml if available; return True on success."""
    if not _PNVML_AVAILABLE:
        return False
    try:
        import pynvml as _pv  # type: ignore[assignment]

        _pv.nvmlInit()
        return True
    except Exception:  # noqa: BLE001 - initialisation failure is non-fatal
        logger.debug("pynvml init failed")
        return False


# ---------------------------------------------------------------------------
# GPUSampler
# ---------------------------------------------------------------------------

class GPUSampler:
    """Background GPU metric sampler using ``pynvml``.

    Emits a ``ProtocolEvent(event_type="gpu.metrics")`` every ``1/2`` second.
    If pynvml is unavailable or initialisation fails, the sampler quietly
    stops and logs a debug message.
    """

    def __init__(
        self,
        sink: AsyncJSONLEventSink,
        interval: float = 0.5,  # 2 Hz
    ) -> None:
        self.sink = sink
        self.interval = interval
        self._task: asyncio.Task[None] | None = None
        self._running = False
        self._initialised = False

    # -- lifecycle -----------------------------------------------------------

    async def start(self) -> None:
        """Initialise pynvml and launch the sampling task."""
        if self._running:
            return
        # Try to initialise pynvml; if it fails, stop gracefully.
        if not _init_pynvml():
            logger.debug("GPU sampler disabled: pynvml not available or init failed")
            return

        self._initialised = True
        self._running = True
        self._task = asyncio.create_task(self._run_loop(), name="gpu-sampler")

    async def stop(self) -> None:
        """Signal the task to stop and wait for exit."""
        self._running = False
        if self._task is not None:
            await self._task
            self._task = None
        self._running = False
        if self._initialised:
            try:
                import pynvml

                pynvml.nvmlShutdown()
            except Exception:  # noqa: BLE001 - best-effort cleanup
                pass

    # -- internals -----------------------------------------------------------

    async def _run_loop(self) -> None:
        """Sample GPU metrics every ``self.interval`` seconds while running."""
        while self._running:
            try:
                with contextlib.suppress(Exception):
                    await self._sample_one()
            except Exception:  # noqa: BLE001 - sampler must never break the run
                pass
            await asyncio.sleep(self.interval)

    def _read_nvml(self) -> dict[str, Any]:
        """Read the current pynvml metrics; returns a dict keyed by metric name."""
        import pynvml

        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        util = pynvml.nvmlDeviceGetUtilizationRates(handle)

        return {
            "gpu_memory_used_mb": memory_info.used,
            "gpu_memory_total_mb": memory_info.total,
            "gpu_utilization_percent": util.gpu,
        }

    async def _sample_one(self) -> None:
        """One sampling tick: read NVML, build event, emit via sink."""
        metrics = self._read_nvml()

        event_dict: dict[str, Any] = {
            "timestamp_ns": __import__("time").time_ns(),
            "concurrency": self.sink.run_id if hasattr(self.sink, "run_id") else "unknown",
        }
        metrics.update(event_dict)

        # Build a ProtocolEvent; the sink will write it to gpu_metrics.jsonl

        event = ProtocolEvent(
            event_type="gpu.metrics",
            run_id=self.sink.run_id,
            **metrics,
        )
        await self.sink.emit(event)