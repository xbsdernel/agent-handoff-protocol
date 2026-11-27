"""Workload driver for closed-loop and open-loop execution (Issue #20).

Supports two execution modes:
- Closed-loop: maintains constant concurrency of N simultaneous
  in-flight workflows.
- Open-loop: dispatches tasks according to a Poisson process with
  rate lambda (exponential inter-arrival times).

Each workflow has an independent timeout guard; exceeding it cancels
the task and logs a ``TIMEOUT_ABORT`` event without killing the entire run.
"""

from __future__ import annotations

import asyncio
import logging
import math
import random
from collections.abc import Callable
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Closed-loop runner
# ---------------------------------------------------------------------------

async def run_closed_loop(
    workflow_fn: Callable[[int], asyncio.Future],
    tasks: list[int],
    concurrency: int,
    max_tasks: int | None = None,
) -> list[Any]:
    """Run *tasks* with at most ``concurrency`` in-flight at any time.

    Parameters
    ----------
    workflow_fn:
        async function ``workflow_fn(task_id)`` that returns a result.
    tasks:
        ordered list of task identifiers (usually ``[0, 1, ..., n-1]``).
    concurrency:
        maximum number of workflows that may run simultaneously.
    max_tasks:
        if provided, stop after this many tasks have completed and cancel
        the remaining in-flight workflows.

    Returns
    -------
    results:
        list of results ordered by ``tasks`` order.  Uncompleted entries
        remain ``None``.
    """
    if concurrency < 1:
        raise ValueError("concurrency must be >= 1")
    if max_tasks is not None and max_tasks < 1:
        raise ValueError("max_tasks must be >= 1")

    results: list[Any] = [None] * len(tasks)

    # Determine how many tasks we'll actually dispatch
    dispatch_limit = len(tasks)
    if max_tasks is not None:
        dispatch_limit = min(len(tasks), max_tasks)

    completed = 0
    # Queue of (index, task_id) still to be dispatched
    remaining: list[tuple[int, int]] = list(enumerate(tasks[:dispatch_limit]))
    # Track in-flight by index
    in_flight: set[int] = set()
    # Signal workers to stop when max_tasks reached
    stop_event = asyncio.Event()

    async def _worker(idx: int, task_id: int) -> None:
        nonlocal completed
        try:
            async with asyncio.Semaphore(concurrency):
                # Execute workflow
                result = await asyncio.wait_for(
                    workflow_fn(task_id), timeout=60.0
                )
                results[task_id] = result
        except TimeoutError:
            results[task_id] = {"status": "TIMEOUT_ABORT", "task_id": task_id}
        except Exception as exc:  # noqa: BLE001 - workflow errors are non-fatal
            results[task_id] = {"status": "ERROR", "task_id": task_id, "error": str(exc)}
        finally:
            in_flight.discard(idx)
            completed += 1
            # Signal stop if we've reached max_tasks
            if max_tasks is not None and completed >= max_tasks:
                stop_event.set()
            # Launch next task from queue if any
            await _launch_next()

    async def _launch_next() -> None:
        nonlocal remaining, in_flight
        if not remaining:
            return
        # If max_tasks already reached, don't launch more
        if max_tasks is not None and completed >= max_tasks:
            return
        idx, task_id = remaining.pop(0)
        asyncio.create_task(_worker(idx, task_id))
        in_flight.add(idx)

    # Launch initial batch of tasks up to concurrency
    for _ in range(concurrency):
        await _launch_next()

    # Wait for all dispatched tasks to complete
    while in_flight or remaining:
        if max_tasks is not None and completed >= max_tasks and not remaining:
            # All in-flight workers will exit via their finally blocks;
            # give them a moment then break.
            if in_flight:
                await asyncio.sleep(0.1)
            break
        await asyncio.sleep(0.05)

    # Any tasks beyond dispatch_limit remain None (already initialized)
    return results


# ---------------------------------------------------------------------------
# Open-loop runner (Poisson arrival process)
# ---------------------------------------------------------------------------

async def run_open_loop(
    workflow_fn: Callable[[int], asyncio.Future],
    tasks: list[int],
    arrival_rate_lambda: float,
    seed: int | None = None,
    max_tasks: int | None = None,
    max_time_s: float | None = None,
) -> list[Any]:
    """Dispatch tasks according to a Poisson process (exponential inter-arrival).

    The inter-arrival time between successive task dispatches is
    ``-ln(U) / λ`` where ``U ~ Uniform(0, 1)``.

    Parameters
    ----------
    workflow_fn:
        async function ``workflow_fn(task_id)`` that returns a result.
    tasks:
        ordered list of task identifiers.
    arrival_rate_lambda:
        Poisson rate λ (events per second).
    seed:
        optional random seed for reproducibility.
    max_tasks:
        stop after this many completed tasks.
    max_time_s:
        stop the entire run after this many seconds.

    Returns
    -------
    results:
        list of results ordered by task submission order.
    """
    if arrival_rate_lambda <= 0:
        raise ValueError("arrival_rate_lambda must be > 0")
    if max_tasks is not None and max_tasks < 1:
        raise ValueError("max_tasks must be >= 1")
    if max_time_s is not None and max_time_s <= 0:
        raise ValueError("max_time_s must be > 0")

    rng = random.Random(seed)
    results: list[Any] = [None] * len(tasks)
    completed = 0
    start_time = asyncio.get_event_loop().time()
    error: str | None = None

    # Dispatch events - each task waits on its own event
    dispatch_events: list[asyncio.Event] = [asyncio.Event() for _ in tasks]
    # Global stop signal for max_tasks
    stop_event = asyncio.Event()

    async def _worker(idx: int, task_id: int) -> None:
        nonlocal completed, error
        # Wait for our dispatch event
        await dispatch_events[idx].wait()
        # If stop_event set before we start, skip this task
        if stop_event.is_set():
            return
        try:
            result = await asyncio.wait_for(
                workflow_fn(task_id), timeout=60.0
            )
            results[task_id] = result
        except TimeoutError:
            results[task_id] = {"status": "TIMEOUT_ABORT", "task_id": task_id}
            error = error or "TIMEOUT_ABORT"
        except Exception as exc:  # noqa: BLE001 - workflow errors are non-fatal
            results[task_id] = {"status": "ERROR", "task_id": task_id, "error": str(exc)}
            error = error or str(exc)
        finally:
            nonlocal completed
            completed += 1
            if max_tasks is not None and completed >= max_tasks and not stop_event.is_set():
                stop_event.set()

    launch_futures: list[asyncio.Task[None]] = []

    # Schedule dispatch times with exponential inter-arrival delays
    for i, task_id in enumerate(tasks):
        # Stop dispatching if max_tasks already reached
        if max_tasks is not None and (i + 1) >= max_tasks:
            break
        # Stop if max_time_s exceeded
        if max_time_s is not None:
            elapsed = asyncio.get_event_loop().time() - start_time
            if elapsed >= max_time_s:
                break

        u = rng.random()
        inter_arrival = -math.log(u) / arrival_rate_lambda
        # Schedule the event to fire after inter_arrival seconds
        asyncio.get_event_loop().call_later(
            inter_arrival, lambda i=i: dispatch_events[i].set()
        )
        launch_futures.append(asyncio.create_task(_worker(i, task_id)))

    # Wait for all launched workers to complete
    if max_time_s is not None:
        elapsed = asyncio.get_event_loop().time() - start_time
        if elapsed < max_time_s:
            remaining = max_time_s - elapsed
            await asyncio.wait_for(
                asyncio.gather(*launch_futures, return_exceptions=True),
                timeout=remaining,
            )
        else:
            for f in launch_futures:
                if not f.done():
                    f.cancel()
    else:
        await asyncio.gather(*launch_futures, return_exceptions=True)

    return results