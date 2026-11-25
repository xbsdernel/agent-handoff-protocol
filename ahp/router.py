"""In-process asynchronous router implementing the AHP 1.0 rules.

The router owns one ``asyncio.Queue`` inbox per registered agent. A consumer
task per agent validates and delivers messages, acknowledges handoffs,
increments hop counts, drops duplicates, applies ACK timeouts with bounded
retries, and emits lifecycle events to an :class:`~telemetry.interfaces.EventSink`.

Typical use::

    router = InProcessRouter(RouterConfig(max_hops=10), event_sink=sink)
    router.register_agent("triage", triage.handle_handoff)
    router.register_agent("billing", billing.handle_handoff)
    result = await router.run_workflow(initial_handoff, timeout_s=120)
    await router.stop()
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from typing import Any
from uuid import UUID

from ahp.models import (
    AckMessage,
    AckStatus,
    AHPErrorCode,
    AHPMessage,
    DoneMessage,
    DoneStatus,
    HandoffMessage,
    MessageEnvelope,
    NeedInfoMessage,
    parse_ahp_message,
)
from ahp.state_machine import (
    InvalidStateTransition,
    WorkflowState,
    WorkflowStateMachine,
)
from telemetry.interfaces import Event, EventSink

#: An agent handler receives one message and optionally returns the next one.
AgentHandler = Callable[[AHPMessage], Awaitable[AHPMessage | None]]


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class RouterError(RuntimeError):
    """Base class for protocol and routing failures."""

    code: AHPErrorCode = AHPErrorCode.INVALID_MESSAGE


class UnknownAgentError(RouterError):
    """The target agent is not registered with the router."""

    code = AHPErrorCode.UNKNOWN_AGENT


class HopLimitExceeded(RouterError):
    """Forwarding the message would exceed the configured hop limit."""

    code = AHPErrorCode.HOP_LIMIT_EXCEEDED


class AckTimeoutError(RouterError):
    """The receiver did not acknowledge the handoff within the retry budget."""

    code = AHPErrorCode.ACK_TIMEOUT


class AgentError(RouterError):
    """An agent handler raised an exception."""

    code = AHPErrorCode.AGENT_ERROR


# ---------------------------------------------------------------------------
# Configuration and telemetry
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class RouterConfig:
    """Protocol limits. Defaults follow the AHP 1.0 specification."""

    max_hops: int = 10
    ack_timeout_s: float = 30.0
    max_retries: int = 2
    retry_backoff_s: float = 0.0
    auto_ack: bool = True

    def __post_init__(self) -> None:
        if self.max_hops < 0:
            raise ValueError("max_hops must be >= 0")
        if self.ack_timeout_s <= 0:
            raise ValueError("ack_timeout_s must be > 0")
        if self.max_retries < 0:
            raise ValueError("max_retries must be >= 0")
        if self.retry_backoff_s < 0:
            raise ValueError("retry_backoff_s must be >= 0")


class RouterEvent(Event):
    """Telemetry event emitted by the router.

    Issue #12 will replace this temporary, router-local event shape with the
    shared strongly-typed event models.
    """

    message_id: str | None = None
    acknowledged_message_id: str | None = None
    message_type: str | None = None
    from_agent: str | None = None
    to_agent: str | None = None
    hop_count: int | None = None
    attempt: int | None = None
    error_code: str | None = None
    state: str | None = None
    detail: str | None = None


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------


def _coerce_message(value: Any) -> AHPMessage:
    if isinstance(value, MessageEnvelope):
        return value
    return parse_ahp_message(value)


def _discard_future(future: asyncio.Future[Any]) -> None:
    """Cancel a pending future or mark a finished one as retrieved.

    Without this, exceptions attached to a waiter that nobody awaits (for
    example because ``send_and_wait_ack`` failed first) are reported by
    asyncio as "Future exception was never retrieved" at garbage collection.
    """

    if not future.done():
        future.cancel()
    elif not future.cancelled():
        future.exception()


class InProcessRouter:
    """Deliver AHP messages between agents inside one event loop."""

    def __init__(
        self,
        config: RouterConfig | None = None,
        event_sink: EventSink | None = None,
        *,
        run_id: str = "run",
    ) -> None:
        self.config = config or RouterConfig()
        self.run_id = run_id
        self._event_sink = event_sink
        self._handlers: dict[str, AgentHandler] = {}
        self._queues: dict[str, asyncio.Queue[AHPMessage]] = {}
        self._consumers: dict[str, asyncio.Task[None]] = {}
        self._seen_ids: set[UUID] = set()
        self._issued_acks: dict[UUID, AckMessage] = {}
        self._pending_acks: dict[UUID, asyncio.Future[AckMessage]] = {}
        self._done_waiters: dict[str, asyncio.Future[DoneMessage]] = {}
        self._states: dict[str, WorkflowStateMachine] = {}
        self._started = False
        self._closed = False

    # -- registration and lifecycle -----------------------------------------

    def register_agent(self, name: str, handler: AgentHandler) -> None:
        """Register ``handler`` under ``name`` so it can receive messages."""

        if self._closed:
            raise RouterError("router is closed")
        if not name:
            raise ValueError("agent name must not be empty")
        if name in self._handlers:
            raise ValueError(f"agent {name!r} is already registered")

        self._handlers[name] = handler
        self._queues[name] = asyncio.Queue()
        if self._started:
            self._ensure_consumer(name)

    async def start(self) -> None:
        """Start one consumer task per registered agent."""

        self._ensure_started()

    async def stop(self) -> None:
        """Cancel consumers and pending waiters. Safe to call twice."""

        self._closed = True
        tasks = list(self._consumers.values())
        self._consumers.clear()
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

        for future in self._pending_acks.values():
            _discard_future(future)
        self._pending_acks.clear()

        for future in self._done_waiters.values():
            _discard_future(future)
        self._done_waiters.clear()

        self._started = False

    async def __aenter__(self) -> InProcessRouter:
        await self.start()
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self.stop()

    def _ensure_started(self) -> None:
        if self._closed:
            raise RouterError("router is closed")
        if self._started:
            return
        self._started = True
        for name in self._handlers:
            self._ensure_consumer(name)

    def _ensure_consumer(self, name: str) -> None:
        if name in self._consumers:
            return
        self._consumers[name] = asyncio.create_task(
            self._consume(name), name=f"ahp-consumer-{name}"
        )

    # -- state and telemetry -------------------------------------------------

    def state_for(self, conversation_id: str) -> WorkflowStateMachine:
        """Return (creating if necessary) the state machine for a conversation."""

        machine = self._states.get(conversation_id)
        if machine is None:
            machine = WorkflowStateMachine(workflow_id=conversation_id)
            self._states[conversation_id] = machine
        return machine

    async def _transition(
        self,
        conversation_id: str,
        target: WorkflowState,
        *,
        reason: str = "",
        hop_count: int | None = None,
    ) -> None:
        machine = self.state_for(conversation_id)
        previous = machine.state
        try:
            machine.transition(target, reason=reason, hop_count=hop_count)
        except InvalidStateTransition as exc:
            await self._emit(
                "workflow.invalid_transition",
                conversation_id=conversation_id,
                error_code=AHPErrorCode.INVALID_MESSAGE,
                state=str(machine.state),
                detail=str(exc),
            )
            return
        await self._emit(
            "workflow.state_changed",
            conversation_id=conversation_id,
            state=str(machine.state),
            detail=f"{previous}->{machine.state}: {reason}"
            if reason
            else f"{previous}->{machine.state}",
        )

    async def _emit(
        self,
        event_type: str,
        *,
        message: MessageEnvelope | None = None,
        conversation_id: str | None = None,
        attempt: int | None = None,
        error_code: AHPErrorCode | None = None,
        state: str | None = None,
        detail: str | None = None,
    ) -> None:
        if self._event_sink is None:
            return
        workflow_id = conversation_id or (message.conversation_id if message else None)
        event = RouterEvent(
            run_id=self.run_id,
            workflow_id=workflow_id,
            event_type=event_type,
            message_id=str(message.message_id) if message else None,
            acknowledged_message_id=(
                str(message.acknowledged_message_id) if isinstance(message, AckMessage) else None
            ),
            message_type=str(message.type) if message else None,
            from_agent=message.from_agent if message else None,
            to_agent=message.to_agent if message else None,
            hop_count=message.hop_count if message else None,
            attempt=attempt,
            error_code=str(error_code) if error_code else None,
            state=state,
            detail=detail,
        )
        try:
            await self._event_sink.emit(event)
        except Exception:  # noqa: BLE001 - telemetry must never break routing
            return

    # -- sending -------------------------------------------------------------

    async def send(
        self, message: AHPMessage | Mapping[str, Any] | str, *, attempt: int = 1
    ) -> None:
        """Validate and enqueue ``message`` without waiting for an ACK."""

        msg = _coerce_message(message)
        self._ensure_started()
        conversation_id = msg.conversation_id

        if msg.hop_count > self.config.max_hops:
            error = HopLimitExceeded(
                f"hop_count {msg.hop_count} exceeds max_hops {self.config.max_hops}"
            )
            await self._fail(conversation_id, error)
            raise error

        if isinstance(msg, AckMessage):
            await self._register_ack(msg)
            return
        if isinstance(msg, DoneMessage):
            await self._complete(conversation_id, msg)
            return

        machine = self.state_for(conversation_id)
        if isinstance(msg, HandoffMessage) and machine.state is WorkflowState.INITIATED:
            await self._transition(
                conversation_id,
                WorkflowState.IN_TRANSIT,
                reason="handoff sent",
                hop_count=msg.hop_count,
            )

        if msg.to_agent not in self._handlers:
            error = UnknownAgentError(f"agent {msg.to_agent!r} is not registered")
            await self._fail(conversation_id, error)
            raise error

        await self._enqueue(msg, attempt=attempt)

    async def _enqueue(self, message: AHPMessage, *, attempt: int = 1) -> None:
        await self._emit("handoff.sent", message=message, attempt=attempt)
        await self._queues[message.to_agent].put(message)

    async def send_and_wait_ack(
        self,
        message: HandoffMessage,
        timeout_s: float | None = None,
    ) -> AckMessage:
        """Deliver a handoff and wait for its ACK, retrying on timeout."""

        if not isinstance(message, HandoffMessage):
            raise TypeError("send_and_wait_ack requires a HandoffMessage")
        self._ensure_started()

        conversation_id = message.conversation_id
        timeout = self.config.ack_timeout_s if timeout_s is None else timeout_s
        if timeout <= 0:
            raise ValueError("timeout_s must be > 0")

        loop = asyncio.get_running_loop()
        future = self._pending_acks.get(message.message_id)
        if future is None or future.done():
            future = loop.create_future()
            self._pending_acks[message.message_id] = future

        attempts = self.config.max_retries + 1
        try:
            for attempt in range(1, attempts + 1):
                await self.send(message, attempt=attempt)
                try:
                    return await asyncio.wait_for(asyncio.shield(future), timeout)
                except TimeoutError:
                    await self._emit(
                        "handoff.ack_timeout",
                        message=message,
                        attempt=attempt,
                        error_code=AHPErrorCode.ACK_TIMEOUT,
                        detail=f"no ACK within {timeout:.3f}s",
                    )
                    if attempt < attempts:
                        await self._emit("handoff.retry", message=message, attempt=attempt)
                        if self.config.retry_backoff_s:
                            await asyncio.sleep(self.config.retry_backoff_s)

            error = AckTimeoutError(f"no ACK for {message.message_id} after {attempts} attempt(s)")
            await self._escalate(conversation_id, error)
            raise error
        finally:
            self._pending_acks.pop(message.message_id, None)
            _discard_future(future)

    async def run_workflow(
        self,
        initial: HandoffMessage,
        *,
        timeout_s: float | None = None,
    ) -> DoneMessage:
        """Run a conversation from ``initial`` until its DONE message arrives."""

        if not isinstance(initial, HandoffMessage):
            raise TypeError("run_workflow requires a HandoffMessage")

        conversation_id = initial.conversation_id
        waiter = self._done_waiter(conversation_id)
        try:
            await self.send_and_wait_ack(initial)
            if timeout_s is None:
                return await waiter
            return await asyncio.wait_for(asyncio.shield(waiter), timeout_s)
        finally:
            self._done_waiters.pop(conversation_id, None)
            _discard_future(waiter)

    async def wait_for_done(
        self,
        conversation_id: str,
        *,
        timeout_s: float | None = None,
    ) -> DoneMessage:
        """Wait for the DONE message of a conversation already in progress."""

        waiter = self._done_waiter(conversation_id)
        try:
            if timeout_s is None:
                return await waiter
            return await asyncio.wait_for(asyncio.shield(waiter), timeout_s)
        finally:
            self._done_waiters.pop(conversation_id, None)
            _discard_future(waiter)

    def _done_waiter(self, conversation_id: str) -> asyncio.Future[DoneMessage]:
        waiter = self._done_waiters.get(conversation_id)
        if waiter is None or waiter.done():
            waiter = asyncio.get_running_loop().create_future()
            self._done_waiters[conversation_id] = waiter
        return waiter

    # -- receiving -----------------------------------------------------------

    async def _consume(self, agent_name: str) -> None:
        queue = self._queues[agent_name]
        while True:
            message = await queue.get()
            try:
                await self._handle_incoming(agent_name, message)
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 - safety net for router bugs
                await self._emit(
                    "agent.error",
                    message=message,
                    error_code=AHPErrorCode.AGENT_ERROR,
                    detail=f"unhandled error while delivering to {agent_name}: {exc}",
                )
                await self._fail(
                    message.conversation_id,
                    AgentError(f"unhandled error while delivering to {agent_name}: {exc}"),
                )
            finally:
                queue.task_done()

    async def _handle_incoming(self, agent_name: str, message: AHPMessage) -> None:
        conversation_id = message.conversation_id

        if message.to_agent != agent_name:
            await self._emit(
                "handoff.misdelivered",
                message=message,
                detail=f"expected {agent_name}, got {message.to_agent}",
            )
            return

        if message.message_id in self._seen_ids:
            await self._emit(
                "handoff.duplicate",
                message=message,
                error_code=AHPErrorCode.DUPLICATE_MESSAGE,
            )
            if isinstance(message, HandoffMessage):
                cached = self._issued_acks.get(message.message_id)
                if cached is not None:
                    await self._deliver_ack(cached)
            return

        self._seen_ids.add(message.message_id)

        if isinstance(message, AckMessage):
            await self._register_ack(message)
            return
        if isinstance(message, DoneMessage):
            await self._complete(conversation_id, message)
            return

        if isinstance(message, NeedInfoMessage):
            await self._transition(
                conversation_id,
                WorkflowState.PROCESSING,
                reason="need_info received",
                hop_count=message.hop_count,
            )
            await self._emit("handoff.need_info", message=message)
        else:
            await self._transition(
                conversation_id,
                WorkflowState.PROCESSING,
                reason="handoff received",
                hop_count=message.hop_count,
            )
            await self._emit("handoff.received", message=message)
            if self.config.auto_ack:
                await self._acknowledge(message)

        handler = self._handlers[agent_name]
        try:
            reply = await handler(message)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - converted to a protocol failure
            await self._emit(
                "agent.error",
                message=message,
                error_code=AHPErrorCode.AGENT_ERROR,
                detail=f"agent {agent_name!r} failed: {exc}",
            )
            await self._fail(
                conversation_id,
                AgentError(f"agent {agent_name!r} failed: {exc}"),
            )
            return

        if reply is None:
            return
        await self._route_reply(
            parent=message,
            current_agent=agent_name,
            reply=_coerce_message(reply),
        )

    async def _route_reply(
        self,
        *,
        parent: AHPMessage,
        current_agent: str,
        reply: AHPMessage,
    ) -> None:
        conversation_id = parent.conversation_id

        if isinstance(reply, AckMessage):
            if reply.acknowledged_message_id != parent.message_id:
                reply = reply.model_copy(update={"acknowledged_message_id": parent.message_id})
            await self._register_ack(reply)
            return

        updates: dict[str, Any] = {
            "from_agent": current_agent,
            "conversation_id": conversation_id,
        }

        if isinstance(reply, HandoffMessage | NeedInfoMessage):
            next_hop = parent.hop_count + 1
            if next_hop > self.config.max_hops:
                await self._fail(
                    conversation_id,
                    HopLimitExceeded(
                        f"hop_count {next_hop} exceeds max_hops {self.config.max_hops}"
                    ),
                )
                return
            updates["hop_count"] = next_hop
            reply = reply.model_copy(update=updates)

            if reply.to_agent not in self._handlers:
                await self._fail(
                    conversation_id,
                    UnknownAgentError(f"agent {reply.to_agent!r} is not registered"),
                )
                return

            target_state = (
                WorkflowState.IN_TRANSIT
                if isinstance(reply, HandoffMessage)
                else WorkflowState.BLOCKED_NEED_INFO
            )
            await self._transition(
                conversation_id,
                target_state,
                reason="message forwarded",
                hop_count=next_hop,
            )
            await self._enqueue(reply)
            return

        # DoneMessage
        reply = reply.model_copy(update=updates)
        await self._complete(conversation_id, reply)

    # -- acknowledgements and completion ------------------------------------

    async def _acknowledge(self, message: HandoffMessage) -> AckMessage:
        ack = AckMessage(
            conversation_id=message.conversation_id,
            from_agent=message.to_agent,
            to_agent=message.from_agent,
            hop_count=message.hop_count,
            acknowledged_message_id=message.message_id,
            status=AckStatus.ACCEPTED,
        )
        await self._register_ack(ack)
        return ack

    async def _register_ack(self, ack: AckMessage) -> None:
        self._issued_acks[ack.acknowledged_message_id] = ack
        await self._deliver_ack(ack)

    async def _deliver_ack(self, ack: AckMessage) -> None:
        waiter = self._pending_acks.get(ack.acknowledged_message_id)
        if waiter is not None and not waiter.done():
            waiter.set_result(ack)
        await self._emit(
            "handoff.acknowledged",
            message=ack,
            detail=str(ack.status),
        )

    async def _complete(self, conversation_id: str, message: DoneMessage) -> None:
        target = (
            WorkflowState.COMPLETED
            if message.status is DoneStatus.RESOLVED
            else WorkflowState.ESCALATED
        )
        await self._transition(
            conversation_id,
            target,
            reason="done received",
            hop_count=message.hop_count,
        )
        event_type = (
            "workflow.completed" if target is WorkflowState.COMPLETED else "workflow.escalated"
        )
        await self._emit(event_type, message=message, state=str(target))

        waiter = self._done_waiters.get(conversation_id)
        if waiter is not None and not waiter.done():
            waiter.set_result(message)

    # -- failures ------------------------------------------------------------

    async def _fail(self, conversation_id: str, error: RouterError) -> None:
        machine = self.state_for(conversation_id)
        if not machine.is_terminal:
            await self._transition(
                conversation_id,
                WorkflowState.FAILED,
                reason=str(error),
                hop_count=machine.hop_count,
            )
        await self._emit(
            "workflow.failed",
            conversation_id=conversation_id,
            error_code=error.code,
            state=str(machine.state),
            detail=str(error),
        )
        self._resolve_done_waiter(conversation_id, error)

    async def _escalate(self, conversation_id: str, error: RouterError) -> None:
        machine = self.state_for(conversation_id)
        if not machine.is_terminal:
            await self._transition(
                conversation_id,
                WorkflowState.ESCALATED,
                reason=str(error),
                hop_count=machine.hop_count,
            )
        await self._emit(
            "workflow.escalated",
            conversation_id=conversation_id,
            error_code=error.code,
            state=str(machine.state),
            detail=str(error),
        )
        self._resolve_done_waiter(conversation_id, error)

    def _resolve_done_waiter(self, conversation_id: str, error: RouterError) -> None:
        waiter = self._done_waiters.get(conversation_id)
        if waiter is not None and not waiter.done():
            waiter.set_exception(error)


__all__ = [
    "AckTimeoutError",
    "AgentError",
    "AgentHandler",
    "HopLimitExceeded",
    "InProcessRouter",
    "RouterConfig",
    "RouterError",
    "RouterEvent",
    "UnknownAgentError",
]
