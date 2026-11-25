"""Unit tests for the in-process AHP router (Issue #4).

Everything is mocked: handlers are plain async functions and the event sink is
an in-memory recorder. No LLM, network or GPU is involved.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from ahp.models import (
    AckMessage,
    AckStatus,
    AHPErrorCode,
    DoneMessage,
    DoneStatus,
    HandoffMessage,
    MessageType,
    NeedInfoMessage,
)
from ahp.router import (
    AckTimeoutError,
    AgentError,
    HopLimitExceeded,
    InProcessRouter,
    RouterConfig,
    UnknownAgentError,
)
from ahp.state_machine import WorkflowState
from telemetry.interfaces import Event, EventSink


class RecordingSink(EventSink):
    def __init__(self) -> None:
        self.events: list[Event] = []

    async def emit(self, event: Event) -> None:
        self.events.append(event)

    async def flush(self) -> None:
        return None

    def event_types(self) -> list[str]:
        return [event.event_type for event in self.events]

    def of_type(self, event_type: str) -> list[Event]:
        return [event for event in self.events if event.event_type == event_type]


def make_handoff(
    conversation_id: str = "conv-1",
    *,
    to_agent: str = "triage",
    **overrides: Any,
) -> HandoffMessage:
    payload: dict[str, Any] = {
        "conversation_id": conversation_id,
        "from_agent": "orchestrator",
        "to_agent": to_agent,
        "goal": "Resolve the customer ticket",
        "summary": "Duplicate charge on the Pro plan.",
    }
    payload.update(overrides)
    return HandoffMessage.model_validate(payload)


def make_done(
    conversation_id: str,
    *,
    from_agent: str = "billing",
    to_agent: str = "orchestrator",
    status: DoneStatus = DoneStatus.RESOLVED,
) -> DoneMessage:
    return DoneMessage(
        conversation_id=conversation_id,
        from_agent=from_agent,
        to_agent=to_agent,
        final_result={"decision": "refund", "amount": 20},
        summary="Refund issued.",
        status=status,
    )


async def test_linear_handoff_completes_and_emits_events() -> None:
    sink = RecordingSink()
    router = InProcessRouter(
        RouterConfig(ack_timeout_s=1.0),
        event_sink=sink,
        run_id="run-test",
    )
    calls: list[tuple[str, MessageType]] = []

    async def triage(message: Any) -> HandoffMessage:
        calls.append(("triage", message.type))
        return HandoffMessage(
            conversation_id=message.conversation_id,
            from_agent="triage",
            to_agent="billing",
            goal="Decide the refund",
            summary="Verified the duplicate charge.",
            facts={"customer_id": "C-1"},
        )

    async def billing(message: Any) -> DoneMessage:
        calls.append(("billing", message.type))
        return make_done(message.conversation_id)

    router.register_agent("triage", triage)
    router.register_agent("billing", billing)

    try:
        result = await router.run_workflow(make_handoff(), timeout_s=2.0)
    finally:
        await router.stop()

    assert result.final_result == {"decision": "refund", "amount": 20}
    assert calls == [("triage", MessageType.HANDOFF), ("billing", MessageType.HANDOFF)]

    machine = router.state_for("conv-1")
    assert machine.state is WorkflowState.COMPLETED
    assert machine.hop_count == 1

    types = sink.event_types()
    assert types.count("handoff.sent") == 2
    assert types.count("handoff.received") == 2
    assert types.count("handoff.acknowledged") == 2
    assert "workflow.completed" in types

    received = sink.of_type("handoff.received")[1]
    assert received.from_agent == "triage"
    assert received.to_agent == "billing"
    assert received.hop_count == 1
    assert received.message_type == "HANDOFF"
    assert received.workflow_id == "conv-1"


async def test_manual_ack_without_auto_ack() -> None:
    router = InProcessRouter(RouterConfig(ack_timeout_s=1.0, auto_ack=False))

    async def receiver(message: Any) -> AckMessage:
        return AckMessage(
            conversation_id=message.conversation_id,
            from_agent="receiver",
            to_agent=message.from_agent,
            acknowledged_message_id=message.message_id,
            status=AckStatus.ACCEPTED,
        )

    router.register_agent("receiver", receiver)
    try:
        ack = await router.send_and_wait_ack(make_handoff(to_agent="receiver"), timeout_s=1.0)
    finally:
        await router.stop()

    assert ack.status is AckStatus.ACCEPTED
    assert router.state_for("conv-1").state is WorkflowState.PROCESSING


async def test_ack_timeout_retries_then_escalates() -> None:
    sink = RecordingSink()
    router = InProcessRouter(
        RouterConfig(ack_timeout_s=0.03, max_retries=2, auto_ack=False),
        event_sink=sink,
    )

    async def silent(message: Any) -> None:
        await asyncio.sleep(5)

    router.register_agent("silent", silent)
    try:
        with pytest.raises(AckTimeoutError):
            await router.send_and_wait_ack(make_handoff(to_agent="silent"))
    finally:
        await router.stop()

    assert router.state_for("conv-1").state is WorkflowState.ESCALATED
    attempts = [event.attempt for event in sink.of_type("handoff.sent")]
    assert attempts == [1, 2, 3]
    assert len(sink.of_type("handoff.ack_timeout")) == 3
    assert len(sink.of_type("handoff.retry")) == 2
    assert len(sink.of_type("workflow.escalated")) == 1


async def test_duplicate_message_ids_are_dropped() -> None:
    sink = RecordingSink()
    router = InProcessRouter(RouterConfig(ack_timeout_s=1.0), event_sink=sink)
    calls = 0

    async def receiver(message: Any) -> DoneMessage:
        nonlocal calls
        calls += 1
        return make_done(
            message.conversation_id,
            from_agent="receiver",
        )

    router.register_agent("receiver", receiver)
    initial = make_handoff(to_agent="receiver")

    waiter = asyncio.create_task(router.wait_for_done("conv-1", timeout_s=2.0))
    await asyncio.sleep(0)
    try:
        await router.send(initial)
        await router.send(initial)
        result = await waiter
        await asyncio.sleep(0.05)
    finally:
        await router.stop()

    assert result.final_result["decision"] == "refund"
    assert calls == 1
    duplicates = sink.of_type("handoff.duplicate")
    assert len(duplicates) == 1
    assert duplicates[0].error_code == AHPErrorCode.DUPLICATE_MESSAGE


async def test_ping_pong_loop_is_stopped_by_hop_limit() -> None:
    sink = RecordingSink()
    router = InProcessRouter(RouterConfig(max_hops=3, ack_timeout_s=1.0), event_sink=sink)

    async def agent_a(message: Any) -> HandoffMessage:
        return HandoffMessage(
            conversation_id=message.conversation_id,
            from_agent="agent_a",
            to_agent="agent_b",
            goal="ping",
            summary="pong",
        )

    async def agent_b(message: Any) -> HandoffMessage:
        return HandoffMessage(
            conversation_id=message.conversation_id,
            from_agent="agent_b",
            to_agent="agent_a",
            goal="pong",
            summary="ping",
        )

    router.register_agent("agent_a", agent_a)
    router.register_agent("agent_b", agent_b)

    try:
        with pytest.raises(HopLimitExceeded):
            await router.run_workflow(make_handoff(to_agent="agent_a"), timeout_s=2.0)
    finally:
        await router.stop()

    machine = router.state_for("conv-1")
    assert machine.state is WorkflowState.FAILED
    assert machine.hop_count == 3
    failures = sink.of_type("workflow.failed")
    assert failures
    assert failures[-1].error_code == AHPErrorCode.HOP_LIMIT_EXCEEDED


async def test_need_info_round_trip() -> None:
    sink = RecordingSink()
    router = InProcessRouter(RouterConfig(ack_timeout_s=1.0), event_sink=sink)
    billing_calls = 0

    async def triage(message: Any) -> HandoffMessage:
        if isinstance(message, NeedInfoMessage):
            assert message.missing_fields == ["customer_id"]
            return HandoffMessage(
                conversation_id=message.conversation_id,
                from_agent="triage",
                to_agent="billing",
                goal="Decide the refund",
                summary="Customer id supplied.",
                facts={"customer_id": "C-9"},
            )
        return HandoffMessage(
            conversation_id=message.conversation_id,
            from_agent="triage",
            to_agent="billing",
            goal="Decide the refund",
            summary="New ticket.",
        )

    async def billing(message: Any) -> HandoffMessage | DoneMessage:
        nonlocal billing_calls
        billing_calls += 1
        if "customer_id" not in message.facts:
            return NeedInfoMessage(
                conversation_id=message.conversation_id,
                from_agent="billing",
                to_agent="triage",
                missing_fields=["customer_id"],
                query="Which customer is this?",
            )
        return make_done(message.conversation_id)

    router.register_agent("triage", triage)
    router.register_agent("billing", billing)

    try:
        result = await router.run_workflow(make_handoff(), timeout_s=2.0)
    finally:
        await router.stop()

    assert result.final_result["decision"] == "refund"
    assert billing_calls == 2
    assert router.state_for("conv-1").hop_count == 3
    assert len(sink.of_type("handoff.need_info")) == 1


async def test_unknown_reply_target_fails_workflow() -> None:
    sink = RecordingSink()
    router = InProcessRouter(RouterConfig(ack_timeout_s=1.0), event_sink=sink)

    async def triage(message: Any) -> HandoffMessage:
        return HandoffMessage(
            conversation_id=message.conversation_id,
            from_agent="triage",
            to_agent="ghost_agent",
            goal="Lost",
            summary="Nobody is listening.",
        )

    router.register_agent("triage", triage)

    try:
        with pytest.raises(UnknownAgentError):
            await router.run_workflow(make_handoff(), timeout_s=2.0)
    finally:
        await router.stop()

    assert router.state_for("conv-1").state is WorkflowState.FAILED
    failures = sink.of_type("workflow.failed")
    assert failures[-1].error_code == AHPErrorCode.UNKNOWN_AGENT


async def test_agent_exception_fails_workflow_without_killing_router() -> None:
    sink = RecordingSink()
    router = InProcessRouter(RouterConfig(ack_timeout_s=1.0), event_sink=sink)

    async def broken(message: Any) -> None:
        raise RuntimeError("kaboom")

    async def healthy(message: Any) -> DoneMessage:
        return make_done(message.conversation_id, from_agent="healthy")

    router.register_agent("broken", broken)
    router.register_agent("healthy", healthy)

    try:
        with pytest.raises(AgentError):
            await router.run_workflow(
                make_handoff("conv-broken", to_agent="broken"),
                timeout_s=2.0,
            )

        result = await router.run_workflow(
            make_handoff("conv-healthy", to_agent="healthy"),
            timeout_s=2.0,
        )
    finally:
        await router.stop()

    assert router.state_for("conv-broken").state is WorkflowState.FAILED
    assert result.final_result["decision"] == "refund"
    assert router.state_for("conv-healthy").state is WorkflowState.COMPLETED
    failures = sink.of_type("workflow.failed")
    assert failures[-1].error_code == AHPErrorCode.AGENT_ERROR
    agent_errors = sink.of_type("agent.error")
    assert len(agent_errors) == 1
    assert agent_errors[0].error_code == AHPErrorCode.AGENT_ERROR
    assert "kaboom" in (agent_errors[0].detail or "")


async def test_escalated_done_marks_conversation_escalated() -> None:
    sink = RecordingSink()
    router = InProcessRouter(RouterConfig(ack_timeout_s=1.0), event_sink=sink)

    async def agent(message: Any) -> DoneMessage:
        return make_done(
            message.conversation_id,
            from_agent="agent",
            status=DoneStatus.ESCALATED_UNRESOLVED,
        )

    router.register_agent("agent", agent)
    try:
        result = await router.run_workflow(
            make_handoff(to_agent="agent"),
            timeout_s=2.0,
        )
    finally:
        await router.stop()

    assert result.status is DoneStatus.ESCALATED_UNRESOLVED
    assert router.state_for("conv-1").state is WorkflowState.ESCALATED
    assert "workflow.escalated" in sink.event_types()
    assert "workflow.completed" not in sink.event_types()


async def test_unknown_initial_agent_is_rejected() -> None:
    router = InProcessRouter()

    with pytest.raises(UnknownAgentError):
        await router.send_and_wait_ack(make_handoff(to_agent="missing"), timeout_s=1.0)

    assert router.state_for("conv-1").state is WorkflowState.FAILED
    await router.stop()


async def test_registering_the_same_agent_twice_is_rejected() -> None:
    router = InProcessRouter()

    async def handler(message: Any) -> None:
        return None

    router.register_agent("agent", handler)
    with pytest.raises(ValueError):
        router.register_agent("agent", handler)
    await router.stop()


def test_invalid_router_config_is_rejected() -> None:
    with pytest.raises(ValueError):
        RouterConfig(max_hops=-1)
    with pytest.raises(ValueError):
        RouterConfig(ack_timeout_s=0)
    with pytest.raises(ValueError):
        RouterConfig(max_retries=-1)
    with pytest.raises(ValueError):
        RouterConfig(retry_backoff_s=-1)
