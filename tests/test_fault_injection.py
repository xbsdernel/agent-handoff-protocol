"""Tests for the fault-injection transport wrapper (Issue #5).

Everything runs in-process with deterministic seeds: no network, GPU or LLM.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Sequence
from typing import Any

import pytest
from pydantic import ValidationError

from ahp.fault_injection import (
    CORRUPTION_STRATEGIES,
    CorruptionPolicy,
    FaultConfig,
    FaultInjectionRouter,
    FaultKind,
    corrupt_document,
)
from ahp.models import (
    AckMessage,
    AckStatus,
    AHPErrorCode,
    DoneMessage,
    HandoffMessage,
    NeedInfoMessage,
    parse_ahp_message,
)
from ahp.router import AckTimeoutError, InProcessRouter, RouterConfig
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


class ForcedRandom:
    """Stand-in for ``random.Random`` that always picks a given strategy."""

    def __init__(self, strategy: str) -> None:
        self.strategy = strategy

    def choice(self, options: Sequence[str]) -> str:
        return self.strategy if self.strategy in options else options[0]


def make_handoff(
    conversation_id: str = "conv-1",
    *,
    from_agent: str = "orchestrator",
    to_agent: str = "receiver",
    **overrides: Any,
) -> HandoffMessage:
    payload: dict[str, Any] = {
        "conversation_id": conversation_id,
        "from_agent": from_agent,
        "to_agent": to_agent,
        "goal": "Resolve the customer ticket",
        "summary": "Duplicate charge on the Pro plan.",
    }
    payload.update(overrides)
    return HandoffMessage.model_validate(payload)


def make_ack(conversation_id: str = "conv-1") -> AckMessage:
    return AckMessage(
        conversation_id=conversation_id,
        from_agent="receiver",
        to_agent="orchestrator",
        acknowledged_message_id=make_handoff(conversation_id).message_id,
        status=AckStatus.ACCEPTED,
    )


def make_need_info(conversation_id: str = "conv-1") -> NeedInfoMessage:
    return NeedInfoMessage(
        conversation_id=conversation_id,
        from_agent="receiver",
        to_agent="orchestrator",
        missing_fields=["customer_id"],
        query="Which customer is this?",
    )


def make_done(conversation_id: str = "conv-1") -> DoneMessage:
    return DoneMessage(
        conversation_id=conversation_id,
        from_agent="receiver",
        to_agent="orchestrator",
        final_result={"decision": "refund"},
        summary="Refund issued.",
    )


# ---------------------------------------------------------------------------
# Configuration and corruption helpers
# ---------------------------------------------------------------------------


def test_zero_fault_config_is_valid() -> None:
    faults = FaultConfig()

    assert faults.drop_rate == 0.0
    assert faults.corruption_rate == 0.0
    assert faults.duplicate_rate == 0.0
    assert faults.delay_range == (0.0, 0.0)
    assert faults.corruption_policy is CorruptionPolicy.DROP
    assert faults.seed is None


def test_config_accepts_string_policy_and_validates_rates() -> None:
    assert (
        FaultConfig(corruption_policy="need_info").corruption_policy is CorruptionPolicy.NEED_INFO
    )

    with pytest.raises(ValueError):
        FaultConfig(drop_rate=-0.1)
    with pytest.raises(ValueError):
        FaultConfig(drop_rate=1.1)
    with pytest.raises(ValueError):
        FaultConfig(corruption_rate=2.0)
    with pytest.raises(ValueError):
        FaultConfig(duplicate_rate=-1.0)
    with pytest.raises(ValueError):
        FaultConfig(delay_range=(-0.1, 0.1))
    with pytest.raises(ValueError):
        FaultConfig(delay_range=(0.2, 0.1))


@pytest.mark.parametrize("strategy", CORRUPTION_STRATEGIES)
def test_every_corruption_strategy_produces_invalid_ahp(strategy: str) -> None:
    name, payload = corrupt_document(make_handoff(), ForcedRandom(strategy))

    assert name == strategy
    with pytest.raises(ValidationError):
        parse_ahp_message(payload)


def test_all_message_types_can_be_corrupted() -> None:
    messages = [make_handoff(), make_ack(), make_need_info(), make_done()]

    for message in messages:
        for strategy in CORRUPTION_STRATEGIES:
            name, payload = corrupt_document(message, ForcedRandom(strategy))

            assert name == strategy
            with pytest.raises(ValidationError):
                parse_ahp_message(payload)


def test_wrapper_is_a_router() -> None:
    assert issubclass(FaultInjectionRouter, InProcessRouter)


# ---------------------------------------------------------------------------
# Router behaviour
# ---------------------------------------------------------------------------


async def test_zero_faults_behaves_like_the_base_router() -> None:
    sink = RecordingSink()
    router = FaultInjectionRouter(
        RouterConfig(ack_timeout_s=1.0),
        event_sink=sink,
        faults=FaultConfig(seed=1),
    )

    async def receiver(message: HandoffMessage) -> DoneMessage:
        return make_done(message.conversation_id)

    router.register_agent("receiver", receiver)

    try:
        result = await router.run_workflow(make_handoff(), timeout_s=2.0)
    finally:
        await router.stop()

    assert result.final_result == {"decision": "refund"}
    assert router.fault_history == []
    assert router.state_for("conv-1").state is WorkflowState.COMPLETED
    assert "handoff.duplicate" not in sink.event_types()


async def test_full_drop_rate_triggers_ack_timeout_and_retries() -> None:
    sink = RecordingSink()
    router = FaultInjectionRouter(
        RouterConfig(ack_timeout_s=0.02, max_retries=2),
        event_sink=sink,
        faults=FaultConfig(drop_rate=1.0, seed=2),
    )

    async def receiver(message: HandoffMessage) -> DoneMessage:
        raise AssertionError("a fully dropped handoff must never be delivered")

    router.register_agent("receiver", receiver)

    try:
        with pytest.raises(AckTimeoutError):
            await router.send_and_wait_ack(make_handoff(), timeout_s=0.02)
    finally:
        await router.stop()

    assert router.state_for("conv-1").state is WorkflowState.ESCALATED
    assert [record.kind for record in router.fault_history] == [FaultKind.DROP] * 3
    assert [record.attempt for record in router.fault_history] == [1, 2, 3]
    assert len(sink.of_type("fault.message_dropped")) == 3
    assert len(sink.of_type("handoff.ack_timeout")) == 3
    assert len(sink.of_type("workflow.escalated")) == 1


async def test_full_corruption_rate_is_rejected_without_crashing() -> None:
    sink = RecordingSink()
    router = FaultInjectionRouter(
        RouterConfig(ack_timeout_s=0.02, max_retries=2),
        event_sink=sink,
        faults=FaultConfig(corruption_rate=1.0, seed=3),
    )
    calls = 0

    async def receiver(message: HandoffMessage) -> DoneMessage:
        nonlocal calls
        calls += 1
        return make_done(message.conversation_id)

    router.register_agent("receiver", receiver)

    try:
        with pytest.raises(AckTimeoutError):
            await router.send_and_wait_ack(make_handoff(), timeout_s=0.02)
    finally:
        await router.stop()

    assert calls == 0
    assert router.state_for("conv-1").state is WorkflowState.ESCALATED
    assert [record.kind for record in router.fault_history] == [FaultKind.CORRUPTION] * 3
    assert all(record.strategy for record in router.fault_history)

    rejections = sink.of_type("handoff.rejected")
    assert len(rejections) == 3
    assert all(event.error_code == AHPErrorCode.SCHEMA_VIOLATION for event in rejections)
    assert "handoff.received" not in sink.event_types()


async def test_corruption_need_info_policy_asks_the_sender_to_resend() -> None:
    sink = RecordingSink()
    router = FaultInjectionRouter(
        RouterConfig(ack_timeout_s=0.02, max_retries=0),
        event_sink=sink,
        faults=FaultConfig(
            corruption_rate=1.0,
            corruption_policy=CorruptionPolicy.NEED_INFO,
            seed=4,
        ),
    )
    received: list[Any] = []

    async def orchestrator(message: Any) -> None:
        received.append(message)
        return None

    async def receiver(message: Any) -> DoneMessage:
        raise AssertionError("a corrupted handoff must never reach the receiver")

    router.register_agent("orchestrator", orchestrator)
    router.register_agent("receiver", receiver)

    try:
        with pytest.raises(AckTimeoutError):
            await router.send_and_wait_ack(make_handoff(), timeout_s=0.02)
        await asyncio.sleep(0.05)
    finally:
        await router.stop()

    assert len(received) == 1
    need_info = received[0]
    assert isinstance(need_info, NeedInfoMessage)
    assert need_info.missing_fields == ["payload"]
    assert need_info.to_agent == "orchestrator"
    assert len(sink.of_type("fault.need_info_sent")) == 1


async def test_corruption_reports_unregistered_senders() -> None:
    sink = RecordingSink()
    router = FaultInjectionRouter(
        event_sink=sink,
        faults=FaultConfig(
            corruption_rate=1.0,
            corruption_policy=CorruptionPolicy.NEED_INFO,
            seed=5,
        ),
    )

    async def receiver(message: Any) -> None:
        return None

    router.register_agent("receiver", receiver)

    try:
        await router.send(make_handoff(from_agent="external_client"))
        await asyncio.sleep(0.02)
    finally:
        await router.stop()

    undeliverable = sink.of_type("fault.need_info_undeliverable")
    assert len(undeliverable) == 1
    assert "external_client" in (undeliverable[0].detail or "")
    assert not sink.of_type("fault.need_info_sent")


async def test_duplicate_injection_is_deduplicated() -> None:
    sink = RecordingSink()
    router = FaultInjectionRouter(
        RouterConfig(ack_timeout_s=1.0),
        event_sink=sink,
        faults=FaultConfig(duplicate_rate=1.0, seed=6),
    )
    calls = 0

    async def receiver(message: HandoffMessage) -> DoneMessage:
        nonlocal calls
        calls += 1
        return make_done(message.conversation_id)

    router.register_agent("receiver", receiver)

    try:
        await router.send(make_handoff())
        await asyncio.sleep(0.05)
    finally:
        await router.stop()

    assert calls == 1
    assert [record.kind for record in router.fault_history] == [FaultKind.DUPLICATION]
    assert len(sink.of_type("fault.message_duplicated")) == 1
    assert len(sink.of_type("handoff.duplicate")) == 1


async def test_delay_injection_delays_delivery() -> None:
    sink = RecordingSink()
    router = FaultInjectionRouter(
        event_sink=sink,
        faults=FaultConfig(delay_range=(0.05, 0.05), seed=7),
    )

    async def receiver(message: Any) -> None:
        return None

    router.register_agent("receiver", receiver)

    started = time.perf_counter()
    try:
        await router.send(make_handoff())
    finally:
        await router.stop()
    elapsed = time.perf_counter() - started

    assert elapsed >= 0.04
    assert [record.kind for record in router.fault_history] == [FaultKind.DELAY]
    assert router.fault_history[0].delay_s == pytest.approx(0.05)
    assert len(sink.of_type("fault.delivery_delayed")) == 1


async def test_fixed_seed_replays_identical_fault_sequence() -> None:
    faults = FaultConfig(
        drop_rate=0.4,
        corruption_rate=0.3,
        duplicate_rate=0.2,
        seed=1234,
    )
    messages = [make_handoff(f"conv-{index}", to_agent="sink_agent") for index in range(25)]

    async def run_once() -> list[Any]:
        router = FaultInjectionRouter(faults=faults)

        async def sink_agent(message: Any) -> None:
            return None

        router.register_agent("sink_agent", sink_agent)
        try:
            for message in messages:
                await router.send(message)
            await asyncio.sleep(0.01)
        finally:
            await router.stop()
        return router.fault_history

    first = await run_once()
    second = await run_once()

    assert first == second
    assert first
    assert len(first) > len(messages) // 4  # the rates actually inject faults
    assert any(record.kind is FaultKind.DROP for record in first)
    assert any(record.kind is FaultKind.CORRUPTION for record in first)
