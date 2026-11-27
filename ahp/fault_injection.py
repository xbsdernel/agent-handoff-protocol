"""Deterministic fault injection for AHP transport robustness testing.

:class:`FaultInjectionRouter` is a drop-in :class:`~ahp.router.InProcessRouter`
that injects controlled transport faults at delivery time:

* **drops** — the message never reaches the recipient, so the sender observes
  an ACK timeout and follows its bounded retry/escalation policy;
* **delays** — an artificial sleep models slow links;
* **corruptions** — the JSON document is mutated into an invalid AHP payload
  (missing field, wrong type, unknown field, bad discriminator or malformed
  JSON) and rejected by validation before any agent sees it;
* **duplications** — an identical copy is delivered twice to exercise the
  ``message_id`` deduplication rule.

All decisions come from one :class:`random.Random` instance seeded through
:class:`FaultConfig`, so a fixed seed reproduces the exact same fault sequence
across runs. Every injected fault is appended to
:attr:`FaultInjectionRouter.fault_history` and emitted to the ``EventSink``.

Example::

    router = FaultInjectionRouter(
        RouterConfig(ack_timeout_s=2.0),
        event_sink=sink,
        faults=FaultConfig(drop_rate=0.1, corruption_rate=0.05, seed=7),
    )
"""

from __future__ import annotations

import asyncio
import random
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Final

from pydantic import ValidationError

from ahp.models import (
    AHPErrorCode,
    AHPMessage,
    HandoffMessage,
    MessageEnvelope,
    NeedInfoMessage,
    parse_ahp_message,
)
from ahp.router import InProcessRouter, RouterConfig
from telemetry.interfaces import EventSink

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


class FaultKind(StrEnum):
    """Categories of injected fault, as recorded in the fault history."""

    DELAY = "delay"
    DROP = "drop"
    CORRUPTION = "corruption"
    DUPLICATION = "duplication"


class CorruptionPolicy(StrEnum):
    """What to do after a corrupted payload fails validation.

    ``DROP`` treats the corrupted delivery as lost and lets the sender's ACK
    timeout/retry policy recover. ``NEED_INFO`` additionally asks the sender for
    a valid re-send, mirroring the protocol rule that a receiver which cannot
    parse a handoff must name what is missing instead of guessing.
    """

    DROP = "drop"
    NEED_INFO = "need_info"


@dataclass(frozen=True, slots=True)
class FaultConfig:
    """Probabilities and limits for :class:`FaultInjectionRouter`.

    ``delay_range`` is ``(low, high)`` seconds sampled uniformly. All rates are
    probabilities in ``[0, 1]``.
    """

    drop_rate: float = 0.0
    delay_range: tuple[float, float] = (0.0, 0.0)
    corruption_rate: float = 0.0
    duplicate_rate: float = 0.0
    corruption_policy: CorruptionPolicy = CorruptionPolicy.DROP
    seed: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "corruption_policy", CorruptionPolicy(self.corruption_policy))

        for name in ("drop_rate", "corruption_rate", "duplicate_rate"):
            value = getattr(self, name)
            if not isinstance(value, int | float) or not 0.0 <= float(value) <= 1.0:
                raise ValueError(f"{name} must be a probability between 0 and 1, got {value!r}")

        low, high = self.delay_range
        if low < 0.0 or high < 0.0:
            raise ValueError(f"delay_range values must be >= 0, got {self.delay_range!r}")
        if high < low:
            raise ValueError(
                f"delay_range must be (low, high) with low <= high, got {self.delay_range!r}"
            )


@dataclass(frozen=True, slots=True)
class FaultRecord:
    """One injected fault, kept for reproducibility checks and assertions."""

    kind: FaultKind
    message_id: str
    attempt: int
    strategy: str = ""
    delay_s: float = 0.0


# ---------------------------------------------------------------------------
# Corruption
# ---------------------------------------------------------------------------

#: Strategies :func:`corrupt_document` may choose from.
CORRUPTION_STRATEGIES: Final[tuple[str, ...]] = (
    "missing_field",
    "wrong_type",
    "unknown_field",
    "invalid_type_tag",
    "malformed_json",
)

#: Payload fields whose removal always makes the document invalid.
_REQUIRED_PAYLOAD_FIELDS: Final[dict[str, tuple[str, ...]]] = {
    "HANDOFF": ("goal", "summary"),
    "ACK": ("acknowledged_message_id",),
    "NEED_INFO": ("missing_fields", "query"),
    "DONE": ("final_result", "summary"),
}


def corrupt_document(message: MessageEnvelope, rng: random.Random) -> tuple[str, Any]:
    """Return ``(strategy, payload)`` where ``payload`` is an invalid AHP message.

    ``payload`` is either a JSON-compatible mapping or a malformed JSON string,
    matching what a corrupted transport could hand to the receiver. The
    strategy is chosen with ``rng`` so a seeded router is reproducible.
    """

    strategy = rng.choice(CORRUPTION_STRATEGIES)

    if strategy == "malformed_json":
        return strategy, '{"type": "HANDOFF", "message_id": '

    document = message.model_dump(mode="json")

    if strategy == "missing_field":
        document.pop(rng.choice(_REQUIRED_PAYLOAD_FIELDS[str(message.type)]), None)
    elif strategy == "wrong_type":
        document["hop_count"] = "not-a-number"
    elif strategy == "unknown_field":
        document["fault_injected"] = True
    else:  # invalid_type_tag
        document["type"] = "UNKNOWN_TYPE"

    return strategy, document


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------


class FaultInjectionRouter(InProcessRouter):
    """An :class:`InProcessRouter` that injects seeded transport faults.

    Faults are applied in :meth:`FaultInjectionRouter._deliver`, the transport
    boundary shared by initial handoffs and agent-to-agent forwards. ``ACK`` and
    ``DONE`` messages are control messages handled inside the reference router
    and bypass this boundary, matching the fact that they are not re-sent by an
    agent's inbox loop.
    """

    def __init__(
        self,
        config: RouterConfig | None = None,
        event_sink: EventSink | None = None,
        *,
        faults: FaultConfig | None = None,
        run_id: str = "run",
    ) -> None:
        super().__init__(config, event_sink, run_id=run_id)
        self.faults = faults or FaultConfig()
        self.fault_history: list[FaultRecord] = []
        self._rng = random.Random(self.faults.seed)

    # -- fault decisions -----------------------------------------------------

    def _roll(self, rate: float) -> bool:
        """Consume one RNG draw and return whether the fault triggers."""

        return rate > 0.0 and self._rng.random() < rate

    def _record(
        self,
        kind: FaultKind,
        message: MessageEnvelope,
        attempt: int,
        *,
        strategy: str = "",
        delay_s: float = 0.0,
    ) -> None:
        self.fault_history.append(
            FaultRecord(
                kind=kind,
                message_id=str(message.message_id),
                attempt=attempt,
                strategy=strategy,
                delay_s=delay_s,
            )
        )

    # -- transport -----------------------------------------------------------

    async def _deliver(self, message: AHPMessage, *, attempt: int = 1) -> None:
        faults = self.faults

        low, high = faults.delay_range
        if high > 0.0:
            delay = self._rng.uniform(low, high)
            self._record(FaultKind.DELAY, message, attempt, delay_s=delay)
            await self._emit(
                "fault.delivery_delayed",
                message=message,
                attempt=attempt,
                detail=f"{delay:.4f}s",
            )
            await asyncio.sleep(delay)

        if self._roll(faults.drop_rate):
            self._record(FaultKind.DROP, message, attempt)
            await self._emit(
                "fault.message_dropped",
                message=message,
                attempt=attempt,
                detail=f"drop_rate={faults.drop_rate}",
            )
            return

        if self._roll(faults.corruption_rate):
            strategy, payload = corrupt_document(message, self._rng)
            self._record(FaultKind.CORRUPTION, message, attempt, strategy=strategy)
            await self._emit(
                "fault.message_corrupted",
                message=message,
                attempt=attempt,
                detail=strategy,
            )
            await self._reject_corrupted(message, payload, strategy, attempt=attempt)
            return

        await super()._deliver(message, attempt=attempt)

        if self._roll(faults.duplicate_rate):
            self._record(FaultKind.DUPLICATION, message, attempt)
            await self._emit("fault.message_duplicated", message=message, attempt=attempt)
            await super()._deliver(message, attempt=attempt)

    async def _reject_corrupted(
        self,
        original: MessageEnvelope,
        payload: Any,
        strategy: str,
        *,
        attempt: int,
    ) -> None:
        """Validate the corrupted payload and turn the rejection into recovery."""

        try:
            parse_ahp_message(payload)
        except ValidationError as exc:
            errors = exc.errors()
            first_error = str(errors[0]["type"]) if errors else "validation_error"
            await self._emit(
                "handoff.rejected",
                message=original,
                attempt=attempt,
                error_code=AHPErrorCode.SCHEMA_VIOLATION,
                detail=f"{strategy}: {first_error}",
                valid_schema=False,
            )
            await self._request_resend(original, attempt=attempt)
            return

        # Defensive: corruption strategies must always produce an invalid
        # document. If one does not, deliver what the transport produced.
        await super()._deliver(parse_ahp_message(payload), attempt=attempt)

    async def _request_resend(self, message: MessageEnvelope, *, attempt: int) -> None:
        """Apply the corruption policy: drop silently or ask the sender again."""

        if self.faults.corruption_policy is not CorruptionPolicy.NEED_INFO:
            return
        if not isinstance(message, HandoffMessage | NeedInfoMessage):
            return
        if message.from_agent not in self._handlers:
            await self._emit(
                "fault.need_info_undeliverable",
                message=message,
                attempt=attempt,
                detail=f"agent {message.from_agent!r} is not registered",
            )
            return

        need_info = NeedInfoMessage(
            conversation_id=message.conversation_id,
            from_agent=message.to_agent,
            to_agent=message.from_agent,
            hop_count=message.hop_count,
            missing_fields=["payload"],
            query="The handoff arrived corrupted; please re-send it.",
        )
        # Recovery traffic is reliable: bypass fault injection so that a 100%
        # corruption rate cannot loop forever.
        await super()._deliver(need_info, attempt=attempt)
        await self._emit(
            "fault.need_info_sent",
            message=need_info,
            attempt=attempt,
            detail="corrupted handoff",
        )


__all__ = [
    "CORRUPTION_STRATEGIES",
    "CorruptionPolicy",
    "FaultConfig",
    "FaultInjectionRouter",
    "FaultKind",
    "FaultRecord",
    "corrupt_document",
]
