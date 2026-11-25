"""Pydantic v2 message models for the Agent Handoff Protocol (AHP).

A message is a JSON object with a shared envelope (identity, routing and
conversation metadata) plus a payload that depends on ``type``. The four
message types form a discriminated union (:data:`AHPMessage`) so a raw JSON
document can be parsed directly into the correct concrete model.

The models are frozen: messages are immutable once created. Forwarding an
existing message (for example to increment ``hop_count``) uses
``model_copy(update=...)``.
"""

from __future__ import annotations

import time
from collections.abc import Mapping
from enum import StrEnum
from typing import Annotated, Any, Final, Literal
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field, JsonValue, TypeAdapter

# ---------------------------------------------------------------------------
# Vocabulary
# ---------------------------------------------------------------------------


class MessageType(StrEnum):
    """Values of the envelope ``type`` discriminator."""

    HANDOFF = "HANDOFF"
    ACK = "ACK"
    NEED_INFO = "NEED_INFO"
    DONE = "DONE"


class AckStatus(StrEnum):
    """Whether a receiver accepted a handoff or is too busy to process it."""

    ACCEPTED = "accepted"
    BUSY = "busy"


class OnFailurePolicy(StrEnum):
    """What the sender wants the receiver to do when the task fails."""

    RETRY = "retry"
    ESCALATE_TO_HUMAN = "escalate_to_human"
    STOP = "stop"


class DoneStatus(StrEnum):
    """Terminal outcome reported by a ``DONE`` message."""

    RESOLVED = "resolved"
    ESCALATED_UNRESOLVED = "escalated_unresolved"


class AHPErrorCode(StrEnum):
    """Stable error codes shared by validation, routing and fault injection."""

    INVALID_MESSAGE = "INVALID_MESSAGE"
    MISSING_FIELD = "MISSING_FIELD"
    INVALID_TYPE = "INVALID_TYPE"
    INVALID_VALUE = "INVALID_VALUE"
    UNKNOWN_MESSAGE_TYPE = "UNKNOWN_MESSAGE_TYPE"
    UNKNOWN_FIELD = "UNKNOWN_FIELD"
    DUPLICATE_MESSAGE = "DUPLICATE_MESSAGE"
    ACK_TIMEOUT = "ACK_TIMEOUT"
    HOP_LIMIT_EXCEEDED = "HOP_LIMIT_EXCEEDED"
    UNKNOWN_AGENT = "UNKNOWN_AGENT"
    SCHEMA_VIOLATION = "SCHEMA_VIOLATION"
    AGENT_ERROR = "AGENT_ERROR"


# ---------------------------------------------------------------------------
# Envelope and payloads
# ---------------------------------------------------------------------------


class MessageEnvelope(BaseModel):
    """Fields every AHP message carries.

    Concrete messages narrow ``type`` to a single literal, which is what makes
    the discriminated union work.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    message_id: UUID = Field(
        default_factory=uuid4,
        description="Globally unique message identifier; used for deduplication.",
    )
    conversation_id: str = Field(
        min_length=1,
        description="Identifier shared by every message in one workflow run.",
    )
    timestamp_ns: int = Field(
        default_factory=time.time_ns,
        ge=0,
        description="Sender clock in nanoseconds since the Unix epoch.",
    )
    from_agent: str = Field(min_length=1, description="Sending agent name.")
    to_agent: str = Field(min_length=1, description="Receiving agent name.")
    hop_count: int = Field(
        default=0,
        ge=0,
        description="Number of handoff hops already taken in this conversation.",
    )
    type: MessageType


class HandoffMessage(MessageEnvelope):
    """A request to perform work, carrying everything the receiver needs."""

    type: Literal[MessageType.HANDOFF] = MessageType.HANDOFF

    goal: str = Field(min_length=1, description="What the receiver must achieve.")
    summary: str = Field(min_length=1, description="Compact context for the receiver.")
    facts: dict[str, JsonValue] = Field(
        default_factory=dict,
        description="Machine-readable facts accumulated so far.",
    )
    done_so_far: list[str] = Field(
        default_factory=list,
        description="Human-readable steps already completed.",
    )
    expected_output: dict[str, JsonValue] = Field(
        default_factory=dict,
        description="Shape or description of the required answer.",
    )
    constraints: dict[str, JsonValue] = Field(
        default_factory=dict,
        description="Deadlines, budgets, policies and other limits.",
    )
    on_failure: OnFailurePolicy = Field(
        default=OnFailurePolicy.ESCALATE_TO_HUMAN,
        description="Preferred recovery policy if the task cannot be completed.",
    )


class AckMessage(MessageEnvelope):
    """Confirmation that a handoff was received."""

    type: Literal[MessageType.ACK] = MessageType.ACK

    acknowledged_message_id: UUID = Field(
        description="``message_id`` of the handoff being acknowledged.",
    )
    status: AckStatus = Field(
        default=AckStatus.ACCEPTED,
        description="Whether the receiver accepted the task or is busy.",
    )


class NeedInfoMessage(MessageEnvelope):
    """A receiver cannot proceed because required information is missing."""

    type: Literal[MessageType.NEED_INFO] = MessageType.NEED_INFO

    missing_fields: list[str] = Field(
        min_length=1,
        description="Names of the fields or facts that must be supplied.",
    )
    query: str = Field(min_length=1, description="Question back to the sender.")


class DoneMessage(MessageEnvelope):
    """Terminal result of a conversation."""

    type: Literal[MessageType.DONE] = MessageType.DONE

    final_result: dict[str, JsonValue] = Field(
        description="The answer, decision or artifact produced by the workflow.",
    )
    summary: str = Field(min_length=1, description="Short human-readable wrap-up.")
    execution_metrics: dict[str, JsonValue] = Field(
        default_factory=dict,
        description="Optional agent-reported counters such as tool calls or cycles.",
    )
    status: DoneStatus = Field(
        default=DoneStatus.RESOLVED,
        description="Whether the task was resolved or escalated unresolved.",
    )


type AHPMessage = Annotated[
    HandoffMessage | AckMessage | NeedInfoMessage | DoneMessage,
    Field(discriminator="type"),
]

#: Runtime adapter used to validate/parse the discriminated union.
AHPMessageAdapter: Final[TypeAdapter[Any]] = TypeAdapter(AHPMessage)


def parse_ahp_message(data: str | bytes | Mapping[str, Any]) -> AHPMessage:
    """Parse a JSON string/bytes document or a mapping into an AHP message.

    Raises :class:`pydantic.ValidationError` when the document is malformed,
    has an unknown ``type`` or violates a field constraint.
    """

    if isinstance(data, str | bytes):
        return AHPMessageAdapter.validate_json(data)
    return AHPMessageAdapter.validate_python(data)


def dump_ahp_message(message: MessageEnvelope) -> str:
    """Serialise an AHP message to compact JSON."""

    return message.model_dump_json()


__all__ = [
    "AHPMessage",
    "AHPMessageAdapter",
    "AHPErrorCode",
    "AckMessage",
    "AckStatus",
    "DoneMessage",
    "DoneStatus",
    "HandoffMessage",
    "MessageEnvelope",
    "MessageType",
    "NeedInfoMessage",
    "OnFailurePolicy",
    "dump_ahp_message",
    "parse_ahp_message",
]
