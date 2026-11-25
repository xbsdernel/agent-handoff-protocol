"""Agent Handoff Protocol (AHP): message models, validation and routing.

This package exposes the protocol vocabulary and the four message types. The
router and state machine live in :mod:`ahp.router` and
:mod:`ahp.state_machine` (Issue #4).
"""

from ahp.interfaces import Agent
from ahp.models import (
    AckMessage,
    AckStatus,
    AHPErrorCode,
    AHPMessage,
    AHPMessageAdapter,
    DoneMessage,
    DoneStatus,
    HandoffMessage,
    MessageEnvelope,
    MessageType,
    NeedInfoMessage,
    OnFailurePolicy,
    dump_ahp_message,
    parse_ahp_message,
)

__all__ = [
    "AHPMessage",
    "AHPMessageAdapter",
    "AHPErrorCode",
    "AckMessage",
    "AckStatus",
    "Agent",
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
