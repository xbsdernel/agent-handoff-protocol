"""Agent Handoff Protocol (AHP): message models, validation, routing and state.

This package exposes the protocol vocabulary, the four message types, the
lifecycle state machine and the in-process router.
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
from ahp.router import (
    AckTimeoutError,
    AgentError,
    HopLimitExceeded,
    InProcessRouter,
    RouterConfig,
    RouterError,
    UnknownAgentError,
)
from ahp.state_machine import (
    InvalidStateTransition,
    WorkflowState,
    WorkflowStateMachine,
)

__all__ = [
    "AHPMessage",
    "AHPMessageAdapter",
    "AHPErrorCode",
    "AckMessage",
    "AckStatus",
    "AckTimeoutError",
    "Agent",
    "AgentError",
    "DoneMessage",
    "DoneStatus",
    "HandoffMessage",
    "HopLimitExceeded",
    "InProcessRouter",
    "InvalidStateTransition",
    "MessageEnvelope",
    "MessageType",
    "NeedInfoMessage",
    "OnFailurePolicy",
    "RouterConfig",
    "RouterError",
    "UnknownAgentError",
    "WorkflowState",
    "WorkflowStateMachine",
    "dump_ahp_message",
    "parse_ahp_message",
]
