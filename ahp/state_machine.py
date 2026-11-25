"""Deterministic lifecycle state machine for one AHP conversation.

The router owns one :class:`WorkflowStateMachine` per ``conversation_id`` and
uses it to track progress for telemetry and debugging. Transitions are strict:
an illegal transition raises :class:`InvalidStateTransition` instead of
silently corrupting the state.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import StrEnum


class WorkflowState(StrEnum):
    """Lifecycle states defined by AHP 1.0."""

    INITIATED = "INITIATED"
    IN_TRANSIT = "IN_TRANSIT"
    PROCESSING = "PROCESSING"
    BLOCKED_NEED_INFO = "BLOCKED_NEED_INFO"
    COMPLETED = "COMPLETED"
    ESCALATED = "ESCALATED"
    FAILED = "FAILED"


TERMINAL_STATES: frozenset[WorkflowState] = frozenset(
    {WorkflowState.COMPLETED, WorkflowState.ESCALATED, WorkflowState.FAILED}
)

#: States reachable from each non-terminal state.
ALLOWED_TRANSITIONS: dict[WorkflowState, frozenset[WorkflowState]] = {
    WorkflowState.INITIATED: frozenset(
        {WorkflowState.IN_TRANSIT, WorkflowState.ESCALATED, WorkflowState.FAILED}
    ),
    WorkflowState.IN_TRANSIT: frozenset(
        {
            WorkflowState.PROCESSING,
            WorkflowState.BLOCKED_NEED_INFO,
            WorkflowState.ESCALATED,
            WorkflowState.FAILED,
        }
    ),
    WorkflowState.PROCESSING: frozenset(
        {
            WorkflowState.IN_TRANSIT,
            WorkflowState.BLOCKED_NEED_INFO,
            WorkflowState.COMPLETED,
            WorkflowState.ESCALATED,
            WorkflowState.FAILED,
        }
    ),
    WorkflowState.BLOCKED_NEED_INFO: frozenset(
        {
            WorkflowState.PROCESSING,
            WorkflowState.IN_TRANSIT,
            WorkflowState.ESCALATED,
            WorkflowState.FAILED,
        }
    ),
    WorkflowState.COMPLETED: frozenset(),
    WorkflowState.ESCALATED: frozenset(),
    WorkflowState.FAILED: frozenset(),
}


class InvalidStateTransition(RuntimeError):
    """Raised when a state machine is asked to make an illegal transition."""

    def __init__(
        self,
        workflow_id: str,
        current: WorkflowState,
        requested: WorkflowState,
    ) -> None:
        super().__init__(
            f"Workflow {workflow_id!r} cannot transition from {current} to {requested}"
        )
        self.workflow_id = workflow_id
        self.current = current
        self.requested = requested


@dataclass(frozen=True, slots=True)
class TransitionRecord:
    """Audit entry for one successful transition."""

    from_state: WorkflowState
    to_state: WorkflowState
    timestamp_ns: int
    hop_count: int
    reason: str = ""


@dataclass(slots=True)
class WorkflowStateMachine:
    """State and hop bookkeeping for a single conversation."""

    workflow_id: str
    state: WorkflowState = WorkflowState.INITIATED
    hop_count: int = 0
    history: list[TransitionRecord] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.workflow_id:
            raise ValueError("workflow_id must not be empty")

    @property
    def is_terminal(self) -> bool:
        return self.state in TERMINAL_STATES

    def can_transition(self, target: WorkflowState) -> bool:
        """Return whether ``target`` is reachable from the current state."""

        return target in ALLOWED_TRANSITIONS[self.state]

    def transition(
        self,
        target: WorkflowState,
        *,
        reason: str = "",
        hop_count: int | None = None,
    ) -> WorkflowState:
        """Move to ``target`` and record the transition.

        Transitioning to the current state is a no-op (it only updates the hop
        counter), which keeps repeated protocol events harmless.
        """

        if hop_count is not None:
            if hop_count < 0:
                raise ValueError("hop_count must be >= 0")
            self.hop_count = max(self.hop_count, hop_count)

        if target == self.state:
            return self.state

        if not self.can_transition(target):
            raise InvalidStateTransition(self.workflow_id, self.state, target)

        record = TransitionRecord(
            from_state=self.state,
            to_state=target,
            timestamp_ns=time.time_ns(),
            hop_count=self.hop_count,
            reason=reason,
        )
        self.state = target
        self.history.append(record)
        return self.state


__all__ = [
    "ALLOWED_TRANSITIONS",
    "TERMINAL_STATES",
    "InvalidStateTransition",
    "TransitionRecord",
    "WorkflowState",
    "WorkflowStateMachine",
]
