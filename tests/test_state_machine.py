"""Unit tests for the AHP lifecycle state machine (Issue #4)."""

from __future__ import annotations

import pytest

from ahp.state_machine import (
    ALLOWED_TRANSITIONS,
    TERMINAL_STATES,
    InvalidStateTransition,
    WorkflowState,
    WorkflowStateMachine,
)


def test_happy_path_reaches_completed() -> None:
    machine = WorkflowStateMachine("conv-1")

    machine.transition(WorkflowState.IN_TRANSIT)
    machine.transition(WorkflowState.PROCESSING)
    machine.transition(WorkflowState.IN_TRANSIT)
    machine.transition(WorkflowState.PROCESSING)
    machine.transition(WorkflowState.COMPLETED, reason="done")

    assert machine.state is WorkflowState.COMPLETED
    assert machine.is_terminal
    assert len(machine.history) == 5
    assert machine.history[-1].reason == "done"


def test_need_info_path() -> None:
    machine = WorkflowStateMachine("conv-1")

    machine.transition(WorkflowState.IN_TRANSIT)
    machine.transition(WorkflowState.PROCESSING)
    machine.transition(WorkflowState.BLOCKED_NEED_INFO)
    machine.transition(WorkflowState.PROCESSING)
    machine.transition(WorkflowState.IN_TRANSIT)
    machine.transition(WorkflowState.PROCESSING)
    machine.transition(WorkflowState.COMPLETED)

    assert machine.state is WorkflowState.COMPLETED


def test_invalid_transition_is_rejected() -> None:
    machine = WorkflowStateMachine("conv-1")

    with pytest.raises(InvalidStateTransition) as excinfo:
        machine.transition(WorkflowState.COMPLETED)

    assert excinfo.value.current is WorkflowState.INITIATED
    assert excinfo.value.requested is WorkflowState.COMPLETED
    assert machine.state is WorkflowState.INITIATED
    assert machine.history == []


def test_same_state_is_a_noop() -> None:
    machine = WorkflowStateMachine("conv-1")
    machine.transition(WorkflowState.IN_TRANSIT)

    machine.transition(WorkflowState.IN_TRANSIT, reason="duplicate")

    assert machine.state is WorkflowState.IN_TRANSIT
    assert len(machine.history) == 1


def test_terminal_states_have_no_outgoing_transitions() -> None:
    for state in TERMINAL_STATES:
        machine = WorkflowStateMachine("conv-1", state=state)
        assert machine.is_terminal
        assert ALLOWED_TRANSITIONS[state] == frozenset()
        with pytest.raises(InvalidStateTransition):
            machine.transition(WorkflowState.PROCESSING)


def test_hop_count_is_monotonic() -> None:
    machine = WorkflowStateMachine("conv-1")

    machine.transition(WorkflowState.IN_TRANSIT, hop_count=2)
    machine.transition(WorkflowState.PROCESSING, hop_count=1)

    assert machine.hop_count == 2
    assert machine.history[-1].hop_count == 2


def test_negative_hop_count_is_rejected() -> None:
    machine = WorkflowStateMachine("conv-1")

    with pytest.raises(ValueError):
        machine.transition(WorkflowState.IN_TRANSIT, hop_count=-1)


def test_empty_workflow_id_is_rejected() -> None:
    with pytest.raises(ValueError):
        WorkflowStateMachine("")
