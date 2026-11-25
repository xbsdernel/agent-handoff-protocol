"""Public interfaces owned by the AHP protocol layer.

The concrete AHP message models (``HandoffMessage``, ``AHPMessage``, ...) are
introduced by Issue #03. They are referenced here under ``TYPE_CHECKING`` only,
so this contract can be imported and used before those models exist, and no
runtime import cycle is created.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from ahp.models import AHPMessage, HandoffMessage


@runtime_checkable
class Agent(Protocol):
    """Contract every workflow agent must satisfy.

    An agent receives a validated handoff message and responds with exactly one
    AHP message (``ACK``, ``NEED_INFO``, another ``HANDOFF``, or ``DONE``).
    """

    name: str

    async def handle_handoff(self, message: HandoffMessage) -> AHPMessage:
        """Process ``message`` and return the next AHP message for the router."""
        ...


__all__ = ["Agent"]
