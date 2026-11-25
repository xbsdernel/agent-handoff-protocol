"""Agent Handoff Protocol (AHP): message models, validation and routing.

The concrete message models and router live in subsequent issues. This package
currently exposes the Day-1 agent contract defined in :mod:`ahp.interfaces`.
"""

from ahp.interfaces import Agent

__all__ = ["Agent"]
