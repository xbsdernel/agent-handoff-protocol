"""Engine-facing contracts owned by the inference platform team.

Every inference engine (vLLM, SGLang, llama-server, or the mock server) is
reached through the same :class:`LLMClient` contract, so switching engines is
only a configuration change. :class:`~engine.models.LLMResult` is re-exported
here for callers that import the contract and its result type together.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from engine.models import LLMResult


@runtime_checkable
class LLMClient(Protocol):
    """Single async entry point for every model call in the harness."""

    async def chat(
        self,
        messages: list[dict[str, Any]],
        schema: dict[str, Any] | None = None,
        temperature: float = 0.0,
    ) -> LLMResult:
        """Stream one completion and return text, usage and timing metadata.

        When ``schema`` is supplied the client must request structured JSON
        output and populate :attr:`LLMResult.structured_data`.
        """
        ...

    async def aclose(self) -> None:
        """Release the underlying HTTP resources."""
        ...


__all__ = ["LLMClient", "LLMResult"]
