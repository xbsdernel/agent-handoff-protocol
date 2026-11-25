"""Engine-facing contracts owned by the inference platform team.

Every inference engine (vLLM, SGLang, llama-server, or the mock server) is
reached through the same :class:`LLMClient` contract, so switching engines is
only a configuration change. The concrete client is implemented in Issue #07.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable


@dataclass(frozen=True, slots=True)
class LLMResult:
    """Normalised result of one streamed chat-completion call.

    ``ttft_ms`` and ``total_latency_ms`` are measured client-side with
    ``time.perf_counter_ns()`` so that the numbers are comparable across
    engines. ``structured_data`` is populated when the call requested a JSON
    schema; ``text`` always holds the raw generated text.
    """

    text: str
    structured_data: dict[str, Any] | None = None
    ttft_ms: float = 0.0
    total_latency_ms: float = 0.0
    tokens_per_second: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    engine_name: str = "unknown"
    model_name: str = "unknown"
    finish_reason: str | None = None


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
