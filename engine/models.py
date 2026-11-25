"""Normalised inference results and engine identity.

``LLMResult`` is the engine-neutral output of one streamed chat completion.
Keeping it in its own module lets the interface contract, the concrete client
and the harness share one type without import cycles.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any


class EngineName(StrEnum):
    """Inference engines the harness can talk to over the OpenAI API."""

    VLLM = "vllm"
    SGLANG = "sglang"
    LLAMA_SERVER = "llama-server"
    MOCK = "mock"
    UNKNOWN = "unknown"


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
    engine_name: str = EngineName.UNKNOWN
    model_name: str = "unknown"
    finish_reason: str | None = None

    @property
    def generation_ms(self) -> float:
        """Time spent decoding after the first token (never negative)."""

        return max(0.0, self.total_latency_ms - self.ttft_ms)


__all__ = ["EngineName", "LLMResult"]
