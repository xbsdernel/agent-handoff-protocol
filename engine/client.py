"""Unified async streaming client for every OpenAI-compatible engine.

The same client talks to the mock server, vLLM, SGLang and llama-server; only
``base_url`` (and the model name) changes. Every call streams, which is what
makes client-side TTFT measurement meaningful:

* ``ttft_ms`` — from immediately before the HTTP request to the first non-empty
  content delta, measured with :func:`time.perf_counter_ns`;
* ``total_latency_ms`` — from request start to the final stream chunk;
* ``tokens_per_second`` — completion tokens divided by the decoding window
  (``total_latency_ms - ttft_ms``);
* token counts — taken from the final streaming usage chunk requested with
  ``stream_options={"include_usage": True}``.

When a JSON schema is supplied, the request asks the engine for structured
output and the returned document is validated with ``jsonschema`` before it is
exposed as :attr:`LLMResult.structured_data`. A malformed or non-conforming
document raises :class:`LLMResponseError` instead of handing bad data to an
agent.

Each completed call emits one standardized ``llm_call`` telemetry event.
"""

from __future__ import annotations

import json
import time
from collections.abc import Mapping
from typing import Any, Final, Literal
from urllib.parse import urlparse

import jsonschema
from openai import AsyncOpenAI

from engine.models import EngineName, LLMResult
from telemetry.interfaces import Event, EventSink

DEFAULT_BASE_URL: Final[str] = "http://127.0.0.1:8000/v1"
DEFAULT_MODEL: Final[str] = "mock-model"
DEFAULT_TIMEOUT_S: Final[float] = 600.0

#: Host ports used by docker/docker-compose.yml, for engine-name inference.
ENGINE_PORTS: Final[Mapping[int, str]] = {
    8001: EngineName.VLLM,
    8002: EngineName.SGLANG,
    8003: EngineName.LLAMA_SERVER,
    8000: EngineName.MOCK,
}

SchemaMode = Literal["json_schema", "json_object"]


class LLMResponseError(RuntimeError):
    """The model returned output that is not valid for the requested schema."""


class LLMCallEvent(Event):
    """Telemetry event emitted once per completion.

    Issue #17 will replace this local event with the shared strongly-typed
    telemetry models; the field names already match that contract.
    """

    engine_name: str = str(EngineName.UNKNOWN)
    model_name: str = "unknown"
    ttft_ms: float = 0.0
    latency_ms: float = 0.0
    tokens_per_second: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    finish_reason: str | None = None
    success: bool = True
    error: str | None = None


def infer_engine_name(base_url: str) -> str:
    """Infer the engine from the conventional local ports.

    Ports follow ``docker/docker-compose.yml`` (8001 vLLM, 8002 SGLang,
    8003 llama-server, 8000 mock). Links to remote hosts deliberately report
    ``"unknown"`` rather than guessing; pass ``engine_name=`` explicitly when
    the engine is not behind one of those ports.
    """

    parsed = urlparse(base_url)
    default_port = 443 if parsed.scheme == "https" else 80
    try:
        port = parsed.port or default_port
    except ValueError:
        return str(EngineName.UNKNOWN)
    name = ENGINE_PORTS.get(port)
    return str(name) if name is not None else str(EngineName.UNKNOWN)


def build_response_format(
    schema: Mapping[str, Any],
    mode: SchemaMode = "json_schema",
) -> dict[str, Any]:
    """Build the ``response_format`` payload for an engine.

    ``json_schema`` is the OpenAI-style structured-output request supported by
    vLLM, SGLang and the mock server. ``json_object`` is the fallback for
    engines that only understand plain JSON mode; the schema is then enforced
    client-side by :meth:`UnifiedLLMClient.chat`.
    """

    if mode == "json_object":
        return {"type": "json_object"}
    if mode != "json_schema":
        raise ValueError(f"unsupported schema mode: {mode!r}")
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "ahp_structured_output",
            "schema": dict(schema),
            "strict": False,
        },
    }


def _usage_value(usage: Any, name: str) -> int:
    if usage is None:
        return 0
    value = getattr(usage, name, None)
    if value is None and isinstance(usage, Mapping):
        value = usage.get(name)
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


class UnifiedLLMClient:
    """Streaming OpenAI-compatible client shared by every agent.

    Parameters
    ----------
    base_url:
        Engine endpoint including ``/v1``, for example
        ``http://localhost:8002/v1`` for SGLang.
    model:
        Model identifier sent with every request. Keep it aligned with the
        engine's served-model name so results are comparable.
    engine_name:
        Overrides port-based inference; used for telemetry tagging.
    event_sink:
        Optional telemetry sink receiving one ``llm_call`` event per call.
    schema_mode:
        ``json_schema`` (default) or ``json_object`` for engines that do not
        accept the structured-output extension.
    http_client:
        Optional preconfigured ``httpx.AsyncClient``. Tests use it to talk to
        the mock app in-process; production code normally leaves it unset.
    """

    def __init__(
        self,
        base_url: str = DEFAULT_BASE_URL,
        model: str = DEFAULT_MODEL,
        *,
        api_key: str = "EMPTY",
        engine_name: str | EngineName | None = None,
        event_sink: EventSink | None = None,
        run_id: str = "run",
        workflow_id: str | None = None,
        timeout_s: float = DEFAULT_TIMEOUT_S,
        max_retries: int = 0,
        include_usage: bool = True,
        schema_mode: SchemaMode = "json_schema",
        default_headers: Mapping[str, str] | None = None,
        client: AsyncOpenAI | None = None,
        http_client: Any | None = None,
    ) -> None:
        if not model:
            raise ValueError("model must not be empty")
        if timeout_s <= 0:
            raise ValueError("timeout_s must be > 0")
        if max_retries < 0:
            raise ValueError("max_retries must be >= 0")
        if schema_mode not in ("json_schema", "json_object"):
            raise ValueError(f"unsupported schema_mode: {schema_mode!r}")

        self.base_url = base_url
        self.model = model
        self.engine_name = (
            str(engine_name) if engine_name is not None else infer_engine_name(base_url)
        )
        self.run_id = run_id
        self.workflow_id = workflow_id
        self.timeout_s = timeout_s
        self.include_usage = include_usage
        self.schema_mode: SchemaMode = schema_mode

        self._event_sink = event_sink
        self._owns_client = client is None
        self._client = client or AsyncOpenAI(
            base_url=base_url,
            api_key=api_key,
            timeout=timeout_s,
            max_retries=max_retries,
            default_headers=dict(default_headers) if default_headers else None,
            http_client=http_client,
        )

    # -- public API ----------------------------------------------------------

    async def chat(
        self,
        messages: list[dict[str, Any]],
        schema: dict[str, Any] | None = None,
        temperature: float = 0.0,
    ) -> LLMResult:
        """Stream one completion and return text, usage and timing metadata."""

        if not messages:
            raise ValueError("messages must not be empty")

        started_ns = time.perf_counter_ns()
        first_token_ns: int | None = None
        parts: list[str] = []
        usage: Any = None
        finish_reason: str | None = None
        structured_data: dict[str, Any] | None = None
        failure: Exception | None = None

        try:
            request: dict[str, Any] = {
                "model": self.model,
                "messages": messages,
                "temperature": temperature,
                "stream": True,
            }
            if self.include_usage:
                request["stream_options"] = {"include_usage": True}
            if schema is not None:
                request["response_format"] = build_response_format(schema, self.schema_mode)

            stream = await self._client.chat.completions.create(**request)
            async for chunk in stream:
                chunk_usage = getattr(chunk, "usage", None)
                if chunk_usage is not None:
                    usage = chunk_usage

                choices = getattr(chunk, "choices", None) or []
                if not choices:
                    continue

                choice = choices[0]
                if choice.finish_reason is not None:
                    finish_reason = str(choice.finish_reason)
                delta = getattr(choice, "delta", None)
                content = getattr(delta, "content", None) if delta is not None else None
                if content:
                    if first_token_ns is None:
                        first_token_ns = time.perf_counter_ns()
                    parts.append(content)

            text = "".join(parts)
            if schema is not None:
                structured_data = self._parse_structured(text, schema)
        except Exception as exc:  # noqa: BLE001 - recorded, then re-raised unchanged
            failure = exc
            raise
        finally:
            finished_ns = time.perf_counter_ns()
            total_latency_ms = (finished_ns - started_ns) / 1_000_000
            ttft_ms = (
                (first_token_ns - started_ns) / 1_000_000 if first_token_ns is not None else 0.0
            )
            prompt_tokens = _usage_value(usage, "prompt_tokens")
            completion_tokens = _usage_value(usage, "completion_tokens")
            total_tokens = _usage_value(usage, "total_tokens")
            if total_tokens == 0:
                total_tokens = prompt_tokens + completion_tokens

            generation_ms = total_latency_ms - ttft_ms
            tokens_per_second = (
                completion_tokens / (generation_ms / 1000)
                if completion_tokens > 0 and generation_ms > 0
                else 0.0
            )
            await self._emit_call(
                success=failure is None,
                ttft_ms=ttft_ms,
                latency_ms=total_latency_ms,
                tokens_per_second=tokens_per_second,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
                finish_reason=finish_reason,
                error=f"{type(failure).__name__}: {failure}" if failure else None,
            )

        return LLMResult(
            text=text,
            structured_data=structured_data,
            ttft_ms=ttft_ms,
            total_latency_ms=total_latency_ms,
            tokens_per_second=tokens_per_second,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            engine_name=self.engine_name,
            model_name=self.model,
            finish_reason=finish_reason,
        )

    async def aclose(self) -> None:
        """Close the async HTTP client if this instance created it."""

        if self._owns_client:
            await self._client.close()

    async def __aenter__(self) -> UnifiedLLMClient:
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self.aclose()

    # -- helpers -------------------------------------------------------------

    @staticmethod
    def _parse_structured(text: str, schema: Mapping[str, Any]) -> dict[str, Any]:
        """Parse and validate a structured response, or raise ``LLMResponseError``."""

        try:
            document = json.loads(text)
        except json.JSONDecodeError as exc:
            raise LLMResponseError(
                f"model output is not valid JSON: {exc.msg} (column {exc.colno})"
            ) from exc

        if not isinstance(document, dict):
            raise LLMResponseError(
                f"structured output must be a JSON object, got {type(document).__name__}"
            )

        try:
            jsonschema.validate(instance=document, schema=dict(schema))
        except jsonschema.ValidationError as exc:
            location = ".".join(str(part) for part in exc.absolute_path) or "<root>"
            raise LLMResponseError(
                f"model output violates the response schema at {location}: {exc.message}"
            ) from exc
        return document

    async def _emit_call(self, **fields: Any) -> None:
        """Emit one telemetry event; telemetry must never break inference."""

        if self._event_sink is None:
            return
        event = LLMCallEvent(
            run_id=self.run_id,
            workflow_id=self.workflow_id,
            event_type="llm_call",
            engine_name=self.engine_name,
            model_name=self.model,
            **fields,
        )
        try:
            await self._event_sink.emit(event)
        except Exception:  # noqa: BLE001 - telemetry is best-effort
            return


__all__ = [
    "DEFAULT_BASE_URL",
    "DEFAULT_MODEL",
    "DEFAULT_TIMEOUT_S",
    "ENGINE_PORTS",
    "LLMCallEvent",
    "LLMResponseError",
    "SchemaMode",
    "UnifiedLLMClient",
    "build_response_format",
    "infer_engine_name",
]
