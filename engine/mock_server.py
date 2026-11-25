"""Mock OpenAI-compatible server for GPU-free development and tests.

Implements the subset of the OpenAI Chat Completions API that the harness
uses:

* ``POST /v1/chat/completions`` with ``stream=true`` (Server-Sent Events) or a
  regular JSON response;
* ``GET /v1/models`` and ``GET /v1/models/{model}``;
* ``GET /health`` and ``GET/POST /mock/config`` for runtime control.

Behaviour can be programmed at three levels, from lowest to highest
precedence:

1. the :class:`MockServerConfig` passed to :func:`create_app` (or the CLI);
2. per-request headers: ``x-mock-mode``, ``x-mock-ttft-ms``,
   ``x-mock-token-delay-ms``, ``x-mock-error`` and ``x-mock-corrupt-json``;
3. structured-output requests, which are answered with a JSON document that
   satisfies the supplied JSON Schema.

Run the naked server with::

    python -m engine.mock_server --port 8000
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import time
import uuid
from collections.abc import AsyncIterator, Mapping
from typing import Any, Literal

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field

DEFAULT_MODEL = "mock-model"
DEFAULT_RESPONSE_TEXT = "This is a mock response from the AHP mock server."

ReplyMode = Literal["template", "echo", "json"]


class MockServerConfig(BaseModel):
    """Programmable defaults for every request served by the mock."""

    model_config = ConfigDict(extra="forbid")

    model: str = DEFAULT_MODEL
    ttft_ms: float = Field(default=50.0, ge=0.0)
    inter_token_delay_ms: float = Field(default=10.0, ge=0.0)
    mode: ReplyMode = "template"
    response_text: str = DEFAULT_RESPONSE_TEXT
    json_response: Any = Field(default_factory=lambda: {"status": "ok"})
    error_status: int | None = Field(default=None, ge=400, le=599)
    error_message: str = "mock server error"
    corrupt_json: bool = False
    always_include_usage: bool = False


class ChatMessage(BaseModel):
    model_config = ConfigDict(extra="allow")

    role: str
    content: str | list[dict[str, Any]] | None = None


class ResponseFormat(BaseModel):
    model_config = ConfigDict(extra="allow")

    type: str = "text"
    json_schema: dict[str, Any] | None = None


class ChatCompletionRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    model: str = DEFAULT_MODEL
    messages: list[ChatMessage] = Field(min_length=1)
    stream: bool = False
    temperature: float | None = None
    max_tokens: int | None = None
    response_format: ResponseFormat | None = None
    stream_options: dict[str, Any] | None = None
    guided_json: dict[str, Any] | None = None


# ---------------------------------------------------------------------------
# JSON Schema sampling
# ---------------------------------------------------------------------------

_MAX_SCHEMA_DEPTH = 32

_UUID_PLACEHOLDER = "00000000-0000-4000-8000-000000000001"


def _resolve_ref(root: Mapping[str, Any], ref: str) -> Mapping[str, Any]:
    if not ref.startswith("#/"):
        raise ValueError(f"Unsupported external $ref: {ref!r}")
    node: Any = root
    for raw_part in ref[2:].split("/"):
        part = raw_part.replace("~1", "/").replace("~0", "~")
        if not isinstance(node, Mapping) or part not in node:
            raise ValueError(f"Unresolvable $ref: {ref!r}")
        node = node[part]
    if not isinstance(node, Mapping):
        raise ValueError(f"$ref does not point to a schema object: {ref!r}")
    return node


def _first_non_null_option(options: list[Any]) -> Any:
    for option in options:
        if isinstance(option, Mapping) and option.get("type") == "null":
            continue
        return option
    return options[0] if options else {}


def _merge_all_of(root: Mapping[str, Any], parts: list[Any]) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for part in parts:
        if not isinstance(part, Mapping):
            continue
        sample = sample_from_schema(part, root=root)
        if isinstance(sample, Mapping):
            merged.update(sample)
    return merged


def sample_from_schema(schema: Mapping[str, Any], root: Mapping[str, Any] | None = None) -> Any:
    """Return a deterministic value that validates against ``schema``.

    The generator covers the JSON Schema features used by the AHP models and
    by tests: ``$ref``/``$defs``, ``const``, ``enum``, ``anyOf``/``oneOf``,
    ``allOf``, objects, arrays, primitives and common string formats.
    Unknown constraints are ignored, so the generated value may not satisfy
    exotic keywords such as ``pattern``.
    """

    if root is None:
        root = schema

    return _sample(schema, root, depth=0)


def _sample(schema: Mapping[str, Any], root: Mapping[str, Any], depth: int) -> Any:
    if depth > _MAX_SCHEMA_DEPTH:
        raise ValueError("JSON Schema nesting is too deep to sample")

    if "$ref" in schema:
        return _sample(_resolve_ref(root, str(schema["$ref"])), root, depth + 1)

    if "const" in schema:
        return schema["const"]

    if "enum" in schema and schema["enum"]:
        return schema["enum"][0]

    if "default" in schema:
        return schema["default"]

    for keyword in ("anyOf", "oneOf"):
        options = schema.get(keyword)
        if isinstance(options, list) and options:
            chosen = _first_non_null_option(options)
            return _sample(chosen, root, depth + 1)

    if isinstance(schema.get("allOf"), list):
        return _merge_all_of(root, schema["allOf"])

    schema_type = schema.get("type")
    if isinstance(schema_type, list):
        schema_type = next((item for item in schema_type if item != "null"), "null")

    if schema_type == "object" or "properties" in schema:
        return _sample_object(schema, root, depth)
    if schema_type == "array" or "items" in schema:
        return _sample_array(schema, root, depth)
    if schema_type == "string":
        return _sample_string(schema)
    if schema_type == "integer":
        return _sample_number(schema, integer=True)
    if schema_type == "number":
        return _sample_number(schema, integer=False)
    if schema_type == "boolean":
        return True
    if schema_type == "null":
        return None

    return "mock"


def _sample_object(
    schema: Mapping[str, Any], root: Mapping[str, Any], depth: int
) -> dict[str, Any]:
    properties = schema.get("properties")
    properties = properties if isinstance(properties, Mapping) else {}
    # Include optional properties too: it produces complete, realistic mock
    # responses and keeps discriminator defaults (``type``, ``status``, ...)
    # in the generated document.
    property_names = list(properties)

    result: dict[str, Any] = {}
    for name in property_names:
        prop_schema = properties.get(name)
        if isinstance(prop_schema, Mapping):
            result[str(name)] = _sample(prop_schema, root, depth + 1)
        elif isinstance(schema.get("additionalProperties"), Mapping):
            result[str(name)] = _sample(schema["additionalProperties"], root, depth + 1)
        else:
            result[str(name)] = None
    return result


def _sample_array(schema: Mapping[str, Any], root: Mapping[str, Any], depth: int) -> list[Any]:
    items = schema.get("items")
    if not isinstance(items, Mapping):
        return []
    min_items = schema.get("minItems", 0)
    count = min_items if isinstance(min_items, int) and min_items > 0 else 0
    return [_sample(items, root, depth + 1) for _ in range(count)]


def _sample_string(schema: Mapping[str, Any]) -> str:
    fmt = schema.get("format")
    if fmt == "uuid":
        return _UUID_PLACEHOLDER
    if fmt == "date-time":
        return "2026-01-01T00:00:00Z"
    if fmt == "date":
        return "2026-01-01"
    if fmt == "time":
        return "00:00:00"

    value = "mock"
    min_length = schema.get("minLength")
    if isinstance(min_length, int) and len(value) < min_length:
        value = value + "x" * (min_length - len(value))
    max_length = schema.get("maxLength")
    if isinstance(max_length, int) and max_length >= 0:
        value = value[:max_length]
    return value


def _sample_number(schema: Mapping[str, Any], *, integer: bool) -> int | float:
    candidates = [schema.get("minimum"), schema.get("exclusiveMinimum"), 0]
    value: int | float = 0
    for candidate in candidates:
        if isinstance(candidate, int | float):
            value = candidate
            break
    if isinstance(schema.get("exclusiveMinimum"), int | float):
        value = max(value, schema["exclusiveMinimum"] + (1 if integer else 0.5))

    maximum = schema.get("maximum")
    if isinstance(maximum, int | float):
        value = min(value, maximum)
    exclusive_maximum = schema.get("exclusiveMaximum")
    if isinstance(exclusive_maximum, int | float):
        value = min(value, exclusive_maximum - (1 if integer else 0.5))

    return int(value) if integer else float(value)


# ---------------------------------------------------------------------------
# Request handling
# ---------------------------------------------------------------------------


class _ResolvedRequest(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    text: str
    ttft_ms: float
    inter_token_delay_ms: float
    error_status: int | None
    corrupt_json: bool
    include_usage: bool


def _message_text(message: ChatMessage) -> str:
    content = message.content
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, Mapping) and isinstance(part.get("text"), str):
                parts.append(part["text"])
        return " ".join(parts)
    return ""


def _last_user_text(messages: list[ChatMessage]) -> str:
    for message in reversed(messages):
        if message.role == "user":
            return _message_text(message)
    return _message_text(messages[-1]) if messages else ""


def _prompt_text(messages: list[ChatMessage]) -> str:
    return "\n".join(_message_text(message) for message in messages)


_TOKEN_PATTERN = re.compile(r"\s+|\S{1,6}")


def _tokenize(text: str) -> list[str]:
    return [match.group(0) for match in _TOKEN_PATTERN.finditer(text)]


def _requested_schema(request: ChatCompletionRequest) -> dict[str, Any] | None:
    if request.guided_json:
        return request.guided_json
    response_format = request.response_format
    if response_format is None:
        return None
    if response_format.type not in {"json_schema", "structured_outputs"}:
        return None
    payload = response_format.json_schema or {}
    nested = payload.get("schema")
    if isinstance(nested, Mapping):
        return dict(nested)
    return dict(payload) or None


def _json_requested(request: ChatCompletionRequest) -> bool:
    response_format = request.response_format
    if _requested_schema(request) is not None:
        return True
    return response_format is not None and response_format.type == "json_object"


def _header_float(headers: Mapping[str, str], name: str, default: float) -> float:
    raw = headers.get(name)
    if raw is None:
        return default
    try:
        return max(0.0, float(raw))
    except ValueError:
        return default


def _header_bool(headers: Mapping[str, str], name: str, default: bool) -> bool:
    raw = headers.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _resolve_request(
    payload: ChatCompletionRequest,
    config: MockServerConfig,
    headers: Mapping[str, str],
) -> _ResolvedRequest:
    mode = headers.get("x-mock-mode", config.mode)
    if mode not in {"template", "echo", "json"}:
        raise ValueError(f"Unknown mock mode: {mode!r}")

    error_status = config.error_status
    corrupt_json = _header_bool(headers, "x-mock-corrupt-json", config.corrupt_json)
    error_header = headers.get("x-mock-error", "").strip().lower()
    if error_header == "corrupt":
        corrupt_json = True
    elif error_header:
        try:
            error_status = int(error_header)
        except ValueError as exc:
            raise ValueError(f"Invalid x-mock-error header: {error_header!r}") from exc

    schema = _requested_schema(payload)
    json_requested = _json_requested(payload)

    if corrupt_json:
        text = '{"corrupt": ' if json_requested else 'corrupt-json{"'
    elif schema is not None:
        text = json.dumps(sample_from_schema(schema), separators=(",", ":"), ensure_ascii=False)
    elif json_requested:
        text = json.dumps(config.json_response, separators=(",", ":"), ensure_ascii=False)
    elif mode == "echo":
        text = _last_user_text(payload.messages)
    elif mode == "json":
        text = json.dumps(config.json_response, separators=(",", ":"), ensure_ascii=False)
    else:
        text = config.response_text

    include_usage = config.always_include_usage or bool(
        payload.stream_options and payload.stream_options.get("include_usage")
    )

    return _ResolvedRequest(
        text=text,
        ttft_ms=_header_float(headers, "x-mock-ttft-ms", config.ttft_ms),
        inter_token_delay_ms=_header_float(
            headers, "x-mock-token-delay-ms", config.inter_token_delay_ms
        ),
        error_status=error_status,
        corrupt_json=corrupt_json,
        include_usage=include_usage,
    )


def _usage(payload: ChatCompletionRequest, text: str) -> dict[str, int]:
    prompt_tokens = max(1, len(_tokenize(_prompt_text(payload.messages))))
    completion_tokens = max(1, len(_tokenize(text)))
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
    }


def _error_payload(status: int, message: str) -> dict[str, Any]:
    error_type = "rate_limit_error" if status == 429 else "server_error"
    return {
        "error": {
            "message": message,
            "type": error_type,
            "param": None,
            "code": status,
        }
    }


def _chunk(
    completion_id: str,
    created: int,
    model: str,
    delta: dict[str, Any],
    *,
    finish_reason: str | None = None,
    usage: dict[str, int] | None = None,
) -> dict[str, Any]:
    body: dict[str, Any] = {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "system_fingerprint": "fp_mock",
        "choices": [{"index": 0, "delta": delta, "logprobs": None, "finish_reason": finish_reason}]
        if usage is None
        else [],
    }
    if usage is not None:
        body["usage"] = usage
    return body


async def _stream_completion(
    payload: ChatCompletionRequest, resolved: _ResolvedRequest
) -> AsyncIterator[str]:
    completion_id = f"chatcmpl-{uuid.uuid4().hex}"
    created = int(time.time())
    model = payload.model or DEFAULT_MODEL
    pieces = _tokenize(resolved.text) or [""]

    await asyncio.sleep(resolved.ttft_ms / 1000.0)

    first = pieces[0]
    yield _sse(_chunk(completion_id, created, model, {"role": "assistant", "content": first}))
    for piece in pieces[1:]:
        if resolved.inter_token_delay_ms:
            await asyncio.sleep(resolved.inter_token_delay_ms / 1000.0)
        yield _sse(_chunk(completion_id, created, model, {"content": piece}))

    yield _sse(_chunk(completion_id, created, model, {}, finish_reason="stop"))
    if resolved.include_usage:
        yield _sse(
            _chunk(
                completion_id,
                created,
                model,
                {},
                usage=_usage(payload, resolved.text),
            )
        )
    yield "data: [DONE]\n\n"


def _sse(body: dict[str, Any]) -> str:
    return f"data: {json.dumps(body, ensure_ascii=False)}\n\n"


def create_app(config: MockServerConfig | None = None) -> FastAPI:
    """Build the mock FastAPI application.

    Passing a config makes the app fully deterministic in tests; the module
    level ``app`` uses :class:`MockServerConfig` defaults.
    """

    app = FastAPI(
        title="AHP Mock OpenAI Server",
        version="0.1.0",
        description="GPU-free OpenAI-compatible server for harness development.",
    )
    app.state.config = config or MockServerConfig()

    @app.get("/health")
    async def health(request: Request) -> dict[str, Any]:
        return {"status": "ok", "model": request.app.state.config.model}

    @app.get("/v1/models")
    async def list_models(request: Request) -> dict[str, Any]:
        model = request.app.state.config.model
        return {
            "object": "list",
            "data": [
                {
                    "id": model,
                    "object": "model",
                    "created": 0,
                    "owned_by": "ahp-mock",
                }
            ],
        }

    @app.get("/v1/models/{model_id}")
    async def get_model(model_id: str) -> dict[str, Any]:
        return {"id": model_id, "object": "model", "created": 0, "owned_by": "ahp-mock"}

    @app.get("/mock/config")
    async def get_mock_config(request: Request) -> dict[str, Any]:
        return request.app.state.config.model_dump()

    @app.post("/mock/config")
    async def update_mock_config(request: Request) -> JSONResponse:
        patch = await request.json()
        if not isinstance(patch, Mapping):
            return JSONResponse(status_code=400, content=_error_payload(400, "Expected an object"))
        try:
            current = request.app.state.config.model_dump()
            request.app.state.config = MockServerConfig.model_validate({**current, **patch})
        except Exception as exc:  # noqa: BLE001 - surfaced to the caller as a 400
            return JSONResponse(status_code=400, content=_error_payload(400, str(exc)))
        return JSONResponse(content=request.app.state.config.model_dump())

    @app.post("/v1/chat/completions")
    async def chat_completions(payload: ChatCompletionRequest, request: Request) -> Any:
        config = request.app.state.config
        try:
            resolved = _resolve_request(payload, config, request.headers)
        except ValueError as exc:
            return JSONResponse(status_code=400, content=_error_payload(400, str(exc)))

        if resolved.error_status is not None:
            return JSONResponse(
                status_code=resolved.error_status,
                content=_error_payload(resolved.error_status, config.error_message),
            )

        if payload.stream:
            return StreamingResponse(
                _stream_completion(payload, resolved),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
            )

        completion_id = f"chatcmpl-{uuid.uuid4().hex}"
        return {
            "id": completion_id,
            "object": "chat.completion",
            "created": int(time.time()),
            "model": payload.model or DEFAULT_MODEL,
            "system_fingerprint": "fp_mock",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": resolved.text},
                    "logprobs": None,
                    "finish_reason": "stop",
                }
            ],
            "usage": _usage(payload, resolved.text),
        }

    return app


app = create_app()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the AHP mock OpenAI server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--ttft-ms", type=float, default=50.0)
    parser.add_argument("--token-delay-ms", type=float, default=10.0)
    parser.add_argument("--mode", choices=["template", "echo", "json"], default="template")
    args = parser.parse_args()

    config = MockServerConfig(
        model=args.model,
        ttft_ms=args.ttft_ms,
        inter_token_delay_ms=args.token_delay_ms,
        mode=args.mode,
    )
    uvicorn.run(create_app(config), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
