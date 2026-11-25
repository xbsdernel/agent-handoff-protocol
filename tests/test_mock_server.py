"""Tests for the mock OpenAI-compatible server (Issue #2).

All tests run on CPU: the server is the only model endpoint involved.
"""

from __future__ import annotations

import json
import threading
import time
import uuid
from collections.abc import AsyncIterator
from typing import Any, Literal

import httpx
import pytest
import uvicorn
from openai import APIStatusError, AsyncOpenAI
from pydantic import BaseModel

from engine.mock_server import MockServerConfig, create_app, sample_from_schema


@pytest.fixture(scope="module")
def mock_root() -> AsyncIterator[str]:
    """Run the mock app on an ephemeral port for the whole module."""

    app = create_app(MockServerConfig(ttft_ms=0.0, inter_token_delay_ms=0.0))
    config = uvicorn.Config(app, host="127.0.0.1", port=0, log_level="warning", access_log=False)
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()

    deadline = time.time() + 15
    while not server.started:
        if time.time() > deadline:
            raise RuntimeError("mock server did not start in time")
        time.sleep(0.01)

    sock = server.servers[0].sockets[0]
    host, port = sock.getsockname()[:2]
    try:
        yield f"http://{host}:{port}"
    finally:
        server.should_exit = True
        thread.join(timeout=10)


@pytest.fixture
async def client(mock_root: str) -> AsyncIterator[AsyncOpenAI]:
    api = AsyncOpenAI(base_url=f"{mock_root}/v1", api_key="test", max_retries=0)
    try:
        yield api
    finally:
        await api.close()


async def _collect(stream: Any) -> tuple[str, Any]:
    text = ""
    usage = None
    async for chunk in stream:
        if chunk.choices:
            delta = chunk.choices[0].delta.content
            if delta:
                text += delta
        if getattr(chunk, "usage", None) is not None:
            usage = chunk.usage
    return text, usage


async def test_openai_client_can_stream(client: AsyncOpenAI) -> None:
    stream = await client.chat.completions.create(
        model="mock-model",
        messages=[{"role": "user", "content": "hello mock"}],
        stream=True,
        stream_options={"include_usage": True},
    )

    text, usage = await _collect(stream)

    assert text
    assert usage is not None
    assert usage.total_tokens == usage.prompt_tokens + usage.completion_tokens


async def test_openai_client_non_streaming(client: AsyncOpenAI) -> None:
    response = await client.chat.completions.create(
        model="mock-model",
        messages=[{"role": "user", "content": "hello mock"}],
    )

    content = response.choices[0].message.content
    assert content
    assert response.usage is not None
    assert response.object == "chat.completion"


async def test_raw_sse_stream_ends_with_done(mock_root: str) -> None:
    payload = {
        "model": "mock-model",
        "messages": [{"role": "user", "content": "stream me"}],
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    async with (
        httpx.AsyncClient(timeout=10) as http,
        http.stream("POST", f"{mock_root}/v1/chat/completions", json=payload) as response,
    ):
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")
        lines = [line async for line in response.aiter_lines() if line]

    assert lines[-1] == "data: [DONE]"
    chunks = [
        json.loads(line.removeprefix("data: ")) for line in lines if line.startswith("data: {")
    ]
    assert any(chunk.get("usage") for chunk in chunks)
    content = "".join(
        chunk["choices"][0]["delta"].get("content", "") for chunk in chunks if chunk.get("choices")
    )
    assert content


async def test_streaming_usage_metadata_is_reported(client: AsyncOpenAI) -> None:
    stream = await client.chat.completions.create(
        model="mock-model",
        messages=[{"role": "user", "content": "count my tokens"}],
        stream=True,
        stream_options={"include_usage": True},
    )

    _, usage = await _collect(stream)

    assert usage is not None
    assert usage.prompt_tokens > 0
    assert usage.completion_tokens > 0
    assert usage.total_tokens == usage.prompt_tokens + usage.completion_tokens


SCHEMA = {
    "type": "object",
    "properties": {
        "type": {"const": "HANDOFF"},
        "message_id": {"type": "string", "format": "uuid"},
        "from_agent": {"type": "string"},
        "decision": {"enum": ["refund", "no_refund"]},
        "facts": {
            "type": "object",
            "properties": {"amount": {"type": "number", "minimum": 0}},
            "required": ["amount"],
            "additionalProperties": False,
        },
        "done_so_far": {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 2,
        },
    },
    "required": [
        "type",
        "message_id",
        "from_agent",
        "decision",
        "facts",
        "done_so_far",
    ],
    "additionalProperties": False,
}


async def test_structured_output_matches_json_schema(client: AsyncOpenAI) -> None:
    response = await client.chat.completions.create(
        model="mock-model",
        messages=[{"role": "user", "content": "produce a handoff"}],
        response_format={
            "type": "json_schema",
            "json_schema": {"name": "handoff", "schema": SCHEMA, "strict": True},
        },
    )

    data = json.loads(response.choices[0].message.content)

    assert data["type"] == "HANDOFF"
    assert data["decision"] == "refund"
    assert data["facts"]["amount"] >= 0
    assert len(data["done_so_far"]) == 2
    assert uuid.UUID(data["message_id"])


async def test_structured_output_streams_as_json(client: AsyncOpenAI) -> None:
    stream = await client.chat.completions.create(
        model="mock-model",
        messages=[{"role": "user", "content": "produce a handoff"}],
        response_format={
            "type": "json_schema",
            "json_schema": {"name": "handoff", "schema": SCHEMA, "strict": True},
        },
        stream=True,
    )

    text, _ = await _collect(stream)

    assert json.loads(text)["type"] == "HANDOFF"


async def test_json_object_mode_returns_configured_json(client: AsyncOpenAI) -> None:
    response = await client.chat.completions.create(
        model="mock-model",
        messages=[{"role": "user", "content": "give me json"}],
        response_format={"type": "json_object"},
    )

    assert json.loads(response.choices[0].message.content) == {"status": "ok"}


async def test_echo_mode(client: AsyncOpenAI) -> None:
    response = await client.chat.completions.create(
        model="mock-model",
        messages=[{"role": "user", "content": "echo this back"}],
        extra_headers={"x-mock-mode": "echo"},
    )

    assert response.choices[0].message.content == "echo this back"


@pytest.mark.parametrize("status", [429, 500])
async def test_http_error_injection(client: AsyncOpenAI, status: int) -> None:
    with pytest.raises(APIStatusError) as excinfo:
        await client.chat.completions.create(
            model="mock-model",
            messages=[{"role": "user", "content": "fail please"}],
            extra_headers={"x-mock-error": str(status)},
        )

    assert excinfo.value.status_code == status


async def test_corrupt_json_injection(client: AsyncOpenAI) -> None:
    response = await client.chat.completions.create(
        model="mock-model",
        messages=[{"role": "user", "content": "give me json"}],
        response_format={"type": "json_object"},
        extra_headers={"x-mock-corrupt-json": "1"},
    )

    with pytest.raises(json.JSONDecodeError):
        json.loads(response.choices[0].message.content)


async def test_latency_simulation(client: AsyncOpenAI) -> None:
    started = time.perf_counter()
    stream = await client.chat.completions.create(
        model="mock-model",
        messages=[{"role": "user", "content": "one two three"}],
        stream=True,
        extra_headers={
            "x-mock-mode": "echo",
            "x-mock-ttft-ms": "40",
            "x-mock-token-delay-ms": "5",
        },
    )
    await _collect(stream)
    elapsed = time.perf_counter() - started

    assert elapsed >= 0.035


async def test_runtime_config_endpoint() -> None:
    app = create_app(MockServerConfig(ttft_ms=0.0, inter_token_delay_ms=0.0))
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as http:
        initial = await http.get("/mock/config")
        assert initial.json()["mode"] == "template"

        updated = await http.post("/mock/config", json={"mode": "echo"})
        assert updated.json()["mode"] == "echo"

        echoed = await http.post(
            "/v1/chat/completions",
            json={
                "model": "mock-model",
                "messages": [{"role": "user", "content": "configured echo"}],
            },
        )
        assert echoed.json()["choices"][0]["message"]["content"] == "configured echo"

        rejected = await http.post("/mock/config", json={"unknown_field": True})
        assert rejected.status_code == 400


async def test_models_and_health_endpoints() -> None:
    app = create_app()
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as http:
        models = await http.get("/v1/models")
        assert models.json()["data"][0]["id"] == "mock-model"

        health = await http.get("/health")
        assert health.json()["status"] == "ok"


class Inner(BaseModel):
    count: int
    tags: list[str]


class Outer(BaseModel):
    name: str
    inner: Inner
    status: Literal["ok", "bad"] = "ok"


def test_sample_from_schema_resolves_defs_and_refs() -> None:
    schema = Outer.model_json_schema()

    sample = sample_from_schema(schema)

    validated = Outer.model_validate(sample)
    assert validated.name
    assert isinstance(validated.inner, Inner)
    assert validated.inner.tags == []


def test_sample_from_schema_handles_nullable_and_constraints() -> None:
    schema = {
        "type": "object",
        "properties": {
            "value": {"anyOf": [{"type": "string"}, {"type": "null"}]},
            "uuid_value": {"type": "string", "format": "uuid"},
            "limited": {"type": "integer", "minimum": 5, "maximum": 9},
        },
        "required": ["value", "uuid_value", "limited"],
    }

    sample = sample_from_schema(schema)

    assert sample["value"] == "mock"
    assert uuid.UUID(sample["uuid_value"])
    assert 5 <= sample["limited"] <= 9


def test_default_app_is_exported() -> None:
    from engine import mock_server

    assert mock_server.app.title == "AHP Mock OpenAI Server"
