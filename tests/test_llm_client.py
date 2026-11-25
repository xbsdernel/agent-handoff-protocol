"""Tests for the unified streaming LLM client (Issue #12).

Everything runs against the in-process mock OpenAI app over an ASGI transport:
no network, GPU or real engine is required.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

import httpx
import pytest
from openai import APIStatusError

from engine.client import (
    LLMCallEvent,
    LLMResponseError,
    UnifiedLLMClient,
    build_response_format,
    infer_engine_name,
)
from engine.interfaces import LLMClient
from engine.mock_server import MockServerConfig, create_app
from telemetry.interfaces import Event, EventSink


class RecordingSink(EventSink):
    def __init__(self) -> None:
        self.events: list[Event] = []

    async def emit(self, event: Event) -> None:
        self.events.append(event)

    async def flush(self) -> None:
        return None


def make_client(
    config: MockServerConfig | None = None,
    **kwargs: Any,
) -> UnifiedLLMClient:
    """Build a client wired to a fresh mock app through ASGI."""

    app = create_app(config or MockServerConfig(ttft_ms=0.0, inter_token_delay_ms=0.0))
    http = httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test")
    kwargs.setdefault("api_key", "test")
    kwargs.setdefault("model", "mock-model")
    return UnifiedLLMClient(base_url="http://test/v1", http_client=http, **kwargs)


@pytest.fixture
async def client() -> AsyncIterator[UnifiedLLMClient]:
    instance = make_client()
    try:
        yield instance
    finally:
        await instance.aclose()


# ---------------------------------------------------------------------------
# Sending, streaming and usage
# ---------------------------------------------------------------------------


async def test_chat_streams_text_with_timing_and_usage(client: UnifiedLLMClient) -> None:
    result = await client.chat([{"role": "user", "content": "hello streaming world"}])

    assert result.text
    assert result.ttft_ms > 0.0
    assert result.total_latency_ms >= result.ttft_ms
    assert result.total_latency_ms > 0.0
    assert result.prompt_tokens > 0
    assert result.completion_tokens > 0
    assert result.total_tokens == result.prompt_tokens + result.completion_tokens
    assert result.tokens_per_second > 0.0
    assert result.finish_reason == "stop"
    assert result.model_name == "mock-model"
    assert result.engine_name == "unknown"


async def test_tokens_per_second_uses_the_decoding_window(client: UnifiedLLMClient) -> None:
    result = await client.chat([{"role": "user", "content": "measure my throughput"}])

    expected = result.completion_tokens / (result.generation_ms / 1000)
    assert result.tokens_per_second == pytest.approx(expected)
    assert result.generation_ms == pytest.approx(result.total_latency_ms - result.ttft_ms)


async def test_ttft_tracks_the_engine_delay() -> None:
    client = make_client(MockServerConfig(ttft_ms=60.0, inter_token_delay_ms=0.0))
    try:
        result = await client.chat([{"role": "user", "content": "wait first"}])
    finally:
        await client.aclose()

    # Windows timers can wake sleeps a few milliseconds early.
    assert result.ttft_ms >= 40.0
    assert result.total_latency_ms >= result.ttft_ms


async def test_usage_can_be_disabled_without_losing_text() -> None:
    client = make_client(include_usage=False)
    try:
        result = await client.chat([{"role": "user", "content": "no usage please"}])
    finally:
        await client.aclose()

    assert result.text
    assert result.prompt_tokens == 0
    assert result.completion_tokens == 0
    assert result.total_tokens == 0
    assert result.tokens_per_second == 0.0


async def test_default_headers_are_forwarded_to_the_engine() -> None:
    client = make_client(default_headers={"x-mock-mode": "echo"})
    try:
        result = await client.chat([{"role": "user", "content": "echo this sentence"}])
    finally:
        await client.aclose()

    assert result.text == "echo this sentence"


async def test_empty_messages_are_rejected(client: UnifiedLLMClient) -> None:
    with pytest.raises(ValueError):
        await client.chat([])


async def test_client_satisfies_the_llm_client_protocol(client: UnifiedLLMClient) -> None:
    assert isinstance(client, LLMClient)


# ---------------------------------------------------------------------------
# Engine identification and response formats
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        ("http://localhost:8001/v1", "vllm"),
        ("http://localhost:8002/v1", "sglang"),
        ("http://localhost:8003/v1", "llama-server"),
        ("http://localhost:8000/v1", "mock"),
        ("http://gpu-box:9000/v1", "unknown"),
        ("https://api.example.com/v1", "unknown"),
    ],
)
def test_engine_name_is_inferred_from_the_compose_ports(url: str, expected: str) -> None:
    assert infer_engine_name(url) == expected


def test_engine_name_can_be_overridden() -> None:
    client = UnifiedLLMClient(
        base_url="http://gpu-box:9000/v1",
        engine_name="vllm",
        client=object(),  # type: ignore[arg-type] - constructor only tags the name
    )
    assert client.engine_name == "vllm"


def test_build_response_format_supports_both_engine_dialects() -> None:
    schema = {"type": "object", "properties": {"ok": {"type": "boolean"}}}

    structured = build_response_format(schema)
    assert structured["type"] == "json_schema"
    assert structured["json_schema"]["schema"] == schema
    assert structured["json_schema"]["strict"] is False

    assert build_response_format(schema, "json_object") == {"type": "json_object"}
    with pytest.raises(ValueError):
        build_response_format(schema, "bogus")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Structured output
# ---------------------------------------------------------------------------

STRUCTURED_SCHEMA = {
    "type": "object",
    "properties": {
        "decision": {"enum": ["refund", "no_refund", "escalate"]},
        "amount": {"type": "number", "minimum": 0},
    },
    "required": ["decision", "amount"],
    "additionalProperties": False,
}


async def test_structured_output_is_parsed_and_validated(client: UnifiedLLMClient) -> None:
    result = await client.chat(
        [{"role": "user", "content": "decide the refund"}],
        schema=STRUCTURED_SCHEMA,
    )

    assert result.structured_data is not None
    assert result.structured_data == json.loads(result.text)
    assert result.structured_data["decision"] == "refund"
    assert result.structured_data["amount"] >= 0


async def test_json_object_mode_still_validates_client_side() -> None:
    client = make_client(
        MockServerConfig(
            ttft_ms=0.0,
            inter_token_delay_ms=0.0,
            json_response={"status": "ok"},
        ),
        schema_mode="json_object",
    )
    try:
        result = await client.chat(
            [{"role": "user", "content": "status please"}],
            schema={
                "type": "object",
                "properties": {"status": {"const": "ok"}},
                "required": ["status"],
            },
        )
    finally:
        await client.aclose()

    assert result.structured_data == {"status": "ok"}


async def test_malformed_structured_output_raises() -> None:
    client = make_client(MockServerConfig(corrupt_json=True), max_retries=0)
    try:
        with pytest.raises(LLMResponseError, match="not valid JSON"):
            await client.chat(
                [{"role": "user", "content": "give me json"}],
                schema=STRUCTURED_SCHEMA,
            )
    finally:
        await client.aclose()


async def test_schema_violations_raise() -> None:
    # The mock ignores `pattern`, so it returns "mock", which this schema rejects.
    schema = {
        "type": "object",
        "properties": {"code": {"type": "string", "pattern": "^Z"}},
        "required": ["code"],
    }
    client = make_client()
    try:
        with pytest.raises(LLMResponseError, match="violates the response schema"):
            await client.chat([{"role": "user", "content": "code"}], schema=schema)
    finally:
        await client.aclose()


# ---------------------------------------------------------------------------
# Telemetry
# ---------------------------------------------------------------------------


async def test_successful_call_emits_one_llm_call_event() -> None:
    sink = RecordingSink()
    client = make_client(event_sink=sink, run_id="run-42", workflow_id="wf-7")
    try:
        result = await client.chat([{"role": "user", "content": "emit an event"}])
    finally:
        await client.aclose()

    assert len(sink.events) == 1
    event = sink.events[0]
    assert isinstance(event, LLMCallEvent)
    assert event.event_type == "llm_call"
    assert event.run_id == "run-42"
    assert event.workflow_id == "wf-7"
    assert event.engine_name == "unknown"
    assert event.model_name == "mock-model"
    assert event.success is True
    assert event.error is None
    assert event.ttft_ms == pytest.approx(result.ttft_ms)
    assert event.latency_ms == pytest.approx(result.total_latency_ms)
    assert event.completion_tokens == result.completion_tokens
    assert event.finish_reason == "stop"


async def test_failed_call_emits_an_error_event_and_reraises() -> None:
    sink = RecordingSink()
    client = make_client(MockServerConfig(error_status=500), event_sink=sink)
    try:
        with pytest.raises(APIStatusError):
            await client.chat([{"role": "user", "content": "fail"}])
    finally:
        await client.aclose()

    assert len(sink.events) == 1
    event = sink.events[0]
    assert isinstance(event, LLMCallEvent)
    assert event.success is False
    assert event.error is not None
    assert "500" in event.error or "APIStatusError" in event.error
    assert event.latency_ms > 0.0


async def test_telemetry_failures_never_break_inference() -> None:
    class BrokenSink(EventSink):
        async def emit(self, event: Event) -> None:
            raise RuntimeError("telemetry down")

        async def flush(self) -> None:
            return None

    client = make_client(event_sink=BrokenSink())
    try:
        result = await client.chat([{"role": "user", "content": "still works"}])
    finally:
        await client.aclose()

    assert result.text
