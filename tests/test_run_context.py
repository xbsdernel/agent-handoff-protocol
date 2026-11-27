"""Tests for the RunContext and environment capture (Issue #17)."""

from __future__ import annotations

import yaml
import shutil
import json
from pathlib import Path

import pytest

from telemetry.run_context import (
    RunContext,
    collect_environment,
    default_run_id,
    _validate_run_id,
)


def test_default_run_id_format() -> None:
    """default_run_id must produce a sortable, collision-resistant string."""
    rid = default_run_id()
    assert isinstance(rid, str)
    assert len(rid) > 0
    # Format: YYYYMMDDTHHMMSS-<8 hex chars>
    assert "T" in rid
    parts = rid.split("-")
    assert len(parts) == 2
    date_time = parts[0]
    assert "T" in date_time


def test_validate_run_id_valid() -> None:
    """Valid run_ids must pass _validate_run_id."""
    _validate_run_id("valid-run-123")
    _validate_run_id("a")
    # These should not raise


def test_validate_run_id_too_long_still_valid() -> None:
    """Run IDs with many characters but no forbidden chars must be valid."""
    _validate_run_id("a" * 100)


def test_validate_run_id_forbidden_chars_fails() -> None:
    """Run IDs with forbidden chars must raise ValueError."""
    import pytest
    bad_ids = ["<invalid>", '"/path', "with\\backslash", "with?question", "with*star"]
    for bid in bad_ids:
        with pytest.raises(ValueError, match="must be a single safe path component"):
            _validate_run_id(bid)


def test_validate_run_id_dot_and_dotdot_fail() -> None:
    """Run IDs of '.' or '..' must raise ValueError."""
    with pytest.raises(ValueError, match="must be a non-empty directory name"):
        _validate_run_id(".")
    with pytest.raises(ValueError, match="must be a non-empty directory name"):
        _validate_run_id("..")


def test_collect_environment_has_all_keys() -> None:
    """collect_environment must return all expected keys."""
    env = collect_environment()
    expected_keys = {
        "created_at",
        "python_version",
        "python_implementation",
        "platform",
        "system",
        "release",
        "machine",
        "processor",
        "cpu_count",
        "hostname",
        "git_commit",
        "git_branch",
        "git_dirty",
        "gpu_model",
        "gpu_driver",
        "docker_image",
    }
    assert set(env.keys()) == expected_keys


def test_collect_environment_git_dirty_is_bool() -> None:
    """git_dirty must be a boolean (or None if git unavailable)."""
    env = collect_environment()
    git_dirty = env["git_dirty"]
    if git_dirty is not None:
        assert isinstance(git_dirty, bool)


def test_collect_environment_gpu_overrides_env() -> None:
    """gpu_model/gpu_driver must respect AHP_GPU_MODEL / AHP_GPU_DRIVER env vars."""
    env = collect_environment(
        env={"AHP_GPU_MODEL": "custom-gpu", "AHP_GPU_DRIVER": "custom-drive"}
    )
    assert env["gpu_model"] == "custom-gpu"
    assert env["gpu_driver"] == "custom-drive"


def test_collect_environment_docker_image_preference() -> None:
    """docker_image prefer explicit arg over AHP_DOCKER_IMAGE env var."""
    env_explicit = collect_environment(docker_image="explicit-image:tag")
    assert env_explicit["docker_image"] == "explicit-image:tag"

    env_env = collect_environment(
        docker_image=None, env={"AHP_DOCKER_IMAGE": "env-image:tag"}
    )
    # When explicit arg is None, fall back to env
    assert env_env["docker_image"] == "env-image:tag"


def test_run_context_create_basic() -> None:
    """RunContext.create must create the directory and write metadata files."""
    ctx = RunContext.create(run_id="test-run-001")
    run_dir = Path("runs") / "test-run-001"
    assert run_dir.is_dir()
    assert (run_dir / "config.yaml").is_file()
    assert (run_dir / "environment.json").is_file()
    assert (run_dir / "events.jsonl").is_file()
    assert (run_dir / "engine_metrics.jsonl").is_file()
    assert (run_dir / "gpu_metrics.jsonl").is_file()

    # Check environment.json content
    env = json.loads((run_dir / "environment.json").read_text())
    assert "created_at" in env
    assert "python_version" in env
    assert "git_commit" in env

    # Cleanup
    import shutil
    shutil.rmtree(run_dir)


def test_run_context_create_with_config() -> None:
    """RunContext.create with a custom config dict."""
    ctx = RunContext.create(
        run_id="test-run-config",
        config={"model": "gpt-4", "temperature": 0.5},
    )
    env = json.loads(
        (Path("runs") / "test-run-config" / "environment.json").read_text()
    )
    # config is stored in config.yaml, not environment.json
    config_yaml = yaml.safe_load(
        (Path("runs") / "test-run-config" / "config.yaml").read_text()
    )
    assert config_yaml == {"model": "gpt-4", "temperature": 0.5}

    import shutil
    shutil.rmtree(Path("runs") / "test-run-config")


def test_run_context_event_sink() -> None:
    """RunContext.event_sink must return a sink wired to events.jsonl."""
    ctx = RunContext.create(run_id="sink-test")
    sink = ctx.event_sink()
    assert sink.path == ctx.events_path
    assert sink.run_id == ctx.run_id


def test_run_context_load_events() -> None:
    """RunContext.load_events must re-validate events from events.jsonl."""
    ctx = RunContext.create(run_id="load-events-test")
    # Write a known event
    (ctx.path / "events.jsonl").write_text(
        '{"event_type": "llm_call", "event_id": "a", "run_id": "load-events-test"}\n',
        encoding="utf-8",
    )
    events = ctx.load_events()
    assert len(events) == 1
    assert events[0].event_type == "llm_call"

    # Write an invalid row to test re-validation
    (ctx.path / "events.jsonl").write_text(
        '{"event_type": "llm_call", "event_id": "a", "run_id": "load-events-test"}\n'
        '{"bad": "row"}\n',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="invalid telemetry event"):
        ctx.load_events()


@pytest.mark.asyncio
async def test_run_context_aclose_sink() -> None:
    """RunContext event_sink aclose must be safe."""
    ctx = RunContext.create(run_id="aclose-test")
    sink = ctx.event_sink()
    await sink.aclose()


# ---------------------------------------------------------------------------
# Integration: full create → emit → flush → aclose roundtrip
# ---------------------------------------------------------------------------


async def test_full_roundtrip() -> None:
    """End-to-end: create run, emit events, flush, aclose, load events back."""
    ctx = RunContext.create(run_id="full-roundtrip-test")
    sink = ctx.event_sink()

    # Emit a mix of event types
    from telemetry.models import LLMCallEvent, WorkflowEvent, SystemEvent, HandoffEvent

    await sink.emit(LLMCallEvent(event_type="llm_call", engine_name="mock", model_name="gpt-4", run_id="full-roundtrip-test"))
    await sink.emit(
        WorkflowEvent(event_type="started", run_id="full-roundtrip-test", workflow_id="wf-1", detail="run started")
    )
    await sink.emit(
        SystemEvent(event_type="system.error", severity="error", run_id="full-roundtrip-test", detail="boom")
    )
    await sink.emit(
        HandoffEvent(
            event_type="handoff",
            run_id="full-roundtrip-test",
            from_agent="a",
            to_agent="b",
            message_type="HANDOFF",
        )
    )

    await sink.flush()
    assert sink.written == 4

    # Close the sink to release the file handle.
    await sink.aclose()

    # Load events back
    loaded = ctx.load_events()
    assert len(loaded) == 4

    # Verify types
    types = [type(e).__name__ for e in loaded]
    assert "LLMCallEvent" in types
    assert "WorkflowEvent" in types
    assert "SystemEvent" in types
    assert "HandoffEvent" in types

    import shutil
    shutil.rmtree(ctx.path)