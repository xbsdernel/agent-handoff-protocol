"""Standardized run-directory layout and environment capture.

Every experiment gets one folder::

    runs/<run_id>/
      config.yaml
      environment.json
      events.jsonl
      engine_metrics.jsonl
      gpu_metrics.jsonl

``environment.json`` records the reproducibility metadata that is easy to lose:
git commit/branch/dirty state, Python and platform details, hostname, CPU, GPU
model and driver, and the Docker image used for the inference engine. Commands
that are unavailable (for example ``nvidia-smi`` on a CPU-only host) are
reported as ``null`` rather than failing the run.
"""

from __future__ import annotations

import json
import os
import platform
import socket
import subprocess
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

import yaml

from telemetry.interfaces import Event
from telemetry.models import load_events
from telemetry.sink import AsyncJSONLEventSink

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUNS_DIR = PROJECT_ROOT / "runs"

METRIC_FILES = ("engine_metrics.jsonl", "gpu_metrics.jsonl")


def _run_command(command: list[str], *, cwd: Path, timeout: float = 5.0) -> str | None:
    try:
        completed = subprocess.run(
            command,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if completed.returncode != 0:
        return None
    return completed.stdout.strip()


def _git_value(root: Path, *args: str) -> str | None:
    return _run_command(["git", "-C", str(root), *args], cwd=root)


def _query_gpu() -> tuple[str | None, str | None]:
    output = _run_command(
        [
            "nvidia-smi",
            "--query-gpu=name,driver_version",
            "--format=csv,noheader",
        ],
        cwd=PROJECT_ROOT,
    )
    if not output:
        return None, None
    first = output.splitlines()[0]
    parts = [part.strip() for part in first.split(",", 1)]
    if len(parts) != 2 or not parts[0]:
        return None, None
    return parts[0], parts[1] or None


def collect_environment(
    *,
    git_root: str | Path | None = None,
    docker_image: str | None = None,
    env: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Capture reproducibility metadata for one run."""

    source = os.environ if env is None else env
    root = Path(git_root) if git_root is not None else PROJECT_ROOT
    gpu_model, gpu_driver = _query_gpu()
    status = _git_value(root, "status", "--porcelain")

    return {
        "created_at": datetime.now(UTC).isoformat(),
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "hostname": socket.gethostname(),
        "git_commit": _git_value(root, "rev-parse", "HEAD"),
        "git_branch": _git_value(root, "rev-parse", "--abbrev-ref", "HEAD"),
        "git_dirty": None if status is None else bool(status),
        "gpu_model": source.get("AHP_GPU_MODEL", gpu_model),
        "gpu_driver": source.get("AHP_GPU_DRIVER", gpu_driver),
        "docker_image": docker_image
        if docker_image is not None
        else source.get("AHP_DOCKER_IMAGE"),
    }


def default_run_id() -> str:
    """Return a sortable, collision-resistant run identifier."""

    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S")
    return f"{stamp}-{uuid4().hex[:8]}"


def _validate_run_id(run_id: str) -> None:
    if not run_id or run_id in {".", ".."}:
        raise ValueError("run_id must be a non-empty directory name")
    if Path(run_id).name != run_id or any(char in run_id for char in '<>:"/\\|?*'):
        raise ValueError(f"run_id must be a single safe path component: {run_id!r}")


@dataclass(frozen=True, slots=True)
class RunContext:
    """Paths and metadata for one standardized run directory."""

    run_id: str
    path: Path
    config: dict[str, Any]
    environment: dict[str, Any]

    @classmethod
    def create(
        cls,
        *,
        run_id: str | None = None,
        root: str | Path = DEFAULT_RUNS_DIR,
        config: Mapping[str, Any] | None = None,
        docker_image: str | None = None,
        git_root: str | Path | None = None,
        env: Mapping[str, str] | None = None,
    ) -> RunContext:
        """Create the directory, write metadata and return the context."""

        resolved_id = run_id or default_run_id()
        _validate_run_id(resolved_id)
        run_path = Path(root) / resolved_id
        run_path.mkdir(parents=True, exist_ok=False)

        config_dict = dict(config or {})
        environment = collect_environment(
            git_root=git_root,
            docker_image=docker_image,
            env=env,
        )

        (run_path / "config.yaml").write_text(
            yaml.safe_dump(config_dict, sort_keys=False),
            encoding="utf-8",
        )
        (run_path / "environment.json").write_text(
            json.dumps(environment, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (run_path / "events.jsonl").touch()
        for name in METRIC_FILES:
            (run_path / name).touch()

        return cls(
            run_id=resolved_id,
            path=run_path,
            config=config_dict,
            environment=environment,
        )

    # -- paths ---------------------------------------------------------------

    @property
    def config_path(self) -> Path:
        return self.path / "config.yaml"

    @property
    def environment_path(self) -> Path:
        return self.path / "environment.json"

    @property
    def events_path(self) -> Path:
        return self.path / "events.jsonl"

    @property
    def engine_metrics_path(self) -> Path:
        return self.path / "engine_metrics.jsonl"

    @property
    def gpu_metrics_path(self) -> Path:
        return self.path / "gpu_metrics.jsonl"

    # -- telemetry -----------------------------------------------------------

    def event_sink(self, **kwargs: Any) -> AsyncJSONLEventSink:
        """Return a JSONL sink wired to this run's ``events.jsonl``."""

        kwargs.setdefault("run_id", self.run_id)
        return AsyncJSONLEventSink(self.events_path, **kwargs)

    def load_events(self) -> list[Event]:
        """Re-validate every event written to this run."""

        return load_events(self.events_path)


__all__ = [
    "DEFAULT_RUNS_DIR",
    "METRIC_FILES",
    "PROJECT_ROOT",
    "RunContext",
    "collect_environment",
    "default_run_id",
]
