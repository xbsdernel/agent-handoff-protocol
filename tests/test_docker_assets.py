"""Static validation for the Docker engine stack (Issue #11).

These tests never start a container, download a model or touch a GPU: they parse
the Compose file, inspect the launcher and check that the documentation stays in
sync with the pinned images, ports and engine flags. GPU runtime validation is a
manual step documented in ``docker/README.md``.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
DOCKER_DIR = REPO_ROOT / "docker"
COMPOSE_PATH = DOCKER_DIR / "docker-compose.yml"
LAUNCHER_PATH = DOCKER_DIR / "launch_engine.sh"
ENV_EXAMPLE_PATH = DOCKER_DIR / ".env.example"
DOCKER_README_PATH = DOCKER_DIR / "README.md"
ROOT_README_PATH = REPO_ROOT / "README.md"

ENGINES = ("vllm", "sglang", "llama-server")

EXPECTED_IMAGES = {
    "vllm": "vllm/vllm-openai:v0.34.0",
    "sglang": "lmsysorg/sglang:v0.5.23-cu130",
    "llama-server": "ghcr.io/ggml-org/llama.cpp:server-cuda-b4738",
}

EXPECTED_HOST_PORTS = {"vllm": 8001, "sglang": 8002, "llama-server": 8003}

#: Environment variables the Compose file interpolates.
COMPOSE_VAR_RE = re.compile(r"\$\{([A-Z][A-Z0-9_]*)")

#: Shell variables the launcher falls back to (``${VAR:-default}``).
SHELL_VAR_RE = COMPOSE_VAR_RE


def _bash_executable() -> str | None:
    """Return a working bash path, or None on hosts without a usable bash."""

    candidates: list[str] = []
    if os.name == "nt":
        # PowerShell resolves "bash" to the WSL stub in System32, which may be
        # unusable when no distro is installed. Prefer Git Bash when present.
        for env in ("ProgramFiles", "ProgramFiles(x86)", "LOCALAPPDATA"):
            base = os.environ.get(env)
            if not base:
                continue
            candidates.append(str(Path(base) / "Git" / "bin" / "bash.exe"))
            candidates.append(str(Path(base) / "Programs" / "Git" / "bin" / "bash.exe"))

    candidates.append(shutil.which("bash") or "")

    for bash in dict.fromkeys(candidates):
        if not bash or not Path(bash).is_file():
            continue
        try:
            proc = subprocess.run([bash, "--version"], capture_output=True, timeout=5)
        except (OSError, subprocess.SubprocessError):
            continue
        if proc.returncode == 0:
            return bash
    return None


BASH = _bash_executable()
requires_bash = pytest.mark.skipif(BASH is None, reason="a working bash is required")


def _git(*args: str) -> subprocess.CompletedProcess[str]:
    git = shutil.which("git")
    assert git is not None, "git must be available for this test"
    return subprocess.run(
        [git, *args],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=30,
    )


def _tracked(relative_path: str) -> bool:
    if shutil.which("git") is None:
        return False
    try:
        return _git("ls-files", "--error-unmatch", "--", relative_path).returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


@pytest.fixture(scope="module")
def compose() -> dict[str, Any]:
    document = yaml.safe_load(COMPOSE_PATH.read_text(encoding="utf-8"))
    assert isinstance(document, dict)
    return document


@pytest.fixture(scope="module")
def services(compose: dict[str, Any]) -> dict[str, Any]:
    services = compose["services"]
    assert isinstance(services, dict)
    return services


def _run_text(service: dict[str, Any]) -> str:
    """Join entrypoint + command into one searchable string."""

    parts: list[str] = []
    for key in ("entrypoint", "command"):
        value = service.get(key)
        if value is None:
            continue
        if isinstance(value, str):
            parts.append(value)
        elif isinstance(value, list):
            parts.extend(str(item) for item in value)
    return " ".join(parts)


def _mounts(service: dict[str, Any]) -> list[tuple[str, str, bool]]:
    """Return (source, target, read_only) for every volume mount."""

    mounts: list[tuple[str, str, bool]] = []
    for volume in service.get("volumes", []):
        if isinstance(volume, dict):
            mounts.append(
                (
                    str(volume.get("source", "")),
                    str(volume.get("target", "")),
                    bool(volume.get("read_only", False)),
                )
            )
        else:
            parts = str(volume).split(":")
            source = parts[0]
            target = parts[1] if len(parts) > 1 else ""
            mode = parts[2].split(",") if len(parts) > 2 else []
            mounts.append((source, target, "ro" in mode))
    return mounts


def _ports(service: dict[str, Any]) -> list[str]:
    return [str(port) for port in service.get("ports", [])]


# ---------------------------------------------------------------------------
# Compose structure
# ---------------------------------------------------------------------------


def test_compose_file_exists_and_names_the_project(compose: dict[str, Any]) -> None:
    assert COMPOSE_PATH.is_file()
    assert compose["name"] == "ahp-engines"
    assert "version" not in compose, "Compose v2 ignores the deprecated version key"


def test_exactly_the_three_engines_are_defined(services: dict[str, Any]) -> None:
    assert set(services) == set(ENGINES)


@pytest.mark.parametrize("engine", ENGINES)
def test_each_engine_has_its_own_profile(services: dict[str, Any], engine: str) -> None:
    assert services[engine]["profiles"] == [engine]


@pytest.mark.parametrize("engine", ENGINES)
def test_images_are_pinned_to_reviewed_tags(services: dict[str, Any], engine: str) -> None:
    image = services[engine]["image"]

    assert image == EXPECTED_IMAGES[engine]
    assert ":latest" not in image
    assert "@sha256:" not in image  # digest pinning is optional; tags are reviewed


@pytest.mark.parametrize("engine", ENGINES)
def test_host_port_maps_to_openai_port_8000(services: dict[str, Any], engine: str) -> None:
    host_port = EXPECTED_HOST_PORTS[engine]
    mappings = _ports(services[engine])

    # Ports are written as "${VLLM_PORT:-8001}:8000", so check the default and
    # the container target separately instead of a literal "8001:8000".
    assert any(f":-{host_port}" in mapping for mapping in mappings), mappings
    assert all(mapping.endswith(":8000") for mapping in mappings)


@pytest.mark.parametrize("engine", ENGINES)
def test_every_engine_serves_the_api_on_container_port_8000(
    services: dict[str, Any], engine: str
) -> None:
    text = _run_text(services[engine])

    assert "--port" in text
    assert "8000" in text


@pytest.mark.parametrize("engine", ENGINES)
def test_every_engine_reserves_an_nvidia_gpu(services: dict[str, Any], engine: str) -> None:
    reservations = services[engine]["deploy"]["resources"]["reservations"]
    devices = reservations["devices"]

    assert any(
        device.get("driver") == "nvidia" and device.get("capabilities") == ["gpu"]
        for device in devices
    )


@pytest.mark.parametrize("engine", ENGINES)
def test_every_engine_mounts_the_shared_hf_cache(services: dict[str, Any], engine: str) -> None:
    mounts = _mounts(services[engine])

    matching = [
        (source, target) for source, target, _ in mounts if target == "/root/.cache/huggingface"
    ]
    assert matching, f"{engine} must mount the host HF cache"
    assert "HF_CACHE_DIR" in matching[0][0]


def test_compose_fails_fast_when_hf_cache_dir_is_missing() -> None:
    text = COMPOSE_PATH.read_text(encoding="utf-8")

    assert "HF_CACHE_DIR:?" in text


@pytest.mark.parametrize("engine", ENGINES)
def test_every_engine_has_a_readiness_healthcheck(services: dict[str, Any], engine: str) -> None:
    healthcheck = services[engine]["healthcheck"]
    test = healthcheck["test"]
    command = " ".join(test) if isinstance(test, list) else str(test)

    assert "/health" in command
    assert "8000" in command
    assert int(str(healthcheck["interval"]).rstrip("s")) > 0
    assert int(str(healthcheck["timeout"]).rstrip("s")) > 0
    assert int(healthcheck["retries"]) > 0
    assert "start_period" in healthcheck


# ---------------------------------------------------------------------------
# Engine-specific flags
# ---------------------------------------------------------------------------


def test_vllm_enables_prefix_caching_and_a_vram_budget(services: dict[str, Any]) -> None:
    text = _run_text(services["vllm"])

    assert "--enable-prefix-caching" in text
    assert "--gpu-memory-utilization" in text
    assert "VLLM_GPU_MEMORY_UTILIZATION" in text


def test_sglang_keeps_radix_attention_and_enables_metrics(
    services: dict[str, Any],
) -> None:
    text = _run_text(services["sglang"])

    # RadixAttention is on by default; make sure nobody disables it.
    assert "--disable-radix-cache" not in text
    assert "--enable-metrics" in text
    assert "--mem-fraction-static" in text
    assert "SGLANG_MEM_FRACTION_STATIC" in text
    assert "sglang.launch_server" in text


def test_llama_server_uses_q8_gguf_parallel_slots_and_metrics(
    services: dict[str, Any],
) -> None:
    text = _run_text(services["llama-server"])

    assert "-np 4" in text
    assert "--metrics" in text
    assert "--n-gpu-layers" in text
    assert "q8_0" in text.lower()
    assert "MODEL_FILE" in text


def test_llama_server_mounts_model_dir_read_only(services: dict[str, Any]) -> None:
    mounts = _mounts(services["llama-server"])

    matching = [(source, read_only) for source, target, read_only in mounts if target == "/models"]
    assert matching, "llama-server must mount MODEL_DIR at /models"
    source, read_only = matching[0]
    assert "MODEL_DIR" in source
    assert read_only is True


@pytest.mark.parametrize("engine", ("vllm", "sglang"))
def test_transformers_engines_share_the_same_model_defaults(
    services: dict[str, Any], engine: str
) -> None:
    text = _run_text(services[engine])

    assert "${MODEL_ID:-Qwen/Qwen2.5-7B-Instruct}" in text
    assert "${SERVED_MODEL_NAME:-qwen2.5-7b-instruct}" in text


def test_llama_server_uses_the_same_served_model_alias(services: dict[str, Any]) -> None:
    text = _run_text(services["llama-server"])

    assert "${SERVED_MODEL_NAME:-qwen2.5-7b-instruct}" in text


def test_compose_gives_transformers_engines_shared_memory(services: dict[str, Any]) -> None:
    for engine in ("vllm", "sglang"):
        assert str(services[engine]["shm_size"]).lower() in {"16g", "16gb"}


# ---------------------------------------------------------------------------
# Configuration template
# ---------------------------------------------------------------------------


def test_env_example_exists_and_is_not_gitignored() -> None:
    assert ENV_EXAMPLE_PATH.is_file()

    gitignore = (REPO_ROOT / ".gitignore").read_text(encoding="utf-8")
    assert "docker/.env" in gitignore
    assert "docker/.env.example" not in gitignore


def test_env_example_documents_every_compose_variable() -> None:
    compose_text = COMPOSE_PATH.read_text(encoding="utf-8")
    env_text = ENV_EXAMPLE_PATH.read_text(encoding="utf-8")
    declared = set(re.findall(r"(?m)^([A-Z][A-Z0-9_]*)=", env_text))

    referenced = set(COMPOSE_VAR_RE.findall(compose_text))
    missing = sorted(referenced - declared)

    assert referenced, "Compose file should interpolate configuration"
    assert not missing, f"missing from docker/.env.example: {missing}"


def test_env_example_documents_launcher_settings() -> None:
    env_text = ENV_EXAMPLE_PATH.read_text(encoding="utf-8")
    launcher_text = LAUNCHER_PATH.read_text(encoding="utf-8")
    declared = set(re.findall(r"(?m)^([A-Z][A-Z0-9_]*)=", env_text))
    referenced = set(SHELL_VAR_RE.findall(launcher_text))

    # The launcher may reference shell built-ins and prompt variables; only its
    # user-facing configuration must be documented.
    for name in ("ENGINE_TIMEOUT", "HF_CACHE_DIR", "MODEL_DIR"):
        assert name in declared
        assert name in referenced


@requires_bash
def test_env_example_is_shell_sourceable() -> None:
    proc = subprocess.run(
        [
            BASH,
            "-c",
            'set -a; source "$1"; set +a; printf "%s" "${HF_CACHE_DIR:-}"',
            "_",
            ".env.example",
        ],
        cwd=DOCKER_DIR,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert proc.returncode == 0, proc.stderr
    assert proc.stdout


# ---------------------------------------------------------------------------
# Launcher
# ---------------------------------------------------------------------------


def test_launcher_is_a_strict_bash_script() -> None:
    text = LAUNCHER_PATH.read_text(encoding="utf-8")

    assert text.startswith("#!/usr/bin/env bash")
    assert "set -euo pipefail" in text


def test_launcher_covers_every_engine_and_endpoint() -> None:
    text = LAUNCHER_PATH.read_text(encoding="utf-8")

    for engine in ENGINES:
        assert engine in text
    for command in ("start", "wait", "stop", "status", "logs"):
        assert command in text
    assert "/health" in text
    assert "/v1/models" in text
    assert "/metrics" in text
    assert "ENGINE_TIMEOUT" in text


@pytest.mark.skipif(
    not _tracked("docker/launch_engine.sh"),
    reason="launcher is not tracked yet (run git add before pytest)",
)
def test_launcher_is_committed_executable() -> None:
    proc = _git("ls-files", "-s", "--", "docker/launch_engine.sh")

    assert proc.returncode == 0
    assert proc.stdout.startswith("100755 "), proc.stdout


@requires_bash
def test_launcher_has_valid_shell_syntax() -> None:
    proc = subprocess.run(
        [BASH, "-n", "launch_engine.sh"],
        cwd=DOCKER_DIR,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert proc.returncode == 0, proc.stderr


@requires_bash
def test_launcher_help_lists_all_engines() -> None:
    proc = subprocess.run(
        [BASH, "./launch_engine.sh", "--help"],
        cwd=DOCKER_DIR,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert proc.returncode == 0, proc.stderr
    output = proc.stdout + proc.stderr
    assert "Usage:" in output
    for engine in ENGINES:
        assert engine in output


@requires_bash
def test_launcher_rejects_unknown_commands() -> None:
    proc = subprocess.run(
        [BASH, "./launch_engine.sh", "bogus-engine"],
        cwd=DOCKER_DIR,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert proc.returncode != 0
    assert "unknown" in (proc.stdout + proc.stderr).lower()


# ---------------------------------------------------------------------------
# Git hygiene and documentation
# ---------------------------------------------------------------------------


def test_models_directory_is_kept_but_weights_are_ignored() -> None:
    assert (DOCKER_DIR / "models" / ".gitkeep").is_file()

    gitignore = (REPO_ROOT / ".gitignore").read_text(encoding="utf-8")
    assert "docker/models/*" in gitignore
    assert "!docker/models/.gitkeep" in gitignore


def test_launcher_keeps_lf_line_endings() -> None:
    assert b"\r\n" not in LAUNCHER_PATH.read_bytes()

    gitattributes = (REPO_ROOT / ".gitattributes").read_text(encoding="utf-8")
    assert "docker/*.sh text eol=lf" in gitattributes


def test_docker_readme_documents_versions_cuda_and_memory() -> None:
    text = DOCKER_README_PATH.read_text(encoding="utf-8")

    for image in EXPECTED_IMAGES.values():
        assert image in text
    for port in EXPECTED_HOST_PORTS.values():
        assert f"http://localhost:{port}/v1" in text
        assert f"http://localhost:{port}/metrics" in text

    assert "CUDA 13.0" in text
    assert "CUDA 12.4" in text
    assert "GPU_MEMORY_UTILIZATION" in text
    assert "--enable-prefix-caching" in text
    assert "RadixAttention" in text
    assert "-np 4" in text
    assert "--metrics" in text
    assert "Q8_0" in text
    assert "launch_engine.sh" in text


def test_root_readme_points_at_the_docker_assets() -> None:
    text = ROOT_README_PATH.read_text(encoding="utf-8")

    assert "| `docker/` |" in text
    assert "docker/README.md" in text
