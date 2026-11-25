#!/usr/bin/env bash
# Start, wait for, inspect and stop one AHP inference engine.
#
# The engine is defined in docker-compose.yml and runs behind an
# OpenAI-compatible API on container port 8000. This script maps it to the
# engine's host port, polls /health and /v1/models until the model is loaded,
# and stops only that engine so sibling engines keep running.
#
# Usage: ./launch_engine.sh <command> [engine]
# Run `./launch_engine.sh --help` for details.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
COMPOSE_FILE="$SCRIPT_DIR/docker-compose.yml"
ENGINES="vllm sglang llama-server"

engine=""
COMPOSE_CMD=()

usage() {
  cat <<'EOF'
Usage: launch_engine.sh <command> [engine]

Commands:
  start  <engine>   Start the engine in the background, then wait until ready (default)
  wait   <engine>   Wait for an already running engine to become ready
  stop   <engine>   Stop and remove only that engine's container
  status <engine>   Show container state and probe readiness
  logs   <engine>   Follow the engine log (TAIL=n sets history, default 200)
  list              List engines, host ports and API endpoints

Engines:
  vllm            vLLM with prefix caching          port 8001
  sglang          SGLang with RadixAttention        port 8002
  llama-server    llama.cpp server (GGUF, -np 4)    port 8003

Examples:
  ./launch_engine.sh vllm
  ./launch_engine.sh start sglang
  ENGINE_TIMEOUT=1800 ./launch_engine.sh start vllm
  ./launch_engine.sh logs llama-server

Environment:
  ENGINE_TIMEOUT       Seconds to wait for readiness (default 900)
  ENGINE_POLL_INTERVAL Seconds between readiness probes (default 2)
  HF_CACHE_DIR         Host Hugging Face cache (default $HOME/.cache/huggingface)
  MODEL_DIR            Host dir with the llama.cpp GGUF (default docker/models)
  VLLM_PORT, SGLANG_PORT, LLAMA_SERVER_PORT
                       Host ports (default 8001, 8002, 8003)
EOF
}

die() {
  printf 'error: %s\n' "$*" >&2
  exit 1
}

load_env() {
  if [[ -f "$SCRIPT_DIR/.env" ]]; then
    set -a
    # shellcheck source=/dev/null
    source "$SCRIPT_DIR/.env"
    set +a
  fi

  HF_CACHE_DIR="${HF_CACHE_DIR:-$HOME/.cache/huggingface}"
  MODEL_DIR="${MODEL_DIR:-$SCRIPT_DIR/models}"
  MODEL_FILE="${MODEL_FILE:-model-q8_0.gguf}"
  ENGINE_TIMEOUT="${ENGINE_TIMEOUT:-900}"
  ENGINE_POLL_INTERVAL="${ENGINE_POLL_INTERVAL:-2}"

  # Resolve relative host paths against docker/ so the script behaves the same
  # from any working directory (Compose also resolves them against docker/).
  [[ "$HF_CACHE_DIR" = /* ]] || HF_CACHE_DIR="$SCRIPT_DIR/${HF_CACHE_DIR#./}"
  [[ "$MODEL_DIR" = /* ]] || MODEL_DIR="$SCRIPT_DIR/${MODEL_DIR#./}"

  export HF_CACHE_DIR MODEL_DIR
}

require_docker() {
  command -v docker >/dev/null 2>&1 || die "docker is not installed or not on PATH"
  if docker compose version >/dev/null 2>&1; then
    COMPOSE_CMD=(docker compose)
  elif command -v docker-compose >/dev/null 2>&1; then
    COMPOSE_CMD=(docker-compose)
  else
    die "Docker Compose v2 is required ('docker compose' not available)"
  fi
}

compose() {
  "${COMPOSE_CMD[@]}" -f "$COMPOSE_FILE" --project-directory "$SCRIPT_DIR" \
    --profile "$engine" "$@"
}

validate_engine() {
  case "$1" in
    vllm | sglang | llama-server) return 0 ;;
    *) die "unknown engine '$1' (expected one of: $ENGINES)" ;;
  esac
}

port_for() {
  case "$1" in
    vllm) printf '%s' "${VLLM_PORT:-8001}" ;;
    sglang) printf '%s' "${SGLANG_PORT:-8002}" ;;
    llama-server) printf '%s' "${LLAMA_SERVER_PORT:-8003}" ;;
  esac
}

# HTTP GET that succeeds for 2xx responses. Prefers curl, falls back to wget.
http_ok() {
  local url="$1"
  if command -v curl >/dev/null 2>&1; then
    curl -fsS --max-time 3 "$url" >/dev/null 2>&1
  elif command -v wget >/dev/null 2>&1; then
    wget -q -T 3 -O /dev/null "$url" 2>/dev/null
  else
    die "curl or wget is required on the host for readiness checks"
  fi
}

wait_ready() {
  local port start deadline elapsed dots
  port="$(port_for "$engine")"
  start=$SECONDS
  deadline=$((start + ENGINE_TIMEOUT))
  dots=0

  printf 'Waiting up to %ss for %s at http://127.0.0.1:%s ...' \
    "$ENGINE_TIMEOUT" "$engine" "$port"

  while ((SECONDS < deadline)); do
    if http_ok "http://127.0.0.1:${port}/health" &&
      http_ok "http://127.0.0.1:${port}/v1/models"; then
      elapsed=$((SECONDS - start))
      printf '\n%s is ready after %ss\n' "$engine" "$elapsed"
      printf '  OpenAI API : http://127.0.0.1:%s/v1\n' "$port"
      printf '  metrics    : http://127.0.0.1:%s/metrics\n' "$port"
      return 0
    fi

    sleep "$ENGINE_POLL_INTERVAL"
    dots=$((dots + 1))
    printf '.'
    if ((dots % 15 == 0)); then
      printf ' %ss\n' "$((SECONDS - start))"
    fi
  done

  printf '\n'
  return 1
}

ensure_model_file() {
  local path="$MODEL_DIR/$MODEL_FILE"
  if [[ ! -f "$path" ]]; then
    printf 'error: llama.cpp weights not found: %s\n\n' "$path" >&2
    printf 'llama-server cannot download a Hugging Face repository.\n' >&2
    printf 'Export %s to a Q8_0 GGUF and place it in MODEL_DIR, or set\n' \
      "${MODEL_ID:-the benchmark checkpoint}" >&2
    printf 'MODEL_DIR/MODEL_FILE in docker/.env (see docker/README.md).\n' >&2
    return 1
  fi
}

ensure_host_dirs() {
  mkdir -p "$HF_CACHE_DIR"
  [[ "$engine" == "llama-server" ]] && mkdir -p "$MODEL_DIR"
  return 0
}

stop_engine() {
  printf 'Stopping %s ...\n' "$engine"
  compose stop "$engine" >/dev/null 2>&1 || true
  compose rm -f "$engine" >/dev/null 2>&1 || true
}

start_engine() {
  ensure_host_dirs
  if [[ "$engine" == "llama-server" ]]; then
    ensure_model_file || return 1
  fi

  printf 'Starting %s (model %s, host port %s) ...\n' \
    "$engine" "${MODEL_ID:-default}" "$(port_for "$engine")"

  if ! compose up -d "$engine"; then
    die "failed to start $engine; is host port $(port_for "$engine") already in use?"
  fi

  if ! wait_ready; then
    printf 'error: %s did not become ready within %ss; last log lines:\n' \
      "$engine" "$ENGINE_TIMEOUT" >&2
    compose logs --tail 80 "$engine" >&2 || true
    stop_engine
    die "$engine failed its readiness check"
  fi
}

status_engine() {
  local port
  port="$(port_for "$engine")"
  compose ps "$engine" || true

  if http_ok "http://127.0.0.1:${port}/health" &&
    http_ok "http://127.0.0.1:${port}/v1/models"; then
    printf '%s: ready (http://127.0.0.1:%s/v1, metrics: http://127.0.0.1:%s/metrics)\n' \
      "$engine" "$port" "$port"
  else
    printf '%s: not ready (container starting or stopped)\n' "$engine"
    return 1
  fi
}

logs_engine() {
  compose logs -f --tail "${TAIL:-200}" "$engine"
}

list_engines() {
  local e
  printf '%-14s %-6s %s\n' "ENGINE" "PORT" "OPENAI BASE URL"
  for e in $ENGINES; do
    printf '%-14s %-6s %s\n' "$e" "$(port_for "$e")" "http://127.0.0.1:$(port_for "$e")/v1"
  done
}

main() {
  local command="${1:-}"
  local target="${2:-}"

  case "$command" in
    -h | --help | help | "")
      usage
      return 0
      ;;
    list)
      load_env
      list_engines
      return 0
      ;;
    start | wait | stop | status | logs)
      engine="$target"
      ;;
    vllm | sglang | llama-server)
      # `./launch_engine.sh vllm` is shorthand for `start vllm`.
      engine="$command"
      command="start"
      ;;
    *)
      usage >&2
      die "unknown command '$command'"
      ;;
  esac

  [[ -n "$engine" ]] || die "missing engine name (expected one of: $ENGINES)"
  validate_engine "$engine"
  load_env
  require_docker

  case "$command" in
    start)
      start_engine
      ;;
    wait)
      wait_ready || die "$engine did not become ready within ${ENGINE_TIMEOUT}s"
      ;;
    stop)
      stop_engine
      ;;
    status)
      status_engine
      ;;
    logs)
      logs_engine
      ;;
  esac
}

main "$@"
