# Docker assets

Reproducible serving stack for the three inference engines compared by the
benchmark. Each engine serves the same checkpoint behind an OpenAI-compatible
API on container port `8000`, exports Prometheus metrics on `/metrics`, and
mounts the same host Hugging Face cache.

| Compose profile / service | Pinned image | CUDA runtime | Host API | Host metrics |
|---|---|---|---|---|
| `vllm` | `vllm/vllm-openai:v0.34.0` | CUDA 13.0.2 | http://localhost:8001/v1 | http://localhost:8001/metrics |
| `sglang` | `lmsysorg/sglang:v0.5.23-cu130` | CUDA 13.0.3 | http://localhost:8002/v1 | http://localhost:8002/metrics |
| `llama-server` | `ghcr.io/ggml-org/llama.cpp:server-cuda-b4738` | CUDA 12.4.0 | http://localhost:8003/v1 | http://localhost:8003/metrics |

Images are pinned by tag; never replace them with `latest`. The CUDA runtime
versions above were read from each image's `CUDA_VERSION` environment variable,
not assumed from the tag.

## Files

| File | Purpose |
|---|---|
| `docker-compose.yml` | One profile per engine, GPU reservations, healthchecks, HF cache mount. |
| `launch_engine.sh` | Start / wait / stop / status / logs for a single engine with a readiness gate. |
| `.env.example` | Template for `docker/.env` (ports, model, cache paths, VRAM budgets). |
| `models/` | Host directory for the llama.cpp Q8_0 GGUF (git-ignored). |

## Prerequisites

- Linux host with an NVIDIA GPU (Ampere or newer recommended).
- NVIDIA driver: **R580 or newer** for the CUDA 13.0 vLLM and SGLang images.
  The llama.cpp CUDA 12.4 image also runs on R580+; its minimum is R550.
  Verify with `nvidia-smi`.
- Docker Engine **26.1+** with Compose v2. The NVIDIA Container Toolkit 1.18+
  defaults to CDI mode, which needs Docker 26.1 or later.
- NVIDIA Container Toolkit installed and configured
  (`nvidia-ctk runtime configure --runtime=docker`).
- Enough free VRAM for one engine at a time (see the memory table below).
- Host `curl` (used by `launch_engine.sh` for readiness probes).

Confirm GPU access before starting an engine:

```bash
docker run --rm --gpus all nvidia/cuda:13.0.0-base-ubuntu24.04 nvidia-smi
```

## Quick start

```bash
cd docker
cp .env.example .env          # edit HF_CACHE_DIR and, for llama.cpp, MODEL_FILE
chmod +x launch_engine.sh     # preserved as 100755 in git

./launch_engine.sh vllm       # or: sglang, llama-server
```

`launch_engine.sh` starts one profile, polls `GET /health` followed by
`GET /v1/models`, and prints the API/metrics URLs once the model is loaded.
If readiness does not succeed within `ENGINE_TIMEOUT` (default 900 s) it dumps
the container log, removes the container, and exits non-zero. Model downloads
on first start are cached in `HF_CACHE_DIR`, so later starts are much faster.

Verify the OpenAI surface and metrics:

```bash
curl -s http://localhost:8001/v1/models | python -m json.tool
curl -s http://localhost:8001/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen2.5-7b-instruct","messages":[{"role":"user","content":"hi"}],"max_tokens":16}'
curl -s http://localhost:8001/metrics | head
```

Manage the engine:

```bash
./launch_engine.sh status vllm
./launch_engine.sh logs vllm
./launch_engine.sh stop vllm     # stops only vLLM; other engines keep running
./launch_engine.sh list
```

Compose can also be used directly:

```bash
export HF_CACHE_DIR="$HOME/.cache/huggingface"
docker compose --profile vllm up -d
docker compose --profile vllm logs -f
docker compose --profile vllm down
```

## Same model on every engine

vLLM and SGLang read `MODEL_ID` and download the checkpoint into the shared
cache. llama.cpp needs a Q8_0 GGUF export of the *same* checkpoint in
`MODEL_DIR`, because it does not download Hugging Face repositories:

```bash
# One-time export, run from the repository root.
hf download Qwen/Qwen2.5-7B-Instruct-GGUF \
  --include '*q8_0.gguf' --local-dir docker/models

# docker/.env
MODEL_DIR=./models
MODEL_FILE=qwen2.5-7b-instruct-q8_0.gguf
```

Using a different checkpoint or a non-Q8_0 quantization invalidates
cross-engine comparisons; the `SERVED_MODEL_NAME` alias is kept identical so
the same client requests work against every engine.

Gated checkpoints (for example `Meta-Llama-3.1-8B-Instruct`) need `HF_TOKEN` in
`docker/.env`. That file is git-ignored.

## Memory allocation

| Engine | Parameter | Default | Notes |
|---|---|---|---|
| vLLM | `VLLM_GPU_MEMORY_UTILIZATION` | `0.90` | Fraction of VRAM for weights + KV cache. Lower it if another process shares the GPU. |
| SGLang | `SGLANG_MEM_FRACTION_STATIC` | `0.85` | Static VRAM fraction for weights + RadixAttention cache. |
| vLLM / SGLang | `MAX_MODEL_LEN` | `8192` | Context window; raising it grows the KV cache. |
| llama.cpp | `LLAMA_N_GPU_LAYERS` | `999` | Offloads every layer that fits. |
| llama.cpp | `LLAMA_CTX_SIZE` | `8192` | Total context, divided across the four `-np 4` slots. |
| vLLM / SGLang | `shm_size` | `16gb` | Shared memory for tensor-parallel workers. |

Approximate single-GPU footprint for a 7B model: vLLM/SGLang ≈ 16–20 GB
(FP16 weights plus KV cache at 8K context, 90% reservation on a 24 GB card);
llama.cpp Q8_0 ≈ 9–11 GB. Run engines one at a time on a 24 GB GPU, or lower
the reservation knobs to co-locate them.

## Engine-specific caching and metrics

- **vLLM** runs with `--enable-prefix-caching`; Prometheus metrics are served
  on `/metrics` by default.
- **SGLang** uses RadixAttention prefix caching by default; `--enable-metrics`
  turns on `/metrics`, including prefix-cache hit-rate gauges.
- **llama.cpp** runs four parallel slots (`-np 4`) and `--metrics`, serving
  `/metrics` from `llama-server`.

All three expose `/health` plus the OpenAI `/v1/models` listing. Compose
healthchecks and `launch_engine.sh` both require `/health` and `/v1/models` to
answer, so "healthy" means the engine is serving requests, not merely that the
process is up. vLLM and llama.cpp return HTTP 503 from `/health` while the model
loads; SGLang starts its HTTP server once the engine is initialized.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `could not select device driver "" with capabilities: [[gpu]]` | Install/configure the NVIDIA Container Toolkit and restart Docker. |
| `unsupported CUDA version` or driver errors | Driver older than R580 for the CUDA 13.0 images; upgrade the driver. |
| Readiness times out on first vLLM start | The model is still downloading; watch `./launch_engine.sh logs vllm` or raise `ENGINE_TIMEOUT`. |
| `CUDA out of memory` | Lower `VLLM_GPU_MEMORY_UTILIZATION` / `SGLANG_MEM_FRACTION_STATIC`, or stop the other engine. |
| `port is already allocated` | Change `VLLM_PORT` / `SGLANG_PORT` / `LLAMA_SERVER_PORT` in `docker/.env`. |
| llama-server exits immediately | `MODEL_FILE` is missing from `MODEL_DIR`; check the path in `docker/.env`. |
| `/metrics` returns 404 | SGLang needs `--enable-metrics` and llama.cpp needs `--metrics`; vLLM serves `/metrics` by default. All are set in the compose file. |
