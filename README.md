# Agent Handoff Protocol (AHP)

**A protocol and benchmarking framework for reliable multi-agent communication and efficient LLM inference.**

## Overview

Multi-agent AI systems rely on multiple specialized agents collaborating to complete complex tasks. However, transferring information between agents introduces challenges such as context loss, ambiguous instructions, malformed messages, and unreliable handoffs.

At the same time, these workflows generate complex inference workloads involving chained requests, parallel execution, and repeated context. Traditional LLM benchmarks may not adequately capture these characteristics.

This project addresses both challenges by designing an **Agent Handoff Protocol (AHP)** and implementing a benchmarking harness to evaluate LLM inference engines under realistic multi-agent workloads.

## Project Objectives

The project has two primary objectives:

1. **Reliable Agent Communication:** Design a structured protocol that standardizes how agents transfer tasks, exchange context, acknowledge messages, and handle communication failures.

2. **Efficient LLM Inference:** Build a benchmarking framework to compare inference engines under realistic multi-agent workloads, evaluating their performance, resource utilization, and reliability.

## Architecture

The project consists of three main components:

### 1. Agent Handoff Protocol (AHP)

A standardized communication specification defining how AI agents transfer work between each other.

It covers:

- Structured handoff messages and required fields.
- Message types such as `HANDOFF`, `ACK`, `NEED_INFO`, and `DONE`.
- Context and task information transfer.
- Message validation and error handling.
- Timeout, retry, escalation, and loop prevention mechanisms.

### 2. Harness Implementation

A software framework that implements the protocol and provides a controlled environment for running multi-agent workflows.

Its responsibilities include:

- Defining and orchestrating specialized agents.
- Routing and validating AHP messages.
- Managing communication failures and retries.
- Integrating with different LLM inference engines through a common interface.
- Executing repeatable workloads and collecting experimental data.

### 3. LLM Inference Evaluation

A systematic evaluation of inference engines using realistic multi-agent workloads.

The evaluation focuses on:

- End-to-end workflow latency.
- Time to First Token (TTFT).
- Throughput under concurrent workloads.
- Token consumption.
- GPU memory utilization.
- Prefix caching effectiveness.
- Structured output reliability.
- Overall workflow success rate.

## Experimental Approach

The framework will execute identical multi-agent workloads across different inference engines while keeping the model, prompts, agent configurations, and workload scenarios consistent.

Potential inference engines include:

- [vLLM](https://github.com/vllm-project/vllm)
- [SGLang](https://github.com/sgl-project/sglang)
- [llama.cpp](https://github.com/ggml-org/llama.cpp)

The experiments aim to identify how different serving architectures handle the unique characteristics of multi-agent inference workloads.

Additionally, comparing structured AHP-based communication against conventional free-text handoffs can help quantify the protocol's impact on reliability.

## Expected Outcomes

- A documented Agent Handoff Protocol specification.
- A working harness implementing AHP.
- Reproducible multi-agent workload scenarios.
- An extensible interface for integrating inference engines.
- Benchmarking results and performance comparisons.
- An analysis of the trade-offs between communication reliability and inference efficiency.

## Project Scope

This project explores the intersection of **Multi-Agent Systems, LLM Infrastructure, and Inference Performance Engineering**.

Rather than building another standalone AI application, the focus is on the infrastructure that makes multi-agent systems more reliable, measurable, and efficient.

## Repository Layout

| Path | Purpose |
|---|---|
| `ahp/` | AHP message models, validation, state machine, router and fault injection |
| `harness/` | Agent runtime, prompts, mock tools and workflows |
| `engine/` | Unified `LLMClient`, engine adapters and mock server |
| `telemetry/` | Event schema, sinks, storage and metric samplers |
| `bench/` | Workload driver and benchmark matrix orchestrator |
| `tests/` | Unit, integration and fuzz tests |
| `docker/` | Pinned Compose profiles, launcher and docs for vLLM, SGLang and llama.cpp |
| `configs/` | YAML experiment configurations |
| `data/` | Frozen workflow task sets |
| `notebooks/` | Analysis notebooks |
| `docs/` | AHP specification |
| `schemas/` | Exported JSON Schemas for constrained decoding |

## Development

Requires Python 3.12 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync                       # create .venv and install locked dependencies
uv run pytest                 # run the test suite
uv run ruff check .           # lint
uv run ruff format --check .  # verify formatting
uv run python -m ahp.schema_export   # regenerate schemas/*.schema.json
```

The Day-1 interface contracts live in `ahp/interfaces.py`, `engine/interfaces.py`
and `telemetry/interfaces.py`; code against them while the concrete
implementations land.

GPU inference engines run in Docker. See `docker/README.md` for the pinned
vLLM, SGLang and llama.cpp images, the `launch_engine.sh` readiness workflow,
and the VRAM allocation settings.

---

**Status:** In Development

**Primary Areas:** Multi-Agent Systems · LLM Inference · Protocol Design · Benchmarking · Performance Engineering
