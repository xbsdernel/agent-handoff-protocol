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

### Engine client

`engine/client.py` provides `UnifiedLLMClient`, a streaming client for any
OpenAI-compatible endpoint. Switching engines is a `base_url` change:

```python
from engine import UnifiedLLMClient

async with UnifiedLLMClient(
    base_url="http://localhost:8002/v1",
    model="qwen2.5-7b-instruct",
) as llm:
    result = await llm.chat(messages, schema=response_schema, temperature=0.0)
    print(result.text, result.ttft_ms, result.tokens_per_second)
```

It measures TTFT and total latency client-side with `perf_counter_ns()`, reads
token counts from streaming usage metadata, validates structured JSON outputs
(raising `LLMResponseError` for malformed or non-conforming documents) and
emits one `llm_call` event per request.

### Agent runtime and tools

`harness/agent.py` provides `BaseAgent`, which implements the AHP `Agent`
protocol on top of any `LLMClient`. On each turn the model returns one action:
a `tool_call` (executed and appended to the prompt) or a `reply`. Only a
validated `HANDOFF`, `NEED_INFO` or `DONE` message is returned to the router.
Invalid output is re-prompted up to two times before raising `AgentReplyError`,
which the router reports as `AGENT_ERROR`.

`harness/tools/sqlite_mock.py` seeds an in-memory SQLite database with
customers, charges and inventory, and exposes `query_customer`,
`query_charges` and `query_inventory` tools:

```python
from harness import BaseAgent, SQLiteMockDB, sqlite_tools

db = SQLiteMockDB()
agent = BaseAgent("triage", "You triage billing tickets.", llm, tools=sqlite_tools(db))
reply = await agent.handle_handoff(incoming_handoff)
```

Call `db.reset()` between scenarios to restore the deterministic fixtures.

### Support triage workflow

`harness/workflows/support_chain.py` implements the canonical linear chain
`customer -> triage_agent -> billing_agent -> reply_agent -> DONE` on top of the
in-process router:

- **Triage** classifies the issue, verifies the customer and hands off to
  billing. If the customer details are missing it can ask the customer or let
  billing ask triage with a `NeedInfoMessage`.
- **Billing** calls `query_charges` and the deterministic `assess_refund` tool,
  then hands the decision to reply.
- **Reply** drafts the customer-facing answer and closes the workflow with a
  validated `DoneMessage`.

Role prompts live in `harness/workflows/prompts/`, role output schemas in
`harness/workflows/schemas.py`, and every summary is capped at 600 characters.
`BaseAgent` merges incoming facts into each outgoing `HANDOFF`, so downstream
agents never repeat earlier work.

```python
from harness import SupportChain, load_support_tickets

chain = SupportChain.from_shared_llm(llm)  # one client for all three roles
ticket = load_support_tickets()[0]

async with chain:
    result = await chain.run_ticket(ticket)
    print(result.decision, result.refund_amount, result.done.status)
```

`data/support_tickets.jsonl` is a frozen set of 50 synthetic scenarios with
`expected_decision` and `expected_refund_amount` ground truth: 12 refunds, 27
no-refund cases and 11 escalations, including six tickets whose input text
deliberately omits the customer ID to exercise the `NeedInfoMessage` path.
`assess_refund` is the single source of truth for those labels.

### Research fork/join workflow

`harness/workflows/research_fork_join.py` implements the parallel fork/join
pattern:

```
user -> lead_researcher -> [topic_researcher_a || topic_researcher_b] -> reviewer_agent -> DONE
```

- **Lead** asks the model for a structured two-scope plan and retries invalid
  output before raising `ResearchPlanError` (`AGENT_ERROR`).
- **Topic researchers** run in separate router queues and are dispatched with
  `asyncio.gather`, so both branches are genuinely in flight at the same time.
- **Reviewer** buffers handoffs keyed by `from_agent`, returns `None` for the
  first branch, and only calls the model once both reports have arrived. It
  merges the facts, lists discrepancies and closes the workflow with a
  validated `DoneMessage`.

Role prompts live in `harness/workflows/prompts/`, schemas in
`harness/workflows/research_schemas.py`, and every summary is capped at 600
characters.

```python
from harness import ResearchForkJoin, load_research_topics

chain = ResearchForkJoin.from_shared_llm(llm)  # one client for all four roles
task = load_research_topics()[0]
result = await chain.run_task(task)
print(result.brief, result.key_facts, result.discrepancies)
```

`data/research_topics.jsonl` is a frozen set of 30 research queries
(`RES-001`–`RES-030`). Each row carries two non-overlapping scopes with
`scope_titles`, `scope_questions` and per-scope `scope_facts`; `key_facts` is
the union of both scopes and `expected_discrepancies` is non-empty for
`RES-004`, `RES-008` and `RES-021`. `ResearchRunResult.matches_ground_truth`
checks that the reviewer's brief covers every key fact.

### Bug triage, fix and test loop

`harness/workflows/bug_fix_loop.py` implements the iterative repair cycle:

```
user -> bug_triage_agent -> developer_agent <-> tester_agent -> DONE
```

- **Triage** turns the stack trace, mock source file and failing test names
  into a reproduction specification.
- **Developer** reads the current file through the `read_file` tool and proposes
  a minimal list of `find`/`replace` edits.
- **Tester** applies the patch atomically in a deterministic `BugSandbox`, runs
  the synthetic unit tests, and either bounces the handoff back with real
  failure logs or closes the workflow with `DONE`.

Loop safety is enforced by the orchestrator, not only by prompts:

- at most `max_repair_cycles` (default 3) developer → tester rounds;
- a repeated patch fingerprint escalates immediately before re-running tests;
- both cases end with `DONE(status="escalated_unresolved")` and an explicit
  `reason` in `final_result`.

```python
from harness import BugFixLoop, load_bug_reports

loop = BugFixLoop.from_shared_llm(llm)  # one client for all three roles
report = load_bug_reports()[0]
result = await loop.run_report(report)
print(result.outcome, result.developer_attempts, result.tests_passed)
```

`data/bug_reports.jsonl` is a frozen set of 25 scenarios (`BUG-001`–`BUG-025`):
15 resolve within one to three attempts, 8 exhaust the three-cycle limit, and 2
repeat an ineffective patch. `BugFixRunResult.matches_ground_truth` checks the
terminal status, the developer attempt count and the escalation reason.

---

**Status:** In Development

**Primary Areas:** Multi-Agent Systems · LLM Inference · Protocol Design · Benchmarking · Performance Engineering
