"""Tests for the research fork/join workflow (Issue #15).

The fake clients are prompt-driven and share an asyncio barrier, so the tests
prove that the two topic researchers really run concurrently and that the
reviewer waits for both handoffs before synthesizing.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest

from ahp.models import DoneMessage, DoneStatus, HandoffMessage
from engine.client import LLMResponseError
from engine.models import LLMResult
from harness.workflows.research_fork_join import (
    LEAD_RESEARCH_PROMPT,
    REVIEWER_RESEARCH_PROMPT,
    TOPIC_AGENT_NAMES,
    TOPIC_RESEARCH_PROMPT,
    LeadResearchAgent,
    ResearchForkJoin,
    ResearchPlanError,
    ResearchTask,
    ReviewerAgent,
    load_research_topics,
)
from harness.workflows.research_schemas import (
    SUMMARY_MAX_CHARS,
    lead_plan_schema,
    reviewer_reply_schema,
    topic_reply_schema,
)

# ---------------------------------------------------------------------------
# Fake clients
# ---------------------------------------------------------------------------


def _incoming_payload(messages: list[dict[str, Any]]) -> dict[str, Any]:
    for message in messages:
        content = str(message.get("content", ""))
        if message.get("role") == "user" and content.startswith(
            "Process this incoming AHP message"
        ):
            return json.loads(content.split("\n\n", 1)[1])
    raise AssertionError("incoming AHP message not found in prompt")


def _research_request(messages: list[dict[str, Any]]) -> dict[str, Any]:
    for message in messages:
        content = str(message.get("content", ""))
        if message.get("role") == "user" and content.startswith("Research request:"):
            return json.loads(content.split("\n", 1)[1])
    raise AssertionError("research request not found in prompt")


class TopicBarrier:
    """Wait until both topic clients are in flight, or fail the test."""

    def __init__(self, timeout_s: float = 2.0) -> None:
        self.started = 0
        self._event = asyncio.Event()
        self.timeout_s = timeout_s

    async def wait(self) -> None:
        self.started += 1
        if self.started >= 2:
            self._event.set()
        await asyncio.wait_for(self._event.wait(), timeout=self.timeout_s)


class ResearchLLM:
    """Prompt-driven fake that plays one research role."""

    def __init__(
        self,
        role: str,
        task: ResearchTask | None = None,
        *,
        barrier: TopicBarrier | None = None,
    ) -> None:
        if role not in {"lead", "topic", "reviewer"}:
            raise ValueError(role)
        self.role = role
        self.task = task
        self.barrier = barrier
        self.calls: list[dict[str, Any]] = []

    async def chat(
        self,
        messages: list[dict[str, Any]],
        schema: dict[str, Any] | None = None,
        temperature: float = 0.0,
    ) -> LLMResult:
        self.calls.append(
            {
                "messages": [dict(message) for message in messages],
                "schema": schema,
                "temperature": temperature,
            }
        )
        if self.role == "lead":
            payload = self._lead(messages)
        elif self.role == "topic":
            payload = await self._topic(messages)
        else:
            payload = self._reviewer(messages)
        return LLMResult(text=json.dumps(payload), structured_data=payload)

    async def aclose(self) -> None:
        return None

    def _task(self) -> ResearchTask:
        if self.task is None:
            raise AssertionError("fake client needs a task")
        return self.task

    def _lead(self, messages: list[dict[str, Any]]) -> dict[str, Any]:
        _research_request(messages)
        task = self._task()
        return {
            "scopes": [
                {
                    "scope_id": "a",
                    "title": task.scope_titles[0],
                    "question": task.scope_questions[0],
                    "focus": "technology",
                },
                {
                    "scope_id": "b",
                    "title": task.scope_titles[1],
                    "question": task.scope_questions[1],
                    "focus": "policy",
                },
            ]
        }

    async def _topic(self, messages: list[dict[str, Any]]) -> dict[str, Any]:
        incoming = _incoming_payload(messages)
        facts = incoming["facts"]
        scope_id = str(facts["scope_id"])
        index = 0 if scope_id == "a" else 1
        if self.barrier is not None:
            await self.barrier.wait()

        task = self._task()
        scope_facts = task.scope_facts[index]
        return {
            "action": "reply",
            "reply": {
                "type": "HANDOFF",
                "to_agent": "reviewer_agent",
                "goal": f"Report scope {scope_id}",
                "summary": f"Findings for {facts['title']}.",
                "facts": {
                    "scope_id": scope_id,
                    "title": facts["title"],
                    "findings": " ".join(scope_facts),
                    "key_facts": list(scope_facts),
                    "sources": [
                        f"source-{task.task_id}-{scope_id}-{position}"
                        for position in range(1, len(scope_facts) + 1)
                    ],
                    "confidence": 0.9,
                },
            },
        }

    def _reviewer(self, messages: list[dict[str, Any]]) -> dict[str, Any]:
        incoming = _incoming_payload(messages)
        findings = incoming["facts"]["findings"]
        task = self._task()

        key_facts: list[str] = []
        sources: list[str] = []
        scope_summaries: list[dict[str, str]] = []
        for finding in findings:
            topic_facts = finding["facts"]
            for fact in topic_facts["key_facts"]:
                if fact not in key_facts:
                    key_facts.append(fact)
            sources.extend(topic_facts["sources"])
            scope_summaries.append(
                {
                    "scope_id": topic_facts["scope_id"],
                    "title": topic_facts["title"],
                    "findings": topic_facts["findings"],
                }
            )

        return {
            "action": "reply",
            "reply": {
                "type": "DONE",
                "to_agent": "user",
                "final_result": {
                    "brief": " ".join(
                        f"{summary['title']}: {summary['findings']}" for summary in scope_summaries
                    ),
                    "key_facts": key_facts,
                    "discrepancies": list(task.expected_discrepancies),
                    "sources": sources,
                    "scope_summaries": scope_summaries,
                },
                "summary": f"Joined {len(findings)} reports for {task.task_id}.",
                "status": "resolved",
            },
        }


class ScriptedLLM:
    """Queue-backed fake used for lead retry tests."""

    def __init__(self, *responses: dict[str, Any] | Exception) -> None:
        self.responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    async def chat(
        self,
        messages: list[dict[str, Any]],
        schema: dict[str, Any] | None = None,
        temperature: float = 0.0,
    ) -> LLMResult:
        self.calls.append(
            {
                "messages": [dict(message) for message in messages],
                "schema": schema,
                "temperature": temperature,
            }
        )
        if not self.responses:
            raise AssertionError("scripted LLM ran out of responses")
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return LLMResult(text=json.dumps(response), structured_data=response)

    async def aclose(self) -> None:
        return None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _task(task_id: str = "RES-001") -> ResearchTask:
    return next(task for task in load_research_topics() if task.task_id == task_id)


def _chain(
    task: ResearchTask,
    *,
    barrier: TopicBarrier | None = None,
) -> tuple[ResearchForkJoin, ResearchLLM, ResearchLLM, ResearchLLM, ResearchLLM]:
    lead_llm = ResearchLLM("lead", task)
    topic_a_llm = ResearchLLM("topic", task, barrier=barrier)
    topic_b_llm = ResearchLLM("topic", task, barrier=barrier)
    reviewer_llm = ResearchLLM("reviewer", task)
    chain = ResearchForkJoin(lead_llm, topic_a_llm, topic_b_llm, reviewer_llm)
    return chain, lead_llm, topic_a_llm, topic_b_llm, reviewer_llm


def _make_lead_handoff(task: ResearchTask) -> HandoffMessage:
    return HandoffMessage(
        conversation_id=f"research-{task.task_id}",
        from_agent="user",
        to_agent="lead_researcher",
        goal=f"Research request {task.task_id}",
        summary=task.query[:100],
        facts={"task_id": task.task_id, "query": task.query},
    )


def _make_topic_handoff(
    task: ResearchTask,
    from_agent: str,
    scope_id: str,
    index: int,
) -> HandoffMessage:
    facts = task.scope_facts[index]
    return HandoffMessage(
        conversation_id="conv-reviewer",
        from_agent=from_agent,
        to_agent="reviewer_agent",
        goal=f"Report scope {scope_id}",
        summary=f"Findings for {task.scope_titles[index]}.",
        facts={
            "scope_id": scope_id,
            "title": task.scope_titles[index],
            "findings": " ".join(facts),
            "key_facts": list(facts),
            "sources": [f"source-{scope_id}"],
            "confidence": 0.9,
        },
    )


# ---------------------------------------------------------------------------
# Concurrency and join
# ---------------------------------------------------------------------------


async def test_parallel_researchers_execute_concurrently() -> None:
    task = _task()
    barrier = TopicBarrier()
    chain, _, topic_a, topic_b, reviewer = _chain(task, barrier=barrier)

    result = await chain.run_task(task)

    assert barrier.started == 2
    assert result.done.status is DoneStatus.RESOLVED
    assert result.matches_ground_truth
    assert len(result.plan.scopes) == 2
    assert {handoff.from_agent for handoff in result.topic_handoffs} == set(TOPIC_AGENT_NAMES)
    assert len(result.messages) == 6
    assert len(topic_a.calls) == 1
    assert len(topic_b.calls) == 1
    assert len(reviewer.calls) == 1
    findings = _incoming_payload(reviewer.calls[0]["messages"])["facts"]["findings"]
    assert len(findings) == 2
    assert all(length <= SUMMARY_MAX_CHARS for length in result.summary_lengths.values())


async def test_reviewer_buffers_until_both_branches_arrive() -> None:
    task = _task()
    reviewer_llm = ResearchLLM("reviewer", task)
    reviewer = ReviewerAgent(reviewer_llm, reply_schema=reviewer_reply_schema())
    first = _make_topic_handoff(task, TOPIC_AGENT_NAMES[0], "a", 0)
    second = _make_topic_handoff(task, TOPIC_AGENT_NAMES[1], "b", 1)

    assert await reviewer.handle_handoff(first) is None
    assert reviewer.pending_branches == 1
    assert reviewer_llm.calls == []

    done = await reviewer.handle_handoff(second)

    assert isinstance(done, DoneMessage)
    assert done.status is DoneStatus.RESOLVED
    assert reviewer.pending_branches == 0
    assert len(reviewer_llm.calls) == 1
    findings = _incoming_payload(reviewer_llm.calls[0]["messages"])["facts"]["findings"]
    assert len(findings) == 2


async def test_reviewer_reset_clears_buffered_branches() -> None:
    task = _task()
    reviewer = ReviewerAgent(
        ResearchLLM("reviewer", task),
        reply_schema=reviewer_reply_schema(),
    )
    await reviewer.handle_handoff(_make_topic_handoff(task, TOPIC_AGENT_NAMES[0], "a", 0))
    assert reviewer.pending_branches == 1

    reviewer.reset()

    assert reviewer.pending_branches == 0


# ---------------------------------------------------------------------------
# Lead planning and retries
# ---------------------------------------------------------------------------


async def test_lead_plan_retries_invalid_output() -> None:
    task = _task()
    valid = {
        "scopes": [
            {
                "scope_id": "a",
                "title": task.scope_titles[0],
                "question": task.scope_questions[0],
            },
            {
                "scope_id": "b",
                "title": task.scope_titles[1],
                "question": task.scope_questions[1],
            },
        ]
    }
    llm = ScriptedLLM({"scopes": []}, valid)
    lead = LeadResearchAgent(llm)

    plan = await lead.plan(_make_lead_handoff(task))

    assert len(plan.scopes) == 2
    assert lead.calls == 2
    assert "rejected" in llm.calls[1]["messages"][-1]["content"]


async def test_lead_plan_retries_llm_response_errors() -> None:
    task = _task()
    valid = {
        "scopes": [
            {
                "scope_id": "a",
                "title": task.scope_titles[0],
                "question": task.scope_questions[0],
            },
            {
                "scope_id": "b",
                "title": task.scope_titles[1],
                "question": task.scope_questions[1],
            },
        ]
    }
    llm = ScriptedLLM(LLMResponseError("bad json"), valid)
    lead = LeadResearchAgent(llm)

    plan = await lead.plan(_make_lead_handoff(task))

    assert len(plan.scopes) == 2
    assert lead.calls == 2
    assert "rejected" in llm.calls[1]["messages"][-1]["content"]


async def test_lead_plan_exhausts_retries() -> None:
    task = _task()
    llm = ScriptedLLM({"scopes": []}, {"scopes": []}, {"scopes": []})
    lead = LeadResearchAgent(llm, max_retries=2)

    with pytest.raises(ResearchPlanError):
        await lead.plan(_make_lead_handoff(task))

    assert lead.calls == 3


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "task",
    load_research_topics(),
    ids=lambda task: task.task_id,
)
async def test_every_dataset_task_matches_ground_truth(task: ResearchTask) -> None:
    chain, _, _, _, reviewer = _chain(task)

    result = await chain.run_task(task)

    assert result.matches_ground_truth, (
        task.task_id,
        sorted(set(task.key_facts) - set(result.key_facts)),
    )
    assert result.done.status is DoneStatus.RESOLVED
    assert len(result.topic_handoffs) == 2
    assert len(result.plan.scopes) == 2
    assert result.discrepancies == task.expected_discrepancies
    assert len(reviewer.calls) == 1
    assert all(length <= SUMMARY_MAX_CHARS for length in result.summary_lengths.values())


async def test_discrepancy_task_surfaces_conflicts() -> None:
    task = _task("RES-004")
    chain, _, _, _, _ = _chain(task)

    result = await chain.run_task(task)

    assert result.discrepancies == task.expected_discrepancies
    assert result.discrepancies


# ---------------------------------------------------------------------------
# Prompts, schemas and configuration
# ---------------------------------------------------------------------------


def test_prompts_and_schemas_load() -> None:
    assert "LEAD RESEARCH AGENT" in LEAD_RESEARCH_PROMPT
    assert "TOPIC RESEARCH AGENT" in TOPIC_RESEARCH_PROMPT
    assert "REVIEWER AGENT" in REVIEWER_RESEARCH_PROMPT
    assert "reviewer_agent" in TOPIC_RESEARCH_PROMPT
    assert "discrepancies" in REVIEWER_RESEARCH_PROMPT
    assert SUMMARY_MAX_CHARS == 600
    assert lead_plan_schema()["properties"]["scopes"]["minItems"] == 2
    assert topic_reply_schema()["properties"]["facts"]["required"]
    assert reviewer_reply_schema()["properties"]["final_result"]["required"]


async def test_agents_request_role_specific_schemas() -> None:
    task = _task()
    chain, _, topic_a, topic_b, reviewer = _chain(task)

    await chain.run_task(task)

    expected_topic = topic_reply_schema(SUMMARY_MAX_CHARS)
    for llm in (topic_a, topic_b):
        assert llm.calls[0]["schema"]["oneOf"][1]["properties"]["reply"] == expected_topic
    assert reviewer.calls[0]["schema"]["oneOf"][1]["properties"]["reply"] == reviewer_reply_schema(
        SUMMARY_MAX_CHARS
    )


def test_from_shared_llm_builds_all_agents() -> None:
    task = _task()
    chain = ResearchForkJoin.from_shared_llm(ResearchLLM("lead", task))

    assert set(chain.agents) == {*TOPIC_AGENT_NAMES, "reviewer_agent"}


def test_constructor_validation() -> None:
    llm = ResearchLLM("lead", _task())
    with pytest.raises(ValueError):
        ResearchForkJoin(llm, llm, llm, llm, summary_max_chars=0)
    with pytest.raises(ValueError):
        ResearchForkJoin(llm, llm, llm, llm, timeout_s=0)
    with pytest.raises(ValueError):
        ResearchForkJoin(llm, llm, llm, llm, max_hops=0)
    with pytest.raises(ValueError):
        ResearchForkJoin(llm, llm, llm, llm, ack_timeout_s=0)
    with pytest.raises(ValueError):
        LeadResearchAgent(llm, max_retries=-1)
    with pytest.raises(ValueError):
        ReviewerAgent(llm, reply_schema=reviewer_reply_schema(), expected_branches=0)


async def test_long_query_and_scope_questions_are_truncated() -> None:
    task = ResearchTask(
        task_id="RES-LONG",
        query="q" * 1000,
        ground_truth_facts={
            "scope_titles": ["A" * 500, "B" * 500],
            "scope_questions": ["Q" * 1000, "R" * 1000],
            "scope_facts": [["fact a"], ["fact b"]],
            "key_facts": ["fact a", "fact b"],
            "expected_discrepancies": [],
        },
    )
    chain, _, _, _, _ = _chain(task)

    result = await chain.run_task(task)

    initial = result.messages[0]
    assert isinstance(initial, HandoffMessage)
    assert len(initial.summary) == SUMMARY_MAX_CHARS
    assert initial.summary.endswith("…")
    assert all(len(handoff.summary) <= SUMMARY_MAX_CHARS for handoff in result.topic_handoffs)
    assert result.matches_ground_truth
