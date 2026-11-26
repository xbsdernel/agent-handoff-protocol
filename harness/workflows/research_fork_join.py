"""Workflow 2: research fork/join with parallel topic researchers.

The workflow fans one broad question out to two independent researchers and
joins their reports in a reviewer:

``user -> lead_researcher -> [topic_researcher_a || topic_researcher_b] -> reviewer_agent -> DONE``

* **Lead** uses structured output to split the query into exactly two
  independent scopes.
* **Topic researchers** run in separate router queues, so the two handoffs are
  processed concurrently by ``asyncio``.
* **Reviewer** buffers incoming topic handoffs and only calls the model once
  both have arrived, then merges the findings and reports discrepancies.

The module also loads the frozen 30-task dataset in ``data/research_topics.jsonl``.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    ValidationError,
    model_validator,
)

from ahp.models import (
    AHPErrorCode,
    AHPMessage,
    DoneMessage,
    HandoffMessage,
    NeedInfoMessage,
)
from ahp.router import InProcessRouter, RouterConfig
from engine.client import LLMResponseError
from engine.interfaces import LLMClient
from harness.agent import BaseAgent
from harness.workflows.research_schemas import (
    SUMMARY_MAX_CHARS,
    lead_plan_schema,
    reviewer_reply_schema,
    topic_reply_schema,
)

PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[2]
DEFAULT_DATASET_PATH: Final[Path] = PROJECT_ROOT / "data" / "research_topics.jsonl"
PROMPTS_DIR: Final[Path] = Path(__file__).with_name("prompts")

TOPIC_AGENT_NAMES: Final[tuple[str, str]] = (
    "topic_researcher_a",
    "topic_researcher_b",
)


def _load_prompt(name: str) -> str:
    return (PROMPTS_DIR / name).read_text(encoding="utf-8").strip()


LEAD_RESEARCH_PROMPT: Final[str] = _load_prompt("lead_research_prompt.txt")
TOPIC_RESEARCH_PROMPT: Final[str] = _load_prompt("topic_research_prompt.txt")
REVIEWER_RESEARCH_PROMPT: Final[str] = _load_prompt("reviewer_research_prompt.txt")


def _json(value: Any) -> str:
    return json.dumps(value, indent=2, sort_keys=True, default=str)


def _truncate(text: str, limit: int) -> str:
    stripped = text.strip()
    if len(stripped) <= limit:
        return stripped
    return stripped[: limit - 1].rstrip() + "…"


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


class ResearchScope(BaseModel):
    """One independent branch of a research plan."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    scope_id: str = Field(min_length=1, pattern=r"^[a-z0-9][a-z0-9_-]*$")
    title: str = Field(min_length=1)
    question: str = Field(min_length=1)
    focus: str = ""


class ResearchPlan(BaseModel):
    """The lead's split of one query into exactly two scopes."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    scopes: list[ResearchScope] = Field(min_length=2, max_length=2)

    @model_validator(mode="after")
    def _validate_scopes(self) -> Self:
        if len({scope.scope_id for scope in self.scopes}) != 2:
            raise ValueError("research scopes must have unique scope_id values")
        if len({scope.title for scope in self.scopes}) != 2:
            raise ValueError("research scopes must have unique titles")
        return self


class ResearchTask(BaseModel):
    """One frozen research query with its ground-truth key facts."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    task_id: str = Field(min_length=1)
    query: str = Field(min_length=1)
    ground_truth_facts: dict[str, JsonValue] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_ground_truth(self) -> Self:
        facts = self.ground_truth_facts
        titles = facts.get("scope_titles")
        if (
            not isinstance(titles, list)
            or len(titles) != 2
            or not all(isinstance(title, str) and title for title in titles)
        ):
            raise ValueError("ground_truth_facts.scope_titles must contain exactly two strings")
        if len(set(titles)) != 2:
            raise ValueError("scope_titles must be unique")

        scope_facts = facts.get("scope_facts")
        if not isinstance(scope_facts, list) or len(scope_facts) != 2:
            raise ValueError("ground_truth_facts.scope_facts must contain exactly two lists")
        for index, group in enumerate(scope_facts):
            if (
                not isinstance(group, list)
                or not group
                or not all(isinstance(item, str) and item for item in group)
            ):
                raise ValueError(f"scope_facts[{index}] must be a non-empty list of strings")

        key_facts = facts.get("key_facts")
        if (
            not isinstance(key_facts, list)
            or not key_facts
            or not all(isinstance(item, str) and item for item in key_facts)
        ):
            raise ValueError("ground_truth_facts.key_facts must be a non-empty list of strings")
        union = set(scope_facts[0]) | set(scope_facts[1])
        if set(key_facts) != union:
            raise ValueError("key_facts must equal the union of the two scope_facts lists")

        questions = facts.get("scope_questions")
        if questions is not None and (
            not isinstance(questions, list)
            or len(questions) != 2
            or not all(isinstance(question, str) and question for question in questions)
        ):
            raise ValueError("ground_truth_facts.scope_questions must contain exactly two strings")

        discrepancies = facts.get("expected_discrepancies", [])
        if not isinstance(discrepancies, list) or not all(
            isinstance(item, str) and item for item in discrepancies
        ):
            raise ValueError("ground_truth_facts.expected_discrepancies must be a list of strings")
        return self

    @property
    def scope_titles(self) -> tuple[str, str]:
        titles = self.ground_truth_facts["scope_titles"]
        assert isinstance(titles, list)
        return (str(titles[0]), str(titles[1]))

    @property
    def scope_facts(self) -> tuple[tuple[str, ...], tuple[str, ...]]:
        groups = self.ground_truth_facts["scope_facts"]
        assert isinstance(groups, list)
        return (tuple(str(item) for item in groups[0]), tuple(str(item) for item in groups[1]))

    @property
    def key_facts(self) -> tuple[str, ...]:
        facts = self.ground_truth_facts["key_facts"]
        assert isinstance(facts, list)
        return tuple(str(item) for item in facts)

    @property
    def scope_questions(self) -> tuple[str, str]:
        questions = self.ground_truth_facts.get("scope_questions")
        if isinstance(questions, list) and len(questions) == 2:
            return (str(questions[0]), str(questions[1]))
        return (f"Investigate {self.scope_titles[0]}", f"Investigate {self.scope_titles[1]}")

    @property
    def expected_discrepancies(self) -> tuple[str, ...]:
        discrepancies = self.ground_truth_facts.get("expected_discrepancies", [])
        assert isinstance(discrepancies, list)
        return tuple(str(item) for item in discrepancies)

    def prompt_record(self) -> dict[str, JsonValue]:
        """The public request fields, excluding the ground-truth answer."""

        return {"task_id": self.task_id, "query": self.query}


def load_research_topics(path: str | Path | None = None) -> list[ResearchTask]:
    """Load the JSONL research dataset, rejecting malformed or duplicate rows."""

    dataset_path = Path(path) if path is not None else DEFAULT_DATASET_PATH
    tasks: list[ResearchTask] = []
    seen: set[str] = set()

    with dataset_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                task = ResearchTask.model_validate_json(stripped)
            except ValueError as exc:
                raise ValueError(f"invalid research task on line {line_number}: {exc}") from exc
            if task.task_id in seen:
                raise ValueError(f"duplicate task_id {task.task_id!r} on line {line_number}")
            seen.add(task.task_id)
            tasks.append(task)

    return tasks


# ---------------------------------------------------------------------------
# Lead planner
# ---------------------------------------------------------------------------


class ResearchPlanError(RuntimeError):
    """The lead could not produce a valid two-scope plan."""

    code = AHPErrorCode.AGENT_ERROR


def _correction_prompt(error: Exception) -> str:
    return (
        f"Your previous response was rejected: {error}\n"
        "Return exactly one JSON object matching the plan schema. It must contain "
        "exactly two scopes with unique scope_id values and non-empty titles and "
        "questions."
    )


class LeadResearchAgent:
    """LLM-backed planner that splits a query into two research scopes."""

    def __init__(self, llm: LLMClient, *, max_retries: int = 2) -> None:
        if max_retries < 0:
            raise ValueError("max_retries must be >= 0")
        self.llm = llm
        self.max_retries = max_retries
        self.calls = 0

    async def plan(self, message: HandoffMessage) -> ResearchPlan:
        """Ask the model for a validated two-scope plan."""

        messages: list[dict[str, str]] = [
            {
                "role": "system",
                "content": LEAD_RESEARCH_PROMPT
                + "\n\nReturn a JSON plan matching this schema:\n"
                + _json(lead_plan_schema()),
            },
            {
                "role": "user",
                "content": "Research request:\n" + _json(message.model_dump(mode="json")),
            },
        ]
        last_error: Exception | None = None

        for _ in range(self.max_retries + 1):
            self.calls += 1
            try:
                result = await self.llm.chat(
                    list(messages),
                    schema=lead_plan_schema(),
                    temperature=0.0,
                )
            except LLMResponseError as exc:
                last_error = exc
                messages.append({"role": "user", "content": _correction_prompt(exc)})
                continue

            try:
                payload = result.structured_data
                if payload is None:
                    if not result.text.strip():
                        raise ValueError("model returned an empty plan")
                    payload = json.loads(result.text)
                if not isinstance(payload, Mapping):
                    raise ValueError(f"plan must be a JSON object, got {type(payload).__name__}")
                return ResearchPlan.model_validate(dict(payload))
            except (ValueError, TypeError, ValidationError) as exc:
                last_error = exc
                if result.text:
                    messages.append({"role": "assistant", "content": result.text})
                messages.append({"role": "user", "content": _correction_prompt(exc)})

        raise ResearchPlanError(
            f"lead_research_agent failed to produce a valid plan after "
            f"{self.max_retries + 1} attempt(s): {last_error}"
        )


# ---------------------------------------------------------------------------
# Topic and reviewer agents
# ---------------------------------------------------------------------------


class TopicResearchAgent(BaseAgent):
    """Investigates one scope and hands findings to the reviewer."""

    def __init__(
        self,
        name: str,
        llm: LLMClient,
        *,
        reply_schema: Mapping[str, Any],
        **kwargs: Any,
    ) -> None:
        super().__init__(
            name,
            TOPIC_RESEARCH_PROMPT,
            llm,
            tools=(),
            reply_schema=reply_schema,
            **kwargs,
        )


class ReviewerAgent(BaseAgent):
    """Joins two topic handoffs into one final brief."""

    def __init__(
        self,
        llm: LLMClient,
        *,
        reply_schema: Mapping[str, Any],
        expected_branches: int = 2,
        **kwargs: Any,
    ) -> None:
        if expected_branches < 1:
            raise ValueError("expected_branches must be >= 1")
        super().__init__(
            "reviewer_agent",
            REVIEWER_RESEARCH_PROMPT,
            llm,
            tools=(),
            reply_schema=reply_schema,
            **kwargs,
        )
        self.expected_branches = expected_branches
        self._buffer: dict[str, HandoffMessage] = {}

    @property
    def pending_branches(self) -> int:
        """How many topic handoffs are buffered and waiting for the join."""

        return len(self._buffer)

    def reset(self) -> None:
        """Clear buffered handoffs before a new workflow run."""

        self._buffer.clear()

    async def handle_handoff(
        self,
        message: HandoffMessage | NeedInfoMessage,
    ) -> AHPMessage | None:
        """Buffer topic reports and synthesize only after the join is complete."""

        if not isinstance(message, HandoffMessage):
            raise TypeError("reviewer_agent only accepts HandoffMessage")
        self._buffer[message.from_agent] = message
        if len(self._buffer) < self.expected_branches:
            return None

        findings = [handoff.model_dump(mode="json") for handoff in self._buffer.values()]
        conversation_id = message.conversation_id
        self._buffer.clear()
        combined = HandoffMessage(
            conversation_id=conversation_id,
            from_agent="user",
            to_agent=self.name,
            goal="Synthesize the parallel research findings into one brief",
            summary=f"Joined {len(findings)} topic reports.",
            facts={"findings": findings},
            expected_output={
                "brief": "string",
                "key_facts": "array[string]",
                "discrepancies": "array[string]",
                "sources": "array[string]",
            },
        )
        return await super().handle_handoff(combined)


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ResearchRunResult:
    """Everything observable about one fork/join research run."""

    task: ResearchTask
    plan: ResearchPlan
    done: DoneMessage
    messages: tuple[AHPMessage, ...]

    @property
    def topic_handoffs(self) -> tuple[HandoffMessage, ...]:
        """Handoffs emitted by the two topic researchers."""

        return tuple(
            message
            for message in self.messages
            if isinstance(message, HandoffMessage) and message.from_agent in TOPIC_AGENT_NAMES
        )

    @property
    def brief(self) -> str:
        return str(self.done.final_result.get("brief", ""))

    @property
    def key_facts(self) -> tuple[str, ...]:
        facts = self.done.final_result.get("key_facts", [])
        if not isinstance(facts, list):
            return ()
        return tuple(str(fact) for fact in facts)

    @property
    def discrepancies(self) -> tuple[str, ...]:
        values = self.done.final_result.get("discrepancies", [])
        if not isinstance(values, list):
            return ()
        return tuple(str(value) for value in values)

    @property
    def matches_ground_truth(self) -> bool:
        return set(self.task.key_facts) <= set(self.key_facts)

    @property
    def summary_lengths(self) -> dict[str, int]:
        lengths: dict[str, int] = {}
        for message in self.messages:
            if isinstance(message, HandoffMessage | DoneMessage):
                lengths[f"{message.from_agent}->{message.to_agent}"] = len(message.summary)
        return lengths


class ResearchForkJoin:
    """Orchestrate the lead -> parallel researchers -> reviewer workflow."""

    def __init__(
        self,
        lead_llm: LLMClient,
        topic_llm_a: LLMClient,
        topic_llm_b: LLMClient,
        reviewer_llm: LLMClient,
        *,
        summary_max_chars: int = SUMMARY_MAX_CHARS,
        timeout_s: float = 60.0,
        max_hops: int = 10,
        ack_timeout_s: float = 5.0,
    ) -> None:
        if summary_max_chars < 1:
            raise ValueError("summary_max_chars must be >= 1")
        if timeout_s <= 0:
            raise ValueError("timeout_s must be > 0")
        if max_hops < 1:
            raise ValueError("max_hops must be >= 1")
        if ack_timeout_s <= 0:
            raise ValueError("ack_timeout_s must be > 0")

        self.summary_max_chars = summary_max_chars
        self.timeout_s = timeout_s
        self.max_hops = max_hops
        self.ack_timeout_s = ack_timeout_s

        self.lead = LeadResearchAgent(lead_llm)
        self.topic_a = TopicResearchAgent(
            TOPIC_AGENT_NAMES[0],
            topic_llm_a,
            reply_schema=topic_reply_schema(summary_max_chars),
        )
        self.topic_b = TopicResearchAgent(
            TOPIC_AGENT_NAMES[1],
            topic_llm_b,
            reply_schema=topic_reply_schema(summary_max_chars),
        )
        self.reviewer = ReviewerAgent(
            reviewer_llm,
            reply_schema=reviewer_reply_schema(summary_max_chars),
        )

    @classmethod
    def from_shared_llm(cls, llm: LLMClient, **kwargs: Any) -> ResearchForkJoin:
        """Use one client for all four agents (typical for a local engine)."""

        return cls(llm, llm, llm, llm, **kwargs)

    @property
    def agents(self) -> dict[str, BaseAgent]:
        return {
            TOPIC_AGENT_NAMES[0]: self.topic_a,
            TOPIC_AGENT_NAMES[1]: self.topic_b,
            "reviewer_agent": self.reviewer,
        }

    async def run_task(self, task: ResearchTask) -> ResearchRunResult:
        """Run one research task through the fork/join chain."""

        self.reviewer.reset()
        conversation_id = f"research-{task.task_id}"
        messages: list[AHPMessage] = []
        plans: list[ResearchPlan] = []

        async def lead_handler(message: AHPMessage) -> None:
            if not isinstance(message, HandoffMessage):
                raise TypeError("lead_researcher only accepts HandoffMessage")
            plan = await self.lead.plan(message)
            plans.append(plan)
            handoffs = self._topic_handoffs(task, plan, conversation_id)
            messages.extend(handoffs)
            await asyncio.gather(*(router.send(handoff) for handoff in handoffs))
            return None

        def wrap(agent: BaseAgent):
            async def handler(message: AHPMessage) -> AHPMessage | None:
                reply = await agent.handle_handoff(message)
                if reply is not None:
                    messages.append(reply)
                return reply

            return handler

        router = InProcessRouter(
            RouterConfig(max_hops=self.max_hops, ack_timeout_s=self.ack_timeout_s),
            run_id=conversation_id,
        )
        router.register_agent("lead_researcher", lead_handler)
        router.register_agent(TOPIC_AGENT_NAMES[0], wrap(self.topic_a))
        router.register_agent(TOPIC_AGENT_NAMES[1], wrap(self.topic_b))
        router.register_agent("reviewer_agent", wrap(self.reviewer))

        initial = self._initial_handoff(task, conversation_id)
        messages.append(initial)
        try:
            done = await router.run_workflow(initial, timeout_s=self.timeout_s)
        finally:
            await router.stop()

        if not plans:
            raise RuntimeError("lead_researcher did not produce a research plan")
        return ResearchRunResult(
            task=task,
            plan=plans[0],
            done=done,
            messages=tuple(messages),
        )

    # -- helpers -------------------------------------------------------------

    def _initial_handoff(self, task: ResearchTask, conversation_id: str) -> HandoffMessage:
        return HandoffMessage(
            conversation_id=conversation_id,
            from_agent="user",
            to_agent="lead_researcher",
            goal=f"Research request {task.task_id}",
            summary=_truncate(task.query, self.summary_max_chars),
            facts={"task_id": task.task_id, "query": task.query},
            expected_output={"scopes": "array of exactly two research scopes"},
            constraints={"summary_max_chars": self.summary_max_chars},
        )

    def _topic_handoffs(
        self,
        task: ResearchTask,
        plan: ResearchPlan,
        conversation_id: str,
    ) -> list[HandoffMessage]:
        handoffs: list[HandoffMessage] = []
        for scope, target in zip(plan.scopes, TOPIC_AGENT_NAMES, strict=True):
            summary = _truncate(
                f"{scope.title}: {scope.question}",
                self.summary_max_chars,
            )
            handoffs.append(
                HandoffMessage(
                    conversation_id=conversation_id,
                    from_agent="lead_researcher",
                    to_agent=target,
                    goal=f"Research scope {scope.scope_id}",
                    summary=summary,
                    facts={
                        "task_id": task.task_id,
                        "scope_id": scope.scope_id,
                        "title": scope.title,
                        "question": scope.question,
                        "focus": scope.focus,
                    },
                    expected_output={
                        "findings": "string",
                        "key_facts": "array[string]",
                        "sources": "array[string]",
                        "confidence": "number",
                    },
                    constraints={"summary_max_chars": self.summary_max_chars},
                )
            )
        return handoffs


__all__ = [
    "DEFAULT_DATASET_PATH",
    "LEAD_RESEARCH_PROMPT",
    "PROJECT_ROOT",
    "REVIEWER_RESEARCH_PROMPT",
    "ResearchForkJoin",
    "ResearchPlan",
    "ResearchPlanError",
    "ResearchRunResult",
    "ResearchScope",
    "ResearchTask",
    "ReviewerAgent",
    "TOPIC_AGENT_NAMES",
    "TOPIC_RESEARCH_PROMPT",
    "TopicResearchAgent",
    "LeadResearchAgent",
    "load_research_topics",
]
