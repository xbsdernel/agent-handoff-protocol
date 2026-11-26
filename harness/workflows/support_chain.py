"""Workflow 1: the sequential customer-support triage chain.

The chain is deliberately linear and inspectable:

``customer -> triage_agent -> billing_agent -> reply_agent -> DONE``

* **Triage** classifies the ticket and verifies the customer. If the customer
  details are missing it emits ``NEED_INFO`` (either to the customer or back to
  triage, depending on where the gap was detected).
* **Billing** queries the seeded SQLite charge history and applies the refund
  policy through the ``assess_refund`` tool.
* **Reply** writes the customer-facing message and closes the workflow with a
  validated ``DONE`` message.

Facts are accumulated automatically by :class:`~harness.agent.BaseAgent`, and
every summary is capped by the role schemas. The module also loads the frozen
50-ticket dataset in ``data/support_tickets.jsonl`` and exposes the deterministic
refund policy used by its ground-truth labels.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator

from ahp.models import (
    AHPMessage,
    DoneMessage,
    HandoffMessage,
    NeedInfoMessage,
)
from ahp.router import InProcessRouter, RouterConfig
from engine.interfaces import LLMClient
from harness.agent import BaseAgent, ToolCallRecord
from harness.tools import SQLiteMockDB, Tool, sqlite_tools
from harness.tools.sqlite_mock import CUSTOMER_ID_PARAMETERS
from harness.workflows.schemas import (
    SUMMARY_MAX_CHARS,
    billing_reply_schema,
    reply_reply_schema,
    triage_reply_schema,
)

PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[2]
DEFAULT_DATASET_PATH: Final[Path] = PROJECT_ROOT / "data" / "support_tickets.jsonl"
PROMPTS_DIR: Final[Path] = Path(__file__).with_name("prompts")

Decision = Literal["refund", "no_refund", "escalate"]


def _load_prompt(name: str) -> str:
    return (PROMPTS_DIR / name).read_text(encoding="utf-8").strip()


TRIAGE_PROMPT: Final[str] = _load_prompt("triage_prompt.txt")
BILLING_PROMPT: Final[str] = _load_prompt("billing_prompt.txt")
REPLY_PROMPT: Final[str] = _load_prompt("reply_prompt.txt")


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class SupportTicket(BaseModel):
    """One frozen support scenario with its ground-truth outcome."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    task_id: str = Field(min_length=1)
    input_text: str = Field(min_length=1)
    customer_id: str | None = None
    expected_decision: Decision
    expected_refund_amount: float = Field(ge=0)
    ground_truth_facts: dict[str, JsonValue] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_ground_truth(self) -> Self:
        if self.expected_decision == "refund":
            if self.expected_refund_amount <= 0:
                raise ValueError("refund tickets require a positive expected_refund_amount")
        elif self.expected_refund_amount != 0:
            raise ValueError("non-refund tickets must have expected_refund_amount 0")

        ground_truth_customer = self.ground_truth_facts.get("customer_id")
        if (
            ground_truth_customer is not None
            and self.customer_id is not None
            and ground_truth_customer != self.customer_id
        ):
            raise ValueError("ground_truth_facts.customer_id must match customer_id")
        return self

    @property
    def requires_customer_lookup(self) -> bool:
        """Whether the input text omits the customer ID on purpose."""

        return self.customer_id is not None and self.customer_id not in self.input_text

    def prompt_record(self) -> dict[str, JsonValue]:
        """Return the authoritative ticket fields shown to triage.

        Ground-truth decisions and refund amounts are deliberately excluded.
        """

        record: dict[str, JsonValue] = {
            "task_id": self.task_id,
            "input_text": self.input_text,
        }
        if self.customer_id is not None:
            record["customer_id"] = self.customer_id
        for key in ("customer_name", "email", "plan"):
            value = self.ground_truth_facts.get(key)
            if value is not None:
                record[key] = value
        return record


def load_support_tickets(path: str | Path | None = None) -> list[SupportTicket]:
    """Load the JSONL ticket dataset, rejecting malformed or duplicate rows."""

    dataset_path = Path(path) if path is not None else DEFAULT_DATASET_PATH
    tickets: list[SupportTicket] = []
    seen: set[str] = set()

    with dataset_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                ticket = SupportTicket.model_validate_json(stripped)
            except ValueError as exc:
                raise ValueError(f"invalid support ticket on line {line_number}: {exc}") from exc
            if ticket.task_id in seen:
                raise ValueError(f"duplicate task_id {ticket.task_id!r} on line {line_number}")
            seen.add(ticket.task_id)
            tickets.append(ticket)

    return tickets


# ---------------------------------------------------------------------------
# Refund policy and billing tools
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class RefundDecision:
    """Deterministic result of the billing policy for one customer."""

    decision: Decision
    refund_amount: float
    customer_id: str
    duplicate_charge_ids: tuple[str, ...] = ()
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "decision": self.decision,
            "refund_amount": self.refund_amount,
            "customer_id": self.customer_id,
            "duplicate_charge_ids": list(self.duplicate_charge_ids),
            "reason": self.reason,
        }


def assess_refund(db: SQLiteMockDB, customer_id: str) -> RefundDecision:
    """Apply the refund policy to one customer's seeded charge history.

    Policy:
    * unknown customer -> escalate;
    * suspected duplicate charges -> refund one duplicate amount;
    * otherwise, any failed charge -> escalate to a human;
    * otherwise -> no refund.
    """

    charges = db.query_charges(customer_id)
    if not charges["found"]:
        return RefundDecision("escalate", 0.0, customer_id, (), "customer_not_found")

    duplicate_ids = tuple(str(charge_id) for charge_id in charges["suspected_duplicate_charge_ids"])
    if duplicate_ids:
        duplicate_set = set(duplicate_ids)
        duplicate_charges = [
            charge for charge in charges["charges"] if str(charge["charge_id"]) in duplicate_set
        ]
        amount = float(duplicate_charges[0]["amount"]) if duplicate_charges else 0.0
        return RefundDecision("refund", amount, customer_id, duplicate_ids, "duplicate_charge")

    if any(str(charge["status"]) == "failed" for charge in charges["charges"]):
        return RefundDecision("escalate", 0.0, customer_id, (), "failed_payment")

    return RefundDecision("no_refund", 0.0, customer_id, (), "no_duplicate_charge")


def assess_refund_tool(db: SQLiteMockDB) -> Tool:
    """Expose :func:`assess_refund` as a billing-agent tool."""

    def handler(customer_id: str) -> dict[str, Any]:
        return assess_refund(db, customer_id).to_dict()

    return Tool(
        name="assess_refund",
        description=(
            "Apply the refund policy to a customer's charge history. Returns the "
            "decision (refund, no_refund or escalate), refund amount, duplicate "
            "charge IDs and reason. Use this instead of inventing a decision."
        ),
        parameters=dict(CUSTOMER_ID_PARAMETERS),
        handler=handler,
    )


def billing_tools(db: SQLiteMockDB) -> list[Tool]:
    """SQLite lookups plus the deterministic refund policy."""

    return [*sqlite_tools(db), assess_refund_tool(db)]


# ---------------------------------------------------------------------------
# Agents
# ---------------------------------------------------------------------------


class TriageAgent(BaseAgent):
    """Classifies the ticket and routes it to billing."""

    def __init__(
        self,
        llm: LLMClient,
        *,
        db: SQLiteMockDB,
        reply_schema: Mapping[str, Any],
        ticket: SupportTicket | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            "triage_agent",
            TRIAGE_PROMPT,
            llm,
            tools=sqlite_tools(db),
            reply_schema=reply_schema,
            **kwargs,
        )
        self.ticket = ticket

    def _initial_messages(
        self,
        message: HandoffMessage | NeedInfoMessage,
    ) -> list[dict[str, str]]:
        messages = super()._initial_messages(message)
        if self.ticket is not None:
            messages.append(
                {
                    "role": "user",
                    "content": "Authoritative ticket record:\n"
                    + json.dumps(self.ticket.prompt_record(), indent=2, sort_keys=True),
                }
            )
        return messages


class BillingAgent(BaseAgent):
    """Inspects charges and applies the refund policy."""

    def __init__(
        self,
        llm: LLMClient,
        *,
        db: SQLiteMockDB,
        reply_schema: Mapping[str, Any],
        **kwargs: Any,
    ) -> None:
        super().__init__(
            "billing_agent",
            BILLING_PROMPT,
            llm,
            tools=billing_tools(db),
            reply_schema=reply_schema,
            **kwargs,
        )


class ReplyAgent(BaseAgent):
    """Writes the customer-facing message and closes the workflow."""

    def __init__(
        self,
        llm: LLMClient,
        *,
        reply_schema: Mapping[str, Any],
        **kwargs: Any,
    ) -> None:
        super().__init__(
            "reply_agent",
            REPLY_PROMPT,
            llm,
            tools=(),
            reply_schema=reply_schema,
            **kwargs,
        )


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SupportRunResult:
    """Everything observable about one end-to-end support run."""

    ticket: SupportTicket
    done: DoneMessage
    messages: tuple[AHPMessage, ...]
    tool_calls: Mapping[str, tuple[ToolCallRecord, ...]]

    @property
    def handoffs(self) -> tuple[HandoffMessage, ...]:
        """All handoffs, including the initial customer -> triage handoff."""

        return tuple(message for message in self.messages if isinstance(message, HandoffMessage))

    @property
    def agent_handoffs(self) -> tuple[HandoffMessage, ...]:
        """Handoffs emitted by agents, excluding the initial customer message."""

        return tuple(message for message in self.handoffs if message.from_agent != "customer")

    @property
    def need_info(self) -> tuple[NeedInfoMessage, ...]:
        return tuple(message for message in self.messages if isinstance(message, NeedInfoMessage))

    @property
    def decision(self) -> str:
        return str(self.done.final_result.get("decision", "escalate"))

    @property
    def refund_amount(self) -> float:
        value = self.done.final_result.get("refund_amount", 0.0)
        return float(value) if isinstance(value, int | float) else 0.0

    @property
    def customer_message(self) -> str:
        return str(self.done.final_result.get("customer_message", ""))

    @property
    def matches_ground_truth(self) -> bool:
        return (
            self.decision == self.ticket.expected_decision
            and abs(self.refund_amount - self.ticket.expected_refund_amount) < 1e-9
        )

    @property
    def summary_lengths(self) -> dict[str, int]:
        """Summary length per sender/receiver pair, for tightness assertions."""

        lengths: dict[str, int] = {}
        for message in self.messages:
            if isinstance(message, HandoffMessage | DoneMessage):
                lengths[f"{message.from_agent}->{message.to_agent}"] = len(message.summary)
        return lengths


class SupportChain:
    """Orchestrate the triage -> billing -> reply support pipeline."""

    def __init__(
        self,
        triage_llm: LLMClient,
        billing_llm: LLMClient,
        reply_llm: LLMClient,
        *,
        db: SQLiteMockDB | None = None,
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

        self.db = db if db is not None else SQLiteMockDB()
        self._owns_db = db is None
        self.summary_max_chars = summary_max_chars
        self.timeout_s = timeout_s
        self.max_hops = max_hops
        self.ack_timeout_s = ack_timeout_s

        self.triage = TriageAgent(
            triage_llm,
            db=self.db,
            reply_schema=triage_reply_schema(summary_max_chars),
        )
        self.billing = BillingAgent(
            billing_llm,
            db=self.db,
            reply_schema=billing_reply_schema(summary_max_chars),
        )
        self.reply = ReplyAgent(
            reply_llm,
            reply_schema=reply_reply_schema(summary_max_chars),
        )

    @classmethod
    def from_shared_llm(cls, llm: LLMClient, **kwargs: Any) -> SupportChain:
        """Use one client for all three agents (typical for a local engine)."""

        return cls(llm, llm, llm, **kwargs)

    @property
    def agents(self) -> dict[str, BaseAgent]:
        return {
            "triage_agent": self.triage,
            "billing_agent": self.billing,
            "reply_agent": self.reply,
        }

    async def run_ticket(self, ticket: SupportTicket) -> SupportRunResult:
        """Run one ticket through the chain until a ``DONE`` message arrives."""

        self.db.reset()
        self.triage.ticket = ticket
        conversation_id = f"support-{ticket.task_id}"
        messages: list[AHPMessage] = []
        tool_calls: dict[str, list[ToolCallRecord]] = {
            "triage_agent": [],
            "billing_agent": [],
            "reply_agent": [],
        }

        def wrap(name: str, agent: BaseAgent):
            async def handler(message: AHPMessage) -> AHPMessage:
                reply = await agent.handle_handoff(message)
                messages.append(reply)
                tool_calls[name].extend(agent.tool_calls)
                return reply

            return handler

        async def customer_handler(message: AHPMessage) -> AHPMessage:
            reply = self._customer_reply(ticket, message)
            messages.append(reply)
            return reply

        router = InProcessRouter(
            RouterConfig(max_hops=self.max_hops, ack_timeout_s=self.ack_timeout_s),
            run_id=conversation_id,
        )
        router.register_agent("triage_agent", wrap("triage_agent", self.triage))
        router.register_agent("billing_agent", wrap("billing_agent", self.billing))
        router.register_agent("reply_agent", wrap("reply_agent", self.reply))
        router.register_agent("customer", customer_handler)

        initial = self._initial_handoff(ticket, conversation_id)
        messages.append(initial)
        try:
            done = await router.run_workflow(initial, timeout_s=self.timeout_s)
        finally:
            await router.stop()

        return SupportRunResult(
            ticket=ticket,
            done=done,
            messages=tuple(messages),
            tool_calls={name: tuple(calls) for name, calls in tool_calls.items()},
        )

    async def aclose(self) -> None:
        """Close the database only when this chain created it."""

        if self._owns_db:
            self.db.close()

    async def __aenter__(self) -> SupportChain:
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self.aclose()

    # -- helpers -------------------------------------------------------------

    def _initial_handoff(self, ticket: SupportTicket, conversation_id: str) -> HandoffMessage:
        facts: dict[str, JsonValue] = {
            "task_id": ticket.task_id,
            "ticket_text": ticket.input_text,
        }
        if ticket.customer_id is not None and ticket.customer_id in ticket.input_text:
            facts["customer_id"] = ticket.customer_id

        summary = ticket.input_text.strip()
        if len(summary) > self.summary_max_chars:
            summary = summary[: self.summary_max_chars - 1].rstrip() + "…"

        return HandoffMessage(
            conversation_id=conversation_id,
            from_agent="customer",
            to_agent="triage_agent",
            goal=f"Resolve support ticket {ticket.task_id}",
            summary=summary or f"Support ticket {ticket.task_id}",
            facts=facts,
            expected_output={
                "decision": "refund | no_refund | escalate",
                "refund_amount": "number",
                "customer_message": "string",
            },
            constraints={"summary_max_chars": self.summary_max_chars},
        )

    @staticmethod
    def _customer_reply(ticket: SupportTicket, message: AHPMessage) -> AHPMessage:
        """Simulate the customer answering a NEED_INFO request."""

        if not isinstance(message, NeedInfoMessage):
            raise ValueError("customer intake only answers NEED_INFO messages")

        supplied: dict[str, JsonValue] = {
            "task_id": ticket.task_id,
            "ticket_text": ticket.input_text,
        }
        for field in message.missing_fields:
            if field == "customer_id" and ticket.customer_id is not None:
                supplied[field] = ticket.customer_id
            elif field in ticket.ground_truth_facts:
                supplied[field] = ticket.ground_truth_facts[field]

        unresolved = [field for field in message.missing_fields if field not in supplied]
        if unresolved:
            raise ValueError(f"customer cannot supply missing fields: {unresolved}")

        return HandoffMessage(
            conversation_id=message.conversation_id,
            from_agent="customer",
            to_agent="triage_agent",
            goal=f"Provide missing details for ticket {ticket.task_id}",
            summary=f"Customer supplied: {', '.join(message.missing_fields)}.",
            facts=supplied,
        )


__all__ = [
    "BILLING_PROMPT",
    "DEFAULT_DATASET_PATH",
    "Decision",
    "PROJECT_ROOT",
    "REPLY_PROMPT",
    "RefundDecision",
    "SupportChain",
    "SupportRunResult",
    "SupportTicket",
    "TRIAGE_PROMPT",
    "BillingAgent",
    "ReplyAgent",
    "TriageAgent",
    "assess_refund",
    "assess_refund_tool",
    "billing_tools",
    "load_support_tickets",
]
