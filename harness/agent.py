"""Base agent runtime: prompts, tool loops and AHP message drafting.

:class:`BaseAgent` is the bridge between the protocol and an LLM. It receives a
validated AHP message, grounds it with local tools (the seeded SQLite mock by
default), and asks the model for exactly one action per turn:

* ``tool_call`` — run a registered Python tool and feed its JSON result back;
* ``reply`` — draft the next AHP message (``HANDOFF``, ``NEED_INFO`` or
  ``DONE``).

Malformed model output never escapes as a half-built protocol message. It is
rejected with a corrective prompt for up to ``max_retries`` retries, then
raised as :class:`AgentReplyError`. The router converts that into an
``AGENT_ERROR`` workflow failure.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Final

import jsonschema

from ahp.models import (
    AHPErrorCode,
    AHPMessage,
    HandoffMessage,
    NeedInfoMessage,
    parse_ahp_message,
)
from engine.client import LLMResponseError
from engine.interfaces import LLMClient
from engine.models import LLMResult
from harness.tools.base import (
    Tool,
    ToolArgumentError,
    ToolRegistry,
    ToolResult,
    UnknownToolError,
)

#: Reply types an agent is allowed to draft. ACKs belong to the router.
REPLY_TYPES: Final[frozenset[str]] = frozenset({"HANDOFF", "NEED_INFO", "DONE"})

#: Actions the model may return on each turn.
ACTION_SCHEMA: Final[dict[str, Any]] = {
    "type": "object",
    "oneOf": [
        {
            "type": "object",
            "properties": {
                "action": {"const": "tool_call"},
                "tool_call": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string", "minLength": 1},
                        "arguments": {"type": "object"},
                    },
                    "required": ["name", "arguments"],
                    "additionalProperties": False,
                },
            },
            "required": ["action", "tool_call"],
            "additionalProperties": False,
        },
        {
            "type": "object",
            "properties": {
                "action": {"const": "reply"},
                "reply": {
                    "type": "object",
                    "properties": {
                        "type": {"enum": sorted(REPLY_TYPES)},
                    },
                    "required": ["type"],
                },
            },
            "required": ["action", "reply"],
            "additionalProperties": False,
        },
    ],
}


def action_schema() -> dict[str, Any]:
    """Return a fresh copy of the generic agent action schema."""

    return deepcopy(ACTION_SCHEMA)


class AgentReplyError(RuntimeError):
    """An agent could not produce a valid AHP reply within its budget."""

    code = AHPErrorCode.AGENT_ERROR


@dataclass(frozen=True, slots=True)
class ToolCallRecord:
    """One executed (or failed) tool invocation, recorded for debugging."""

    tool: str
    arguments: dict[str, Any]
    result: Any = None
    error: str | None = None

    @property
    def ok(self) -> bool:
        """Whether the tool ran without raising."""

        return self.error is None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable summary."""

        return {
            "tool": self.tool,
            "ok": self.ok,
            "arguments": dict(self.arguments),
            "result": self.result,
            "error": self.error,
        }


def _json(value: Any) -> str:
    """Pretty, deterministic JSON for prompts; never fails on exotic values."""

    return json.dumps(value, indent=2, sort_keys=True, default=str)


def _tool_result_record(result: ToolResult) -> ToolCallRecord:
    return ToolCallRecord(
        tool=result.tool,
        arguments=result.arguments,
        result=result.result,
        error=result.error,
    )


class BaseAgent:
    """LLM-backed agent implementing :class:`ahp.interfaces.Agent`.

    Parameters
    ----------
    name:
        Agent name used as ``from_agent`` in drafted messages.
    system_prompt:
        Role instructions for the model.
    llm:
        Any :class:`engine.interfaces.LLMClient`.
    tools:
        Tools the model may call, or an existing :class:`ToolRegistry`.
    max_retries:
        Corrective re-prompts after invalid model output. ``2`` means three
        total attempts before :class:`AgentReplyError`.
    max_tool_iterations:
        Safety limit on tool calls in one ``handle_handoff`` invocation.
    temperature:
        Sampling temperature forwarded to the client.

    ``tool_calls`` holds the records of the most recent call. A single agent
    instance is expected to process one conversation message at a time, which
    matches the router's per-agent queue.
    """

    def __init__(
        self,
        name: str,
        system_prompt: str,
        llm: LLMClient,
        tools: ToolRegistry | Iterable[Tool] = (),
        *,
        max_retries: int = 2,
        max_tool_iterations: int = 8,
        temperature: float = 0.0,
        response_schema: Mapping[str, Any] | None = None,
    ) -> None:
        if not name:
            raise ValueError("agent name must not be empty")
        if not system_prompt.strip():
            raise ValueError("system_prompt must not be empty")
        if max_retries < 0:
            raise ValueError("max_retries must be >= 0")
        if max_tool_iterations < 1:
            raise ValueError("max_tool_iterations must be >= 1")
        if temperature < 0:
            raise ValueError("temperature must be >= 0")

        self.name = name
        self.system_prompt = system_prompt
        self.llm = llm
        self.tools = tools if isinstance(tools, ToolRegistry) else ToolRegistry(tools)
        self.max_retries = max_retries
        self.max_tool_iterations = max_tool_iterations
        self.temperature = temperature
        self.response_schema: dict[str, Any] = (
            deepcopy(dict(response_schema)) if response_schema is not None else action_schema()
        )
        self.tool_calls: list[ToolCallRecord] = []

    # -- protocol entry point ------------------------------------------------

    async def handle_handoff(self, message: HandoffMessage | NeedInfoMessage) -> AHPMessage:
        """Process one incoming AHP message and return exactly one reply.

        ``NeedInfoMessage`` is accepted because a downstream agent may ask this
        agent for missing details; it is answered with a fresh action, usually
        an updated ``HANDOFF``.
        """

        if not isinstance(message, HandoffMessage | NeedInfoMessage):
            raise TypeError("BaseAgent.handle_handoff expects HandoffMessage or NeedInfoMessage")

        self.tool_calls = []
        messages = self._initial_messages(message)
        invalid_outputs = 0
        tool_invocations = 0

        while True:
            try:
                result = await self.llm.chat(
                    list(messages),
                    schema=self.response_schema,
                    temperature=self.temperature,
                )
            except LLMResponseError as exc:
                # The unified client validates structured output itself; a
                # malformed document is a bad model output and gets re-prompted
                # exactly like a locally detected schema violation.
                invalid_outputs += 1
                if invalid_outputs > self.max_retries:
                    raise self._retry_limit_error(exc) from exc
                messages.append({"role": "user", "content": self._correction_prompt(exc)})
                continue

            try:
                action = self._parse_action(result)
                if action["action"] == "tool_call":
                    tool, arguments = self._resolve_tool_call(action)
                    reply: AHPMessage | None = None
                else:
                    tool = None
                    arguments = None
                    reply = self._draft_reply(message, action["reply"])
            except (
                ValueError,
                TypeError,
                KeyError,
                jsonschema.ValidationError,
                UnknownToolError,
                ToolArgumentError,
            ) as exc:
                invalid_outputs += 1
                if invalid_outputs > self.max_retries:
                    raise self._retry_limit_error(exc) from exc
                messages.append({"role": "assistant", "content": self._raw_response(result)})
                messages.append({"role": "user", "content": self._correction_prompt(exc)})
                continue

            if reply is not None:
                return reply

            tool_invocations += 1
            if tool_invocations > self.max_tool_iterations:
                raise AgentReplyError(
                    f"agent {self.name!r} exceeded the tool-call limit "
                    f"({self.max_tool_iterations}) without drafting a reply"
                )

            assert tool is not None and arguments is not None
            record = _tool_result_record(await self.tools.invoke(tool, arguments))
            self.tool_calls.append(record)
            messages.append({"role": "assistant", "content": _json(action)})
            messages.append(
                {
                    "role": "user",
                    "content": "Tool result:\n" + _json(record.to_dict()),
                }
            )

    # -- prompt construction -------------------------------------------------

    def _initial_messages(self, message: HandoffMessage | NeedInfoMessage) -> list[dict[str, str]]:
        return [
            {"role": "system", "content": self._system_prompt()},
            {
                "role": "user",
                "content": (
                    "Process this incoming AHP message and return exactly one action.\n\n"
                    + _json(message.model_dump(mode="json"))
                ),
            },
        ]

    def _system_prompt(self) -> str:
        tools = self.tools.definitions()
        if tools:
            tool_context = "Available tools:\n" + _json(tools)
        else:
            tool_context = "No tools are available; reply immediately."

        return (
            f"{self.system_prompt.strip()}\n\n"
            f"You are {self.name!r}, an agent in an Agent Handoff Protocol (AHP) workflow.\n"
            "Return exactly one JSON object per turn, without markdown fences or commentary:\n"
            '- {"action": "tool_call", "tool_call": {"name": "<tool>", '
            '"arguments": {<validated arguments>}}}\n'
            '- {"action": "reply", "reply": {<AHP message payload>}}\n\n'
            "Reply payload rules:\n"
            "- HANDOFF requires type, to_agent, goal and summary. Add facts so the "
            "receiver never has to repeat your work.\n"
            "- NEED_INFO requires type, missing_fields (non-empty) and query; to_agent "
            "defaults to the sender.\n"
            "- DONE requires type, final_result and summary; to_agent defaults to the "
            "sender. Use status='escalated_unresolved' when the task cannot be resolved.\n"
            "- Never emit ACK: the router acknowledges handoffs itself.\n\n"
            f"{tool_context}"
        )

    # -- model output handling -----------------------------------------------

    def _retry_limit_error(self, error: BaseException) -> AgentReplyError:
        attempts = self.max_retries + 1
        return AgentReplyError(
            f"agent {self.name!r} failed to produce a valid AHP action "
            f"after {attempts} attempt(s): {error}"
        )

    @staticmethod
    def _raw_response(result: LLMResult) -> str:
        if result.text:
            return result.text
        if result.structured_data is not None:
            return _json(result.structured_data)
        return ""

    @staticmethod
    def _correction_prompt(error: Exception) -> str:
        return (
            f"Your previous response was rejected: {error}\n"
            "Return exactly one JSON object that follows the action schema. "
            "Use only the listed tools and include every required AHP field."
        )

    def _parse_action(self, result: LLMResult) -> dict[str, Any]:
        payload = result.structured_data
        if payload is None:
            if not result.text.strip():
                raise ValueError("model returned an empty response")
            payload = json.loads(result.text)
        if not isinstance(payload, Mapping):
            raise ValueError(f"action must be a JSON object, got {type(payload).__name__}")
        action = dict(payload)
        jsonschema.validate(instance=action, schema=self.response_schema)
        return action

    def _resolve_tool_call(self, action: Mapping[str, Any]) -> tuple[Tool, dict[str, Any]]:
        tool_call = action["tool_call"]
        if not isinstance(tool_call, Mapping):
            raise ValueError("tool_call must be an object")
        return self.tools.resolve(str(tool_call["name"]), tool_call["arguments"])

    def _draft_reply(
        self,
        incoming: HandoffMessage | NeedInfoMessage,
        reply_payload: Mapping[str, Any],
    ) -> AHPMessage:
        if not isinstance(reply_payload, Mapping):
            raise ValueError("reply must be a JSON object")
        payload = dict(reply_payload)
        reply_type = payload.get("type")
        if reply_type not in REPLY_TYPES:
            raise ValueError(f"reply type must be one of {sorted(REPLY_TYPES)}, got {reply_type!r}")

        # Identity and routing metadata are owned by the agent and router, never
        # by the model. Dropping message_id/timestamp also guarantees uniqueness.
        payload["conversation_id"] = incoming.conversation_id
        payload["from_agent"] = self.name
        payload["hop_count"] = 0
        payload.pop("message_id", None)
        payload.pop("timestamp_ns", None)
        if reply_type in {"NEED_INFO", "DONE"} and "to_agent" not in payload:
            payload["to_agent"] = incoming.from_agent

        return parse_ahp_message(payload)


__all__ = [
    "ACTION_SCHEMA",
    "AgentReplyError",
    "BaseAgent",
    "REPLY_TYPES",
    "ToolCallRecord",
    "action_schema",
]
