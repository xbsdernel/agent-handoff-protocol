"""Local tool abstractions shared by every harness agent.

Agents use tools to ground their decisions in deterministic data while the LLM
only decides *which* tool to call and *how* to interpret the result. A
:class:`Tool` is a JSON-Schema-described Python callable; a
:class:`ToolRegistry` looks tools up by name, validates model-supplied
arguments and records failures instead of crashing the agent.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import dataclass, field
from typing import Any, Final

import jsonschema

#: Schema used when a tool takes no arguments.
EMPTY_PARAMETERS: Final[dict[str, Any]] = {
    "type": "object",
    "properties": {},
    "additionalProperties": False,
}


class ToolError(RuntimeError):
    """Base class for tool-resolution problems."""


class UnknownToolError(ToolError):
    """The model asked for a tool that is not registered."""


class ToolArgumentError(ToolError):
    """The model supplied arguments that violate the tool's JSON Schema."""


@dataclass(frozen=True, slots=True)
class ToolResult:
    """Outcome of one tool invocation, safe to serialise into an LLM prompt.

    ``error`` is populated when the tool itself raised. Tool failures are
    returned as data rather than propagated so the model can recover (for
    example by choosing another tool or reporting the failure).
    """

    tool: str
    arguments: dict[str, Any] = field(default_factory=dict)
    result: Any = None
    error: str | None = None

    @property
    def ok(self) -> bool:
        """Whether the tool ran without raising."""

        return self.error is None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable view used in prompts and telemetry."""

        return {
            "tool": self.tool,
            "ok": self.ok,
            "arguments": dict(self.arguments),
            "result": self.result,
            "error": self.error,
        }


@dataclass(frozen=True, slots=True)
class Tool:
    """A named Python callable described by a JSON Schema.

    ``handler`` is invoked with the validated arguments as keyword arguments.
    Both synchronous and asynchronous handlers are supported.
    """

    name: str
    description: str
    handler: Callable[..., Any]
    parameters: dict[str, Any] = field(default_factory=lambda: dict(EMPTY_PARAMETERS))

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("tool name must not be empty")
        if not callable(self.handler):
            raise TypeError(f"handler for tool {self.name!r} must be callable")

    def definition(self) -> dict[str, Any]:
        """Return the OpenAI-style function definition for prompt injection."""

        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": dict(self.parameters),
            },
        }


class ToolRegistry:
    """Ordered collection of tools with validation and safe invocation."""

    def __init__(self, tools: Iterable[Tool] = ()) -> None:
        self._tools: dict[str, Tool] = {}
        for tool in tools:
            self.add(tool)

    def add(self, tool: Tool) -> None:
        """Register ``tool``; duplicate names are rejected."""

        if tool.name in self._tools:
            raise ValueError(f"tool {tool.name!r} is already registered")
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool:
        """Return the tool registered as ``name`` or raise ``UnknownToolError``."""

        try:
            return self._tools[name]
        except KeyError as exc:
            available = ", ".join(sorted(self._tools)) or "<none>"
            raise UnknownToolError(f"unknown tool {name!r}; available tools: {available}") from exc

    def resolve(
        self,
        name: str,
        arguments: Mapping[str, Any] | None = None,
    ) -> tuple[Tool, dict[str, Any]]:
        """Look up ``name`` and validate ``arguments`` against its schema."""

        tool = self.get(name)
        if arguments is None:
            raw: Any = {}
        else:
            raw = arguments
        if not isinstance(raw, Mapping):
            raise ToolArgumentError(
                f"arguments for tool {name!r} must be an object, got {type(raw).__name__}"
            )
        resolved = dict(raw)
        try:
            jsonschema.validate(instance=resolved, schema=tool.parameters)
        except jsonschema.ValidationError as exc:
            location = ".".join(str(part) for part in exc.absolute_path) or "<root>"
            raise ToolArgumentError(
                f"invalid arguments for tool {name!r} at {location}: {exc.message}"
            ) from exc
        return tool, resolved

    async def invoke(self, tool: Tool, arguments: Mapping[str, Any]) -> ToolResult:
        """Run ``tool`` and capture any exception as a :class:`ToolResult`."""

        resolved = dict(arguments)
        try:
            value = tool.handler(**resolved)
            if inspect.isawaitable(value):
                value = await value
        except Exception as exc:  # noqa: BLE001 - failures are returned as data
            return ToolResult(
                tool=tool.name,
                arguments=resolved,
                error=f"{type(exc).__name__}: {exc}",
            )
        return ToolResult(tool=tool.name, arguments=resolved, result=value)

    async def execute(
        self,
        name: str,
        arguments: Mapping[str, Any] | None = None,
    ) -> ToolResult:
        """Resolve, validate and invoke one tool in a single call."""

        tool, resolved = self.resolve(name, arguments)
        return await self.invoke(tool, resolved)

    def definitions(self) -> list[dict[str, Any]]:
        """Return every tool definition in registration order."""

        return [tool.definition() for tool in self._tools.values()]

    @property
    def names(self) -> tuple[str, ...]:
        """Registered tool names in registration order."""

        return tuple(self._tools)

    def __contains__(self, name: object) -> bool:
        return name in self._tools

    def __iter__(self) -> Iterator[Tool]:
        return iter(self._tools.values())

    def __len__(self) -> int:
        return len(self._tools)


__all__ = [
    "EMPTY_PARAMETERS",
    "Tool",
    "ToolArgumentError",
    "ToolError",
    "ToolRegistry",
    "ToolResult",
    "UnknownToolError",
]
