"""Workflow 3: bug triage, fix and test with a bounded repair loop.

The workflow runs an iterative cycle:

``user -> bug_triage_agent -> developer_agent <-> tester_agent -> DONE``

* **Triage** turns the stack trace and failing tests into a reproduction
  specification and hands it to the developer.
* **Developer** reads the current mock source file and proposes a minimal
  ``find``/``replace`` patch.
* **Tester** applies the patch to a :class:`BugSandbox`, runs the synthetic unit
  tests and either bounces the handoff back to the developer with real failure
  logs or closes the workflow with a ``DONE`` message.

Loop safety lives in the orchestrator, not only in the prompt:

* at most ``max_repair_cycles`` developer -> tester rounds;
* a repeated patch signature is detected immediately;
* both conditions force a ``DONE`` with ``status="escalated_unresolved"``.

The module also loads the frozen 25-scenario dataset in
``data/bug_reports.jsonl``.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Literal, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    ValidationError,
    model_validator,
)

from ahp.models import (
    AHPMessage,
    DoneMessage,
    DoneStatus,
    HandoffMessage,
    NeedInfoMessage,
)
from ahp.router import InProcessRouter, RouterConfig
from engine.interfaces import LLMClient
from harness.agent import BaseAgent, ToolCallRecord
from harness.tools.base import EMPTY_PARAMETERS, Tool
from harness.workflows.bug_fix_schemas import (
    MAX_REPAIR_CYCLES,
    SUMMARY_MAX_CHARS,
    developer_reply_schema,
    tester_reply_schema,
    triage_reply_schema,
)

PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[2]
DEFAULT_DATASET_PATH: Final[Path] = PROJECT_ROOT / "data" / "bug_reports.jsonl"
PROMPTS_DIR: Final[Path] = Path(__file__).with_name("prompts")

FAILURE_LOG_MAX_CHARS: Final[int] = 2000
EscalationReason = Literal["max_repair_cycles", "repeated_patch"]


def _load_prompt(name: str) -> str:
    return (PROMPTS_DIR / name).read_text(encoding="utf-8").strip()


BUG_TRIAGE_PROMPT: Final[str] = _load_prompt("bug_triage_prompt.txt")
DEVELOPER_PROMPT: Final[str] = _load_prompt("developer_prompt.txt")
TESTER_PROMPT: Final[str] = _load_prompt("tester_prompt.txt")


def _json(value: Any) -> str:
    return json.dumps(value, indent=2, sort_keys=True, default=str)


def _truncate(text: str, limit: int) -> str:
    stripped = text.strip()
    if len(stripped) <= limit:
        return stripped
    return stripped[: limit - 1].rstrip() + "…"


def _patch_signature(patch: Any) -> str:
    """Canonical fingerprint used to detect a developer resubmitting a patch."""

    return json.dumps(patch, sort_keys=True, separators=(",", ":"), default=str)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class PatchEdit(BaseModel):
    """One exact ``find``/``replace`` edit in a developer patch."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    find: str = Field(min_length=1)
    replace: str = Field(min_length=1)


class BugTest(BaseModel):
    """A synthetic unit test expressed as required and forbidden substrings."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(min_length=1)
    description: str = ""
    requires: list[str] = Field(min_length=1)
    forbids: list[str] = Field(default_factory=list)


class BugReport(BaseModel):
    """One frozen bug scenario with its ground-truth repair path."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    task_id: str = Field(min_length=1)
    title: str = Field(min_length=1)
    error_stack_trace: str = Field(min_length=1)
    file_path: str = Field(min_length=1)
    broken_code: str = Field(min_length=1)
    failing_tests: list[BugTest] = Field(min_length=1)
    patches: list[list[PatchEdit]] = Field(min_length=1)
    expected_outcome: Literal["resolved", "escalated_unresolved"]
    expected_developer_attempts: int = Field(ge=1)
    expected_escalation_reason: EscalationReason | None = None

    @model_validator(mode="after")
    def _validate_ground_truth(self) -> Self:
        if len(self.patches) < self.expected_developer_attempts:
            raise ValueError("patches must cover every expected developer attempt")
        if self.expected_outcome == "resolved":
            if self.expected_escalation_reason is not None:
                raise ValueError("resolved bugs must not declare an escalation reason")
            if self.expected_developer_attempts > MAX_REPAIR_CYCLES:
                raise ValueError("resolved bugs must fit within the repair-cycle limit")
        else:
            if self.expected_escalation_reason is None:
                raise ValueError("escalated bugs must declare expected_escalation_reason")
            if self.expected_developer_attempts < 1:
                raise ValueError("escalated bugs must have at least one attempt")

        if self.expected_escalation_reason == "repeated_patch":
            if self.expected_developer_attempts < 2:
                raise ValueError("repeated-patch escalation needs at least two attempts")
            first = self.patches[0]
            second = self.patches[1]
            if first != second:
                raise ValueError("repeated-patch scenarios must repeat the first patch")
        return self

    @property
    def failing_test_names(self) -> tuple[str, ...]:
        return tuple(test.name for test in self.failing_tests)

    def prompt_record(self) -> dict[str, JsonValue]:
        """The bug report shown to triage, excluding the repair path."""

        return {
            "task_id": self.task_id,
            "title": self.title,
            "error_stack_trace": self.error_stack_trace,
            "file_path": self.file_path,
            "broken_code": self.broken_code,
            "failing_tests": [
                {"name": test.name, "description": test.description} for test in self.failing_tests
            ],
        }


def load_bug_reports(path: str | Path | None = None) -> list[BugReport]:
    """Load the JSONL bug dataset, rejecting malformed or duplicate rows."""

    dataset_path = Path(path) if path is not None else DEFAULT_DATASET_PATH
    reports: list[BugReport] = []
    seen: set[str] = set()

    with dataset_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                report = BugReport.model_validate_json(stripped)
            except ValueError as exc:
                raise ValueError(f"invalid bug report on line {line_number}: {exc}") from exc
            if report.task_id in seen:
                raise ValueError(f"duplicate task_id {report.task_id!r} on line {line_number}")
            seen.add(report.task_id)
            reports.append(report)

    return reports


# ---------------------------------------------------------------------------
# Sandbox and synthetic verification
# ---------------------------------------------------------------------------


class PatchApplyError(ValueError):
    """A developer patch could not be applied to the current source file."""


@dataclass(frozen=True, slots=True)
class TestOutcome:
    """Result of one synthetic unit test."""

    name: str
    passed: bool
    log: str

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "passed": self.passed, "log": self.log}


@dataclass(frozen=True, slots=True)
class VerificationResult:
    """Everything the tester observed for one developer attempt."""

    attempt: int
    applied: bool
    passed: bool
    tests: tuple[TestOutcome, ...] = ()
    error: str | None = None
    failure_log: str = ""

    @property
    def total_tests(self) -> int:
        return len(self.tests)

    @property
    def passed_tests(self) -> int:
        return sum(1 for test in self.tests if test.passed)

    @property
    def failing_tests(self) -> tuple[str, ...]:
        return tuple(test.name for test in self.tests if not test.passed)

    def to_dict(self) -> dict[str, Any]:
        return {
            "attempt": self.attempt,
            "applied": self.applied,
            "passed": self.passed,
            "tests_passed": self.passed_tests,
            "total_tests": self.total_tests,
            "tests": [test.to_dict() for test in self.tests],
            "failure_log": self.failure_log,
            "error": self.error,
        }


class BugSandbox:
    """Deterministic mock codebase plus synthetic unit-test runner."""

    def __init__(self, report: BugReport | None = None) -> None:
        self._report: BugReport | None = None
        self._code = ""
        if report is not None:
            self.reset(report)

    def reset(self, report: BugReport) -> None:
        self._report = report
        self._code = report.broken_code

    @property
    def report(self) -> BugReport:
        if self._report is None:
            raise RuntimeError("bug sandbox has no report; call reset() first")
        return self._report

    @property
    def code(self) -> str:
        return self._code

    def read_file(self) -> dict[str, Any]:
        """Return the current file content and the failing test catalogue."""

        report = self.report
        return {
            "file_path": report.file_path,
            "code": self._code,
            "failing_tests": [
                {"name": test.name, "description": test.description}
                for test in report.failing_tests
            ],
        }

    def apply_patch(self, edits: list[PatchEdit] | list[Mapping[str, Any]]) -> dict[str, Any]:
        """Apply all edits atomically, rejecting ambiguous or missing anchors."""

        validated = self._coerce_edits(edits)
        candidate = self._code
        for edit in validated:
            occurrences = candidate.count(edit.find)
            if occurrences != 1:
                raise PatchApplyError(
                    f"find string must occur exactly once, found {occurrences}: {edit.find!r}"
                )
            candidate = candidate.replace(edit.find, edit.replace, 1)

        self._code = candidate
        return {
            "file_path": self.report.file_path,
            "edits_applied": len(validated),
        }

    def run_tests(self) -> dict[str, Any]:
        """Evaluate every synthetic test against the current source file."""

        outcomes: list[TestOutcome] = []
        for test in self.report.failing_tests:
            missing = [item for item in test.requires if item not in self._code]
            forbidden = [item for item in test.forbids if item in self._code]
            passed = not missing and not forbidden
            details: list[str] = []
            if missing:
                details.append("missing: " + ", ".join(repr(item) for item in missing))
            if forbidden:
                details.append("forbidden: " + ", ".join(repr(item) for item in forbidden))
            log = f"{test.name}: {'PASS' if passed else 'FAIL'}"
            if details:
                log += " (" + "; ".join(details) + ")"
            outcomes.append(TestOutcome(name=test.name, passed=passed, log=log))

        failing = [outcome for outcome in outcomes if not outcome.passed]
        failure_log = "\n".join(outcome.log for outcome in failing)
        return {
            "passed": not failing,
            "total_tests": len(outcomes),
            "tests_passed": len(outcomes) - len(failing),
            "tests": [outcome.to_dict() for outcome in outcomes],
            "failure_log": _truncate(failure_log, FAILURE_LOG_MAX_CHARS),
        }

    @staticmethod
    def _coerce_edits(
        edits: list[PatchEdit] | list[Mapping[str, Any]],
    ) -> list[PatchEdit]:
        if not isinstance(edits, list) or not edits:
            raise PatchApplyError("patch must be a non-empty list of edits")
        validated: list[PatchEdit] = []
        for index, edit in enumerate(edits):
            if isinstance(edit, PatchEdit):
                validated.append(edit)
                continue
            if not isinstance(edit, Mapping):
                raise PatchApplyError(f"patch[{index}] must be an object")
            try:
                validated.append(PatchEdit(find=str(edit["find"]), replace=str(edit["replace"])))
            except (KeyError, ValidationError, TypeError) as exc:
                raise PatchApplyError(f"patch[{index}] is invalid: {exc}") from exc
        return validated


def read_file_tool(sandbox: BugSandbox) -> Tool:
    """Expose the current mock file to triage and developer agents."""

    def handler() -> dict[str, Any]:
        return sandbox.read_file()

    return Tool(
        name="read_file",
        description=(
            "Read the current mock source file plus the failing test names and "
            "descriptions. Call this before proposing a patch."
        ),
        parameters=dict(EMPTY_PARAMETERS),
        handler=handler,
    )


# ---------------------------------------------------------------------------
# Agents
# ---------------------------------------------------------------------------


class BugTriageAgent(BaseAgent):
    """Turns the bug report into a reproduction specification."""

    def __init__(
        self,
        llm: LLMClient,
        *,
        sandbox: BugSandbox,
        reply_schema: Mapping[str, Any],
        **kwargs: Any,
    ) -> None:
        super().__init__(
            "bug_triage_agent",
            BUG_TRIAGE_PROMPT,
            llm,
            tools=[read_file_tool(sandbox)],
            reply_schema=reply_schema,
            **kwargs,
        )


class DeveloperAgent(BaseAgent):
    """Inspects the mock file and proposes a patch."""

    def __init__(
        self,
        llm: LLMClient,
        *,
        sandbox: BugSandbox,
        reply_schema: Mapping[str, Any],
        **kwargs: Any,
    ) -> None:
        super().__init__(
            "developer_agent",
            DEVELOPER_PROMPT,
            llm,
            tools=[read_file_tool(sandbox)],
            reply_schema=reply_schema,
            **kwargs,
        )


class TesterAgent(BaseAgent):
    """Applies patches, runs synthetic tests and reports the outcome."""

    def __init__(
        self,
        llm: LLMClient,
        *,
        sandbox: BugSandbox,
        reply_schema: Mapping[str, Any],
        summary_max_chars: int = SUMMARY_MAX_CHARS,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            "tester_agent",
            TESTER_PROMPT,
            llm,
            tools=(),
            reply_schema=reply_schema,
            **kwargs,
        )
        self.sandbox = sandbox
        self.summary_max_chars = summary_max_chars
        self.verifications: list[VerificationResult] = []

    def reset(self) -> None:
        """Clear verification history before a new workflow run."""

        self.verifications.clear()

    async def handle_handoff(
        self,
        message: HandoffMessage | NeedInfoMessage,
    ) -> AHPMessage:
        """Verify the developer's patch, then let the model report the result."""

        if not isinstance(message, HandoffMessage):
            raise TypeError("tester_agent only accepts HandoffMessage")

        attempt = self._attempt_number(message.facts.get("attempt"))
        verification, records = self._verify(message.facts.get("patch"), attempt)
        self.verifications.append(verification)

        facts = dict(message.facts)
        facts["verification"] = verification.to_dict()
        bug_id = str(message.facts.get("bug_id", "unknown"))
        synthetic = HandoffMessage(
            conversation_id=message.conversation_id,
            from_agent=message.from_agent,
            to_agent=self.name,
            goal=f"Report verification for {bug_id}",
            summary=_truncate(
                f"Attempt {attempt}: {'passed' if verification.passed else 'failed'} "
                f"({verification.passed_tests}/{verification.total_tests} tests).",
                self.summary_max_chars,
            ),
            facts=facts,
            expected_output={
                "verification": "object",
                "passed": "boolean",
            },
        )

        try:
            reply = await super().handle_handoff(synthetic)
        finally:
            self.tool_calls = records
        return reply

    @staticmethod
    def _attempt_number(value: Any) -> int:
        if isinstance(value, int) and value >= 1:
            return value
        return 1

    def _verify(
        self,
        patch: Any,
        attempt: int,
    ) -> tuple[VerificationResult, list[ToolCallRecord]]:
        records: list[ToolCallRecord] = []
        try:
            edits = BugSandbox._coerce_edits(patch)
            apply_result = self.sandbox.apply_patch(edits)
        except (PatchApplyError, ValidationError, TypeError, KeyError) as exc:
            error = f"{type(exc).__name__}: {exc}"
            records.append(
                ToolCallRecord(
                    tool="apply_patch",
                    arguments={"patch": patch},
                    error=error,
                )
            )
            return (
                VerificationResult(
                    attempt=attempt,
                    applied=False,
                    passed=False,
                    error=error,
                    failure_log=_truncate(error, FAILURE_LOG_MAX_CHARS),
                ),
                records,
            )

        records.append(
            ToolCallRecord(
                tool="apply_patch",
                arguments={"patch": patch},
                result=apply_result,
            )
        )
        run_result = self.sandbox.run_tests()
        records.append(ToolCallRecord(tool="run_tests", arguments={}, result=run_result))

        tests = tuple(
            TestOutcome(
                name=str(item["name"]),
                passed=bool(item["passed"]),
                log=str(item["log"]),
            )
            for item in run_result["tests"]
        )
        return (
            VerificationResult(
                attempt=attempt,
                applied=True,
                passed=bool(run_result["passed"]),
                tests=tests,
                failure_log=str(run_result["failure_log"]),
            ),
            records,
        )


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class BugFixRunResult:
    """Everything observable about one bug-fix run."""

    report: BugReport
    done: DoneMessage
    messages: tuple[AHPMessage, ...]
    tool_calls: Mapping[str, tuple[ToolCallRecord, ...]]
    developer_attempts: int
    tester_runs: int
    escalation_reason: str | None = None

    @property
    def outcome(self) -> str:
        return str(self.done.status)

    @property
    def resolved(self) -> bool:
        return self.done.status is DoneStatus.RESOLVED

    @property
    def tests_passed(self) -> int:
        value = self.done.final_result.get("tests_passed", 0)
        return int(value) if isinstance(value, int) else 0

    @property
    def total_tests(self) -> int:
        value = self.done.final_result.get("total_tests", len(self.report.failing_tests))
        return int(value) if isinstance(value, int) else len(self.report.failing_tests)

    @property
    def failure_log(self) -> str:
        return str(self.done.final_result.get("failure_log", ""))

    @property
    def matches_ground_truth(self) -> bool:
        if self.outcome != self.report.expected_outcome:
            return False
        if self.developer_attempts != self.report.expected_developer_attempts:
            return False
        return self.escalation_reason == self.report.expected_escalation_reason

    @property
    def developer_handoffs(self) -> tuple[HandoffMessage, ...]:
        """Developer -> tester handoffs, one per repair attempt."""

        return tuple(
            message
            for message in self.messages
            if isinstance(message, HandoffMessage) and message.from_agent == "developer_agent"
        )

    @property
    def tester_handoffs(self) -> tuple[HandoffMessage, ...]:
        """Tester -> developer failure handoffs."""

        return tuple(
            message
            for message in self.messages
            if isinstance(message, HandoffMessage) and message.from_agent == "tester_agent"
        )

    @property
    def summary_lengths(self) -> dict[str, int]:
        lengths: dict[str, int] = {}
        for message in self.messages:
            if isinstance(message, HandoffMessage | DoneMessage):
                lengths[f"{message.from_agent}->{message.to_agent}"] = len(message.summary)
        return lengths


class BugFixLoop:
    """Orchestrate the triage -> developer <-> tester repair loop."""

    def __init__(
        self,
        triage_llm: LLMClient,
        developer_llm: LLMClient,
        tester_llm: LLMClient,
        *,
        sandbox: BugSandbox | None = None,
        summary_max_chars: int = SUMMARY_MAX_CHARS,
        max_repair_cycles: int = MAX_REPAIR_CYCLES,
        timeout_s: float = 60.0,
        max_hops: int = 20,
        ack_timeout_s: float = 5.0,
    ) -> None:
        if summary_max_chars < 1:
            raise ValueError("summary_max_chars must be >= 1")
        if max_repair_cycles < 1:
            raise ValueError("max_repair_cycles must be >= 1")
        if timeout_s <= 0:
            raise ValueError("timeout_s must be > 0")
        if max_hops < 1:
            raise ValueError("max_hops must be >= 1")
        if ack_timeout_s <= 0:
            raise ValueError("ack_timeout_s must be > 0")

        self.summary_max_chars = summary_max_chars
        self.max_repair_cycles = max_repair_cycles
        self.timeout_s = timeout_s
        self.max_hops = max_hops
        self.ack_timeout_s = ack_timeout_s
        self.sandbox = sandbox if sandbox is not None else BugSandbox()

        self.triage = BugTriageAgent(
            triage_llm,
            sandbox=self.sandbox,
            reply_schema=triage_reply_schema(summary_max_chars),
        )
        self.developer = DeveloperAgent(
            developer_llm,
            sandbox=self.sandbox,
            reply_schema=developer_reply_schema(summary_max_chars),
        )
        self.tester = TesterAgent(
            tester_llm,
            sandbox=self.sandbox,
            reply_schema=tester_reply_schema(summary_max_chars),
            summary_max_chars=summary_max_chars,
        )

    @classmethod
    def from_shared_llm(cls, llm: LLMClient, **kwargs: Any) -> BugFixLoop:
        """Use one client for all three agents (typical for a local engine)."""

        return cls(llm, llm, llm, **kwargs)

    @property
    def agents(self) -> dict[str, BaseAgent]:
        return {
            "bug_triage_agent": self.triage,
            "developer_agent": self.developer,
            "tester_agent": self.tester,
        }

    async def run_report(self, report: BugReport) -> BugFixRunResult:
        """Run one bug report through the repair loop until ``DONE``."""

        self.sandbox.reset(report)
        self.tester.reset()
        conversation_id = f"bugfix-{report.task_id}"
        messages: list[AHPMessage] = []
        tool_calls: dict[str, list[ToolCallRecord]] = {
            "bug_triage_agent": [],
            "developer_agent": [],
            "tester_agent": [],
        }
        attempts = 0
        seen_patches: set[str] = set()
        escalation_reason: str | None = None

        def wrap(name: str, agent: BaseAgent):
            async def handler(message: AHPMessage) -> AHPMessage:
                reply = await agent.handle_handoff(message)
                messages.append(reply)
                tool_calls[name].extend(agent.tool_calls)
                return reply

            return handler

        async def developer_handler(message: AHPMessage) -> AHPMessage:
            nonlocal attempts
            attempts += 1
            reply = await self.developer.handle_handoff(message)
            messages.append(reply)
            tool_calls["developer_agent"].extend(self.developer.tool_calls)
            return reply

        async def tester_handler(message: AHPMessage) -> AHPMessage:
            nonlocal escalation_reason
            if not isinstance(message, HandoffMessage):
                raise TypeError("tester_agent only accepts HandoffMessage")

            signature = _patch_signature(message.facts.get("patch"))
            if signature in seen_patches:
                escalation_reason = "repeated_patch"
                done = self._escalation_done(
                    conversation_id,
                    report,
                    attempts,
                    escalation_reason,
                )
                messages.append(done)
                return done
            seen_patches.add(signature)

            reply = await self.tester.handle_handoff(message)
            tool_calls["tester_agent"].extend(self.tester.tool_calls)

            if isinstance(reply, HandoffMessage) and attempts >= self.max_repair_cycles:
                escalation_reason = "max_repair_cycles"
                done = self._escalation_done(
                    conversation_id,
                    report,
                    attempts,
                    escalation_reason,
                )
                messages.append(done)
                return done

            messages.append(reply)
            return reply

        router = InProcessRouter(
            RouterConfig(max_hops=self.max_hops, ack_timeout_s=self.ack_timeout_s),
            run_id=conversation_id,
        )
        router.register_agent("bug_triage_agent", wrap("bug_triage_agent", self.triage))
        router.register_agent("developer_agent", developer_handler)
        router.register_agent("tester_agent", tester_handler)

        initial = self._initial_handoff(report, conversation_id)
        messages.append(initial)
        try:
            done = await router.run_workflow(initial, timeout_s=self.timeout_s)
        finally:
            await router.stop()

        return BugFixRunResult(
            report=report,
            done=done,
            messages=tuple(messages),
            tool_calls={name: tuple(calls) for name, calls in tool_calls.items()},
            developer_attempts=attempts,
            tester_runs=len(self.tester.verifications),
            escalation_reason=escalation_reason,
        )

    # -- helpers -------------------------------------------------------------

    def _initial_handoff(self, report: BugReport, conversation_id: str) -> HandoffMessage:
        summary = _truncate(
            f"{report.title}: {report.error_stack_trace}",
            self.summary_max_chars,
        )
        return HandoffMessage(
            conversation_id=conversation_id,
            from_agent="user",
            to_agent="bug_triage_agent",
            goal=f"Diagnose and fix bug {report.task_id}",
            summary=summary,
            facts=report.prompt_record(),
            expected_output={
                "reproduction_steps": "array[string]",
                "suspected_cause": "string",
                "patch": "array of find/replace edits",
            },
            constraints={
                "summary_max_chars": self.summary_max_chars,
                "max_repair_cycles": self.max_repair_cycles,
            },
        )

    def _escalation_done(
        self,
        conversation_id: str,
        report: BugReport,
        attempts: int,
        reason: str,
    ) -> DoneMessage:
        verification = self.tester.verifications[-1] if self.tester.verifications else None
        final_result: dict[str, JsonValue] = {
            "bug_id": report.task_id,
            "outcome": "escalated_unresolved",
            "reason": reason,
            "attempts": attempts,
            "tests_passed": verification.passed_tests if verification else 0,
            "total_tests": (
                verification.total_tests if verification else len(report.failing_tests)
            ),
            "failure_log": verification.failure_log if verification else "",
        }
        return DoneMessage(
            conversation_id=conversation_id,
            from_agent="tester_agent",
            to_agent="user",
            final_result=final_result,
            summary=_truncate(
                f"Escalated {report.task_id} after {attempts} attempt(s): {reason}.",
                self.summary_max_chars,
            ),
            execution_metrics={
                "developer_attempts": attempts,
                "tester_runs": len(self.tester.verifications),
                "escalation_reason": reason,
            },
            status=DoneStatus.ESCALATED_UNRESOLVED,
        )


__all__ = [
    "BUG_TRIAGE_PROMPT",
    "BugFixLoop",
    "BugFixRunResult",
    "BugReport",
    "BugSandbox",
    "BugTest",
    "BugTriageAgent",
    "DEFAULT_DATASET_PATH",
    "DEVELOPER_PROMPT",
    "DeveloperAgent",
    "EscalationReason",
    "FAILURE_LOG_MAX_CHARS",
    "PatchApplyError",
    "PatchEdit",
    "PROJECT_ROOT",
    "TESTER_PROMPT",
    "TestOutcome",
    "TesterAgent",
    "VerificationResult",
    "load_bug_reports",
    "read_file_tool",
]
