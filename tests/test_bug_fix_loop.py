"""Tests for the bug triage/fix/test loop (Issue #16).

The fake clients are prompt-driven. The developer replays the dataset's patch
sequence, while the tester reacts to the deterministic verification result the
workflow puts in its incoming facts. This exercises the real loop-control code
without an external model.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from ahp.models import DoneMessage, DoneStatus, HandoffMessage
from engine.models import LLMResult
from harness.workflows.bug_fix_loop import (
    BUG_TRIAGE_PROMPT,
    DEVELOPER_PROMPT,
    TESTER_PROMPT,
    BugFixLoop,
    BugReport,
    BugSandbox,
    BugTest,
    DeveloperAgent,
    PatchApplyError,
    PatchEdit,
    load_bug_reports,
    read_file_tool,
)
from harness.workflows.bug_fix_loop import (
    TesterAgent as BugTesterAgent,
)
from harness.workflows.bug_fix_schemas import (
    MAX_REPAIR_CYCLES,
    SUMMARY_MAX_CHARS,
    developer_reply_schema,
    triage_reply_schema,
)
from harness.workflows.bug_fix_schemas import (
    tester_reply_schema as build_tester_reply_schema,
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


class BugLLM:
    """Prompt-driven fake that plays one role in the repair loop."""

    def __init__(self, role: str, report: BugReport | None = None) -> None:
        if role not in {"triage", "developer", "tester"}:
            raise ValueError(role)
        self.role = role
        self.report = report
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
        if self.role == "triage":
            payload = self._triage(messages)
        elif self.role == "developer":
            payload = self._developer(messages)
        else:
            payload = self._tester(messages)
        return LLMResult(text=json.dumps(payload), structured_data=payload)

    async def aclose(self) -> None:
        return None

    def _report(self) -> BugReport:
        if self.report is None:
            raise AssertionError("fake client needs a report")
        return self.report

    def _triage(self, messages: list[dict[str, Any]]) -> dict[str, Any]:
        incoming = _incoming_payload(messages)
        facts = incoming["facts"]
        report = self._report()
        return {
            "action": "reply",
            "reply": {
                "type": "HANDOFF",
                "to_agent": "developer_agent",
                "goal": f"Fix {report.task_id}",
                "summary": f"Reproduced {report.task_id}.",
                "facts": {
                    "bug_id": report.task_id,
                    "file_path": facts["file_path"],
                    "reproduction_steps": [
                        "Open the failing file",
                        "Run the reported failing test",
                    ],
                    "suspected_cause": "The implementation does not match the test.",
                    "failing_tests": list(report.failing_test_names),
                },
            },
        }

    def _developer(self, messages: list[dict[str, Any]]) -> dict[str, Any]:
        incoming = _incoming_payload(messages)
        facts = incoming["facts"]
        report = self._report()
        index = min(len(self.calls) - 1, len(report.patches) - 1)
        patch = report.patches[index]
        return {
            "action": "reply",
            "reply": {
                "type": "HANDOFF",
                "to_agent": "tester_agent",
                "goal": f"Verify attempt {index + 1}",
                "summary": f"Patch attempt {index + 1} for {report.task_id}.",
                "facts": {
                    "bug_id": report.task_id,
                    "file_path": facts["file_path"],
                    "attempt": index + 1,
                    "rationale": "Apply the minimal fix for the failing test.",
                    "patch": [edit.model_dump() for edit in patch],
                },
            },
        }

    def _tester(self, messages: list[dict[str, Any]]) -> dict[str, Any]:
        incoming = _incoming_payload(messages)
        facts = incoming["facts"]
        report = self._report()
        verification = facts["verification"]
        attempt = int(facts.get("attempt", verification.get("attempt", 1)))
        patch = facts.get("patch", [])

        if verification["passed"]:
            return {
                "action": "reply",
                "reply": {
                    "type": "DONE",
                    "to_agent": "user",
                    "final_result": {
                        "bug_id": report.task_id,
                        "attempts": attempt,
                        "patch": patch,
                        "tests_passed": verification["tests_passed"],
                        "total_tests": verification["total_tests"],
                        "failure_log": "",
                    },
                    "summary": f"{report.task_id} fixed on attempt {attempt}.",
                    "status": "resolved",
                },
            }

        failing = [test["name"] for test in verification["tests"] if not test["passed"]] or [
            "patch_application"
        ]
        return {
            "action": "reply",
            "reply": {
                "type": "HANDOFF",
                "to_agent": "developer_agent",
                "goal": f"Repair attempt {attempt}",
                "summary": f"Attempt {attempt} failed {len(failing)} test(s).",
                "facts": {
                    "bug_id": report.task_id,
                    "attempt": attempt,
                    "failure_log": verification["failure_log"],
                    "failing_tests": failing,
                    "suggested_next_step": "Fix the failing assertions.",
                },
            },
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _report(task_id: str) -> BugReport:
    return next(report for report in load_bug_reports() if report.task_id == task_id)


def _loop(
    report: BugReport,
    **kwargs: Any,
) -> tuple[BugFixLoop, BugLLM, BugLLM, BugLLM]:
    triage = BugLLM("triage", report)
    developer = BugLLM("developer", report)
    tester = BugLLM("tester", report)
    return BugFixLoop(triage, developer, tester, **kwargs), triage, developer, tester


# ---------------------------------------------------------------------------
# End-to-end dataset
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "report",
    load_bug_reports(),
    ids=lambda report: report.task_id,
)
async def test_every_dataset_scenario_matches_ground_truth(report: BugReport) -> None:
    loop, _, _, _ = _loop(report)

    result = await loop.run_report(report)

    assert result.matches_ground_truth, (
        report.task_id,
        result.outcome,
        result.developer_attempts,
        result.escalation_reason,
    )
    assert result.developer_attempts == report.expected_developer_attempts
    assert result.outcome == report.expected_outcome
    assert len(result.developer_handoffs) == report.expected_developer_attempts
    assert all(length <= SUMMARY_MAX_CHARS for length in result.summary_lengths.values())

    tester_tools = [call.tool for call in result.tool_calls["tester_agent"]]
    assert tester_tools.count("apply_patch") == result.tester_runs
    assert tester_tools.count("run_tests") == result.tester_runs

    if report.expected_outcome == "resolved":
        assert result.resolved
        assert isinstance(result.done, DoneMessage)
        assert result.done.status is DoneStatus.RESOLVED
        assert result.tests_passed == result.total_tests
        assert result.tester_runs == report.expected_developer_attempts
        assert result.escalation_reason is None
    else:
        assert not result.resolved
        assert result.done.status is DoneStatus.ESCALATED_UNRESOLVED
        assert result.failure_log
        expected_runs = (
            1
            if report.expected_escalation_reason == "repeated_patch"
            else report.expected_developer_attempts
        )
        assert result.tester_runs == expected_runs


async def test_multi_cycle_fix_bounces_between_developer_and_tester() -> None:
    report = _report("BUG-011")
    loop, _, developer, tester = _loop(report)

    result = await loop.run_report(report)

    assert result.resolved
    assert result.developer_attempts == 2
    assert len(result.developer_handoffs) == 2
    assert len(result.tester_handoffs) == 1
    assert len(tester.calls) == 2
    assert len(developer.calls) == 2
    assert result.tester_handoffs[0].facts["failing_tests"] == ["test_area_circle"]


async def test_max_repair_cycles_hard_terminates() -> None:
    report = _report("BUG-016")
    loop, _, _, _ = _loop(report, max_repair_cycles=2)

    result = await loop.run_report(report)

    assert result.outcome == "escalated_unresolved"
    assert result.developer_attempts == 2
    assert result.tester_runs == 2
    assert result.escalation_reason == "max_repair_cycles"
    assert result.done.status is DoneStatus.ESCALATED_UNRESOLVED


async def test_repeated_patch_escalates_before_the_tester_runs() -> None:
    report = _report("BUG-024")
    loop, _, developer, tester = _loop(report)

    result = await loop.run_report(report)

    assert result.outcome == "escalated_unresolved"
    assert result.developer_attempts == 2
    assert result.tester_runs == 1
    assert len(tester.calls) == 1
    assert len(developer.calls) == 2
    assert result.escalation_reason == "repeated_patch"


async def test_patch_application_failure_is_reported_and_escalated() -> None:
    report = BugReport(
        task_id="BUG-MISSING",
        title="Patch anchor is missing",
        error_stack_trace="AssertionError: broken",
        file_path="src/x.py",
        broken_code="def x():\n    return 1\n",
        failing_tests=[BugTest(name="test_x", requires=["return 2"])],
        patches=[[PatchEdit(find="does-not-exist", replace="return 2")]],
        expected_outcome="escalated_unresolved",
        expected_developer_attempts=1,
        expected_escalation_reason="max_repair_cycles",
    )
    loop, _, _, _ = _loop(report, max_repair_cycles=1)

    result = await loop.run_report(report)

    assert result.outcome == "escalated_unresolved"
    assert result.escalation_reason == "max_repair_cycles"
    assert "PatchApplyError" in result.failure_log
    apply_call = result.tool_calls["tester_agent"][0]
    assert apply_call.tool == "apply_patch"
    assert not apply_call.ok
    assert [call.tool for call in result.tool_calls["tester_agent"]] == ["apply_patch"]


# ---------------------------------------------------------------------------
# Sandbox
# ---------------------------------------------------------------------------


def test_sandbox_applies_patches_and_runs_tests() -> None:
    report = _report("BUG-011")
    sandbox = BugSandbox(report)

    first = sandbox.apply_patch(report.patches[0])
    after_first = sandbox.run_tests()
    second = sandbox.apply_patch(report.patches[1])
    after_second = sandbox.run_tests()

    assert first["edits_applied"] == 1
    assert second["edits_applied"] == 1
    assert after_first["passed"] is False
    assert after_first["tests_passed"] == 1
    assert after_second["passed"] is True
    assert after_second["tests_passed"] == after_second["total_tests"]
    assert after_second["failure_log"] == ""


def test_sandbox_patch_failure_is_atomic() -> None:
    report = _report("BUG-001")
    sandbox = BugSandbox(report)
    before = sandbox.code

    with pytest.raises(PatchApplyError, match="exactly once"):
        sandbox.apply_patch(
            [
                PatchEdit(find="return amount * rate", replace="x"),
                PatchEdit(find="missing", replace="y"),
            ]
        )

    assert sandbox.code == before


def test_sandbox_rejects_ambiguous_and_invalid_patches() -> None:
    report = _report("BUG-001")
    sandbox = BugSandbox(report)

    with pytest.raises(PatchApplyError):
        sandbox.apply_patch([])
    with pytest.raises(PatchApplyError):
        sandbox.apply_patch([{"find": "amount", "replace": "x"}])
    with pytest.raises(PatchApplyError):
        sandbox.apply_patch([{"replace": "x"}])
    with pytest.raises(PatchApplyError):
        sandbox.apply_patch(["not-an-edit"])


def test_sandbox_read_file_reflects_current_code() -> None:
    report = _report("BUG-001")
    sandbox = BugSandbox(report)
    sandbox.apply_patch(report.patches[0])

    record = sandbox.read_file()

    assert record["file_path"] == report.file_path
    assert "return amount * (1 + rate)" in record["code"]
    assert record["failing_tests"] == [
        {"name": test.name, "description": test.description} for test in report.failing_tests
    ]


def test_read_file_tool_definition() -> None:
    sandbox = BugSandbox(_report("BUG-001"))
    tool = read_file_tool(sandbox)

    assert tool.name == "read_file"
    assert tool.parameters["properties"] == {}
    assert tool.handler()["file_path"] == "src/billing/tax.py"


# ---------------------------------------------------------------------------
# Prompts, schemas and configuration
# ---------------------------------------------------------------------------


def test_prompts_and_schemas_load() -> None:
    assert "BUG TRIAGE AGENT" in BUG_TRIAGE_PROMPT
    assert "DEVELOPER AGENT" in DEVELOPER_PROMPT
    assert "TESTER AGENT" in TESTER_PROMPT
    assert MAX_REPAIR_CYCLES == 3
    assert SUMMARY_MAX_CHARS == 600
    assert triage_reply_schema()["properties"]["to_agent"]["const"] == "developer_agent"
    assert developer_reply_schema()["properties"]["to_agent"]["const"] == "tester_agent"
    tester = build_tester_reply_schema()
    assert len(tester["oneOf"]) == 2
    assert tester["oneOf"][0]["properties"]["to_agent"]["const"] == "developer_agent"
    assert tester["oneOf"][1]["properties"]["to_agent"]["const"] == "user"


async def test_agents_request_role_specific_schemas() -> None:
    report = _report("BUG-001")
    loop, triage, developer, tester = _loop(report)

    await loop.run_report(report)

    assert triage.calls[0]["schema"]["oneOf"][1]["properties"]["reply"] == triage_reply_schema()
    assert (
        developer.calls[0]["schema"]["oneOf"][1]["properties"]["reply"] == developer_reply_schema()
    )
    assert (
        tester.calls[0]["schema"]["oneOf"][1]["properties"]["reply"] == build_tester_reply_schema()
    )


def test_from_shared_llm_builds_all_agents() -> None:
    report = _report("BUG-001")
    loop = BugFixLoop.from_shared_llm(BugLLM("triage", report))

    assert set(loop.agents) == {
        "bug_triage_agent",
        "developer_agent",
        "tester_agent",
    }


def test_constructor_validation() -> None:
    report = _report("BUG-001")
    llm = BugLLM("triage", report)
    with pytest.raises(ValueError):
        BugFixLoop(llm, llm, llm, summary_max_chars=0)
    with pytest.raises(ValueError):
        BugFixLoop(llm, llm, llm, max_repair_cycles=0)
    with pytest.raises(ValueError):
        BugFixLoop(llm, llm, llm, timeout_s=0)
    with pytest.raises(ValueError):
        BugFixLoop(llm, llm, llm, max_hops=0)
    with pytest.raises(ValueError):
        BugFixLoop(llm, llm, llm, ack_timeout_s=0)


async def test_long_report_is_truncated() -> None:
    report = _report("BUG-001").model_copy(
        update={
            "title": "T" * 1000,
            "error_stack_trace": "S" * 1000,
        }
    )
    loop, _, _, _ = _loop(report)

    result = await loop.run_report(report)

    initial = result.messages[0]
    assert isinstance(initial, HandoffMessage)
    assert len(initial.summary) == SUMMARY_MAX_CHARS
    assert initial.summary.endswith("…")
    assert result.resolved
    assert all(length <= SUMMARY_MAX_CHARS for length in result.summary_lengths.values())


async def test_developer_agent_exposes_read_file_tool() -> None:
    report = _report("BUG-001")
    sandbox = BugSandbox(report)
    agent = DeveloperAgent(
        BugLLM("developer", report),
        sandbox=sandbox,
        reply_schema=developer_reply_schema(),
    )

    assert agent.tools.names == ("read_file",)


async def test_tester_reset_clears_verification_history() -> None:
    report = _report("BUG-001")
    sandbox = BugSandbox(report)
    tester = BugTesterAgent(
        BugLLM("tester", report),
        sandbox=sandbox,
        reply_schema=build_tester_reply_schema(),
    )
    await tester.handle_handoff(
        HandoffMessage(
            conversation_id="conv",
            from_agent="developer_agent",
            to_agent="tester_agent",
            goal="verify",
            summary="verify",
            facts={
                "bug_id": report.task_id,
                "file_path": report.file_path,
                "attempt": 1,
                "rationale": "fix",
                "patch": [edit.model_dump() for edit in report.patches[0]],
            },
        )
    )
    assert len(tester.verifications) == 1

    tester.reset()

    assert tester.verifications == []
