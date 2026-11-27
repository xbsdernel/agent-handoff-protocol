"""Tests for the frozen bug-report dataset (Issue #16)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from harness.workflows.bug_fix_loop import (
    BugReport,
    BugTest,
    PatchEdit,
    load_bug_reports,
)


@pytest.fixture(scope="module")
def reports() -> list[BugReport]:
    return load_bug_reports()


def test_dataset_has_25_unique_ordered_reports(reports: list[BugReport]) -> None:
    assert len(reports) == 25
    assert len({report.task_id for report in reports}) == 25
    assert [report.task_id for report in reports] == [f"BUG-{index:03d}" for index in range(1, 26)]


def test_outcome_distribution(reports: list[BugReport]) -> None:
    resolved = [report for report in reports if report.expected_outcome == "resolved"]
    escalated = [report for report in reports if report.expected_outcome == "escalated_unresolved"]
    repeated = [
        report for report in reports if report.expected_escalation_reason == "repeated_patch"
    ]
    max_cycles = [
        report for report in reports if report.expected_escalation_reason == "max_repair_cycles"
    ]

    assert len(resolved) == 15
    assert len(escalated) == 10
    assert len(repeated) == 2
    assert len(max_cycles) == 8


def test_attempt_distribution(reports: list[BugReport]) -> None:
    counts = {attempts: 0 for attempts in (1, 2, 3)}
    for report in reports:
        counts[report.expected_developer_attempts] += 1
    assert counts == {1: 10, 2: 5, 3: 10}


def test_patch_paths_produce_the_declared_outcome(reports: list[BugReport]) -> None:
    for report in reports:
        code = report.broken_code
        assert not _all_pass(code, report), f"{report.task_id} passes on broken code"

        if report.expected_escalation_reason == "repeated_patch":
            assert report.patches[0] == report.patches[1]
            code = _apply(code, report.patches[0])
            assert not _all_pass(code, report)
            continue

        for attempt, patch in enumerate(report.patches, start=1):
            code = _apply(code, patch)
            results = _run_tests(code, report)
            if report.expected_outcome == "resolved" and attempt == (
                report.expected_developer_attempts
            ):
                assert all(results), f"{report.task_id} should resolve on attempt {attempt}"
            if attempt < report.expected_developer_attempts:
                assert not all(results), (
                    f"{report.task_id} should still fail before attempt {attempt + 1}"
                )

        if report.expected_outcome == "escalated_unresolved":
            assert not _all_pass(code, report), f"{report.task_id} should stay unresolved"


def test_prompt_record_excludes_the_repair_path(reports: list[BugReport]) -> None:
    record = reports[0].prompt_record()
    assert record["task_id"] == reports[0].task_id
    assert record["broken_code"] == reports[0].broken_code
    assert "patches" not in record
    assert "expected_outcome" not in record
    assert "expected_developer_attempts" not in record
    assert isinstance(record["failing_tests"], list)
    assert set(record["failing_tests"][0]) == {"name", "description"}


def test_test_catalog_is_unique_per_report(reports: list[BugReport]) -> None:
    for report in reports:
        names = report.failing_test_names
        assert len(names) == len(set(names))
        assert all(test.requires for test in report.failing_tests)
        assert all(item for test in report.failing_tests for item in test.requires)


# ---------------------------------------------------------------------------
# Model validation
# ---------------------------------------------------------------------------


def _valid_report_kwargs() -> dict:
    return {
        "task_id": "BUG-X",
        "title": "title",
        "error_stack_trace": "trace",
        "file_path": "src/x.py",
        "broken_code": "def x():\n    return 1\n",
        "failing_tests": [
            BugTest(name="test_x", requires=["return 2"]),
        ],
        "patches": [[PatchEdit(find="return 1", replace="return 2")]],
        "expected_outcome": "resolved",
        "expected_developer_attempts": 1,
    }


def test_resolved_report_rejects_escalation_reason() -> None:
    with pytest.raises(ValidationError, match="escalation reason"):
        BugReport(
            **_valid_report_kwargs(),
            expected_escalation_reason="max_repair_cycles",
        )


def test_escalated_report_requires_escalation_reason() -> None:
    kwargs = _valid_report_kwargs()
    kwargs["expected_outcome"] = "escalated_unresolved"
    with pytest.raises(ValidationError, match="expected_escalation_reason"):
        BugReport(**kwargs)


def test_patches_must_cover_every_expected_attempt() -> None:
    kwargs = _valid_report_kwargs()
    kwargs["expected_developer_attempts"] = 2
    with pytest.raises(ValidationError, match="cover every"):
        BugReport(**kwargs)


def test_repeated_patch_scenario_must_repeat_the_first_patch() -> None:
    kwargs = _valid_report_kwargs()
    kwargs.update(
        expected_outcome="escalated_unresolved",
        expected_developer_attempts=2,
        expected_escalation_reason="repeated_patch",
        patches=[
            [PatchEdit(find="return 1", replace="return 2")],
            [PatchEdit(find="return 2", replace="return 3")],
        ],
    )
    with pytest.raises(ValidationError, match="repeat the first patch"):
        BugReport(**kwargs)


def test_resolved_report_must_fit_the_cycle_limit() -> None:
    kwargs = _valid_report_kwargs()
    kwargs.update(
        expected_developer_attempts=4,
        patches=[[PatchEdit(find="return 1", replace="return 2")] for _ in range(4)],
    )
    with pytest.raises(ValidationError, match="repair-cycle limit"):
        BugReport(**kwargs)


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------


def test_loader_rejects_duplicate_task_ids(tmp_path: Path, reports: list[BugReport]) -> None:
    path = tmp_path / "bugs.jsonl"
    row = reports[0].model_dump(mode="json")
    path.write_text(json.dumps(row) + "\n" + json.dumps(row) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="duplicate task_id"):
        load_bug_reports(path)


def test_loader_reports_the_bad_line(tmp_path: Path) -> None:
    path = tmp_path / "bugs.jsonl"
    path.write_text('{"task_id": "BUG-X"}\n', encoding="utf-8")

    with pytest.raises(ValueError, match="line 1"):
        load_bug_reports(path)


def test_loader_skips_blank_lines(tmp_path: Path, reports: list[BugReport]) -> None:
    path = tmp_path / "bugs.jsonl"
    path.write_text(
        "\n" + json.dumps(reports[0].model_dump(mode="json")) + "\n\n",
        encoding="utf-8",
    )

    loaded = load_bug_reports(path)
    assert [report.task_id for report in loaded] == [reports[0].task_id]


# ---------------------------------------------------------------------------
# Helpers mirroring the sandbox rules
# ---------------------------------------------------------------------------


def _apply(code: str, patch: list[PatchEdit]) -> str:
    candidate = code
    for edit in patch:
        assert candidate.count(edit.find) == 1, edit.find
        candidate = candidate.replace(edit.find, edit.replace, 1)
    return candidate


def _run_tests(code: str, report: BugReport) -> list[bool]:
    return [
        all(item in code for item in test.requires)
        and not any(item in code for item in test.forbids)
        for test in report.failing_tests
    ]


def _all_pass(code: str, report: BugReport) -> bool:
    return all(_run_tests(code, report))
