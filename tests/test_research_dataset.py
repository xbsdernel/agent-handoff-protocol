"""Tests for the frozen research-topic dataset (Issue #15)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from harness.workflows.research_fork_join import (
    ResearchPlan,
    ResearchScope,
    ResearchTask,
    load_research_topics,
)


@pytest.fixture(scope="module")
def tasks() -> list[ResearchTask]:
    return load_research_topics()


def test_dataset_has_30_unique_ordered_tasks(tasks: list[ResearchTask]) -> None:
    assert len(tasks) == 30
    assert len({task.task_id for task in tasks}) == 30
    assert [task.task_id for task in tasks] == [f"RES-{index:03d}" for index in range(1, 31)]


def test_every_task_has_two_independent_scopes(tasks: list[ResearchTask]) -> None:
    for task in tasks:
        titles = task.scope_titles
        assert len(titles) == 2
        assert titles[0] != titles[1]
        assert all(title.strip() for title in titles)

        first, second = task.scope_facts
        assert first
        assert second
        assert set(first).isdisjoint(second)
        assert all(fact.strip() for fact in (*first, *second))


def test_key_facts_are_the_union_of_scope_facts(tasks: list[ResearchTask]) -> None:
    for task in tasks:
        first, second = task.scope_facts
        assert set(task.key_facts) == set(first) | set(second)
        assert len(task.key_facts) == len(set(task.key_facts))


def test_scope_questions_are_present(tasks: list[ResearchTask]) -> None:
    for task in tasks:
        questions = task.scope_questions
        assert len(questions) == 2
        assert all(question.strip() for question in questions)


def test_expected_discrepancies_are_explicit(tasks: list[ResearchTask]) -> None:
    with_discrepancies = {
        task.task_id: task.expected_discrepancies for task in tasks if task.expected_discrepancies
    }
    assert set(with_discrepancies) == {"RES-004", "RES-008", "RES-021"}
    assert all(len(values) == 1 for values in with_discrepancies.values())


def test_prompt_record_excludes_the_answer(tasks: list[ResearchTask]) -> None:
    record = tasks[0].prompt_record()
    assert record == {"task_id": tasks[0].task_id, "query": tasks[0].query}
    assert "scope_facts" not in record
    assert "key_facts" not in record


# ---------------------------------------------------------------------------
# Model validation
# ---------------------------------------------------------------------------


def test_research_plan_requires_two_unique_scopes() -> None:
    scope = ResearchScope(scope_id="a", title="Alpha", question="What is alpha?")
    with pytest.raises(ValidationError):
        ResearchPlan(scopes=[scope])
    with pytest.raises(ValidationError):
        ResearchPlan(scopes=[scope, scope])
    with pytest.raises(ValidationError):
        ResearchPlan(scopes=[scope, ResearchScope(scope_id="b", title="Alpha", question="Q?")])


def test_research_scope_rejects_invalid_ids() -> None:
    with pytest.raises(ValidationError):
        ResearchScope(scope_id="Bad ID", title="Alpha", question="Q?")


def test_research_task_rejects_inconsistent_key_facts() -> None:
    with pytest.raises(ValidationError, match="union"):
        ResearchTask(
            task_id="RES-X",
            query="query",
            ground_truth_facts={
                "scope_titles": ["A", "B"],
                "scope_facts": [["a"], ["b"]],
                "key_facts": ["a"],
            },
        )


def test_research_task_rejects_missing_scope_titles() -> None:
    with pytest.raises(ValidationError, match="scope_titles"):
        ResearchTask(
            task_id="RES-X",
            query="query",
            ground_truth_facts={
                "scope_facts": [["a"], ["b"]],
                "key_facts": ["a", "b"],
            },
        )


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------


def test_loader_rejects_duplicate_task_ids(tmp_path: Path, tasks: list[ResearchTask]) -> None:
    path = tmp_path / "research.jsonl"
    row = tasks[0].model_dump(mode="json")
    path.write_text(json.dumps(row) + "\n" + json.dumps(row) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="duplicate task_id"):
        load_research_topics(path)


def test_loader_reports_the_bad_line(tmp_path: Path) -> None:
    path = tmp_path / "research.jsonl"
    path.write_text(
        json.dumps({"task_id": "RES-X", "query": "query", "ground_truth_facts": {}}) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="line 1"):
        load_research_topics(path)


def test_loader_skips_blank_lines(tmp_path: Path, tasks: list[ResearchTask]) -> None:
    path = tmp_path / "research.jsonl"
    path.write_text(
        "\n" + json.dumps(tasks[0].model_dump(mode="json")) + "\n\n",
        encoding="utf-8",
    )

    loaded = load_research_topics(path)
    assert [task.task_id for task in loaded] == [tasks[0].task_id]
