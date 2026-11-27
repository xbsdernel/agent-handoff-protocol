"""Automated ground-truth scorer for workflow runs (Issue #19).

Compares the final ``DoneMessage["final_result"]`` of a workflow run against
the frozen scenario dataset (``data/bug_reports.jsonl`` or
``data/research_topics.jsonl``) and classifies the outcome into one of:

- ``SUCCESS`` – all expected conditions met
- ``INCORRECT_DECISION`` – decision disagreed with ground truth
- ``SCHEMA_VIOLATION`` – final_result does not conform to the expected schema
- ``TIMEOUT_ABORT`` – workflow exceeded hop/ retry limits
- ``HOP_LIMIT_EXCEEDED`` – router exceeded max_hops

The scorer produces a ``scores.json`` file in the run directory, one entry
per report, keyed by ``task_id``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Outcome classification
# ---------------------------------------------------------------------------

OUTCOME_CHOICES = (
    "SUCCESS",
    "INCORRECT_DECISION",
    "SCHEMA_VIOLATION",
    "TIMEOUT_ABORT",
    "HOP_LIMIT_EXCEEDED",
)


def _classify_final_result(
    final_result: dict[str, Any],
    dataset_entry: dict[str, Any],
) -> str:
    """Classify a ``final_result`` against a dataset entry.

    The logic is rule-based and intentionally simple so that it can run
    without an LLM or external process.

    Returns one of ``OUTCOME_CHOICES``.
    """
    # --- Schema violation check first ---
    # If final_result is missing required keys that the dataset expects,
    # we flag it as a schema violation before doing anything else.
    expected_keys = set(dataset_entry.keys())
    result_keys = set(final_result.keys())
    if not expected_keys.issubset(result_keys):
        # Missing expected top-level keys -> schema violation
        return "SCHEMA_VIOLATION"

    # --- SUCCESS check ---
    # The simplest case: every dataset key appears in final_result with
    # a matching value (deep equality for top-level; tolerant of extra keys).
    all_match = all(
        final_result.get(k) == v for k, v in dataset_entry.items()
    )
    if all_match:
        return "SUCCESS"

    # --- INCORRECT_DECISION ---
    # The decision (e.g. refund yes/no, resolve/escalate) differs.
    # Heuristic: if the dataset has an "expected_decision" or "expected_outcome"
    # field and the final_result has a "decision" or "status" field that
    # disagrees, flag as incorrect.
    dataset_decision = dataset_entry.get("expected_decision") or dataset_entry.get("expected_outcome")
    result_decision = final_result.get("decision") or final_result.get("status")
    if dataset_decision is not None and result_decision is not None and str(dataset_decision) != str(result_decision):
        return "INCORRECT_DECISION"

    # --- TIMEOUT / HOP LIMIT checks ---
    # If the dataset mentions a timeout or hop limit, and the final result
    # contains a reason matching that, classify accordingly.
    for timeout_key in ("max_hops", "timeout_s", "max_steps"):
        if timeout_key in dataset_entry and timeout_key in final_result:
            return "TIMEOUT_ABORT" if timeout_key == "timeout_s" else "HOP_LIMIT_EXCEEDED"

    # Default fallback when we can't determine a tighter class:
    return "INCORRECT_DECISION"


# ---------------------------------------------------------------------------
# Main scorer function
# ---------------------------------------------------------------------------

def score_run(
    run_dir: str | Path,
    dataset_path: str | Path,
    *,
    output_path: str | Path | None = None,
) -> dict[str, str]:
    """Score every report in *dataset_path* against the run in *run_dir*.

    Parameters
    ----------
    run_dir:
        Path to the run directory created by ``RunContext.create()``.
        Must contain ``events.jsonl`` (parsed and validated).
    dataset_path:
        Path to the frozen dataset (``bug_reports.jsonl`` or
        ``research_topics.jsonl``).
    output_path:
        Optional path for the output ``scores.json``. If ``None``, derived
        from the run directory name.

    Returns
    -------
    results:
        Dict mapping ``task_id`` → outcome string (one of ``OUTCOME_CHOICES``).
    """
    run_path = Path(run_dir)
    ds_path = Path(dataset_path)

    # --- Load dataset ---
    dataset_reports: list[dict[str, Any]] = []
    with ds_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                report = json.loads(line)
                if "task_id" in report:
                    dataset_reports.append(report)
            except json.JSONDecodeError:
                continue

    # --- Load run events and build a lookup per task_id ---
    events_path = run_path / "events.jsonl"
    events_by_task: dict[str, list] = {}
    if events_path.is_file():
        from telemetry.models import load_events as _load_events

        for event in _load_events(events_path):
            run_stem = run_path.name
            events_by_task.setdefault(run_stem, []).append(event)

    # --- Score each dataset report ---
    results: dict[str, str] = {}
    for report in dataset_reports:
        task_id = report["task_id"]

        # Look for a matching final_result in the run's output.
        final_result: dict[str, Any] | None = None

        # Try reading final_result from the run's environment.json
        env_path = run_path / "environment.json"
        if env_path.is_file():
            try:
                env = json.loads(env_path.read_text(encoding="utf-8"))
                final_result = env.get("final_result")
            except Exception:
                final_result = None

        # Classify
        if final_result is not None:
            outcome = _classify_final_result(final_result, report)
        else:
            # No final_result found — we cannot classify; default to
            # TIMEOUT_ABORT as a conservative fallback.
            outcome = "TIMEOUT_ABORT"

        results[task_id] = outcome

    # --- Write scores.json ---
    output_path = run_path / "scores.json" if output_path is None else Path(output_path)

    with Path(output_path).open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2, sort_keys=True)

    return results