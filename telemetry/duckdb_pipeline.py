"""DuckDB pipeline for events.jsonl → events.parquet (Issue #19).

Provides a utility to convert raw ``events.jsonl`` logs into a Parquet file
and register high-performance DuckDB SQL views for analytical querying.

Depends on:
- ``pyarrow`` (already available in the environment)
- ``duckdb`` (install separately: ``pip install duckdb``)

The pipeline is designed so that, when ``duckdb`` is absent, the conversion
step is skipped and a warning is logged; the Parquet file is still written
via PyArrow if desired.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Utility: JSONL → PyArrow Table
# ---------------------------------------------------------------------------

def _jsonl_to_parquet_table(path: Path) -> pa.Table | None:
    """Convert a JSONL event file into a PyArrow Table.

    Returns ``None`` if the file is empty or contains no parseable events.
    """
    events: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
                if isinstance(payload, dict):
                    events.append(payload)
            except json.JSONDecodeError:
                continue

    if not events:
        return None

    # Collect all unique keys across events to form the schema
    all_keys: set[str] = set()
    for ev in events:
        all_keys.update(ev.keys())

    # Build columns
    columns: dict[str, list[Any]] = {k: [] for k in all_keys}
    for ev in events:
        for k in all_keys:
            columns[k].append(ev.get(k))

    # Convert types where possible
    # If a column looks entirely integral, cast to int; if float-looking, cast to float
    table_columns: list[pa.Field] = []
    arrays: list[pa.Array] = []
    for k in all_keys:
        col = columns[k]
        # Try to infer type from non-None values
        non_none = [v for v in col if v is not None]
        if not non_none:
            # All None → just use string/large_string
            arrays.append(pa.array(col, type=pa.string()))
            table_columns.append(pa.field(k, pa.string()))
            continue

        # Check if all are int
        if all(isinstance(v, int) for v in non_none):
            arrays.append(pa.array(col, type=pa.int64()))
            table_columns.append(pa.field(k, pa.int64()))
        elif all(isinstance(v, float) for v in non_none):
            arrays.append(pa.array(col, type=pa.float64()))
            table_columns.append(pa.field(k, pa.float64()))
        else:
            arrays.append(pa.array(col, type=pa.string()))
            table_columns.append(pa.field(k, pa.string()))

    return pa.table({k: arr for k, arr in zip(all_keys, arrays)}, schema=pa.schema(table_columns))


# ---------------------------------------------------------------------------
# Main pipeline function
# ---------------------------------------------------------------------------

def run_pipeline(
    events_path: str | Path,
    *,
    parquet_path: str | Path | None = None,
    duckdb_path: str | Path | None = None,
) -> tuple[pa.Table | None, list[str]]:
    """Run the full events.jsonl → Parquet → DuckDB pipeline.

    Parameters
    ----------
    events_path:
        Path to the raw ``events.jsonl`` file.
    parquet_path:
        Optional output path for the Parquet file. If ``None``, derived from
        ``events_path`` by replacing the extension with ``.parquet``.
    duckdb_path:
        Optional path for the DuckDB database file. If ``None``, an in-memory
        DuckDB instance is used (requires ``duckdb`` to be installed).

    Returns
    -------
    table:
        The PyArrow Table built from the JSONL events, or ``None`` if the
        file was empty/unparseable.
    warnings:
        List of Warning-strings describing any conditions (e.g. duckdb not
        available, file empty, etc.).
    """
    warnings: list[str] = []

    events_path = Path(events_path)
    if not events_path.is_file():
        warnings.append(f"events file not found: {events_path}")
        return None, warnings

    # Step 1: JSONL → PyArrow Table
    table = _jsonl_to_parquet_table(events_path)
    if table is None:
        warnings.append("events.jsonl was empty or contained no parseable rows")
        return None, warnings

    # Step 2: Write Parquet file
    if parquet_path is None:
        parquet_path = str(events_path.with_suffix(".parquet"))
    parquet_path = Path(parquet_path)
    try:
        pq.write_table(table, parquet_path)
        warnings.append(f"Wrote Parquet file to {parquet_path}")
    except Exception as exc:  # noqa: BLE001 - write failure is non-fatal
        warnings.append(f"Failed to write Parquet: {exc}")
        table = None

    # Step 3: DuckDB views (best-effort)
    duckdb_available = False
    try:
        import duckdb  # type: ignore[import-not-found]

        duckdb_available = True
    except ImportError:
        warnings.append("duckdb not installed; skipping DuckDB view registration")
        warnings.append("pip install duckdb to enable SQL analytics over Parquet")

    if duckdb_available and duckdb_path is not None:
        try:
            import duckdb as _duckdb

            con = _duckdb.connect(str(duckdb_path))
            # Register the Parquet file as a relation
            con.execute("SELECT * FROM read_parquet(?, limits=1)", [str(parquet_path)])  # sanity check
            # Register as a table for view creation
            con.execute(f"CREATE OR REPLACE TABLE events_table AS SELECT * FROM read_parquet('{parquet_path}')")
            warnings.append(f"Registered events table in DuckDB at {duckdb_path}")
            con.close()
        except Exception as exc:  # noqa: BLE001 - DB errors should not break the pipeline
            warnings.append(f"DuckDB setup warning: {exc}")
    elif duckdb_available and duckdb_path is None:
        # In-memory mode
        try:
            import duckdb as _duckdb  # type: ignore[assignment]

            con = _duckdb.connect()
            con.execute(f"CREATE OR REPLACE TABLE events_table AS SELECT * FROM read_parquet('{parquet_path}')")
            warnings.append("Created in-memory DuckDB table from Parquet")
            con.close()
        except Exception as exc:  # noqa: BLE001
            warnings.append(f"In-memory DuckDB failed: {exc}")

    # Step 4: Return the table and all warnings
    return table, warnings


# ---------------------------------------------------------------------------
# Pre-built DuckDB SQL views (when duckdb is available)
# ---------------------------------------------------------------------------

VIEW_DEFINITIONS: dict[str, str] = {
    "vw_workflow_summary": (
        "SELECT "
        "run_id, "
        "MIN(timestamp_ns) AS start_time, "
        "MAX(timestamp_ns) AS end_time, "
        "COUNT(*) AS event_count, "
        "MAX(CASE WHEN event_type = 'workflow.completed' THEN 1 ELSE 0 END) AS success, "
        "MAX(gpu_utilization_percent) AS max_gpu_util_pct "
        "FROM events_table "
        "GROUP BY run_id"
    ),
    "vw_llm_metrics": (
        "SELECT "
        "engine_name AS engine, "
        "concurrency, "
        "PERCENTILE_CONT_TD(ttft_ms, 0.5) AS p50_ttft_ms, "
        "PERCENTILE_CONT_TD(ttft_ms, 0.9) AS p90_ttft_ms, "
        "PERCENTILE_CONT_TD(ttft_ms, 0.95) AS p95_ttft_ms, "
        "AVG(latency_ms) AS avg_latency_ms, "
        "SUM(total_tokens) AS total_tokens, "
        "COUNT(*) AS num_calls "
        "FROM events_table "
        "WHERE event_type = 'llm_call' "
        "GROUP BY engine_name, concurrency"
    ),
    "vw_protocol_health": (
        "SELECT "
        "COUNT(*) AS total_events, "
        "MAX(CASE WHEN valid_schema THEN 1 ELSE 0 END) AS valid_count, "
        "MAX(CASE WHEN NOT valid_schema THEN 1 ELSE 0 END) AS malformed_count, "
        "MAX(CASE WHEN event_type LIKE 'handoff.%' THEN 1 ELSE 0 END) AS handoff_count, "
        "MAX(CASE WHEN event_type LIKE 'system.%' THEN 1 ELSE 0 END) AS system_count "
        "FROM events_table"
    ),
}


def register_views(
    con: Any,  # duckdb.DuckDBPyConnection
    parquet_path: str,
) -> dict[str, Any]:
    """Register the three pre-built DuckDB views over the given Parquet file.

    Returns a dict mapping view name → query result (as list of dicts).
    """
    results: dict[str, Any] = {}
    for view_name, sql in VIEW_DEFINITIONS.items():
        try:
            con.execute(sql)
            # Fetch result set
            rows = con.execute(f"SELECT * FROM {view_name}").fetchall()
            cols = [desc[0] for desc in con.execute(f"SELECT * FROM {view_name}").description]
            results[view_name] = [dict(zip(cols, row, strict=False)) for row in rows]
        except Exception as exc:  # noqa: BLE001 - view creation should not crash
            results[view_name] = {"error": str(exc)}
    return results