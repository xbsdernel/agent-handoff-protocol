"""SQLite-backed mock business data for agent tools.

The database gives agents realistic grounding without any external service:
customer accounts, billing charges and inventory are seeded deterministically
into an in-memory (or temporary file) SQLite database. Every helper returns
plain JSON-serialisable dictionaries, so results can be appended to an LLM
prompt unchanged.

Money is stored as integer cents and returned as dollar floats with two
decimals, which keeps arithmetic exact in the database and friendly in
prompts. :meth:`SQLiteMockDB.reset` restores the seed state, so tests that
mutate the database stay isolated.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Final

from harness.tools.base import Tool

CUSTOMER_ID_PARAMETERS: Final[dict[str, Any]] = {
    "type": "object",
    "properties": {
        "customer_id": {
            "type": "string",
            "minLength": 1,
            "description": "Customer identifier such as CUST-001.",
        }
    },
    "required": ["customer_id"],
    "additionalProperties": False,
}

ITEM_ID_PARAMETERS: Final[dict[str, Any]] = {
    "type": "object",
    "properties": {
        "item_id": {
            "type": "string",
            "minLength": 1,
            "description": "Inventory item identifier such as ITEM-001.",
        }
    },
    "required": ["item_id"],
    "additionalProperties": False,
}

SCHEMA_SQL: Final[str] = """
CREATE TABLE IF NOT EXISTS customers (
    customer_id TEXT PRIMARY KEY,
    name        TEXT NOT NULL,
    email       TEXT NOT NULL,
    plan        TEXT NOT NULL,
    joined_at   TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS charges (
    charge_id    TEXT PRIMARY KEY,
    customer_id  TEXT NOT NULL REFERENCES customers(customer_id),
    amount_cents INTEGER NOT NULL,
    currency     TEXT NOT NULL,
    status       TEXT NOT NULL,
    created_at   TEXT NOT NULL,
    description  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS inventory (
    item_id         TEXT PRIMARY KEY,
    name            TEXT NOT NULL,
    category        TEXT NOT NULL,
    unit_price_cents INTEGER NOT NULL,
    quantity        INTEGER NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_charges_customer ON charges(customer_id);
"""

#: Deterministic seed rows: (customer_id, name, email, plan, joined_at).
CUSTOMER_SEED: Final[tuple[tuple[str, str, str, str, str], ...]] = (
    ("CUST-001", "Ava Thompson", "ava.thompson@example.com", "pro", "2024-03-12"),
    ("CUST-002", "Ben Carter", "ben.carter@example.com", "starter", "2025-01-08"),
    ("CUST-003", "Chloe Nguyen", "chloe.nguyen@example.com", "pro", "2023-11-21"),
    ("CUST-004", "Diego Ramirez", "diego.ramirez@example.com", "enterprise", "2022-07-04"),
    ("CUST-005", "Emma Patel", "emma.patel@example.com", "starter", "2025-06-15"),
    ("CUST-006", "Farah Hassan", "farah.hassan@example.com", "pro", "2024-09-30"),
)

#: Deterministic seed rows:
#: (charge_id, customer_id, amount_cents, currency, status, created_at, description).
CHARGE_SEED: Final[tuple[tuple[str, str, int, str, str, str, str], ...]] = (
    ("CHG-1001", "CUST-001", 4900, "USD", "paid", "2026-10-01", "Pro monthly subscription"),
    ("CHG-1002", "CUST-001", 4900, "USD", "paid", "2026-10-01", "Pro monthly subscription"),
    ("CHG-1003", "CUST-001", 4900, "USD", "paid", "2026-11-01", "Pro monthly subscription"),
    ("CHG-2001", "CUST-002", 1900, "USD", "paid", "2026-11-02", "Starter monthly subscription"),
    ("CHG-3001", "CUST-003", 4900, "USD", "paid", "2026-10-15", "Pro monthly subscription"),
    ("CHG-3002", "CUST-003", 4900, "USD", "refunded", "2026-09-15", "Pro monthly subscription"),
    ("CHG-4001", "CUST-004", 29900, "USD", "paid", "2026-11-01", "Enterprise monthly subscription"),
    ("CHG-5001", "CUST-005", 1900, "USD", "paid", "2026-10-20", "Starter monthly subscription"),
    ("CHG-5002", "CUST-005", 1900, "USD", "paid", "2026-10-20", "Starter monthly subscription"),
    ("CHG-6001", "CUST-006", 4900, "USD", "failed", "2026-11-05", "Pro monthly subscription"),
)

#: Deterministic seed rows:
#: (item_id, name, category, unit_price_cents, quantity).
INVENTORY_SEED: Final[tuple[tuple[str, str, str, int, int], ...]] = (
    ("ITEM-001", "AHP Pro Sensor", "hardware", 12900, 42),
    ("ITEM-002", "AHP Starter Hub", "hardware", 5900, 0),
    ("ITEM-003", "AHP Enterprise Rack", "hardware", 89900, 7),
)


def _dollars(cents: int) -> float:
    return round(cents / 100, 2)


class SQLiteMockDB:
    """Seeded SQLite fixture used by the mock tools.

    Parameters
    ----------
    path:
        ``":memory:"`` (default) for a throw-away database, or a filesystem
        path when a test needs persistence across connections.
    auto_seed:
        When true (default) the deterministic fixtures are loaded on creation
        and by every :meth:`reset`.
    """

    def __init__(self, path: str | Path = ":memory:", *, auto_seed: bool = True) -> None:
        self.path = str(path)
        self.auto_seed = auto_seed
        self._conn = sqlite3.connect(self.path)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA foreign_keys = ON")
        self._conn.executescript(SCHEMA_SQL)
        self._conn.commit()
        if self.auto_seed:
            self.seed()

    # -- lifecycle -----------------------------------------------------------

    def seed(self) -> None:
        """Insert the fixture rows, replacing rows with the same primary key."""

        self._conn.executemany(
            "INSERT OR REPLACE INTO customers "
            "(customer_id, name, email, plan, joined_at) VALUES (?, ?, ?, ?, ?)",
            CUSTOMER_SEED,
        )
        self._conn.executemany(
            "INSERT OR REPLACE INTO charges "
            "(charge_id, customer_id, amount_cents, currency, status, created_at, description) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            CHARGE_SEED,
        )
        self._conn.executemany(
            "INSERT OR REPLACE INTO inventory "
            "(item_id, name, category, unit_price_cents, quantity) VALUES (?, ?, ?, ?, ?)",
            INVENTORY_SEED,
        )
        self._conn.commit()

    def reset(self) -> None:
        """Delete every row and restore the seed state.

        This is the isolation primitive for tests: rows inserted by one test
        never leak into the next one using the same database handle.
        """

        self._conn.execute("DELETE FROM charges")
        self._conn.execute("DELETE FROM inventory")
        self._conn.execute("DELETE FROM customers")
        self._conn.commit()
        if self.auto_seed:
            self.seed()

    def close(self) -> None:
        """Close the underlying connection."""

        self._conn.close()

    def __enter__(self) -> SQLiteMockDB:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    # -- low-level helpers ---------------------------------------------------

    def execute(
        self,
        sql: str,
        params: Sequence[Any] = (),
    ) -> list[dict[str, Any]] | int:
        """Execute raw SQL, returning rows for queries and rowcount otherwise.

        Intended for fixtures, assertions and mutations in tests. Application
        code should prefer the typed ``query_*`` helpers.
        """

        cursor = self._conn.execute(sql, tuple(params))
        if cursor.description is None:
            self._conn.commit()
            return cursor.rowcount
        rows = [dict(row) for row in cursor.fetchall()]
        self._conn.commit()
        return rows

    # -- agent-facing tools --------------------------------------------------

    def query_customer(self, customer_id: str) -> dict[str, Any]:
        """Look up one customer account by identifier."""

        row = self._conn.execute(
            "SELECT customer_id, name, email, plan, joined_at FROM customers WHERE customer_id = ?",
            (customer_id,),
        ).fetchone()
        if row is None:
            return {"customer_id": customer_id, "found": False}
        return {"found": True, **dict(row)}

    def query_charges(self, customer_id: str) -> dict[str, Any]:
        """Return a customer's charges in dollars plus suspected duplicates."""

        customer = self._conn.execute(
            "SELECT customer_id FROM customers WHERE customer_id = ?",
            (customer_id,),
        ).fetchone()
        if customer is None:
            return {
                "customer_id": customer_id,
                "found": False,
                "charges": [],
                "charge_count": 0,
                "total_amount": 0.0,
                "suspected_duplicate_charge_ids": [],
            }

        rows = self._conn.execute(
            "SELECT charge_id, amount_cents, currency, status, created_at, description "
            "FROM charges WHERE customer_id = ? ORDER BY created_at, charge_id",
            (customer_id,),
        ).fetchall()
        charges: list[dict[str, Any]] = []
        groups: dict[tuple[int, str, str], list[str]] = {}
        for row in rows:
            record = dict(row)
            amount_cents = int(record.pop("amount_cents"))
            record["amount"] = _dollars(amount_cents)
            charges.append(record)
            groups.setdefault(
                (amount_cents, str(record["created_at"]), str(record["description"])),
                [],
            ).append(str(record["charge_id"]))

        duplicates = sorted(
            charge_id
            for charge_ids in groups.values()
            if len(charge_ids) > 1
            for charge_id in charge_ids
        )
        return {
            "customer_id": customer_id,
            "found": True,
            "charges": charges,
            "charge_count": len(charges),
            "total_amount": _dollars(sum(int(row["amount_cents"]) for row in rows)),
            "suspected_duplicate_charge_ids": duplicates,
        }

    def query_inventory(self, item_id: str) -> dict[str, Any]:
        """Look up stock and pricing for one inventory item."""

        row = self._conn.execute(
            "SELECT item_id, name, category, unit_price_cents, quantity "
            "FROM inventory WHERE item_id = ?",
            (item_id,),
        ).fetchone()
        if row is None:
            return {"item_id": item_id, "found": False}
        record = dict(row)
        unit_price_cents = int(record.pop("unit_price_cents"))
        quantity = int(record["quantity"])
        return {
            "found": True,
            **record,
            "quantity": quantity,
            "unit_price": _dollars(unit_price_cents),
            "in_stock": quantity > 0,
        }


def sqlite_tools(db: SQLiteMockDB) -> list[Tool]:
    """Build the three mock business tools bound to ``db``."""

    return [
        Tool(
            name="query_customer",
            description="Look up a customer account by customer_id (for example CUST-001).",
            parameters=dict(CUSTOMER_ID_PARAMETERS),
            handler=db.query_customer,
        ),
        Tool(
            name="query_charges",
            description=(
                "List a customer's billing charges in USD and flag suspected duplicate "
                "charges. Use this before deciding on a refund."
            ),
            parameters=dict(CUSTOMER_ID_PARAMETERS),
            handler=db.query_charges,
        ),
        Tool(
            name="query_inventory",
            description="Check stock quantity and unit price for an inventory item (ITEM-001).",
            parameters=dict(ITEM_ID_PARAMETERS),
            handler=db.query_inventory,
        ),
    ]


__all__ = [
    "CHARGE_SEED",
    "CUSTOMER_ID_PARAMETERS",
    "CUSTOMER_SEED",
    "INVENTORY_SEED",
    "ITEM_ID_PARAMETERS",
    "SCHEMA_SQL",
    "SQLiteMockDB",
    "sqlite_tools",
]
