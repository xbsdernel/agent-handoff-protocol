"""Tests for the seeded SQLite mock database and its tool wrappers (Issue #13).

Every test uses an in-memory database or a pytest temporary directory, so the
suite stays offline and isolated.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from harness.tools import SQLiteMockDB, ToolRegistry, sqlite_tools
from harness.tools.base import ToolArgumentError, UnknownToolError
from harness.tools.sqlite_mock import CHARGE_SEED, CUSTOMER_SEED, INVENTORY_SEED


@pytest.fixture
def db() -> SQLiteMockDB:
    database = SQLiteMockDB()
    try:
        yield database
    finally:
        database.close()


# ---------------------------------------------------------------------------
# Seeding and queries
# ---------------------------------------------------------------------------


def test_seed_loads_every_fixture_table(db: SQLiteMockDB) -> None:
    counts = db.execute(
        "SELECT (SELECT COUNT(*) FROM customers) AS customers, "
        "(SELECT COUNT(*) FROM charges) AS charges, "
        "(SELECT COUNT(*) FROM inventory) AS inventory"
    )

    assert counts == [
        {
            "customers": len(CUSTOMER_SEED),
            "charges": len(CHARGE_SEED),
            "inventory": len(INVENTORY_SEED),
        }
    ]


def test_query_customer_returns_a_complete_account(db: SQLiteMockDB) -> None:
    customer = db.query_customer("CUST-001")

    assert customer == {
        "customer_id": "CUST-001",
        "found": True,
        "name": "Ava Thompson",
        "email": "ava.thompson@example.com",
        "plan": "pro",
        "joined_at": "2024-03-12",
    }


def test_query_customer_reports_unknown_ids_without_raising(db: SQLiteMockDB) -> None:
    assert db.query_customer("CUST-999") == {"customer_id": "CUST-999", "found": False}


def test_query_charges_returns_dollar_amounts_and_totals(db: SQLiteMockDB) -> None:
    result = db.query_charges("CUST-001")

    assert result["found"] is True
    assert result["charge_count"] == 3
    assert result["total_amount"] == 147.0
    assert [charge["amount"] for charge in result["charges"]] == [49.0, 49.0, 49.0]
    assert [charge["charge_id"] for charge in result["charges"]] == [
        "CHG-1001",
        "CHG-1002",
        "CHG-1003",
    ]
    assert all(isinstance(charge["amount"], float) for charge in result["charges"])


def test_query_charges_flags_duplicate_payments(db: SQLiteMockDB) -> None:
    ava = db.query_charges("CUST-001")
    chloe = db.query_charges("CUST-003")

    assert ava["suspected_duplicate_charge_ids"] == ["CHG-1001", "CHG-1002"]
    assert chloe["suspected_duplicate_charge_ids"] == []
    assert [charge["status"] for charge in chloe["charges"]] == ["refunded", "paid"]


def test_query_charges_for_unknown_customer_is_empty(db: SQLiteMockDB) -> None:
    result = db.query_charges("CUST-999")

    assert result == {
        "customer_id": "CUST-999",
        "found": False,
        "charges": [],
        "charge_count": 0,
        "total_amount": 0.0,
        "suspected_duplicate_charge_ids": [],
    }


def test_query_inventory_exposes_stock_and_price(db: SQLiteMockDB) -> None:
    in_stock = db.query_inventory("ITEM-001")
    out_of_stock = db.query_inventory("ITEM-002")

    assert in_stock["found"] is True
    assert in_stock["name"] == "AHP Pro Sensor"
    assert in_stock["unit_price"] == 129.0
    assert in_stock["quantity"] == 42
    assert in_stock["in_stock"] is True

    assert out_of_stock["quantity"] == 0
    assert out_of_stock["in_stock"] is False


def test_query_inventory_reports_unknown_items(db: SQLiteMockDB) -> None:
    assert db.query_inventory("ITEM-999") == {"item_id": "ITEM-999", "found": False}


# ---------------------------------------------------------------------------
# Reset isolation and persistence
# ---------------------------------------------------------------------------


def test_reset_restores_mutated_rows(db: SQLiteMockDB) -> None:
    db.execute(
        "INSERT INTO customers (customer_id, name, email, plan, joined_at) VALUES (?, ?, ?, ?, ?)",
        ("CUST-777", "Test User", "test@example.com", "starter", "2026-01-01"),
    )
    db.execute("DELETE FROM charges")
    db.execute("UPDATE inventory SET quantity = 0")

    assert db.query_customer("CUST-777")["found"] is True
    assert db.query_charges("CUST-001")["found"] is True  # customer remains
    assert db.query_charges("CUST-001")["charge_count"] == 0
    assert db.query_inventory("ITEM-001")["in_stock"] is False

    db.reset()

    assert db.query_customer("CUST-777")["found"] is False
    assert db.query_charges("CUST-001")["charge_count"] == len(
        [row for row in CHARGE_SEED if row[1] == "CUST-001"]
    )
    assert db.query_inventory("ITEM-001")["in_stock"] is True


def test_reset_on_unseeded_database_stays_empty() -> None:
    database = SQLiteMockDB(auto_seed=False)
    try:
        assert database.query_customer("CUST-001")["found"] is False
        database.seed()
        assert database.query_customer("CUST-001")["found"] is True

        database.reset()

        assert database.query_customer("CUST-001")["found"] is False
        assert database.query_inventory("ITEM-001")["found"] is False
    finally:
        database.close()


def test_file_database_persists_between_connections(tmp_path: Path) -> None:
    path = tmp_path / "mock.db"
    first = SQLiteMockDB(path, auto_seed=False)
    try:
        first.seed()
        assert first.query_inventory("ITEM-001")["found"] is True
        first.execute("DELETE FROM inventory")
    finally:
        first.close()

    second = SQLiteMockDB(path, auto_seed=False)
    try:
        assert second.query_inventory("ITEM-001")["found"] is False

        second.reset()
        assert second.query_inventory("ITEM-001")["found"] is False

        second.seed()
        assert second.query_inventory("ITEM-001")["found"] is True
    finally:
        second.close()


# ---------------------------------------------------------------------------
# Tool definitions and registry integration
# ---------------------------------------------------------------------------


def test_sqlite_tools_expose_openai_style_definitions(db: SQLiteMockDB) -> None:
    tools = {tool.name: tool for tool in sqlite_tools(db)}

    assert set(tools) == {"query_customer", "query_charges", "query_inventory"}
    for tool in tools.values():
        definition = tool.definition()
        assert definition["type"] == "function"
        assert definition["function"]["name"] == tool.name
        assert definition["function"]["parameters"]["type"] == "object"

    assert tools["query_customer"].parameters["required"] == ["customer_id"]
    assert tools["query_inventory"].parameters["required"] == ["item_id"]


async def test_registry_executes_sqlite_tools(db: SQLiteMockDB) -> None:
    registry = ToolRegistry(sqlite_tools(db))

    customer = await registry.execute("query_customer", {"customer_id": "CUST-002"})
    charges = await registry.execute("query_charges", {"customer_id": "CUST-005"})
    inventory = await registry.execute("query_inventory", {"item_id": "ITEM-003"})

    assert customer.ok is True
    assert customer.result["name"] == "Ben Carter"
    assert charges.result["suspected_duplicate_charge_ids"] == ["CHG-5001", "CHG-5002"]
    assert inventory.result["unit_price"] == 899.0


def test_registry_rejects_unknown_tools_and_bad_arguments(db: SQLiteMockDB) -> None:
    registry = ToolRegistry(sqlite_tools(db))

    with pytest.raises(UnknownToolError, match="unknown tool"):
        registry.resolve("refund_customer", {"customer_id": "CUST-001"})

    with pytest.raises(ToolArgumentError, match="required property"):
        registry.resolve("query_customer", {})

    with pytest.raises(ToolArgumentError, match="Additional properties"):
        registry.resolve("query_customer", {"customer_id": "CUST-001", "extra": True})

    with pytest.raises(ToolArgumentError, match="must be an object"):
        registry.resolve("query_customer", ["CUST-001"])  # type: ignore[arg-type]
