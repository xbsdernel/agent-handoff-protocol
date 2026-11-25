"""Tests for AHP JSON Schema export (Issue #3)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from ahp.models import AckMessage, DoneMessage, HandoffMessage, NeedInfoMessage
from ahp.schema_export import (
    DRAFT_07,
    SCHEMA_NAMES,
    build_ahp_json_schemas,
    export_ahp_json_schemas,
    main,
    to_draft07,
)
from engine.mock_server import sample_from_schema


def _walk_refs(node: Any) -> list[str]:
    refs: list[str] = []
    if isinstance(node, dict):
        for key, value in node.items():
            if key == "$ref" and isinstance(value, str):
                refs.append(value)
            else:
                refs.extend(_walk_refs(value))
    elif isinstance(node, list):
        for item in node:
            refs.extend(_walk_refs(item))
    return refs


def _resolve(document: dict[str, Any], ref: str) -> Any:
    assert ref.startswith("#/")
    node: Any = document
    for raw_part in ref[2:].split("/"):
        part = raw_part.replace("~1", "/").replace("~0", "~")
        node = node[part]
    return node


def test_all_schemas_are_exported() -> None:
    schemas = build_ahp_json_schemas()

    assert set(schemas) == set(SCHEMA_NAMES)
    assert all(schema["$schema"] == DRAFT_07 for schema in schemas.values())


def test_schemas_are_draft07_and_references_resolve() -> None:
    for name, schema in build_ahp_json_schemas().items():
        serialized = json.dumps(schema)
        assert "$defs" not in serialized, name
        assert "#/$defs/" not in serialized, name
        assert "discriminator" not in schema, name
        for ref in _walk_refs(schema):
            assert _resolve(schema, ref) is not None, f"{name}: {ref}"


def test_converter_renames_defs_and_schema_uri() -> None:
    converted = to_draft07(
        {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "$defs": {"Thing": {"type": "string"}},
            "type": "object",
            "properties": {"thing": {"$ref": "#/$defs/Thing"}},
            "discriminator": {"propertyName": "type"},
        }
    )

    assert converted["$schema"] == DRAFT_07
    assert "definitions" in converted
    assert converted["properties"]["thing"]["$ref"] == "#/definitions/Thing"
    assert "discriminator" not in converted


@pytest.mark.parametrize(
    ("name", "model"),
    [
        ("handoff", HandoffMessage),
        ("ack", AckMessage),
        ("need_info", NeedInfoMessage),
        ("done", DoneMessage),
    ],
)
def test_generated_samples_validate_against_models(name: str, model: type) -> None:
    schema = build_ahp_json_schemas()[name]

    sample = sample_from_schema(schema)

    model.model_validate(sample)


def test_union_schema_sample_is_a_valid_message() -> None:
    schema = build_ahp_json_schemas()["ahp_message"]

    sample = sample_from_schema(schema)

    assert sample["type"] in {"HANDOFF", "ACK", "NEED_INFO", "DONE"}
    assert HandoffMessage.model_validate(sample)


def test_export_writes_one_file_per_schema(tmp_path: Path) -> None:
    written = export_ahp_json_schemas(tmp_path)

    assert set(written) == set(SCHEMA_NAMES)
    for name, path in written.items():
        assert path.exists()
        assert path.parent == tmp_path
        assert json.loads(path.read_text(encoding="utf-8"))["$schema"] == DRAFT_07
        assert name in path.name


def test_cli_writes_schemas(tmp_path: Path) -> None:
    assert main(["--output-dir", str(tmp_path)]) == 0
    assert (tmp_path / "ahp_message.schema.json").exists()


def test_handoff_schema_has_expected_required_fields() -> None:
    schema = build_ahp_json_schemas()["handoff"]

    assert set(schema["required"]) == {
        "conversation_id",
        "from_agent",
        "to_agent",
        "goal",
        "summary",
    }
    assert schema["properties"]["type"]["const"] == "HANDOFF"
