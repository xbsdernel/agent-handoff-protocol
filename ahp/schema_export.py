"""Export AHP message models as JSON Schema for constrained decoding.

Engines such as vLLM and SGLang accept a JSON Schema to force the model's
output to be valid AHP. Pydantic emits Draft 2020-12 with ``$defs``; this
module converts the documents to Draft-07 (``definitions``) and rewrites local
references so the schemas work with the widest range of engines.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Final

from ahp.models import AckMessage, AHPMessageAdapter, DoneMessage, HandoffMessage, NeedInfoMessage

DRAFT_07: Final[str] = "http://json-schema.org/draft-07/schema#"

#: Schemas exported by :func:`build_ahp_json_schemas`, in file-name order.
SCHEMA_NAMES: Final[tuple[str, ...]] = (
    "handoff",
    "ack",
    "need_info",
    "done",
    "ahp_message",
)


def _convert(node: Any) -> Any:
    if isinstance(node, dict):
        converted: dict[str, Any] = {}
        for key, value in node.items():
            if key == "$defs":
                converted["definitions"] = _convert(value)
            elif key in {"$schema", "discriminator"}:
                # Draft-07 has no $schema here and no discriminator keyword;
                # the oneOf branches already describe the union.
                continue
            else:
                converted[key] = _convert(value)
        return converted

    if isinstance(node, list):
        return [_convert(item) for item in node]

    if isinstance(node, str) and node.startswith("#/$defs/"):
        return "#/definitions/" + node.removeprefix("#/$defs/")

    return node


def to_draft07(schema: Mapping[str, Any]) -> dict[str, Any]:
    """Return a Draft-07 copy of a Pydantic-generated JSON Schema."""

    converted = _convert(dict(schema))
    converted["$schema"] = DRAFT_07
    return converted


def build_ahp_json_schemas() -> dict[str, dict[str, Any]]:
    """Build Draft-07 schemas for each message type and for the union."""

    return {
        "handoff": to_draft07(HandoffMessage.model_json_schema()),
        "ack": to_draft07(AckMessage.model_json_schema()),
        "need_info": to_draft07(NeedInfoMessage.model_json_schema()),
        "done": to_draft07(DoneMessage.model_json_schema()),
        "ahp_message": to_draft07(AHPMessageAdapter.json_schema()),
    }


def export_ahp_json_schemas(output_dir: str | Path = "schemas") -> dict[str, Path]:
    """Write one ``<name>.schema.json`` file per schema and return the paths."""

    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)

    written: dict[str, Path] = {}
    for name, schema in build_ahp_json_schemas().items():
        path = directory / f"{name}.schema.json"
        path.write_text(
            json.dumps(schema, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        written[name] = path
    return written


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Export AHP JSON Schemas (Draft-07)")
    parser.add_argument(
        "--output-dir",
        default="schemas",
        help="Directory that receives the *.schema.json files (default: schemas).",
    )
    args = parser.parse_args(argv)
    for name, path in export_ahp_json_schemas(args.output_dir).items():
        print(f"{name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
