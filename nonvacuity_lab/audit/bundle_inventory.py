from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator


def iter_json_pointers(value: Any, pointer: str = "") -> Iterator[tuple[str, Any]]:
    if isinstance(value, dict):
        for key, item in value.items():
            escaped = str(key).replace("~", "~0").replace("/", "~1")
            yield from iter_json_pointers(item, f"{pointer}/{escaped}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            yield from iter_json_pointers(item, f"{pointer}/{index}")
    else:
        yield pointer or "/", value


def _named_pointers(value: Any, pointer: str = "") -> Iterator[tuple[str, Any]]:
    if isinstance(value, dict):
        for key, item in value.items():
            escaped = str(key).replace("~", "~0").replace("/", "~1")
            child = f"{pointer}/{escaped}"
            yield child, item
            yield from _named_pointers(item, child)
    elif isinstance(value, list):
        for index, item in enumerate(value):
            child = f"{pointer}/{index}"
            yield child, item
            yield from _named_pointers(item, child)


def build_bundle_inventory(root: Path) -> list[dict[str, Any]]:
    rows = []
    for path in sorted(Path(root).rglob("*.json")):
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        relative = path.relative_to(root).as_posix()
        for pointer, value in _named_pointers(raw):
            if "witness" in pointer.rsplit("/", 1)[-1].lower():
                rows.append({"kind": "WITNESS_POINTER", "artifact": relative, "json_pointer": pointer, "value_type": type(value).__name__})
        if "obligation" in relative.lower() or (isinstance(raw, dict) and raw.get("obligation_id")):
            rows.append({"kind": "OBLIGATION_ARTIFACT", "artifact": relative, "obligation_id": raw.get("obligation_id") if isinstance(raw, dict) else None})
    return rows


def resolve_f5_witness_pointer(rows: list[dict[str, Any]]) -> dict[str, Any]:
    candidates = [row for row in rows if row["kind"] == "WITNESS_POINTER"]
    if not candidates:
        raise ValueError("F5_WITNESS_POINTER_UNRESOLVED")
    return min(candidates, key=lambda row: (0 if "proof_summary" in row["artifact"] else 1, len(row["json_pointer"]), row["artifact"]))


def resolve_f6_obligation_artifact(rows: list[dict[str, Any]]) -> dict[str, Any]:
    candidates = [row for row in rows if row["kind"] == "OBLIGATION_ARTIFACT"]
    if not candidates:
        raise ValueError("F6_OBLIGATION_ARTIFACT_UNRESOLVED")
    return sorted(candidates, key=lambda row: (row.get("obligation_id") is None, row["artifact"]))[0]
