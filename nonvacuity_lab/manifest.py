"""Load and resolve campaign/mutation manifests without side effects."""

from __future__ import annotations

import json
from dataclasses import asdict
from enum import Enum
from pathlib import Path
from typing import Any

from .schema import CampaignConfig, MutationManifest


def load_campaign(path: Path) -> CampaignConfig:
    resolved = Path(path).resolve()
    raw = _read_campaign_with_extends(resolved, seen=set())
    _validate_campaign_schema(raw)
    return CampaignConfig.from_mapping(
        raw,
        base_dir=resolved.parent,
    )


def load_mutation(path: Path) -> MutationManifest:
    resolved = Path(path).resolve()
    return MutationManifest.from_mapping(_read_json(resolved), base_dir=resolved.parent)


def serializable_manifest(value: CampaignConfig | MutationManifest) -> dict[str, Any]:
    return _jsonable(asdict(value))


def write_resolved_manifest(
    path: Path,
    value: CampaignConfig | MutationManifest,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(serializable_manifest(value), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _read_json(path: Path) -> dict[str, Any]:
    def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"JSON 含重复键 {key!r}: {path}")
            result[key] = value
        return result

    raw = json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=reject_duplicate_keys,
    )
    if not isinstance(raw, dict):
        raise ValueError(f"manifest 必须为 JSON object: {path}")
    return raw


def _read_campaign_with_extends(path: Path, *, seen: set[Path]) -> dict[str, Any]:
    if path in seen:
        raise ValueError(f"campaign extends 循环: {path}")
    seen.add(path)
    raw = _read_json(path)
    parent_ref = raw.pop("extends", None)
    if parent_ref is None:
        return raw
    parent_path = Path(str(parent_ref))
    parent_path = (
        parent_path if parent_path.is_absolute() else path.parent / parent_path
    ).resolve()
    parent = _read_campaign_with_extends(parent_path, seen=seen)
    merged = {**parent, **raw}
    merged["mutations"] = [
        *list(parent.get("mutations", ())),
        *list(raw.get("mutations", ())),
    ]
    return merged


def _validate_campaign_schema(raw: dict[str, Any]) -> None:
    try:
        import jsonschema
    except ImportError as exc:
        raise ValueError(
            '加载 campaign 需要 jsonschema；请安装 ".[formal]"'
        ) from exc
    schema_path = (
        Path(__file__).resolve().parent.parent
        / "configs"
        / "nonvacuity"
        / "schema.json"
    )
    schema = _read_json(schema_path)
    try:
        jsonschema.Draft202012Validator(schema).validate(raw)
    except jsonschema.ValidationError as exc:
        field = "/".join(str(item) for item in exc.absolute_path) or "/"
        raise ValueError(f"campaign JSON Schema 校验失败 ({field}): {exc.message}") from exc


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value
