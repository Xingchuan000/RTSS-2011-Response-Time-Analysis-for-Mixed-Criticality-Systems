from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path


class ConfigError(ValueError):
    pass


def canonical_config_bytes(config: dict) -> bytes:
    value = deepcopy(config)
    value.pop("config_sha256", None)
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def config_hash(config: dict) -> str:
    return hashlib.sha256(canonical_config_bytes(config)).hexdigest()


def write_resolved_campaign(path: Path, config: dict) -> None:
    value = deepcopy(config)
    value["config_sha256"] = config_hash(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def verify_config_hash(config: dict) -> None:
    expected = config.get("config_sha256")
    actual = config_hash(config)
    if expected != actual:
        raise ConfigError(f"config hash mismatch: {expected} != {actual}")


def validate_config_kind(config: dict) -> None:
    kind = str(config.get("config_kind", ""))
    if kind == "TEMPLATE":
        if config.get("enabled"):
            raise ConfigError("template campaign cannot be enabled")
        return
    if kind != "RESOLVED":
        raise ConfigError("config_kind must be TEMPLATE or RESOLVED")
    if "resolver_receipt" not in config:
        raise ConfigError("resolved campaign requires resolver_receipt")
    for mutation in config.get("mutations", []):
        if mutation.get("enabled") and "resolved_target" not in mutation and mutation.get("mutation_class") not in {"BASELINE", "ENVELOPE_GRADIENT"}:
            raise ConfigError(f"{mutation.get('mutation_id')} missing resolved_target")
