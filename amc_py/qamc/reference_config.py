"""Validation helpers for frozen q-AMC reference experiment artifacts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _tree_fingerprint(root: Path) -> str:
    entries = [
        (str(item.relative_to(root)), _sha256_file(item))
        for item in sorted(candidate for candidate in root.rglob("*") if candidate.is_file())
        if ".git" not in item.parts
    ]
    return hashlib.sha256(_canonical(entries)).hexdigest()


def load_and_validate_frozen_reference(path: str | Path) -> dict[str, Any]:
    """Fail closed if a frozen reference or either bound source artifact changed."""

    frozen_path = Path(path)
    payload = json.loads(frozen_path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != "qamc_reference_experiment_config_v2":
        raise ValueError("QAMC_REFERENCE_CONFIG_NOT_FROZEN_V2")
    claimed = payload.get("fingerprint")
    unsigned = dict(payload)
    unsigned.pop("fingerprint", None)
    if claimed != hashlib.sha256(_canonical(unsigned)).hexdigest():
        raise ValueError("QAMC_REFERENCE_CONFIG_FINGERPRINT_MISMATCH")

    source = Path(str(payload.get("source_config_path", "")))
    if not source.is_file():
        raise FileNotFoundError("QAMC_REFERENCE_SOURCE_CONFIG_MISSING")
    if _sha256_file(source) != payload.get("source_config_sha256"):
        raise ValueError("QAMC_REFERENCE_SOURCE_CONFIG_HASH_MISMATCH")
    reward = payload.get("reward_artifact")
    if not isinstance(reward, dict):
        raise ValueError("QAMC_REFERENCE_REWARD_ARTIFACT_INVALID")
    reward_path = Path(str(reward.get("path", "")))
    if not reward_path.is_file():
        raise FileNotFoundError("QAMC_REFERENCE_REWARD_ARTIFACT_MISSING")
    if _sha256_file(reward_path) != reward.get("sha256"):
        raise ValueError("QAMC_REFERENCE_REWARD_ARTIFACT_HASH_MISMATCH")
    if _tree_fingerprint(source.parent) != payload.get("source_tree_fingerprint"):
        raise ValueError("QAMC_REFERENCE_SOURCE_TREE_FINGERPRINT_MISMATCH")
    if not isinstance(payload.get("normalized"), dict):
        raise ValueError("QAMC_REFERENCE_NORMALIZED_CONFIG_MISSING")
    return payload


def assert_reference_matches_values(
    frozen: dict[str, Any],
    values: dict[str, Any],
) -> None:
    """Require CLI-effective values to equal the frozen reference values."""

    normalized = frozen["normalized"]
    mismatches = [
        key
        for key, actual in values.items()
        if key in normalized and normalized[key] is not None and normalized[key] != actual
    ]
    if mismatches:
        raise ValueError(
            "QAMC_REFERENCE_EFFECTIVE_CONFIG_MISMATCH:" + ",".join(sorted(mismatches))
        )


def validate_qamc_model_artifact(
    model_path: str | Path,
    *,
    frozen_reference: dict[str, Any],
    profile_manifest_path: str | Path,
    profile_spec_fingerprint: str,
) -> dict[str, Any]:
    """Bind a DQN checkpoint to the q-AMC run configuration that produced it."""

    model = Path(model_path)
    config_path = model.parent / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError("QAMC_MODEL_RUN_CONFIG_MISSING")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("runtime_config", {}).get("semantics") != "Q_AMC":
        raise ValueError("QAMC_MODEL_RUNTIME_SEMANTICS_MISMATCH")
    qamc = config.get("qamc")
    if not isinstance(qamc, dict):
        raise ValueError("QAMC_MODEL_METADATA_MISSING")
    manifest = json.loads(Path(profile_manifest_path).read_text(encoding="utf-8"))
    expected = {
        "reference_config_fingerprint": frozen_reference["fingerprint"],
        "profile_manifest_fingerprint": manifest.get("fingerprint"),
        "profile_spec_fingerprint": profile_spec_fingerprint,
    }
    mismatches = [key for key, value in expected.items() if qamc.get(key) != value]
    if mismatches:
        raise ValueError(
            "QAMC_MODEL_ARTIFACT_FINGERPRINT_MISMATCH:"
            + ",".join(sorted(mismatches))
        )
    return config


__all__ = [
    "assert_reference_matches_values",
    "load_and_validate_frozen_reference",
    "validate_qamc_model_artifact",
]
