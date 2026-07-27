"""Validation helpers for frozen q-AMC reference and model artifacts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from .effective_config import canonical_sha256


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _verify_file_hash(path_value: object, expected: object, error: str) -> Path:
    path = Path(str(path_value))
    if not path.is_file():
        raise FileNotFoundError(error.replace("_HASH_", "_").replace("_MISMATCH", "_MISSING"))
    if _sha256_file(path) != expected:
        raise ValueError(error)
    return path


def load_and_validate_frozen_reference(path: str | Path) -> dict[str, Any]:
    """Fail closed if a frozen v3 reference or any bounded artifact changed."""

    return _load_and_validate_frozen_reference(Path(path).resolve(), seen=set())


def _load_and_validate_frozen_reference(
    path: Path,
    *,
    seen: set[Path],
) -> dict[str, Any]:
    if path in seen:
        raise ValueError("QAMC_REFERENCE_DERIVATION_CYCLE")
    seen = {*seen, path}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != "qamc_reference_experiment_config_v3":
        raise ValueError("QAMC_REFERENCE_CONFIG_NOT_FROZEN_V3")
    claimed = payload.get("fingerprint")
    unsigned = dict(payload)
    unsigned.pop("fingerprint", None)
    if claimed != canonical_sha256(unsigned):
        raise ValueError("QAMC_REFERENCE_CONFIG_FINGERPRINT_MISMATCH")

    source = _verify_file_hash(
        payload.get("source_config_path"),
        payload.get("source_config_sha256"),
        "QAMC_REFERENCE_SOURCE_CONFIG_HASH_MISMATCH",
    )
    reward = payload.get("reward_artifact")
    if not isinstance(reward, dict):
        raise ValueError("QAMC_REFERENCE_REWARD_ARTIFACT_INVALID")
    reward_path = _verify_file_hash(
        reward.get("path"),
        reward.get("sha256"),
        "QAMC_REFERENCE_REWARD_ARTIFACT_HASH_MISMATCH",
    )
    effective = payload.get("effective_reference_config")
    if not isinstance(effective, dict):
        raise ValueError("QAMC_REFERENCE_EFFECTIVE_CONFIG_MISSING")
    if canonical_sha256(effective) != payload.get(
        "effective_reference_config_fingerprint"
    ):
        raise ValueError("QAMC_REFERENCE_EFFECTIVE_CONFIG_FINGERPRINT_MISMATCH")

    bound = payload.get("bound_artifacts")
    if not isinstance(bound, dict):
        raise ValueError("QAMC_REFERENCE_BOUND_ARTIFACTS_INVALID")
    expected_required = {
        "config.json": _sha256_file(source),
        "reward_config": _sha256_file(reward_path),
    }
    for name, expected in expected_required.items():
        if bound.get(name) != expected:
            raise ValueError(f"QAMC_REFERENCE_BOUND_ARTIFACT_HASH_MISMATCH:{name}")
    for optional_name in (
        "feature_names.json",
        "action_definitions.json",
        "model_best_metadata.json",
    ):
        if optional_name not in bound:
            continue
        optional_path = source.parent / optional_name
        if not optional_path.is_file():
            raise FileNotFoundError(
                f"QAMC_REFERENCE_BOUND_ARTIFACT_MISSING:{optional_name}"
            )
        if _sha256_file(optional_path) != bound[optional_name]:
            raise ValueError(
                f"QAMC_REFERENCE_BOUND_ARTIFACT_HASH_MISMATCH:{optional_name}"
            )

    source_payload = json.loads(source.read_text(encoding="utf-8"))
    derivation = source_payload.get("reference_variant_derivation")
    if derivation is not None:
        if not isinstance(derivation, dict) or derivation.get(
            "schema_version"
        ) != "qamc_observation_reference_derivation_v1":
            raise ValueError("QAMC_REFERENCE_DERIVATION_SCHEMA_INVALID")
        base_path = Path(
            str(derivation.get("base_frozen_reference_path"))
        ).resolve()
        if base_path == path:
            raise ValueError("QAMC_REFERENCE_DERIVATION_CYCLE")
        base = _load_and_validate_frozen_reference(base_path, seen=seen)
        if base["fingerprint"] != derivation.get(
            "base_frozen_reference_fingerprint"
        ):
            raise ValueError("QAMC_REFERENCE_DERIVATION_BASE_MISMATCH")
        base_effective = base["effective_reference_config"]
        derived_effective = payload["effective_reference_config"]
        changed = {
            key
            for key in set(base_effective) | set(derived_effective)
            if derived_effective.get(key) != base_effective.get(key)
        }
        if changed != {"observation_mode", "observation_dim"}:
            raise ValueError("QAMC_REFERENCE_DERIVATION_ILLEGAL_OVERRIDE")
    return payload


def assert_reference_matches_values(
    frozen: dict[str, Any],
    values: dict[str, Any],
) -> None:
    """Require CLI-effective values to equal the frozen canonical values."""

    effective = frozen["effective_reference_config"]
    mismatches = [
        key
        for key, actual in values.items()
        if key in effective and effective[key] is not None and effective[key] != actual
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
    expected_taskset_fingerprint: str,
    expected_profile_fingerprint: str,
    expected_action_dim: int,
    expected_observation_dim: int,
    expected_observation_mode: str,
    expected_observation_schema_fingerprint: str,
    expected_action_space_fingerprint: str,
    expected_semantic_version: str,
    expected_demand_mapping_version: str,
) -> dict[str, Any]:
    """Bind a DQN checkpoint to every effective q-AMC run dimension."""

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
        "taskset_fingerprint": expected_taskset_fingerprint,
        "profile_fingerprint": expected_profile_fingerprint,
        "action_dim": expected_action_dim,
        "observation_dim": expected_observation_dim,
        "observation_mode": expected_observation_mode,
        "observation_schema_fingerprint": (
            expected_observation_schema_fingerprint
        ),
        "action_space_fingerprint": expected_action_space_fingerprint,
        "semantic_version": expected_semantic_version,
        "demand_mapping_version": expected_demand_mapping_version,
    }
    mismatches = [key for key, value in expected.items() if qamc.get(key) != value]
    if mismatches:
        raise ValueError(
            "QAMC_MODEL_ARTIFACT_BINDING_MISMATCH:" + ",".join(sorted(mismatches))
        )
    return config


__all__ = [
    "assert_reference_matches_values",
    "load_and_validate_frozen_reference",
    "validate_qamc_model_artifact",
]
