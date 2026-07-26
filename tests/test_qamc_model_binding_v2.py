from __future__ import annotations

import json
from pathlib import Path

import pytest

from amc_py.qamc.reference_config import validate_qamc_model_artifact


EXPECTED = {
    "reference_config_fingerprint": "reference",
    "profile_manifest_fingerprint": "manifest",
    "profile_spec_fingerprint": "spec",
    "taskset_fingerprint": "taskset",
    "profile_fingerprint": "profile",
    "action_dim": 24,
    "observation_dim": 128,
    "action_space_fingerprint": "actions",
    "semantic_version": "qamc_budget_overlay_v5",
    "demand_mapping_version": "wcet_capped_component_split_v1",
}


def _validate(tmp_path: Path, metadata: dict[str, object]) -> None:
    model = tmp_path / "model.pt"
    model.write_bytes(b"checkpoint")
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "runtime_config": {"semantics": "Q_AMC"},
                "qamc": metadata,
            }
        ),
        encoding="utf-8",
    )
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"fingerprint": "manifest"}), encoding="utf-8")
    validate_qamc_model_artifact(
        model,
        frozen_reference={"fingerprint": "reference"},
        profile_manifest_path=manifest,
        profile_spec_fingerprint="spec",
        expected_taskset_fingerprint="taskset",
        expected_profile_fingerprint="profile",
        expected_action_dim=24,
        expected_observation_dim=128,
        expected_action_space_fingerprint="actions",
        expected_semantic_version="qamc_budget_overlay_v5",
        expected_demand_mapping_version="wcet_capped_component_split_v1",
    )


def test_complete_qamc_model_binding_passes(tmp_path: Path) -> None:
    _validate(tmp_path, dict(EXPECTED))


@pytest.mark.parametrize("field", sorted(EXPECTED))
def test_each_qamc_model_binding_mismatch_is_rejected(
    tmp_path: Path,
    field: str,
) -> None:
    metadata = dict(EXPECTED)
    metadata[field] = "wrong"
    with pytest.raises(ValueError, match="ARTIFACT_BINDING_MISMATCH"):
        _validate(tmp_path, metadata)
