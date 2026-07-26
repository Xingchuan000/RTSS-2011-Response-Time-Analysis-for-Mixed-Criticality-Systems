from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from amc_py.models import Criticality, Task
from amc_py.qamc.profile_spec import QAmcProfileSpec, write_profile_spec
from amc_py.qamc.profiles import (
    build_qamc_profile_bundle,
    load_profile_bundle_from_manifest,
    write_profile_bundle,
)
from amc_py.qamc.reference_config import (
    assert_reference_matches_values,
    load_and_validate_frozen_reference,
)
from scripts.freeze_qamc_reference_config import freeze_reference_config


def _manifest_fingerprint(payload: dict) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(raw.encode()).hexdigest()


def test_profile_manifest_binds_spec_entry_and_taskset(tmp_path: Path) -> None:
    task = Task("L", 10, 10, 6, 6, Criticality.LO)
    spec = QAmcProfileSpec()
    spec_path = tmp_path / "spec.json"
    write_profile_spec(spec, spec_path)
    bundle = build_qamc_profile_bundle(
        [task], taskset_fingerprint="taskset-a", spec=spec
    )
    profile_path = tmp_path / "profile.json"
    write_profile_bundle(bundle, profile_path)
    manifest = {
        "schema_version": "qamc_profile_manifest_v1",
        "spec_path": str(spec_path),
        "spec_fingerprint": spec.fingerprint,
        "profiles": {
            "taskset-a": {
                "path": str(profile_path),
                "fingerprint": bundle.fingerprint,
            }
        },
    }
    manifest["fingerprint"] = _manifest_fingerprint(manifest)
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    loaded = load_profile_bundle_from_manifest(
        manifest_path, taskset_fingerprint="taskset-a", spec_path=spec_path
    )
    assert loaded.fingerprint == bundle.fingerprint

    manifest["profiles"]["taskset-a"]["fingerprint"] = "tampered"
    unsigned = dict(manifest)
    unsigned.pop("fingerprint")
    manifest["fingerprint"] = _manifest_fingerprint(unsigned)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="ENTRY_FINGERPRINT"):
        load_profile_bundle_from_manifest(
            manifest_path, taskset_fingerprint="taskset-a", spec_path=spec_path
        )


def test_frozen_reference_detects_artifact_and_effective_config_drift(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    reward = run_dir / "reward.json"
    reward.write_text('{"reward": 1}', encoding="utf-8")
    config = {
        "action_space": "single",
        "q_network_type": "mlp",
        "action_feature_mode": "static_v1",
        "include_explicit_noop": False,
        "budget_increase_ratio": 0.1,
        "budget_decrease_ratio": 0.05,
        "budget_rounding_mode": "ceil_floor",
        "min_budget_delta": 1,
        "budget_floor_ratio": 0.5,
        "check_safety": True,
        "observation_mode": "v11_full_10d",
        "reward_mode": "mendes",
        "agent_period": 1000,
        "reward_config_path": "reward.json",
        "selector": {"metric": "lc_qos"},
    }
    (run_dir / "config.json").write_text(json.dumps(config), encoding="utf-8")
    frozen_path = tmp_path / "frozen.json"
    freeze_reference_config(run_dir, frozen_path)

    frozen = load_and_validate_frozen_reference(frozen_path)
    assert_reference_matches_values(frozen, {"action_space": "single"})
    with pytest.raises(ValueError, match="EFFECTIVE_CONFIG_MISMATCH"):
        assert_reference_matches_values(frozen, {"action_space": "pair"})

    reward.write_text('{"reward": 2}', encoding="utf-8")
    with pytest.raises(ValueError, match="REWARD_ARTIFACT_HASH_MISMATCH"):
        load_and_validate_frozen_reference(frozen_path)


def test_qamc_is_not_registered_as_a_formally_proved_route() -> None:
    root = Path(__file__).resolve().parents[1] / "formal_toolchain"
    occurrences = [
        path
        for path in root.rglob("*.py")
        if "Q_AMC" in path.read_text(encoding="utf-8")
    ]
    assert occurrences == []
