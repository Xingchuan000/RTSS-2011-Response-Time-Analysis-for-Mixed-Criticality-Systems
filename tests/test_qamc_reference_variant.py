from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from amc_py.qamc.effective_config import QAmcReferenceEffectiveConfig
from amc_py.qamc.reference_config import load_and_validate_frozen_reference
from scripts.derive_qamc_observation_reference import derive_reference_variant
from scripts.freeze_qamc_reference_config import freeze_reference_config


def _base_frozen(tmp_path: Path, task_count: int = 12) -> Path:
    reward = tmp_path / "reward.json"
    reward.write_text("{}\n", encoding="utf-8")
    run = tmp_path / "base_run"
    run.mkdir()
    effective = QAmcReferenceEffectiveConfig(
        schema_version="qamc_reference_effective_config_v1",
        action_space="single",
        q_network_type="mlp",
        action_feature_mode="static_v1",
        include_explicit_noop=False,
        budget_increase_ratio=0.02,
        budget_decrease_ratio=0.02,
        budget_rounding_mode="ceil_floor",
        min_budget_delta=1,
        budget_floor_ratio=0.9,
        check_safety=True,
        step_guard_semantics="checked",
        observation_mode="v11_full_10d",
        reward_mode="mendes",
        reward_config_path=str(reward.resolve()),
        reward_config_sha256=hashlib.sha256(reward.read_bytes()).hexdigest(),
        agent_period=25000,
        save_best_by="qos_recovery_stable",
        selector_contract_version="selector_contract_v1",
        enable_deploy_cap_mask=True,
        deploy_cap_mask_ratio=4.0,
        deploy_cap_mask_criticality="lo",
        forbid_decreasing_hi_budgets=True,
        action_dim=24,
        observation_dim=10 * task_count + 8,
    )
    (run / "config.json").write_text(
        json.dumps(
            {
                "effective_reference_config": effective.to_jsonable(),
                "tasks": [{"name": f"T{index}"} for index in range(task_count)],
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "base.frozen.json"
    freeze_reference_config(run, output)
    return output


def test_o2_reference_variant_has_12_task_schema_of_152(
    tmp_path: Path,
) -> None:
    base = _base_frozen(tmp_path)
    variant = tmp_path / "variant"
    derive_reference_variant(
        base_frozen_path=base,
        output_dir=variant,
        observation_mode="v14_qamc_full_12d",
    )
    feature_schema = json.loads(
        (variant / "feature_names.json").read_text(encoding="utf-8")
    )
    assert feature_schema["observation_dim"] == 152
    assert len(feature_schema["feature_names"]) == 152

    frozen_variant = tmp_path / "variant.frozen.json"
    freeze_reference_config(variant, frozen_variant)
    validated = load_and_validate_frozen_reference(frozen_variant)
    assert validated["effective_reference_config"]["observation_dim"] == 152


@pytest.mark.parametrize("field,value", [("reward_mode", "wrong"), ("budget_floor_ratio", 0.8)])
def test_reference_variant_rejects_non_observation_override(
    tmp_path: Path,
    field: str,
    value: object,
) -> None:
    base = _base_frozen(tmp_path)
    variant = tmp_path / "variant"
    derive_reference_variant(
        base_frozen_path=base,
        output_dir=variant,
        observation_mode="v14_qamc_full_12d",
    )
    config_path = variant / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["effective_reference_config"][field] = value
    config_path.write_text(json.dumps(config), encoding="utf-8")
    frozen_variant = tmp_path / "variant.frozen.json"
    freeze_reference_config(variant, frozen_variant)
    with pytest.raises(
        ValueError,
        match="QAMC_REFERENCE_DERIVATION_ILLEGAL_OVERRIDE",
    ):
        load_and_validate_frozen_reference(frozen_variant)


def test_reference_variant_rejects_changed_base_fingerprint(
    tmp_path: Path,
) -> None:
    base = _base_frozen(tmp_path)
    variant = tmp_path / "variant"
    derive_reference_variant(
        base_frozen_path=base,
        output_dir=variant,
        observation_mode="v14_qamc_full_12d",
    )
    frozen_variant = tmp_path / "variant.frozen.json"
    freeze_reference_config(variant, frozen_variant)
    base_payload = json.loads(base.read_text(encoding="utf-8"))
    base_payload["legacy_upgrade"]["tampered"] = True
    base.write_text(json.dumps(base_payload), encoding="utf-8")
    with pytest.raises(
        ValueError,
        match="QAMC_REFERENCE_CONFIG_FINGERPRINT_MISMATCH",
    ):
        load_and_validate_frozen_reference(frozen_variant)


def test_learning_reference_variant_binds_reward_and_selector(tmp_path: Path) -> None:
    base = _base_frozen(tmp_path)
    project = tmp_path / "project"
    reward_dir = project / "configs" / "reward_modes"
    reward_dir.mkdir(parents=True)
    reward = reward_dir / "interval_lo_equiv_jne_v2_job_weighted.json"
    reward.write_text(
        json.dumps(
            {
                "event_weights": {
                    "job_start": 0.0,
                    "lo_overrun": 0.0,
                    "hi_overrun": 0.0,
                },
                "step_reward_formula": "-delta_lo_equiv_jne",
            }
        ),
        encoding="utf-8",
    )
    variant = tmp_path / "learning_variant"
    derive_reference_variant(
        base_frozen_path=base,
        output_dir=variant,
        observation_mode="v14_qamc_full_12d",
        reward_mode="interval_lo_equiv_jne_v2_job_weighted",
        save_best_by="lo_quality_qos_best",
        project_root=project,
    )
    frozen_variant = tmp_path / "learning_variant.frozen.json"
    freeze_reference_config(variant, frozen_variant)
    validated = load_and_validate_frozen_reference(frozen_variant)
    effective = validated["effective_reference_config"]
    assert effective["observation_mode"] == "v14_qamc_full_12d"
    assert effective["observation_dim"] == 152
    assert effective["reward_mode"] == "interval_lo_equiv_jne_v2_job_weighted"
    assert effective["reward_config_path"] == str(reward.resolve())
    assert effective["reward_config_sha256"] == hashlib.sha256(
        reward.read_bytes()
    ).hexdigest()
    assert effective["save_best_by"] == "lo_quality_qos_best"


def test_learning_reference_rejects_undeclared_override(tmp_path: Path) -> None:
    base = _base_frozen(tmp_path)
    project = tmp_path / "project"
    reward_dir = project / "configs" / "reward_modes"
    reward_dir.mkdir(parents=True)
    reward = reward_dir / "new_reward.json"
    reward.write_text(
        json.dumps(
            {
                "event_weights": {
                    "job_start": 0.0,
                    "lo_overrun": 0.0,
                    "hi_overrun": 0.0,
                },
                "step_reward_formula": "-delta_lo_equiv_jne",
            }
        ),
        encoding="utf-8",
    )
    variant = tmp_path / "learning_variant"
    derive_reference_variant(
        base_frozen_path=base,
        output_dir=variant,
        observation_mode="v14_qamc_full_12d",
        reward_mode="new_reward",
        save_best_by="lo_quality_qos_best",
        project_root=project,
    )
    config_path = variant / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["effective_reference_config"]["budget_floor_ratio"] = 0.8
    config_path.write_text(json.dumps(config), encoding="utf-8")
    frozen_variant = tmp_path / "learning_variant.frozen.json"
    freeze_reference_config(variant, frozen_variant)
    with pytest.raises(
        ValueError,
        match="QAMC_REFERENCE_DERIVATION_ILLEGAL_OVERRIDE",
    ):
        load_and_validate_frozen_reference(frozen_variant)
