from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from amc_py.qamc.effective_config import QAmcReferenceEffectiveConfig
from amc_py.qamc.reference_config import load_and_validate_frozen_reference
from scripts.freeze_qamc_reference_config import freeze_reference_config


def _write_current_run(root: Path, *, selector: str = "qos_recovery_stable") -> Path:
    run = root / "run"
    run.mkdir()
    reward = root / "reward.json"
    reward.write_text('{"reward": 1}\n', encoding="utf-8")
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
        save_best_by=selector,
        selector_contract_version="selector_contract_v1",
        enable_deploy_cap_mask=True,
        deploy_cap_mask_ratio=4.0,
        deploy_cap_mask_criticality="lo",
        forbid_decreasing_hi_budgets=True,
        action_dim=24,
        observation_dim=128,
    )
    (run / "config.json").write_text(
        json.dumps({"effective_reference_config": effective.to_jsonable()}),
        encoding="utf-8",
    )
    return run


def _legacy_config() -> dict[str, object]:
    return {
        "action_space": "single",
        "q_network_type": "mlp",
        "action_feature_mode": "static_v1",
        "include_explicit_noop": False,
        "budget_increase_ratio": 0.02,
        "budget_decrease_ratio": 0.02,
        "budget_floor_ratio": 0.9,
        "observation_mode": "v11_full_10d",
        "reward_mode": "mendes",
        "runtime_config": {"agent_period": 25000},
        "save_best_by": "qos_recovery_stable",
        "enable_deploy_cap_mask": True,
        "deploy_cap_mask_ratio": 4.0,
        "deploy_cap_mask_criticality": "lo",
        "forbid_decreasing_hi_budgets": True,
        "action_space_size": 24,
        "observation_dim": 128,
    }


def test_current_config_freezes_without_recursive_lookup(tmp_path: Path) -> None:
    run = _write_current_run(tmp_path)
    output = tmp_path / "frozen.json"
    frozen = freeze_reference_config(run, output)
    assert frozen["schema_version"] == "qamc_reference_experiment_config_v3"
    assert load_and_validate_frozen_reference(output)["fingerprint"] == frozen["fingerprint"]


def test_legacy_config_requires_explicit_upgrade_flag(tmp_path: Path) -> None:
    run = tmp_path / "run"
    run.mkdir()
    (run / "config.json").write_text(json.dumps(_legacy_config()), encoding="utf-8")
    with pytest.raises(ValueError, match="EFFECTIVE_CONFIG_MISSING"):
        freeze_reference_config(run, tmp_path / "frozen.json")


def test_legacy_upgrade_records_provenance_and_project_reward(
    tmp_path: Path,
) -> None:
    project = tmp_path / "project"
    reward_dir = project / "configs" / "reward_modes"
    reward_dir.mkdir(parents=True)
    (reward_dir / "mendes.json").write_text("{}\n", encoding="utf-8")
    run = tmp_path / "run"
    run.mkdir()
    (run / "config.json").write_text(json.dumps(_legacy_config()), encoding="utf-8")
    frozen = freeze_reference_config(
        run,
        tmp_path / "frozen.json",
        allow_legacy_upgrade=True,
        project_root=project,
    )
    assert frozen["legacy_upgrade"]["performed"] is True
    assert frozen["reward_artifact"]["path"] == str(
        (reward_dir / "mendes.json").resolve()
    )


def test_save_best_by_is_audited(tmp_path: Path) -> None:
    run = _write_current_run(tmp_path, selector="lo_degraded_completion_ratio")
    with pytest.raises(ValueError, match="SELECTOR_NOT_COMPATIBLE"):
        freeze_reference_config(run, tmp_path / "frozen.json")


def test_later_log_file_does_not_invalidate_frozen_reference(
    tmp_path: Path,
) -> None:
    run = _write_current_run(tmp_path)
    output = tmp_path / "frozen.json"
    freeze_reference_config(run, output)
    (run / "later.log").write_text("new log\n", encoding="utf-8")
    load_and_validate_frozen_reference(output)


def test_bound_config_hash_change_is_rejected(tmp_path: Path) -> None:
    run = _write_current_run(tmp_path)
    output = tmp_path / "frozen.json"
    freeze_reference_config(run, output)
    (run / "config.json").write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="SOURCE_CONFIG_HASH_MISMATCH"):
        load_and_validate_frozen_reference(output)
