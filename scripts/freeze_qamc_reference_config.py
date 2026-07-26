"""Freeze a C-AMC-sem training run into the canonical q-AMC reference contract."""

from __future__ import annotations

import argparse
from dataclasses import fields
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from amc_py.qamc.effective_config import (
    QAmcReferenceEffectiveConfig,
    canonical_sha256,
)
from amc_py.qamc.selector_contract import selector_is_compatible
from amc_py.runtime_models import RuntimeSemantics


LEGACY_FIXED_PROVENANCE: dict[str, dict[str, Any]] = {
    "budget_rounding_mode": {
        "value": "ceil_floor",
        "source": "amc_py.rl.env.AmcBudgetEnv default/effective runtime",
    },
    "min_budget_delta": {
        "value": 1,
        "source": "amc_py.rl.env.AmcBudgetEnv default/effective runtime",
    },
    "check_safety": {
        "value": True,
        "source": "train_dqn_amc environment construction",
    },
    "step_guard_semantics": {
        "value": "checked",
        "source": "train launcher effective arguments",
    },
}

LEGACY_PATHS: dict[str, tuple[str, ...]] = {
    "action_space": ("action_space",),
    "q_network_type": ("q_network_type",),
    "action_feature_mode": ("action_feature_mode",),
    "include_explicit_noop": ("include_explicit_noop",),
    "budget_increase_ratio": ("budget_increase_ratio",),
    "budget_decrease_ratio": ("budget_decrease_ratio",),
    "budget_floor_ratio": ("budget_floor_ratio",),
    "observation_mode": ("observation_mode",),
    "reward_mode": ("reward_mode",),
    "agent_period": ("runtime_config", "agent_period"),
    "save_best_by": ("save_best_by",),
    "enable_deploy_cap_mask": ("enable_deploy_cap_mask",),
    "deploy_cap_mask_ratio": ("deploy_cap_mask_ratio",),
    "deploy_cap_mask_criticality": ("deploy_cap_mask_criticality",),
    "forbid_decreasing_hi_budgets": ("forbid_decreasing_hi_budgets",),
    "action_dim": ("action_space_size",),
    "observation_dim": ("observation_dim",),
}


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _required(raw: dict[str, Any], path: tuple[str, ...]) -> Any:
    value: Any = raw
    for key in path:
        if not isinstance(value, dict) or key not in value:
            raise ValueError("QAMC_LEGACY_REQUIRED_PATH_MISSING:" + ".".join(path))
        value = value[key]
    return value


def _validate_effective_reference_config(effective: dict[str, Any]) -> None:
    expected = {field.name for field in fields(QAmcReferenceEffectiveConfig)}
    missing = sorted(expected - set(effective))
    if missing:
        raise ValueError(
            "QAMC_REFERENCE_EFFECTIVE_FIELDS_MISSING:" + ",".join(missing)
        )
    try:
        contract = QAmcReferenceEffectiveConfig(
            **{name: effective[name] for name in expected}
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("QAMC_REFERENCE_EFFECTIVE_CONFIG_INVALID") from exc
    if contract.schema_version != "qamc_reference_effective_config_v1":
        raise ValueError("QAMC_REFERENCE_EFFECTIVE_CONFIG_SCHEMA_INVALID")
    reward_path = Path(contract.reward_config_path)
    if not reward_path.is_absolute():
        raise ValueError("QAMC_REFERENCE_REWARD_PATH_NOT_ABSOLUTE")
    if not reward_path.is_file():
        raise FileNotFoundError("QAMC_REFERENCE_REWARD_ARTIFACT_MISSING")
    if _sha256_file(reward_path) != contract.reward_config_sha256:
        raise ValueError("QAMC_REFERENCE_REWARD_ARTIFACT_HASH_MISMATCH")
    if not contract.check_safety:
        raise ValueError("QAMC_REFERENCE_REQUIRES_CHECK_SAFETY")
    if contract.step_guard_semantics != "checked":
        raise ValueError("QAMC_REFERENCE_UNCHECKED_STEP_GUARD_FORBIDDEN")


def _validate_selector_compatibility(selector: str) -> None:
    if not selector_is_compatible(selector, RuntimeSemantics.Q_AMC):
        raise ValueError(f"QAMC_SELECTOR_NOT_COMPATIBLE:{selector}")


def _upgrade_legacy_reference_config(
    *,
    raw: dict[str, Any],
    run_root: Path,
    project_root: Path | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if project_root is None:
        raise ValueError("QAMC_LEGACY_UPGRADE_PROJECT_ROOT_REQUIRED")
    values = {name: _required(raw, path) for name, path in LEGACY_PATHS.items()}
    reward_path = (
        project_root
        / "configs"
        / "reward_modes"
        / f"{values['reward_mode']}.json"
    ).resolve()
    if not reward_path.is_file():
        raise FileNotFoundError(f"QAMC_REFERENCE_REWARD_ARTIFACT_MISSING:{reward_path}")
    for name, provenance in LEGACY_FIXED_PROVENANCE.items():
        values[name] = provenance["value"]
    values.update(
        {
            "schema_version": "qamc_reference_effective_config_v1",
            "reward_config_path": str(reward_path),
            "reward_config_sha256": _sha256_file(reward_path),
            "selector_contract_version": "selector_contract_v1",
        }
    )
    effective = QAmcReferenceEffectiveConfig(**values).to_jsonable()
    metadata = {
        "performed": True,
        "source_run_root": str(run_root),
        "fixed_values": LEGACY_FIXED_PROVENANCE,
        "explicit_paths": {
            name: ".".join(path) for name, path in sorted(LEGACY_PATHS.items())
        },
    }
    return effective, metadata


def freeze_reference_config(
    reference_run_dir: str | Path,
    output: str | Path,
    *,
    allow_legacy_upgrade: bool = False,
    project_root: str | Path | None = None,
) -> dict[str, Any]:
    root = Path(reference_run_dir).resolve()
    config_path = root / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError("QAMC_REFERENCE_CONFIG_MISSING")
    try:
        raw = json.loads(config_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError("QAMC_REFERENCE_CONFIG_INVALID_JSON") from exc
    if not isinstance(raw, dict):
        raise ValueError("QAMC_REFERENCE_CONFIG_MUST_BE_OBJECT")

    effective = raw.get("effective_reference_config")
    legacy_upgrade: dict[str, Any] = {"performed": False}
    if not isinstance(effective, dict):
        if not allow_legacy_upgrade:
            raise ValueError("QAMC_REFERENCE_EFFECTIVE_CONFIG_MISSING")
        effective, legacy_upgrade = _upgrade_legacy_reference_config(
            raw=raw,
            run_root=root,
            project_root=(
                Path(project_root).resolve() if project_root is not None else None
            ),
        )
    _validate_effective_reference_config(effective)
    _validate_selector_compatibility(str(effective["save_best_by"]))

    reward_path = Path(str(effective["reward_config_path"])).resolve()
    bound_artifacts = {
        "config.json": _sha256_file(config_path),
        "reward_config": _sha256_file(reward_path),
    }
    for optional_name in (
        "feature_names.json",
        "action_definitions.json",
        "model_best_metadata.json",
    ):
        artifact_path = root / optional_name
        if artifact_path.is_file():
            bound_artifacts[optional_name] = _sha256_file(artifact_path)

    result: dict[str, Any] = {
        "schema_version": "qamc_reference_experiment_config_v3",
        "source_run_root": str(root),
        "source_config_path": str(config_path),
        "source_config_sha256": _sha256_file(config_path),
        "effective_reference_config": effective,
        "effective_reference_config_fingerprint": canonical_sha256(effective),
        "reward_artifact": {
            "path": str(reward_path),
            "sha256": _sha256_file(reward_path),
        },
        "bound_artifacts": bound_artifacts,
        "legacy_upgrade": legacy_upgrade,
    }
    result["fingerprint"] = canonical_sha256(result)
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-run-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--allow-legacy-upgrade", action="store_true")
    args = parser.parse_args()
    try:
        result = freeze_reference_config(
            args.reference_run_dir,
            args.output,
            allow_legacy_upgrade=args.allow_legacy_upgrade,
            project_root=args.project_root,
        )
    except (FileNotFoundError, ValueError) as exc:
        print(str(exc))
        return 2
    print(
        json.dumps(
            {"output": args.output, "fingerprint": result["fingerprint"]},
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
