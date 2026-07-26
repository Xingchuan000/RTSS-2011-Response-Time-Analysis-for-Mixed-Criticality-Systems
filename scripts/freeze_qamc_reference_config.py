"""Freeze a real C-AMC-sem training configuration for q-AMC experiments.

This command intentionally refuses to infer a reference configuration from
CLI defaults.  It is an audit artifact generator, not a training command.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


REQUIRED_NORMALIZED_KEYS = (
    "action_space",
    "q_network_type",
    "action_feature_mode",
    "include_explicit_noop",
    "budget_increase_ratio",
    "budget_decrease_ratio",
    "budget_rounding_mode",
    "min_budget_delta",
    "budget_floor_ratio",
    "check_safety",
    "observation_mode",
    "reward_mode",
    "agent_period",
)


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def _tree_fingerprint(root: Path) -> str:
    entries: list[tuple[str, str]] = []
    for path in sorted(p for p in root.rglob("*") if p.is_file() and ".git" not in p.parts):
        entries.append((str(path.relative_to(root)), hashlib.sha256(path.read_bytes()).hexdigest()))
    return hashlib.sha256(_canonical(entries)).hexdigest()


def _find_key(value: Any, names: set[str]) -> Any:
    if isinstance(value, dict):
        for key, item in value.items():
            if key in names:
                return item
            result = _find_key(item, names)
            if result is not None:
                return result
    elif isinstance(value, list):
        for item in value:
            result = _find_key(item, names)
            if result is not None:
                return result
    return None


def _contains_c_amc_only_selector(value: Any) -> bool:
    if isinstance(value, dict):
        for key, item in value.items():
            text = str(key).lower()
            if "degraded" in text or "c_amc" in text or "c-amc" in text:
                return True
            if _contains_c_amc_only_selector(item):
                return True
    elif isinstance(value, list):
        return any(_contains_c_amc_only_selector(item) for item in value)
    elif isinstance(value, str):
        lowered = value.lower()
        return "lo_degraded" in lowered or "c_amc" in lowered or "c-amc" in lowered
    return False


def freeze_reference_config(reference_run_dir: str | Path, output: str | Path) -> dict[str, Any]:
    root = Path(reference_run_dir)
    config_path = root / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError("QAMC_REFERENCE_CONFIG_MISSING")
    try:
        raw = json.loads(config_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError("QAMC_REFERENCE_CONFIG_INVALID_JSON") from exc
    if not isinstance(raw, dict):
        raise ValueError("QAMC_REFERENCE_CONFIG_MUST_BE_OBJECT")

    # These aliases cover the existing training configuration naming variants
    # without introducing a default value when the source is absent.
    aliases = {
        "action_space": {"action_space", "dqn_action_space"},
        "q_network_type": {"q_network_type", "network_type", "q_network"},
        "action_feature_mode": {"action_feature_mode", "action_features"},
        "include_explicit_noop": {"include_explicit_noop"},
        "budget_increase_ratio": {"budget_increase_ratio"},
        "budget_decrease_ratio": {"budget_decrease_ratio"},
        "budget_rounding_mode": {"budget_rounding_mode", "rounding_mode"},
        "min_budget_delta": {"min_budget_delta"},
        "budget_floor_ratio": {"budget_floor_ratio"},
        "check_safety": {"check_safety"},
        "observation_mode": {"observation_mode"},
        "reward_mode": {"reward_mode"},
        "agent_period": {"agent_period"},
    }
    normalized: dict[str, Any] = {}
    missing: list[str] = []
    for normalized_key, names in aliases.items():
        value = _find_key(raw, names)
        if value is None:
            missing.append(normalized_key)
        else:
            normalized[normalized_key] = value
    if missing:
        raise ValueError("QAMC_REFERENCE_REQUIRED_FIELDS_MISSING:" + ",".join(missing))

    reward_candidate = _find_key(raw, {"reward_config_path", "reward_artifact", "reward_file"})
    reward_path: Path | None = None
    if isinstance(reward_candidate, str):
        reward_path = Path(reward_candidate)
        if not reward_path.is_absolute():
            reward_path = root / reward_path
    if reward_path is None:
        candidates = sorted(root.glob("*reward*.json"))
        reward_path = candidates[0] if candidates else None
    if reward_path is None or not reward_path.is_file():
        raise FileNotFoundError("QAMC_REFERENCE_REWARD_ARTIFACT_MISSING")

    selector = _find_key(raw, {"selector", "checkpoint_selector", "validation_selector"})
    if _contains_c_amc_only_selector(selector):
        raise ValueError("QAMC_SELECTOR_USES_C_AMC_ONLY_METRIC")
    normalized["enable_deploy_cap_mask"] = _find_key(raw, {"enable_deploy_cap_mask"})
    normalized["deploy_cap_mask_ratio"] = _find_key(raw, {"deploy_cap_mask_ratio"})
    normalized["deploy_cap_mask_criticality"] = _find_key(raw, {"deploy_cap_mask_criticality"})
    normalized["forbid_decreasing_hi_budgets"] = _find_key(raw, {"forbid_decreasing_hi_budgets"})
    normalized["network"] = _find_key(raw, {"network", "network_config"})
    normalized["replay"] = _find_key(raw, {"replay", "replay_config"})
    normalized["epsilon"] = _find_key(raw, {"epsilon", "epsilon_config"})
    normalized["validation"] = _find_key(raw, {"validation", "validation_config"})
    normalized["selector"] = selector
    normalized["workload_taskset_scenario_protocol"] = _find_key(raw, {"workload", "taskset", "scenario"})
    normalized["training_episode_organization"] = _find_key(raw, {"episodes", "training_episodes", "episode_organization"})

    artifact = {
        "path": str(reward_path),
        "sha256": _sha256_file(reward_path),
    }
    result = {
        "schema_version": "qamc_reference_experiment_config_v2",
        "source_config_path": str(config_path.resolve()),
        "source_config_sha256": _sha256_file(config_path),
        "source_tree_fingerprint": _tree_fingerprint(root),
        "reference_config_raw": raw,
        "normalized": normalized,
        "reward_artifact": artifact,
        "selector_artifact": {"selector": selector, "qamc_compatible": True},
    }
    result["fingerprint"] = hashlib.sha256(_canonical(result)).hexdigest()
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-run-dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    try:
        result = freeze_reference_config(args.reference_run_dir, args.output)
    except (FileNotFoundError, ValueError) as exc:
        print(str(exc))
        return 2
    print(json.dumps({"output": args.output, "fingerprint": result["fingerprint"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
