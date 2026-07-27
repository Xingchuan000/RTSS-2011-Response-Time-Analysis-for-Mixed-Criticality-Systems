"""Derive a q-AMC observation-only reference variant from a frozen O0 reference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from amc_py.qamc.reference_config import load_and_validate_frozen_reference
from amc_py.rl.feature_config import FeatureConfig, is_qamc_observation_mode
from amc_py.rl.observation_metadata import (
    build_observation_feature_names_from_task_names,
)
from amc_py.rl.observation_schema import observation_schema_fingerprint


def derive_reference_variant(
    *,
    base_frozen_path: Path,
    output_dir: Path,
    observation_mode: str,
) -> None:
    if not is_qamc_observation_mode(observation_mode):
        raise ValueError("QAMC_REFERENCE_VARIANT_MODE_REQUIRED")

    base_frozen_path = base_frozen_path.resolve()
    frozen = load_and_validate_frozen_reference(base_frozen_path)
    base_effective = dict(frozen["effective_reference_config"])
    source_config_path = Path(str(frozen["source_config_path"]))
    source_config = json.loads(source_config_path.read_text(encoding="utf-8"))
    tasks = source_config.get("tasks")
    if not isinstance(tasks, list) or not tasks:
        raise ValueError("QAMC_REFERENCE_TASKS_MISSING")
    try:
        task_names = tuple(str(task["name"]) for task in tasks)
    except (KeyError, TypeError) as exc:
        raise ValueError("QAMC_REFERENCE_TASKS_MISSING") from exc

    feature_names = build_observation_feature_names_from_task_names(
        task_names,
        FeatureConfig(observation_mode=observation_mode),
    )
    observation_dim = len(feature_names)
    derived_effective = dict(base_effective)
    derived_effective["observation_mode"] = observation_mode
    derived_effective["observation_dim"] = observation_dim
    changed = {
        key
        for key in set(base_effective) | set(derived_effective)
        if derived_effective.get(key) != base_effective.get(key)
    }
    if changed != {"observation_mode", "observation_dim"}:
        raise ValueError("QAMC_REFERENCE_VARIANT_ILLEGAL_OVERRIDE")

    schema_hash = observation_schema_fingerprint(
        observation_mode=observation_mode,
        feature_names=feature_names,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    derived_config = {
        "effective_reference_config": derived_effective,
        "reference_variant_derivation": {
            "schema_version": "qamc_observation_reference_derivation_v1",
            "base_frozen_reference_path": str(base_frozen_path),
            "base_frozen_reference_fingerprint": frozen["fingerprint"],
            "allowed_overrides": ["observation_mode", "observation_dim"],
            "observation_mode": observation_mode,
            "observation_dim": observation_dim,
        },
        "tasks": tasks,
        "observation_mode": observation_mode,
        "observation_dim": observation_dim,
    }
    (output_dir / "config.json").write_text(
        json.dumps(derived_config, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / "feature_names.json").write_text(
        json.dumps(
            {
                "schema_version": "observation_feature_schema_v1",
                "observation_mode": observation_mode,
                "observation_dim": observation_dim,
                "feature_names": list(feature_names),
                "fingerprint": schema_hash,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-frozen-reference", type=Path, required=True)
    parser.add_argument("--observation-mode", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    try:
        derive_reference_variant(
            base_frozen_path=args.base_frozen_reference,
            output_dir=args.output_dir,
            observation_mode=args.observation_mode,
        )
    except (FileNotFoundError, ValueError) as exc:
        print(str(exc))
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
