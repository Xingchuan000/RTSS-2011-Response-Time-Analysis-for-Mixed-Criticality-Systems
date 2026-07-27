"""Derive a bounded q-AMC learning-reference variant from a frozen reference.

The historical observation-only path remains supported.  Optional reward and
checkpoint-selector overrides are recorded explicitly and are validated by the
frozen-reference loader; arbitrary effective-config changes remain forbidden.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from amc_py.qamc.reference_config import load_and_validate_frozen_reference
from amc_py.qamc.selector_contract import selector_is_compatible
from amc_py.rl.feature_config import FeatureConfig, is_qamc_observation_mode
from amc_py.rl.observation_metadata import (
    build_observation_feature_names_from_task_names,
)
from amc_py.rl.observation_schema import observation_schema_fingerprint
from amc_py.rl.reward_config import reward_config_dir
from amc_py.runtime_models import RuntimeSemantics


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def derive_reference_variant(
    *,
    base_frozen_path: Path,
    output_dir: Path,
    observation_mode: str,
    reward_mode: str | None = None,
    save_best_by: str | None = None,
    project_root: Path | None = None,
) -> None:
    """Create a bounded derived reference.

    With no reward/selector arguments this emits the original
    ``qamc_observation_reference_derivation_v1`` contract.  When either is
    supplied, the derived config uses
    ``qamc_learning_reference_derivation_v1`` and declares the exact fields
    that changed.
    """

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

    reward_path: Path | None = None
    if reward_mode is not None:
        root = PROJECT_ROOT if project_root is None else project_root.resolve()
        # Prefer the supplied project root so the generated contract binds the
        # artifact that the user's training checkout will actually load.
        candidate = root / "configs" / "reward_modes" / f"{reward_mode}.json"
        if not candidate.is_file():
            # Keep direct Python/test use convenient when project_root is not
            # supplied explicitly.
            candidate = reward_config_dir() / f"{reward_mode}.json"
        reward_path = candidate.resolve()
        if not reward_path.is_file():
            raise FileNotFoundError(
                f"QAMC_REFERENCE_REWARD_ARTIFACT_MISSING:{reward_path}"
            )
        derived_effective["reward_mode"] = reward_mode
        derived_effective["reward_config_path"] = str(reward_path)
        derived_effective["reward_config_sha256"] = _sha256_file(reward_path)

    if save_best_by is not None:
        if not selector_is_compatible(save_best_by, RuntimeSemantics.Q_AMC):
            raise ValueError(f"QAMC_SELECTOR_NOT_COMPATIBLE:{save_best_by}")
        derived_effective["save_best_by"] = save_best_by

    changed = {
        key
        for key in set(base_effective) | set(derived_effective)
        if derived_effective.get(key) != base_effective.get(key)
    }
    allowed = {
        "observation_mode",
        "observation_dim",
        "reward_mode",
        "reward_config_path",
        "reward_config_sha256",
        "save_best_by",
    }
    if not changed or not changed.issubset(allowed):
        raise ValueError("QAMC_REFERENCE_VARIANT_ILLEGAL_OVERRIDE")
    if (changed & {"reward_mode", "reward_config_path", "reward_config_sha256"}) and not {
        "reward_mode",
        "reward_config_path",
        "reward_config_sha256",
    }.issubset(changed):
        raise ValueError("QAMC_REFERENCE_REWARD_OVERRIDE_INCOMPLETE")

    schema_hash = observation_schema_fingerprint(
        observation_mode=observation_mode,
        feature_names=feature_names,
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    learning_override = reward_mode is not None or save_best_by is not None
    derivation_schema = (
        "qamc_learning_reference_derivation_v1"
        if learning_override
        else "qamc_observation_reference_derivation_v1"
    )
    derivation = {
        "schema_version": derivation_schema,
        "base_frozen_reference_path": str(base_frozen_path),
        "base_frozen_reference_fingerprint": frozen["fingerprint"],
        "allowed_overrides": sorted(changed),
        "observation_mode": observation_mode,
        "observation_dim": observation_dim,
    }
    if reward_mode is not None and reward_path is not None:
        derivation.update(
            {
                "reward_mode": reward_mode,
                "reward_config_path": str(reward_path),
                "reward_config_sha256": _sha256_file(reward_path),
            }
        )
    if save_best_by is not None:
        derivation["save_best_by"] = save_best_by

    derived_config = {
        "effective_reference_config": derived_effective,
        "reference_variant_derivation": derivation,
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
    parser.add_argument("--reward-mode")
    parser.add_argument("--save-best-by")
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    try:
        derive_reference_variant(
            base_frozen_path=args.base_frozen_reference,
            output_dir=args.output_dir,
            observation_mode=args.observation_mode,
            reward_mode=args.reward_mode,
            save_best_by=args.save_best_by,
            project_root=args.project_root,
        )
    except (FileNotFoundError, ValueError) as exc:
        print(str(exc))
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
