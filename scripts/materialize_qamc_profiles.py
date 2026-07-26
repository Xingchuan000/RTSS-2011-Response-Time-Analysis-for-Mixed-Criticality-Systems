"""Materialize q-AMC profiles from a registry or an actual training config."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from amc_py.models import Criticality, Task
from amc_py.qamc.effective_config import canonical_sha256
from amc_py.qamc.profile_spec import load_profile_spec
from amc_py.qamc.profiles import (
    build_qamc_profile_bundle,
    compute_taskset_fingerprint,
    write_profile_bundle,
)


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _parse_task(raw: dict[str, Any]) -> Task:
    return Task(
        name=str(raw["name"]),
        period=int(raw["period"]),
        deadline=int(raw["deadline"]),
        c_lo=int(raw["c_lo"]),
        c_hi=int(raw["c_hi"]),
        criticality=Criticality(raw["criticality"]),
    )


def load_tasks_from_reference_config(path: str | Path) -> list[Task]:
    config_path = Path(path)
    raw = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("QAMC_REFERENCE_CONFIG_MUST_BE_OBJECT")
    raw_tasks = raw.get("tasks")
    if raw_tasks is None:
        initial_taskset = raw.get("initial_taskset")
        raw_tasks = (
            initial_taskset.get("tasks")
            if isinstance(initial_taskset, dict)
            else None
        )
    if not isinstance(raw_tasks, list) or not raw_tasks:
        raise ValueError("QAMC_REFERENCE_TASKS_MISSING")
    if not all(isinstance(item, dict) for item in raw_tasks):
        raise ValueError("QAMC_REFERENCE_TASKS_INVALID")
    tasks = [_parse_task(item) for item in raw_tasks]
    names = [task.name for task in tasks]
    if len(names) != len(set(names)):
        raise ValueError("QAMC_REFERENCE_TASK_NAMES_NOT_UNIQUE")
    return tasks


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _new_manifest(spec_path: Path, spec_fingerprint: str) -> dict[str, Any]:
    return {
        "schema_version": "qamc_profile_manifest_v1",
        "spec_path": str(spec_path.resolve()),
        "spec_fingerprint": spec_fingerprint,
        "profiles": {},
    }


def _validate_existing_manifest(
    manifest: dict[str, Any],
    *,
    spec_fingerprint: str,
) -> None:
    if manifest.get("schema_version") != "qamc_profile_manifest_v1":
        raise ValueError("QAMC_UNSUPPORTED_PROFILE_MANIFEST_SCHEMA")
    if manifest.get("spec_fingerprint") != spec_fingerprint:
        raise ValueError("QAMC_PROFILE_MANIFEST_SPEC_FINGERPRINT_MISMATCH")
    if not isinstance(manifest.get("profiles"), dict):
        raise ValueError("QAMC_PROFILE_MANIFEST_PROFILES_INVALID")


def materialize(
    tasksets_path: str | Path | None,
    spec_path: str | Path,
    output_dir: str | Path,
    *,
    reference_run_dir: str | Path | None = None,
    reference_config: str | Path | None = None,
    append_manifest: bool = False,
) -> dict[str, Any]:
    selected = sum(
        source is not None
        for source in (tasksets_path, reference_run_dir, reference_config)
    )
    if selected != 1:
        raise ValueError("QAMC_PROFILE_EXACTLY_ONE_SOURCE_REQUIRED")

    source_rows: list[tuple[list[Task], Path | None]] = []
    if reference_run_dir is not None:
        config_path = Path(reference_run_dir).resolve() / "config.json"
        source_rows.append((load_tasks_from_reference_config(config_path), config_path))
    elif reference_config is not None:
        config_path = Path(reference_config).resolve()
        source_rows.append((load_tasks_from_reference_config(config_path), config_path))
    else:
        registry_path = Path(tasksets_path)  # type: ignore[arg-type]
        raw_registry = json.loads(registry_path.read_text(encoding="utf-8"))
        if isinstance(raw_registry, dict):
            raw_registry = raw_registry.get("tasksets", raw_registry)
        if not isinstance(raw_registry, list):
            raise ValueError("QAMC_TASKSET_REGISTRY_MUST_BE_LIST")
        for item in raw_registry:
            raw_tasks = item.get("tasks", item) if isinstance(item, dict) else item
            if not isinstance(raw_tasks, list):
                raise ValueError("QAMC_TASKSET_REGISTRY_ENTRY_INVALID")
            tasks = [_parse_task(raw) for raw in raw_tasks]
            computed = compute_taskset_fingerprint(tasks)
            claimed = (
                str(item.get("taskset_fingerprint", computed))
                if isinstance(item, dict)
                else computed
            )
            if claimed != computed:
                raise ValueError("QAMC_TASKSET_REGISTRY_FINGERPRINT_MISMATCH")
            source_rows.append((tasks, registry_path.resolve()))

    spec_file = Path(spec_path)
    spec = load_profile_spec(spec_file)
    root = Path(output_dir)
    manifest_path = root / "manifest.json"
    if append_manifest and manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        _validate_existing_manifest(manifest, spec_fingerprint=spec.fingerprint)
        manifest.pop("fingerprint", None)
    else:
        manifest = _new_manifest(spec_file, spec.fingerprint)

    for tasks, source_path in source_rows:
        names = [task.name for task in tasks]
        if len(names) != len(set(names)):
            raise ValueError("QAMC_REFERENCE_TASK_NAMES_NOT_UNIQUE")
        taskset_fingerprint = compute_taskset_fingerprint(tasks)
        bundle = build_qamc_profile_bundle(
            tasks,
            taskset_fingerprint=taskset_fingerprint,
            spec=spec,
        )
        profile_path = root / taskset_fingerprint / "profile.json"
        write_profile_bundle(bundle, profile_path)
        entry: dict[str, Any] = {
            "path": f"{taskset_fingerprint}/profile.json",
            "fingerprint": bundle.fingerprint,
            "task_count": len(tasks),
            "lo_profile_count": len(bundle.profiles),
            "hi_tasks_are_single_level_and_excluded": True,
        }
        if source_path is not None:
            entry.update(
                {
                    "source_config_path": str(source_path),
                    "source_config_sha256": _sha256_file(source_path),
                }
            )
        manifest["profiles"][taskset_fingerprint] = entry

    manifest["fingerprint"] = canonical_sha256(manifest)
    _atomic_write_json(manifest_path, manifest)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--tasksets")
    source.add_argument("--reference-run-dir")
    source.add_argument("--reference-config")
    parser.add_argument("--spec", default="configs/qamc_profile_spec_v2.json")
    parser.add_argument("--output-dir", default="outputs/qamc_profiles")
    parser.add_argument("--append-manifest", action="store_true")
    args = parser.parse_args()
    result = materialize(
        args.tasksets,
        args.spec,
        args.output_dir,
        reference_run_dir=args.reference_run_dir,
        reference_config=args.reference_config,
        append_manifest=args.append_manifest,
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
