from __future__ import annotations

import json
from pathlib import Path

import pytest

from amc_py.models import Criticality, Task
from amc_py.qamc.profile_spec import QAmcProfileSpec, write_profile_spec
from amc_py.qamc.profiles import (
    compute_taskset_fingerprint,
    load_profile_bundle_from_manifest,
)
from scripts.materialize_qamc_profiles import materialize


def _reference_config(path: Path) -> list[Task]:
    tasks = [
        Task("H", 20, 20, 5, 8, Criticality.HI),
        Task("L", 25, 25, 9, 9, Criticality.LO),
    ]
    path.write_text(
        json.dumps(
            {
                "tasks": [
                    {
                        "name": task.name,
                        "period": task.period,
                        "deadline": task.deadline,
                        "c_lo": task.c_lo,
                        "c_hi": task.c_hi,
                        "criticality": task.criticality.value,
                    }
                    for task in tasks
                ]
            }
        ),
        encoding="utf-8",
    )
    return tasks


def test_materialize_from_training_config_tasks_and_excludes_hi(
    tmp_path: Path,
) -> None:
    config = tmp_path / "config.json"
    tasks = _reference_config(config)
    spec_path = tmp_path / "spec.json"
    write_profile_spec(QAmcProfileSpec(), spec_path)
    output = tmp_path / "profiles"
    manifest = materialize(
        None,
        spec_path,
        output,
        reference_config=config,
    )
    fingerprint = compute_taskset_fingerprint(tasks)
    assert fingerprint in manifest["profiles"]
    bundle = load_profile_bundle_from_manifest(
        output / "manifest.json",
        taskset_fingerprint=fingerprint,
        spec_path=spec_path,
    )
    assert set(bundle.profiles) == {"L"}
    assert manifest["profiles"][fingerprint]["path"] == f"{fingerprint}/profile.json"


def test_append_manifest_is_idempotent(tmp_path: Path) -> None:
    config = tmp_path / "config.json"
    _reference_config(config)
    spec_path = tmp_path / "spec.json"
    write_profile_spec(QAmcProfileSpec(), spec_path)
    output = tmp_path / "profiles"
    first = materialize(
        None,
        spec_path,
        output,
        reference_config=config,
        append_manifest=True,
    )
    second = materialize(
        None,
        spec_path,
        output,
        reference_config=config,
        append_manifest=True,
    )
    assert first == second


def test_manifest_path_escape_is_rejected(tmp_path: Path) -> None:
    config = tmp_path / "config.json"
    tasks = _reference_config(config)
    spec_path = tmp_path / "spec.json"
    write_profile_spec(QAmcProfileSpec(), spec_path)
    output = tmp_path / "profiles"
    manifest = materialize(None, spec_path, output, reference_config=config)
    fingerprint = compute_taskset_fingerprint(tasks)
    manifest["profiles"][fingerprint]["path"] = "../outside.json"
    unsigned = dict(manifest)
    unsigned.pop("fingerprint")
    from amc_py.qamc.effective_config import canonical_sha256

    manifest["fingerprint"] = canonical_sha256(unsigned)
    (output / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="ESCAPES_MANIFEST_ROOT"):
        load_profile_bundle_from_manifest(
            output / "manifest.json",
            taskset_fingerprint=fingerprint,
            spec_path=spec_path,
        )
