"""Materialize one q-AMC profile per taskset from a JSON taskset registry."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from amc_py.models import Criticality, Task
from amc_py.qamc.profile_spec import load_profile_spec
from amc_py.qamc.profiles import build_qamc_profile_bundle, write_profile_bundle


def _taskset_fingerprint(tasks: list[Task]) -> str:
    payload = [(t.name, t.period, t.deadline, t.c_lo, t.c_hi, t.criticality.value) for t in tasks]
    # Keep the same fingerprint recipe as amc_py.dqn.experiment so a
    # materialized profile can be resolved by the training/evaluation factory.
    return hashlib.sha256(str(payload).encode("utf-8")).hexdigest()[:12]


def _parse_task(raw: dict[str, Any]) -> Task:
    return Task(
        name=str(raw["name"]), period=int(raw["period"]), deadline=int(raw["deadline"]),
        c_lo=int(raw["c_lo"]), c_hi=int(raw["c_hi"]), criticality=Criticality(raw["criticality"]),
    )


def materialize(tasksets_path: str | Path, spec_path: str | Path, output_dir: str | Path) -> dict[str, Any]:
    tasksets = json.loads(Path(tasksets_path).read_text(encoding="utf-8"))
    if isinstance(tasksets, dict):
        tasksets = tasksets.get("tasksets", tasksets)
    if not isinstance(tasksets, list):
        raise ValueError("QAMC_TASKSET_REGISTRY_MUST_BE_LIST")
    spec = load_profile_spec(spec_path)
    root = Path(output_dir)
    entries: dict[str, dict[str, str]] = {}
    for item in tasksets:
        raw_tasks = item.get("tasks", item) if isinstance(item, dict) else item
        tasks = [_parse_task(raw) for raw in raw_tasks]
        fingerprint = str(item.get("taskset_fingerprint", _taskset_fingerprint(tasks))) if isinstance(item, dict) else _taskset_fingerprint(tasks)
        bundle = build_qamc_profile_bundle(tasks, taskset_fingerprint=fingerprint, spec=spec)
        path = root / fingerprint / "profile.json"
        write_profile_bundle(bundle, path)
        entries[fingerprint] = {"path": str(path), "fingerprint": bundle.fingerprint}
    payload: dict[str, Any] = {
        "schema_version": "qamc_profile_manifest_v1",
        "spec_path": str(Path(spec_path).resolve()),
        "spec_fingerprint": spec.fingerprint,
        "profiles": entries,
    }
    payload["fingerprint"] = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    root.mkdir(parents=True, exist_ok=True)
    (root / "manifest.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasksets", required=True)
    parser.add_argument("--spec", default="configs/qamc_profile_spec_v2.json")
    parser.add_argument("--output-dir", default="outputs/qamc_profiles")
    args = parser.parse_args()
    print(json.dumps(materialize(args.tasksets, args.spec, args.output_dir), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
