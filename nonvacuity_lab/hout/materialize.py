from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import shutil

from ..canonical import file_hash
from .schema import HoutProfile


@dataclass(frozen=True)
class MaterializedHoutProfile:
    scenario_file: Path
    runtime_config_file: Path
    determinism_receipt_file: Path
    profile: HoutProfile


def materialize_hout_profile(profile: HoutProfile, workspace: Path) -> MaterializedHoutProfile:
    profile.validate()
    target = workspace / "hout_inputs"
    target.mkdir(parents=True, exist_ok=False)
    scenario = target / "scenarios.json"
    scenario.write_text(json.dumps(list(profile.scenario_seeds), indent=2) + "\n", encoding="utf-8")
    runtime = target / "runtime_config.json"
    shutil.copy2(profile.runtime_config_path, runtime)
    receipt = target / "determinism_receipt.json"
    receipt.write_text(json.dumps({
        "profile_id": profile.profile_id,
        "taskset_sha256": file_hash(Path(profile.taskset_path)),
        "scenario_file_sha256": file_hash(scenario),
        "demand_trace_sha256": None if profile.demand_trace_path is None else file_hash(Path(profile.demand_trace_path)),
        "runtime_config_sha256": file_hash(runtime), "horizon": profile.horizon,
        "controller_release_times": list(profile.controller_release_times),
        "worker_count": profile.worker_count, "random_seed": profile.random_seed,
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return MaterializedHoutProfile(scenario, runtime, receipt, profile)
