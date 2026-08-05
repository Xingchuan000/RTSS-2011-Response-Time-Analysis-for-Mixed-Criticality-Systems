"""Prepare one practical paired-HOUT profile in a resolved v2 campaign.

The script intentionally targets research workflows: it copies the taskset
artifact resolved from Phase-3 audit data, writes a deterministic ordinary HOUT
runtime config, binds selected mutations to the profile, and reseals the config.
It does not invent scenario seeds; callers must supply a JSON list or an
existing HOUT manifest containing ``scenario_seeds``.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from nonvacuity_lab.config_io import write_resolved_campaign
from nonvacuity_lab.config_resolver import _discover_symbolic_binding


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--profile-id", required=True)
    parser.add_argument("--scenario-source", type=Path, required=True)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--experiment-name", choices=("small_stress", "small_nominal", "rtss11", "automotive", "mc_fairgen"))
    group.add_argument("--experiment-factory")
    parser.add_argument("--bind-mutation", action="append", default=[])
    parser.add_argument("--horizon", type=int, default=50_000_000)
    parser.add_argument("--agent-period", type=int, default=100)
    parser.add_argument("--worker-count", type=int, default=1)
    parser.add_argument("--required-scenario", type=int, action="append", default=[])
    parser.add_argument("--factory-kwargs", default="{}", help="JSON object merged into experiment_factory_kwargs")
    parser.add_argument("--input-root", type=Path)
    return parser.parse_args()


def _scenario_seeds(path: Path) -> list[int]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(raw, list):
        values = raw
    elif isinstance(raw, dict):
        values = raw.get("scenario_seeds", raw.get("scenarios"))
    else:
        values = None
    if not isinstance(values, list) or not values:
        raise ValueError("scenario source must contain a non-empty scenario seed list")
    return [int(item) for item in values]


def _target_for_seed(config: dict, seed: int) -> dict:
    for mutation in config.get("mutations", []):
        if mutation.get("seed") == seed:
            target = mutation.get("resolved_target")
            if isinstance(target, dict) and target.get("tree_path"):
                return target
    raise ValueError(f"resolved target not found for seed {seed}; run resolve first")


def main() -> int:
    args = _parse_args()
    config_path = args.config.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("schema_version") != "nonvacuity_campaign_v2" or config.get("config_kind") != "RESOLVED":
        raise ValueError("--config must be a resolved nonvacuity_campaign_v2 file")
    target = _target_for_seed(config, args.seed)
    binding = _discover_symbolic_binding(target)
    source_taskset = Path(binding["taskset_path"]).resolve()
    input_root = (args.input_root or (config_path.parent / "inputs")).resolve()
    seed_root = input_root / f"s{args.seed}"
    seed_root.mkdir(parents=True, exist_ok=True)
    taskset = seed_root / "taskset.json"
    shutil.copy2(source_taskset, taskset)

    kwargs = json.loads(args.factory_kwargs)
    if not isinstance(kwargs, dict):
        raise ValueError("--factory-kwargs must decode to an object")
    # All supported generated/provider workloads accept fixed_taskset_seed;
    # small fixtures ignore seed anyway.  Do not add it to custom factories.
    if args.experiment_name in {"rtss11", "automotive", "mc_fairgen"}:
        kwargs.setdefault("fixed_taskset_seed", int(args.seed))
    runtime = {
        "ordinary_hout_factory": "amc_py.evaluation.ordinary_tree_hout:build_ordinary_tree_hout_runner",
        "scenario_runner_factory": "amc_py.evaluation.c_amc_sem_tree_scenario:run_c_amc_sem_tree_scenario",
        "horizon": int(args.horizon),
        "agent_period": int(args.agent_period),
        "runtime_semantics": "C_AMC_SEM",
        "taskset_seed": int(args.seed),
        "scenario_seed_drives_bundle": True,
        "action_space": "single",
        "budget_increase_ratio": 0.02,
        "budget_decrease_ratio": 0.02,
        "budget_floor_ratio": 0.0,
        "forbid_decreasing_hi_budgets": True,
        "mask_detail_mode": "full",
        "enable_deploy_cap_mask": True,
        "deploy_cap_mask_ratio": 4.0,
        "deploy_cap_mask_criticality": "lo",
        "experiment_factory_kwargs": kwargs,
    }
    if args.experiment_name:
        runtime["experiment_name"] = args.experiment_name
    else:
        runtime["experiment_factory"] = args.experiment_factory
    runtime_path = seed_root / "hout_runtime.json"
    runtime_path.write_text(json.dumps(runtime, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    scenarios = _scenario_seeds(args.scenario_source.resolve())
    required = [int(item) for item in args.required_scenario]
    missing = sorted(set(required) - set(scenarios))
    if missing:
        raise ValueError(f"required scenarios missing from scenario source: {missing}")
    command = [
        "python", "scripts/run_nonvacuity_hout.py",
        "--seed-dir", "{seed_dir}",
        "--tree", "{tree_path}",
        "--scenario-file", "{scenario_file}",
        "--runtime-config", "{runtime_config}",
        "--taskset", "{taskset}",
        "--output-dir", "{output_dir}",
    ]
    profile = {
        "taskset_path": str(taskset),
        "runtime_config_path": str(runtime_path),
        "scenario_seeds": scenarios,
        "required_scenarios": required,
        "horizon": int(args.horizon),
        "controller_release_times": [0],
        "worker_count": int(args.worker_count),
        "random_seed": int(args.seed),
        "base_command": command,
        "mutated_command": list(command),
    }
    config.setdefault("hout_profiles", {})[args.profile_id] = profile
    bind_ids = set(args.bind_mutation)
    if bind_ids:
        known = {str(item.get("mutation_id")) for item in config.get("mutations", [])}
        unknown = sorted(bind_ids - known)
        if unknown:
            raise ValueError(f"unknown mutation ids: {unknown}")
        for mutation in config.get("mutations", []):
            if str(mutation.get("mutation_id")) in bind_ids:
                mutation["hout_profile_id"] = args.profile_id
    write_resolved_campaign(config_path, config)
    print(json.dumps({
        "status": "HOUT_PROFILE_PREPARED",
        "config": str(config_path),
        "profile_id": args.profile_id,
        "seed": args.seed,
        "scenario_count": len(scenarios),
        "taskset": str(taskset),
        "runtime_config": str(runtime_path),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
