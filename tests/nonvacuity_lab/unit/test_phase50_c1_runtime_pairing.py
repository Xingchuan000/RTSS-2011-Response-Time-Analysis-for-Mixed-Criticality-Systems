from __future__ import annotations

import json
from pathlib import Path

from nonvacuity_lab.runners.paired_hout import _materialize_profile_inputs
from nonvacuity_lab.v2_runner import _v2_mutation_to_v1


def _write_inputs(tmp_path: Path) -> tuple[Path, Path]:
    runtime = tmp_path / "hout_runtime.json"
    runtime.write_text(
        json.dumps({
            "budget_increase_ratio": 0.02,
            "budget_decrease_ratio": 0.02,
            "horizon": 100,
        }),
        encoding="utf-8",
    )
    taskset = tmp_path / "taskset.json"
    taskset.write_text(json.dumps({"ordered_tasks": []}), encoding="utf-8")
    return runtime, taskset


def test_v2_c1_derives_mutated_runtime_override(tmp_path: Path):
    runtime, taskset = _write_inputs(tmp_path)
    mutation = {
        "mutation_id": "C1_action_ratio",
        "mutation_class": "ACTION_SEMANTICS",
        "seed": 185,
        "tree_variant": "best_overall",
        "hout_profile_id": "s185_h5",
        "mutator": {
            "kind": "action_step",
            "parameters": {
                "direction": "inc_only",
                "before_ratio": 0.02,
                "after_ratio": 0.05,
            },
        },
        "activation": {"mode": "hout"},
        "expected": {"allowed_result_statuses": ["DEPLOYED_TREE_PROVED"]},
    }
    config = {
        "hout_profiles": {
            "s185_h5": {
                "runtime_config_path": str(runtime),
                "taskset_path": str(taskset),
                "scenario_seeds": [1],
            }
        }
    }
    translated = _v2_mutation_to_v1(mutation, config=config, base_dir=tmp_path)
    assert translated["metadata"]["hout"]["mutated_runtime_overrides"] == {
        "budget_increase_ratio": 0.05
    }


def test_hout_materializes_distinct_base_and_mutated_runtime_configs(tmp_path: Path):
    runtime, taskset = _write_inputs(tmp_path)
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    materialized = _materialize_profile_inputs(
        {
            "scenario_seeds": [1, 2],
            "runtime_config_path": str(runtime),
            "taskset_path": str(taskset),
            "mutated_runtime_overrides": {"budget_increase_ratio": 0.05},
        },
        workspace_root=workspace,
    )
    base_path = Path(materialized["base_runtime_config"])
    mutated_path = Path(materialized["mutated_runtime_config"])
    assert base_path != mutated_path
    base = json.loads(base_path.read_text(encoding="utf-8"))
    mutated = json.loads(mutated_path.read_text(encoding="utf-8"))
    assert base["budget_increase_ratio"] == 0.02
    assert mutated["budget_increase_ratio"] == 0.05
    assert base["budget_decrease_ratio"] == mutated["budget_decrease_ratio"] == 0.02
