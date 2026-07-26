"""Frozen formal-semantics binding and mutable-runtime decoupling tests."""

from pathlib import Path
import json
import shutil

from formal_toolchain.binding.action_binding import bind_action_runtime
from formal_toolchain.binding.event_runtime_binding import bind_event_runtime
from formal_toolchain.binding.observation_binding import bind_observation_runtime
from formal_toolchain.binding.removal_binding import bind_removal_runtime
from formal_toolchain.adapters.source_manifest import build_source_manifest
from formal_toolchain.adapters.tree_artifact import inspect_tree_artifact
from formal_toolchain.bridge.effect_compiler import compile_effect_ir
from formal_toolchain.bridge.model_bounds import _legacy_test_bounds
from formal_toolchain.bridge.runtime_branch_map import (
    PATH_SPECS,
    _path_row,
    build_normal_runtime_path_coverage,
    build_runtime_branch_map,
)
from formal_toolchain.bridge.transition_compiler import compile_source_guards
from formal_toolchain.core.hashing import sha256_object
from formal_toolchain.semantics.frozen_runtime_contract import (
    CONTRACT_VERSION,
    frozen_contract_manifest,
    frozen_contract_files,
)

ROOT = Path(__file__).parents[3]


def _copy_sources(tmp_path: Path) -> Path:
    shutil.copytree(ROOT / "formal_toolchain", tmp_path / "formal_toolchain")
    relatives = (
        "amc_py/event_models.py", "amc_py/event_runtime.py", "amc_py/runtime_models.py",
        "amc_py/runtime_scenarios.py", "amc_py/rl/env.py", "amc_py/rl/actions.py",
        "amc_py/rl/safety.py", "amc_py/rl/observation.py", "amc_py/rl/feature_state.py",
        "amc_py/rl/feature_config.py", "amc_py/viper/fixed_point.py",
        "amc_py/viper/integer_tree.py", "amc_py/viper/tree_policy.py",
    )
    for relative in relatives:
        source = ROOT / relative
        destination = tmp_path / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    return tmp_path


def _fresh_path_map(root: Path) -> dict:
    paths = {spec[0]: _path_row(root, spec) for spec in PATH_SPECS}
    coverage = build_normal_runtime_path_coverage(root)
    contract = frozen_contract_manifest(root)
    return {
        "schema_version": "phase_k_transition_path_map_v3_frozen_semantics",
        "source_hash": build_source_manifest(root)["semantic_hash"],
        "formal_semantics_contract_version": CONTRACT_VERSION,
        "formal_semantics_contract_hash": contract["semantic_hash"],
        "mutable_runtime_binding": "NON_BLOCKING_AUDIT_ONLY",
        "paths": paths,
        "coverage": coverage,
        "path_map_hash": sha256_object({
            "paths": paths,
            "coverage": coverage["artifact_hash"],
            "formal_semantics_contract_hash": contract["semantic_hash"],
        }),
    }


def test_real_source_binders_and_observation_pass():
    event = bind_event_runtime(ROOT)
    action = bind_action_runtime(ROOT)
    removal = bind_removal_runtime(ROOT)
    observation = bind_observation_runtime(
        ROOT,
        ROOT / "tests/formal/fixtures/synthetic_p0/feature_names.json",
        runtime_feature_names=[name for name in json.loads((
            ROOT / "tests/formal/fixtures/synthetic_p0/feature_names.json"
        ).read_text())["feature_names"]],
        ordered_tasks=["SYN_HI", "SYN_LO"],
        feature_task_order=["SYN_HI", "SYN_LO"],
    )
    assert [item["status"] for item in (event, action, removal, observation)] == ["PASS"] * 4


def test_c_amc_sem_hi_release_binds_frozen_non_response_path():
    fixture = _fresh_path_map(ROOT)
    branch_map = build_runtime_branch_map(
        ROOT,
        source_hash=fixture["source_hash"],
        path_map=fixture,
    )
    assert branch_map["status"] == "PASS"
    assert branch_map["mutable_runtime_binding"] == "NON_BLOCKING_AUDIT_ONLY"

    row = next(item for item in branch_map["paths"] if item["case_id"] == "HI_RELEASE")
    guards = {(item["test_source"], item["polarity"]) for item in row["guard_ir"]}
    assert ("task.criticality is Criticality.HI", True) in guards
    assert ("_is_c_amc_semantics(self.config.semantics)", True) in guards
    response_guard = (
        "_is_response_based_semantics(self.config.semantics) and "
        "task.criticality is Criticality.HI"
    )
    assert (response_guard, False) in guards

    bounds = _legacy_test_bounds()
    compiled_guard = compile_source_guards(row["guard_ir"])
    compiled_effect = compile_effect_ir(
        row["effect_ir"], bounds=bounds, guard_ir=row["guard_ir"]
    )
    assert "(= config_semantics 1)" in compiled_guard.formula
    assert "(= task_criticality 1)" in compiled_guard.formula
    assert "(= c_queue_event_count_post (+ c_queue_event_count 2))" in compiled_effect.queue_equations


def test_mutable_qamc_runtime_changes_are_non_blocking(tmp_path: Path):
    root = _copy_sources(tmp_path)
    before = build_source_manifest(root)
    assert bind_event_runtime(root)["status"] == "PASS"
    assert bind_removal_runtime(root)["status"] == "PASS"

    mutable_runtime = root / "amc_py/event_runtime.py"
    mutable_runtime.write_text(
        mutable_runtime.read_text(encoding="utf-8")
        + "\n# q-AMC experimental branch added after formal freeze\n",
        encoding="utf-8",
    )
    after = build_source_manifest(root)
    assert after["semantic_hash"] == before["semantic_hash"]
    assert after["implementation_audit_hash"] != before["implementation_audit_hash"]
    assert bind_event_runtime(root)["status"] == "PASS"
    assert bind_removal_runtime(root)["status"] == "PASS"
    assert all(_path_row(root, spec)["case_id"] == spec[2] for spec in PATH_SPECS)


def test_frozen_semantics_mutation_is_rejected(tmp_path: Path):
    root = _copy_sources(tmp_path)
    before = build_source_manifest(root)
    frozen_runtime = root / "formal_toolchain/semantics/frozen_c_amc_sem_event_runtime.py"
    frozen_runtime.write_text(
        frozen_runtime.read_text(encoding="utf-8").replace(
            "job.executed_time <= budget", "job.executed_time < budget", 1
        ),
        encoding="utf-8",
    )
    after = build_source_manifest(root)
    assert after["semantic_hash"] != before["semantic_hash"]
    assert bind_removal_runtime(root)["status"] in {"FAIL", "UNRESOLVED"}


def test_policy_and_fixed_point_mutations_remain_blocking(tmp_path: Path):
    root = _copy_sources(tmp_path)
    policy_path = root / "amc_py/viper/tree_policy.py"
    policy_path.write_text(
        policy_path.read_text().replace("return None, base", "return 0, base", 1),
        encoding="utf-8",
    )
    assert bind_action_runtime(root)["status"] in {"FAIL", "UNRESOLVED"}

    fixed_point = ROOT / "tests/formal/fixtures/synthetic_p0/fixed_point_config.json"
    mutated_fixed = tmp_path / "fixed_point_config.json"
    mutated_fixed.write_text(
        fixed_point.read_text().replace('"scale":1000', '"scale":999'),
        encoding="utf-8",
    )
    artifact = tmp_path / "artifact"
    artifact.mkdir()
    for name in (
        "artifact_manifest.json", "integer_tree.json", "feature_names.json",
        "action_definitions.json", "metadata.json",
    ):
        shutil.copy2(ROOT / "tests/formal/fixtures/synthetic_p0" / name, artifact / name)
    shutil.copy2(mutated_fixed, artifact / "fixed_point_config.json")
    try:
        inspect_tree_artifact(
            artifact,
            expected_state_dim=28,
            expected_action_dim=24,
            expected_seed=None,
        )
    except ValueError:
        pass
    else:
        raise AssertionError("fixed-point mutation 未被 artifact verifier 检测")
