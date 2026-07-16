"""对真实 amc_py 源码副本做绑定和语义 mutation 验收。"""

from pathlib import Path
import json
import shutil

from formal_toolchain.binding.action_binding import bind_action_runtime
from formal_toolchain.binding.event_runtime_binding import bind_event_runtime
from formal_toolchain.binding.observation_binding import bind_observation_runtime
from formal_toolchain.binding.removal_binding import bind_removal_runtime
from formal_toolchain.adapters.tree_artifact import inspect_tree_artifact
from formal_toolchain.bridge.effect_compiler import compile_effect_ir
from formal_toolchain.bridge.model_bounds import _legacy_test_bounds
from formal_toolchain.bridge.runtime_branch_map import build_runtime_branch_map
from formal_toolchain.bridge.transition_compiler import compile_source_guards

ROOT = Path(__file__).parents[3]


def _copy_sources(tmp_path: Path) -> Path:
    for relative in ("amc_py/event_models.py", "amc_py/event_runtime.py", "amc_py/rl/env.py",
                     "amc_py/rl/actions.py", "amc_py/viper/tree_policy.py",
                     "amc_py/rl/observation.py"):
        source = ROOT / relative
        destination = tmp_path / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    return tmp_path


def test_real_source_binders_and_observation_pass():
    root = ROOT
    event = bind_event_runtime(root)
    action = bind_action_runtime(root)
    removal = bind_removal_runtime(root)
    observation = bind_observation_runtime(root, ROOT / "tests/formal/fixtures/synthetic_p0/feature_names.json",
        runtime_feature_names=[name for name in __import__("json").loads((ROOT / "tests/formal/fixtures/synthetic_p0/feature_names.json").read_text())["feature_names"]],
        ordered_tasks=["SYN_HI", "SYN_LO"], feature_task_order=["SYN_HI", "SYN_LO"])
    assert [item["status"] for item in (event, action, removal, observation)] == ["PASS"] * 4


def test_c_amc_sem_hi_release_binds_non_response_path_and_two_queue_pushes():
    """回归锁定 C-AMC-sem HI release 的真实 CFG 和 timing summary。

    这里同时检查源码 guard 与 EffectIR，防止 demand 公式虽然
    正确，path 却退回 AMC-RA/AMC-RH 并多压入 response-expiry。
    """
    fixture = json.loads((
        ROOT / "tests/formal/fixtures/synthetic_p0/phase_k_case_map.json"
    ).read_text(encoding="utf-8"))
    branch_map = build_runtime_branch_map(
        ROOT, source_hash=fixture["source_hash"], path_map=fixture)
    assert branch_map["status"] == "PASS"

    row = next(item for item in branch_map["paths"] if item["case_id"] == "HI_RELEASE")
    guards = {(item["test_source"], item["polarity"]) for item in row["guard_ir"]}
    assert ("task.criticality is Criticality.HI", True) in guards
    assert ("_is_c_amc_semantics(self.config.semantics)", True) in guards
    response_guard = (
        "_is_response_based_semantics(self.config.semantics) and "
        "task.criticality is Criticality.HI"
    )
    assert (response_guard, False) in guards
    assert all("_schedule_response_time_expiry_for_hi_job" not in item["source"]
               for item in row["effect_ir"])

    bounds = _legacy_test_bounds()
    compiled_guard = compile_source_guards(row["guard_ir"])
    compiled_effect = compile_effect_ir(
        row["effect_ir"], bounds=bounds, guard_ir=row["guard_ir"])
    assert "(= config_semantics 1)" in compiled_guard.formula
    assert "(= task_criticality 1)" in compiled_guard.formula
    assert "(= c_queue_event_count_post (+ c_queue_event_count 2))" \
        in compiled_effect.queue_equations


def test_real_source_mutations_are_rejected(tmp_path: Path):
    root = _copy_sources(tmp_path)
    event_path = root / "amc_py/event_models.py"
    event_path.write_text(event_path.read_text().replace("EventType.JOB_COMPLETION: 1,", "EventType.JOB_COMPLETION: 6,"), encoding="utf-8")
    assert bind_event_runtime(root)["status"] == "FAIL"

    removal_path = root / "amc_py/event_runtime.py"
    removal_path.write_text(removal_path.read_text().replace("job.executed_time <= budget", "job.executed_time < budget", 1), encoding="utf-8")
    assert bind_removal_runtime(root)["status"] == "FAIL"

    policy_path = root / "amc_py/viper/tree_policy.py"
    policy_path.write_text(policy_path.read_text().replace("return None, base", "return 0, base", 1), encoding="utf-8")
    # action binder 的 IR/semantic summary 不得继续声称 fallback 是 None。
    assert bind_action_runtime(root)["status"] in {"FAIL", "UNRESOLVED"}

    primary_path = root / "amc_py/event_runtime.py"
    primary_path.write_text(primary_path.read_text().replace("c_amc_sem_primary_on_switch_time", "disabled_primary_on_switch_time"), encoding="utf-8")
    assert bind_event_runtime(root)["status"] in {"FAIL", "UNRESOLVED"}

    mismatch_path = root / "amc_py/rl/env.py"
    mismatch_path.write_text(mismatch_path.read_text().replace("self._actions", "self._different_actions", 1), encoding="utf-8")
    assert bind_action_runtime(root)["status"] in {"FAIL", "UNRESOLVED"}

    fixed_point = ROOT / "tests/formal/fixtures/synthetic_p0/fixed_point_config.json"
    mutated_fixed = tmp_path / "fixed_point_config.json"
    mutated_fixed.write_text(fixed_point.read_text().replace('"scale":1000', '"scale":999'), encoding="utf-8")
    artifact = tmp_path / "artifact"
    artifact.mkdir()
    for name in ("artifact_manifest.json", "integer_tree.json", "feature_names.json", "action_definitions.json", "metadata.json"):
        shutil.copy2(ROOT / "tests/formal/fixtures/synthetic_p0" / name, artifact / name)
    shutil.copy2(mutated_fixed, artifact / "fixed_point_config.json")
    try:
        inspect_tree_artifact(artifact, expected_state_dim=28, expected_action_dim=24, expected_seed=None)
    except ValueError:
        pass
    else:
        raise AssertionError("fixed-point mutation 未被 artifact verifier 检测")
