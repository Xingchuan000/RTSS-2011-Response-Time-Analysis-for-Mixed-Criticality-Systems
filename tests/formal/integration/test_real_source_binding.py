"""对真实 amc_py 源码副本做绑定和语义 mutation 验收。"""

from pathlib import Path
import shutil

from formal_toolchain.binding.action_binding import bind_action_runtime
from formal_toolchain.binding.event_runtime_binding import bind_event_runtime
from formal_toolchain.binding.observation_binding import bind_observation_runtime
from formal_toolchain.binding.removal_binding import bind_removal_runtime
from formal_toolchain.adapters.tree_artifact import inspect_tree_artifact

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
