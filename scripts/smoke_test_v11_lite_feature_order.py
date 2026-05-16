"""v11 observation 消融特征顺序冒烟测试。

验证目标：
1. 所有新增 v11 消融模式的每任务特征确实来自 v11_full_10d 删除指定列；
2. 所有新增模式的全局 8 维特征与 v11_full_10d 完全一致；
3. 不重新定义同名特征的语义，只验证“固定顺序子集”这一核心约束。
"""

from __future__ import annotations

from amc_py.dqn.experiment import build_env_from_experiment_config, build_small_stress_experiment_config
from amc_py.rl.feature_config import FeatureConfig
from amc_py.runtime_models import RuntimeSemantics


def _build_env(*, observation_mode: str):
    """构造用于特征顺序检查的统一环境。"""

    return build_env_from_experiment_config(
        build_small_stress_experiment_config(),
        seed=0,
        end_time=120,
        agent_period=10,
        semantics=RuntimeSemantics.AMC_PLUS,
        action_space="single",
        reward_mode="mendes",
        feature_config=FeatureConfig(observation_mode=observation_mode),
    )


def _split_state(state_vector: tuple[float, ...], *, n_tasks: int, per_task_dim: int) -> tuple[list[list[float]], list[float]]:
    """把状态向量拆成每任务块与全局块，便于做列级对照。"""

    task_values = list(state_vector[: n_tasks * per_task_dim])
    global_values = list(state_vector[n_tasks * per_task_dim :])
    per_task_blocks = [task_values[idx * per_task_dim : (idx + 1) * per_task_dim] for idx in range(n_tasks)]
    return per_task_blocks, global_values


def _assert_close_list(left: list[float], right: list[float], *, label: str) -> None:
    """使用严格小阈值做浮点近似比较。"""

    if len(left) != len(right):
        raise AssertionError(f"{label} 长度不一致: {len(left)} != {len(right)}")
    for idx, (lval, rval) in enumerate(zip(left, right, strict=True)):
        if abs(float(lval) - float(rval)) >= 1e-9:
            raise AssertionError(f"{label} 第 {idx} 个值不一致: {lval} != {rval}")


def _extract(mode: str) -> tuple[list[list[float]], list[float], int]:
    """reset 同一个 seed，并返回拆分后的 per-task/global 特征。"""

    env = _build_env(observation_mode=mode)
    obs = env.reset(seed=0)
    n_tasks = len(env.ordered_tasks)
    per_task_dim = (len(obs.state_vector) - 8) // n_tasks if mode != "v10_basic" else 2
    per_task_blocks, global_values = _split_state(obs.state_vector, n_tasks=n_tasks, per_task_dim=per_task_dim)
    return per_task_blocks, global_values, n_tasks


def _run_case(*, mode: str, kept_indices: list[int]) -> None:
    """验证某个消融模式是否等于 v11_full_10d 的固定列子集。"""

    full_blocks, full_global, _ = _extract("v11_full_10d")
    lite_blocks, lite_global, _ = _extract(mode)

    for task_idx, (full_row, lite_row) in enumerate(zip(full_blocks, lite_blocks, strict=True)):
        expected_row = [full_row[idx] for idx in kept_indices]
        _assert_close_list(expected_row, lite_row, label=f"{mode} task[{task_idx}]")

    _assert_close_list(full_global, lite_global, label=f"{mode} global")
    print(f"{mode} feature order: PASS")


def main() -> None:
    """依次验证所有新增 v11 消融模式的特征顺序。"""

    _run_case(mode="v11_no_risk_9d", kept_indices=[0, 1, 2, 3, 4, 6, 7, 8, 9])
    _run_case(mode="v11_no_util_9d", kept_indices=[0, 1, 2, 3, 4, 5, 6, 7, 8])
    _run_case(mode="v11_no_max_9d", kept_indices=[0, 1, 2, 4, 5, 6, 7, 8, 9])
    _run_case(mode="v11_no_priority_9d", kept_indices=[0, 1, 2, 3, 4, 5, 6, 7, 9])
    _run_case(mode="v11_no_risk_no_util_8d", kept_indices=[0, 1, 2, 3, 4, 6, 7, 8])
    _run_case(mode="v11_lite_6d", kept_indices=[0, 1, 2, 4, 6, 7])


if __name__ == "__main__":
    main()
