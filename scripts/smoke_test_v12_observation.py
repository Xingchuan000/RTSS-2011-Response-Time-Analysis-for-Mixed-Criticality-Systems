"""v12 observation 冒烟测试脚本。

验证目标：
1. v10/v11/v12 三种 observation_mode 的状态维度正确；
2. reset 与多步 step 后所有观测值都位于 [0, 1]。
"""

from __future__ import annotations

from amc_py.dqn.experiment import build_env_from_experiment_config, build_small_stress_experiment_config
from amc_py.rl.feature_config import FeatureConfig
from amc_py.runtime_models import RuntimeSemantics


def _build_env(*, observation_mode: str):
    """构造用于 observation 冒烟检查的统一环境。"""

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


def _first_valid_action_id(env) -> int:
    """返回当前时刻第一个合法动作 id。"""

    mask = env.valid_action_mask()
    return next(i for i, valid in enumerate(mask) if valid)


def _assert_unit_interval(state_vector: tuple[float, ...]) -> None:
    """断言观测向量全部位于 [0, 1]。"""

    if not all(0.0 <= float(value) <= 1.0 for value in state_vector):
        raise AssertionError("state_vector 存在不在 [0,1] 的特征值")


def _run_mode(mode: str) -> None:
    """执行单个模式的 reset + 多步 step 冒烟。"""

    env = _build_env(observation_mode=mode)
    obs = env.reset(seed=0)
    n_tasks = len(env.ordered_tasks)

    if mode == "v10_basic":
        expected_dim = 2 * n_tasks
    elif mode == "v11_full_10d":
        expected_dim = 10 * n_tasks + 8
    elif mode == "v12_full_14d":
        expected_dim = 14 * n_tasks + 8
    else:
        raise ValueError(f"unsupported mode: {mode}")

    if len(obs.state_vector) != expected_dim:
        raise AssertionError(f"{mode} reset 维度错误: got={len(obs.state_vector)}, expected={expected_dim}")
    _assert_unit_interval(obs.state_vector)

    for _ in range(5):
        if env._done:  # noqa: SLF001
            break
        action_id = _first_valid_action_id(env)
        result = env.step(action_id)
        if len(result.observation.state_vector) != expected_dim:
            raise AssertionError(
                f"{mode} step 维度错误: got={len(result.observation.state_vector)}, expected={expected_dim}"
            )
        _assert_unit_interval(result.observation.state_vector)

    print(f"{mode}: PASS")


def main() -> None:
    """依次执行三种 observation mode 的冒烟测试。"""

    for mode in ("v10_basic", "v11_full_10d", "v12_full_14d"):
        _run_mode(mode)


if __name__ == "__main__":
    main()
