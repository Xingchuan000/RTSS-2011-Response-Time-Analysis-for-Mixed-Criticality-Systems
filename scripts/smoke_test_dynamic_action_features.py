"""dynamic_v1 动作特征最小冒烟测试。"""

from __future__ import annotations

import numpy as np

from amc_py.dqn.experiment import build_env_from_experiment_config, build_small_stress_experiment_config
from amc_py.runtime_models import RuntimeSemantics


def main() -> None:
    """验证 dynamic_v1 特征形状、有限性与 step 后可继续计算。"""

    cfg = build_small_stress_experiment_config()
    env = build_env_from_experiment_config(
        cfg,
        seed=123,
        end_time=100,
        agent_period=10,
        semantics=RuntimeSemantics.AMC_PLUS,
        action_space="single",
        include_explicit_noop=True,
    )
    obs = env.reset(seed=123)
    assert len(obs.state_vector) > 0

    names = env.get_action_feature_names("dynamic_v1")
    f0 = env.get_action_feature_matrix("dynamic_v1")
    assert len(f0) == env.action_space_size
    assert all(len(row) == len(names) for row in f0)
    assert np.isfinite(np.asarray(f0, dtype=np.float64)).all()

    mask = env.valid_action_mask()
    action_id = next((idx for idx, valid in enumerate(mask) if valid), None)
    assert action_id is not None
    result = env.step(action_id)

    if not result.done:
        f1 = env.get_action_feature_matrix("dynamic_v1")
        assert len(f1) == len(f0)
        assert all(len(row) == len(names) for row in f1)
        assert np.isfinite(np.asarray(f1, dtype=np.float64)).all()

    print("dynamic action features smoke ok")


if __name__ == "__main__":
    main()
