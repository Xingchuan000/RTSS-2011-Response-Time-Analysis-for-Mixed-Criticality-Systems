"""Action-aware Q network 最小闭环 smoke 测试。"""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

import torch

from amc_py.dqn import ActionAwareQNetwork, DqnBudgetAgent, DqnConfig, Transition
from amc_py.dqn.experiment import build_env_from_experiment_config, build_small_stress_experiment_config
from amc_py.runtime_models import RuntimeSemantics


def main() -> None:
    """执行 action-aware 网络、环境特征、agent 训练与存取闭环测试。"""

    anet = ActionAwareQNetwork(observation_dim=10, action_feature_dim=7, hidden_layers=(16, 16))
    assert anet(torch.zeros((2, 10)), torch.zeros((5, 7))).shape == (2, 5)
    assert anet(torch.zeros((2, 10)), torch.zeros((2, 5, 7))).shape == (2, 5)

    cfg = build_small_stress_experiment_config()
    env = build_env_from_experiment_config(
        cfg,
        seed=0,
        end_time=100,
        agent_period=10,
        semantics=RuntimeSemantics.AMC_PLUS,
        action_space="single",
        include_explicit_noop=True,
    )
    obs = env.reset(seed=0)
    action_features = env.get_action_feature_matrix("static_v1")
    action_feature_names = env.get_action_feature_names("static_v1")

    agent = DqnBudgetAgent(
        observation_dim=len(obs.state_vector),
        action_dim=env.action_space_size,
        config=DqnConfig(
            q_network_type="action_aware",
            action_feature_mode="static_v1",
            hidden_layers=(8, 8),
            min_replay_size=2,
            batch_size=2,
        ),
        action_features=action_features,
        action_feature_names=action_feature_names,
    )

    mask = env.valid_action_mask()
    action_id = agent.select_action_id(obs.state_vector, valid_action_mask=mask, training=False)
    assert action_id is not None

    for _ in range(3):
        agent.remember(
            Transition(
                state=obs.state_vector,
                action_id=int(action_id),
                reward=1.0,
                next_state=obs.state_vector,
                done=False,
                valid_action_mask=tuple(mask),
                next_valid_action_mask=tuple(mask),
            )
        )
    loss = agent.optimize_one_step()
    assert loss is not None

    with TemporaryDirectory() as tmpdir:
        model_path = Path(tmpdir) / "model.pt"
        agent.save(model_path)
        loaded = DqnBudgetAgent.load(model_path)
        loaded.set_action_features(action_features, action_feature_names)
        action_id_loaded = loaded.select_action_id(obs.state_vector, valid_action_mask=mask, training=False)
        assert action_id_loaded is not None

    print("action-aware q network smoke ok")


if __name__ == "__main__":
    main()
