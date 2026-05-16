"""Taskwise DQN v2 的结构化 smoke test。

本脚本严格围绕任务文档中的 v2 验收要求，覆盖四类检查：
1. `TaskwiseDqnNetwork` 在四种 v1/v2 组合下的输出 shape；
2. `taskwise-v2` agent 的 create / save / load；
3. 非法 v2 配置必须显式报错；
4. 打开 task embedding + action bias 后的 1 episode 端到端训练。

设计原则：
- 不做自动跳过、自动降级、自动切换配置的兜底逻辑；
- 任一断言失败都直接抛异常；
- 所有边界都锁定在 `single + explicit noop + v11_full_10d + taskwise`。
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory

import torch

from amc_py.dqn import DqnBudgetAgent, DqnConfig
from amc_py.dqn.network import TaskwiseDqnNetwork


def _repo_root() -> Path:
    """解析仓库根目录。"""

    return Path(__file__).resolve().parents[1]


def _assert_raises_value_error(factory, expected_message_fragment: str) -> None:
    """断言指定调用必须抛出 ValueError，且报错文本包含关键片段。"""

    try:
        factory()
    except ValueError as exc:
        message = str(exc)
        assert expected_message_fragment in message, (
            f"报错信息未包含预期片段：expected={expected_message_fragment!r}, got={message!r}"
        )
        return
    raise AssertionError("预期抛出 ValueError，但实际没有抛错")


def _run_network_shape_tests() -> None:
    """验证 v1/v2 四种组合都能输出固定的 `[batch, 25]`。"""

    variants = [
        ("taskwise_v1", False, False),
        ("taskwise_v2_task_embedding_only", True, False),
        ("taskwise_v2_action_bias_only", False, True),
        ("taskwise_v2_full", True, True),
    ]
    for variant_name, use_task_embedding, use_action_bias in variants:
        net = TaskwiseDqnNetwork(
            task_count=12,
            per_task_feature_dim=10,
            global_feature_dim=8,
            action_dim=25,
            noop_action_id=24,
            use_task_embedding=use_task_embedding,
            task_embedding_dim=8,
            use_action_bias=use_action_bias,
            action_bias_init=0.0,
        )
        x = torch.zeros(4, 128, dtype=torch.float32)
        y = net(x)
        assert y.shape == (4, 25), f"{variant_name} 输出形状错误：{tuple(y.shape)}"


def _run_agent_save_load_test() -> None:
    """验证 taskwise-v2 agent 保存/恢复后仍保留 embedding 与 action bias 结构。"""

    config = DqnConfig(
        network_arch="taskwise",
        task_count=12,
        per_task_feature_dim=10,
        global_feature_dim=8,
        taskwise_use_task_embedding=True,
        taskwise_task_embedding_dim=8,
        taskwise_use_action_bias=True,
        taskwise_action_bias_init=0.0,
    )
    agent = DqnBudgetAgent(
        observation_dim=128,
        action_dim=25,
        config=config,
        noop_action_id=24,
        device="cpu",
    )
    sample_input = torch.zeros(1, 128, dtype=torch.float32)
    q_before = agent.policy_network(sample_input).detach().cpu()

    with TemporaryDirectory(prefix="taskwise_v2_agent_save_load_") as tmp_dir:
        model_path = Path(tmp_dir) / "taskwise_v2_agent.pt"
        agent.save(model_path)
        loaded = DqnBudgetAgent.load(model_path, device="cpu")
        q_after = loaded.policy_network(sample_input).detach().cpu()

        assert loaded.config.network_arch == "taskwise"
        assert loaded.config.taskwise_use_task_embedding is True
        assert loaded.config.taskwise_use_action_bias is True
        assert isinstance(loaded.policy_network, TaskwiseDqnNetwork)
        assert loaded.policy_network.use_task_embedding is True
        assert loaded.policy_network.use_action_bias is True
        assert loaded.policy_network.task_embedding is not None
        assert loaded.policy_network.action_bias is not None
        assert q_before.shape == q_after.shape == (1, 25)


def _run_invalid_config_tests() -> None:
    """验证 v2 关键非法配置不会被静默接受。"""

    _assert_raises_value_error(
        lambda: TaskwiseDqnNetwork(
            task_count=12,
            per_task_feature_dim=10,
            global_feature_dim=8,
            action_dim=25,
            noop_action_id=24,
            use_task_embedding=True,
            task_embedding_dim=0,
        ),
        "task_embedding_dim",
    )
    _assert_raises_value_error(
        lambda: DqnBudgetAgent(
            observation_dim=128,
            action_dim=25,
            config=DqnConfig(
                network_arch="taskwise",
                task_count=12,
                per_task_feature_dim=10,
                global_feature_dim=8,
                taskwise_use_task_embedding=True,
                taskwise_task_embedding_dim=0,
            ),
            noop_action_id=24,
            device="cpu",
        ),
        "taskwise_task_embedding_dim",
    )


def _run_training_smoke_test() -> None:
    """运行 1 episode 端到端训练，并检查 v2 配置与产物是否写出。"""

    repo_root = _repo_root()
    with TemporaryDirectory(prefix="taskwise_v2_training_smoke_") as tmp_dir:
        output_dir = Path(tmp_dir) / "outputs" / "smoke_taskwise_v2_seed409"
        command = [
            sys.executable,
            "-u",
            "scripts/train_dqn_amc.py",
            "--workload",
            "mc_fairgen",
            "--mc-fairgen-mode",
            "paper_learnable_headroom",
            "--mc-fairgen-num-tasks",
            "12",
            "--mc-fairgen-hi-ratio",
            "0.5",
            "--mc-fairgen-period-source",
            "automotive",
            "--fixed-taskset-seed",
            "409",
            "--train-seed-mode",
            "per-episode",
            "--episodes",
            "1",
            "--end-time",
            "100000",
            "--agent-period",
            "50000",
            "--validation-seeds",
            "200:201",
            "--validate-every",
            "1",
            "--validation-end-time",
            "100000",
            "--validation-workers",
            "1",
            "--checkpoint",
            "1",
            "--save-best-by",
            "pareto_relative_score",
            "--reward-mode",
            "interval_v1",
            "--action-space",
            "single",
            "--budget-increase-ratio",
            "0.025",
            "--budget-decrease-ratio",
            "0.015",
            "--include-explicit-noop",
            "--budget-floor-ratio",
            "0.9",
            "--observation-mode",
            "v11_full_10d",
            "--network-arch",
            "taskwise",
            "--taskwise-use-task-embedding",
            "--taskwise-task-embedding-dim",
            "8",
            "--taskwise-use-action-bias",
            "--taskwise-action-bias-init",
            "0.0",
            "--ema-alpha",
            "0.2",
            "--overrun-ema-alpha",
            "0.1",
            "--history-k",
            "8",
            "--event-window",
            "10",
            "--max-cost-weight",
            "0.7",
            "--risk-max-scale",
            "3.0",
            "--include-safety-margin",
            "--output-dir",
            str(output_dir),
        ]
        env = os.environ.copy()
        env["PYTHONPATH"] = str(repo_root)
        subprocess.run(command, cwd=repo_root, env=env, check=True)

        config_path = output_dir / "config.json"
        model_best_path = output_dir / "model_best.pt"
        model_final_path = output_dir / "model_final.pt"
        train_metrics_path = output_dir / "train_metrics.csv"
        validation_metrics_path = output_dir / "validation_metrics.csv"
        validation_summary_path = output_dir / "validation_unified_summary.csv"
        assert config_path.exists(), f"未生成 {config_path}"
        assert model_best_path.exists(), f"未生成 {model_best_path}"
        assert model_final_path.exists(), f"未生成 {model_final_path}"
        assert train_metrics_path.exists(), f"未生成 {train_metrics_path}"
        assert validation_metrics_path.exists(), f"未生成 {validation_metrics_path}"
        assert validation_summary_path.exists(), f"未生成 {validation_summary_path}"

        with config_path.open("r", encoding="utf-8") as f:
            config_payload = json.load(f)
        assert config_payload["network_arch"] == "taskwise"
        assert config_payload["taskwise_use_task_embedding"] is True
        assert config_payload["taskwise_task_embedding_dim"] == 8
        assert config_payload["taskwise_use_action_bias"] is True
        assert config_payload["taskwise_action_bias_init"] == 0.0
        assert int(config_payload["observation_dim"]) == 128
        assert int(config_payload["action_space_size"]) == 25


def main() -> None:
    """顺序执行 taskwise-v2 smoke test。"""

    _run_network_shape_tests()
    _run_agent_save_load_test()
    _run_invalid_config_tests()
    _run_training_smoke_test()
    print("Taskwise DQN v2 smoke test: PASS")


if __name__ == "__main__":
    main()
