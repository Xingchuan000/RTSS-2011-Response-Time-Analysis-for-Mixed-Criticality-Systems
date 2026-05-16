"""Taskwise DQN 的结构化 smoke test。

本脚本按实现计划中的验收顺序覆盖四层检查：
1. TaskwiseDqnNetwork 的 shape test；
2. DqnBudgetAgent 的 create / save / load test；
3. 非法配置必须报错的 test；
4. 1 episode 端到端训练 smoke test。

脚本设计原则：
- 默认直接执行全部四层检查，不额外提供“自动降级/跳过训练”的兜底分支；
- 任一环节失败都抛异常并中断，确保 smoke test 真正具备守门意义；
- 所有断言都围绕 taskwise 第一版的明确边界，不扩展到其他 observation/action 组合。
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


def _run_network_shape_test() -> None:
    """检查 taskwise 网络输出维度是否与动作空间严格一致。"""

    net = TaskwiseDqnNetwork(
        task_count=12,
        per_task_feature_dim=10,
        global_feature_dim=8,
        action_dim=25,
        noop_action_id=24,
    )
    x = torch.zeros(3, 128)
    y = net(x)
    assert y.shape == (3, 25), f"TaskwiseDqnNetwork 输出形状错误：{tuple(y.shape)}"


def _run_agent_save_load_test() -> None:
    """检查 taskwise agent 能否正常创建、保存并恢复。"""

    config = DqnConfig(
        network_arch="taskwise",
        task_count=12,
        per_task_feature_dim=10,
        global_feature_dim=8,
    )
    agent = DqnBudgetAgent(
        observation_dim=128,
        action_dim=25,
        config=config,
        noop_action_id=24,
    )
    sample_state = tuple(float(value) for value in torch.zeros(128).tolist())
    sample_mask = tuple(True for _ in range(25))
    q_before = agent.policy_network(torch.zeros(1, 128, dtype=torch.float32)).detach().cpu()

    with TemporaryDirectory(prefix="taskwise_agent_save_load_") as tmp_dir:
        model_path = Path(tmp_dir) / "taskwise_agent.pt"
        agent.save(model_path)
        loaded = DqnBudgetAgent.load(model_path, device="cpu")
        q_after = loaded.policy_network(torch.zeros(1, 128, dtype=torch.float32)).detach().cpu()
        assert loaded.config.network_arch == "taskwise"
        assert loaded.observation_dim == 128
        assert loaded.action_dim == 25
        assert loaded.select_action_id(sample_state, valid_action_mask=sample_mask, training=False) is not None
        assert q_before.shape == q_after.shape == (1, 25)


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


def _run_invalid_config_tests() -> None:
    """检查 taskwise 的关键非法配置都会被拒绝。"""

    _assert_raises_value_error(
        lambda: TaskwiseDqnNetwork(
            task_count=12,
            per_task_feature_dim=10,
            global_feature_dim=8,
            action_dim=24,
            noop_action_id=23,
        ),
        "action_dim",
    )
    _assert_raises_value_error(
        lambda: DqnBudgetAgent(
            observation_dim=127,
            action_dim=25,
            config=DqnConfig(
                network_arch="taskwise",
                task_count=12,
                per_task_feature_dim=10,
                global_feature_dim=8,
            ),
            noop_action_id=24,
        ),
        "observation_dim",
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
            ),
            noop_action_id=23,
        ),
        "noop_action_id",
    )


def _run_training_smoke_test() -> None:
    """运行 1 episode 的端到端训练 smoke test，并校验关键产物。"""

    repo_root = _repo_root()
    with TemporaryDirectory(prefix="taskwise_training_smoke_") as tmp_dir:
        output_dir = Path(tmp_dir) / "outputs" / "smoke_taskwise_seed409"
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
        # 训练子进程需要继承当前解释器环境变量，只覆盖 `PYTHONPATH` 指向仓库根目录。
        # 这样既能复用当前 conda 环境中的 `python/torch`，也能确保脚本按源码目录导入本仓库包。
        env = os.environ.copy()
        env["PYTHONPATH"] = str(repo_root)
        subprocess.run(command, cwd=repo_root, env=env, check=True)

        config_path = output_dir / "config.json"
        model_best_path = output_dir / "model_best.pt"
        model_final_path = output_dir / "model_final.pt"
        train_metrics_path = output_dir / "train_metrics.csv"
        validation_summary_path = output_dir / "validation_unified_summary.csv"
        assert config_path.exists(), f"未生成 {config_path}"
        assert model_best_path.exists(), f"未生成 {model_best_path}"
        assert model_final_path.exists(), f"未生成 {model_final_path}"
        assert train_metrics_path.exists(), f"未生成 {train_metrics_path}"
        assert validation_summary_path.exists(), f"未生成 {validation_summary_path}"

        with config_path.open("r", encoding="utf-8") as f:
            config_payload = json.load(f)
        assert config_payload["network_arch"] == "taskwise"
        assert int(config_payload["observation_dim"]) == 128
        assert int(config_payload["action_space_size"]) == 25


def main() -> None:
    """顺序执行四层 smoke test。"""

    _run_network_shape_test()
    _run_agent_save_load_test()
    _run_invalid_config_tests()
    _run_training_smoke_test()
    print("Taskwise DQN smoke test: PASS")


if __name__ == "__main__":
    main()
