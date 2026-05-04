"""最小 DQN 训练 smoke 脚本。"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from amc_py.dqn import DqnBudgetAgent, DqnConfig, Transition
from amc_py.models import Criticality, Task
from amc_py.rl.env import AmcBudgetEnv
from amc_py.runtime_models import RuntimeConfig, RuntimeSemantics
from amc_py.runtime_scenarios import ExecutionScenario, make_table_scenario


def _build_small_taskset() -> list[Task]:
    """构造 smoke 训练使用的小任务集。"""

    return [
        Task("T1", period=10, deadline=10, c_lo=2, c_hi=3, criticality=Criticality.HI),
        Task("T2", period=15, deadline=15, c_lo=2, c_hi=2, criticality=Criticality.LO),
        Task("T3", period=20, deadline=20, c_lo=3, c_hi=3, criticality=Criticality.LO),
    ]


def _build_stress_scenario() -> ExecutionScenario:
    """构造可稳定触发 runtime 压力行为的场景。"""

    return make_table_scenario(
        actual_costs={
            ("T2", 0): 5,
            ("T2", 1): 5,
            ("T2", 2): 4,
            ("T2", 3): 5,
            ("T3", 1): 5,
            ("T1", 0): 3,
            ("T1", 2): 3,
        },
        default_hi="c_lo",
        default_lo="c_lo",
    )


def build_parser() -> argparse.ArgumentParser:
    """构建命令行参数解析器。"""

    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--end-time", type=int, default=100)
    parser.add_argument("--agent-period", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/dqn_smoke"))
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--gamma", type=float, default=0.99)
    return parser


def main() -> None:
    """运行最小 DQN 训练闭环并导出日志与模型。"""

    args = build_parser().parse_args()

    tasks = _build_small_taskset()
    scenario = _build_stress_scenario()
    runtime_config = RuntimeConfig(end_time=args.end_time, semantics=RuntimeSemantics.AMC_PLUS)
    env = AmcBudgetEnv(
        ordered_tasks=tasks,
        scenario=scenario,
        runtime_config=runtime_config,
        agent_period=args.agent_period,
        check_safety=True,
    )

    config = DqnConfig(
        gamma=args.gamma,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        # smoke 脚本需要在很少的 episode 内验证优化闭环，因此最小回放门槛与 batch 对齐。
        min_replay_size=args.batch_size,
        seed=args.seed,
        network_seed=args.seed,
        exploration_seed=args.seed,
        replay_seed=args.seed,
    )

    initial_obs = env.reset(seed=args.seed)
    agent = DqnBudgetAgent(
        observation_dim=len(initial_obs.state_vector),
        action_dim=env.action_space_size,
        config=config,
    )

    rows: list[dict[str, int | float | str | None]] = []
    global_step = 0

    for episode in range(args.episodes):
        obs = env.reset(seed=args.seed + episode)
        done = False
        while not done:
            mask = env.valid_action_mask() if hasattr(env, "valid_action_mask") else None
            valid_action_count = 0 if mask is None else sum(mask)
            masked_action_count = 0 if mask is None else len(mask) - valid_action_count
            action_id = agent.select_action_id(
                obs.state_vector,
                valid_action_mask=mask,
                training=True,
            )
            result = env.step(action_id)
            next_mask = env.valid_action_mask() if not result.done else tuple(False for _ in range(env.action_space_size))

            loss: float | None = None
            if action_id is not None:
                # 仅在存在合法动作时记录 DQN 样本，避免把 NoOp 映射为伪动作编号参与训练。
                transition = Transition(
                    state=obs.state_vector,
                    action_id=action_id,
                    reward=result.reward,
                    next_state=result.observation.state_vector,
                    done=result.done,
                    valid_action_mask=tuple(mask) if mask is not None else tuple(True for _ in range(env.action_space_size)),
                    next_valid_action_mask=next_mask,
                )
                agent.remember(transition)
                loss = agent.optimize_one_step()

            info = result.info
            rows.append(
                {
                    "episode": episode,
                    "step": global_step,
                    "sim_time": info.get("time"),
                    "reward": result.reward,
                    "loss": "" if loss is None else loss,
                    "epsilon": agent.current_epsilon,
                    "action_id": "" if action_id is None else action_id,
                    "accepted": info.get("accepted"),
                    "valid_action_count": valid_action_count,
                    "masked_action_count": masked_action_count,
                    "noop_due_to_no_valid_action": action_id is None,
                    "reject_reason": "no_valid_action" if action_id is None else info.get("reject_reason", ""),
                    "mode_changes": info.get("mode_changes"),
                    "lo_cancellations": info.get("lo_cancellations"),
                    "deadline_misses": info.get("deadline_misses"),
                }
            )

            obs = result.observation
            done = result.done
            global_step += 1

    args.output_dir.mkdir(parents=True, exist_ok=True)
    log_path = args.output_dir / "train_log.csv"
    model_path = args.output_dir / "model.pt"

    fieldnames = [
        "episode",
        "step",
        "sim_time",
        "reward",
        "loss",
        "epsilon",
        "action_id",
        "accepted",
        "valid_action_count",
        "masked_action_count",
        "noop_due_to_no_valid_action",
        "reject_reason",
        "mode_changes",
        "lo_cancellations",
        "deadline_misses",
    ]
    with log_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    agent.save(model_path)


if __name__ == "__main__":
    main()
