"""正式 DQN 训练命令行入口。"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict
from pathlib import Path

from amc_py.dqn import (
    DqnBudgetAgent,
    DqnConfig,
    Transition,
    build_env_from_experiment_config,
    build_small_nominal_experiment_config,
    build_small_stress_experiment_config,
    resolve_experiment_bundle,
)
from amc_py.models import Task
from amc_py.runtime_models import RuntimeSemantics


def _parse_hidden_layers(raw_value: str | None) -> tuple[int, ...] | None:
    """将逗号分隔的隐藏层字符串解析为整数元组。"""

    if raw_value is None or raw_value == "":
        return None
    return tuple(int(part.strip()) for part in raw_value.split(",") if part.strip())


def _serialize_tasks(tasks: list[Task]) -> list[dict[str, int | str]]:
    """将任务集转换为可写入 JSON 的结构。"""

    return [
        {
            "name": task.name,
            "period": task.period,
            "deadline": task.deadline,
            "c_lo": task.c_lo,
            "c_hi": task.c_hi,
            "criticality": task.criticality.value,
        }
        for task in tasks
    ]


def build_parser() -> argparse.ArgumentParser:
    """构建正式训练 CLI 的命令行参数解析器。"""

    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--end-time", type=int, default=100)
    parser.add_argument("--agent-period", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=6)
    parser.add_argument("--hidden-layers", type=str, default=None)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--target-update-freq", type=int, default=5)
    parser.add_argument("--epsilon-start", type=float, default=1.0)
    parser.add_argument("--epsilon-end", type=float, default=0.05)
    parser.add_argument("--epsilon-decay-steps", type=int, default=1000)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/dqn_amc"))
    parser.add_argument("--checkpoint", type=int, default=0)
    parser.add_argument("--scenario", choices=["nominal", "stress"], default="stress")
    return parser


def main() -> None:
    """运行正式 DQN 训练并产出完整目录结构。"""

    args = build_parser().parse_args()
    experiment_config = (
        build_small_nominal_experiment_config()
        if args.scenario == "nominal"
        else build_small_stress_experiment_config()
    )
    bundle = resolve_experiment_bundle(experiment_config, args.seed)
    env = build_env_from_experiment_config(
        experiment_config,
        seed=args.seed,
        end_time=args.end_time,
        agent_period=args.agent_period,
        semantics=RuntimeSemantics.AMC_PLUS,
    )

    hidden_layers = _parse_hidden_layers(args.hidden_layers)
    config = DqnConfig(
        gamma=args.gamma,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        min_replay_size=args.batch_size,
        target_update_freq=args.target_update_freq,
        epsilon_start=args.epsilon_start,
        epsilon_end=args.epsilon_end,
        epsilon_decay_steps=args.epsilon_decay_steps,
        hidden_layers=hidden_layers,
        seed=args.seed,
    )

    initial_obs = env.reset(seed=args.seed)
    agent = DqnBudgetAgent(
        observation_dim=len(initial_obs.state_vector),
        action_dim=env.action_space_size,
        config=config,
        hidden_layers=hidden_layers,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = args.output_dir / "checkpoints"
    if args.checkpoint > 0:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, int | float | str | bool]] = []
    global_step = 0
    for episode in range(args.episodes):
        obs = env.reset(seed=args.seed + episode)
        done = False
        episode_reward = 0.0
        while not done:
            mask = env.valid_action_mask()
            valid_action_count = sum(mask)
            masked_action_count = len(mask) - valid_action_count
            action_id = agent.select_action_id(
                obs.state_vector,
                valid_action_mask=mask,
                training=True,
            )
            result = env.step(action_id)

            loss: float | None = None
            if action_id is not None:
                transition = Transition(
                    state=obs.state_vector,
                    action_id=action_id,
                    reward=result.reward,
                    next_state=result.observation.state_vector,
                    done=result.done,
                )
                agent.remember(transition)
                loss = agent.optimize_one_step()

            info = result.info
            accepted = bool(info.get("accepted"))
            rejected = action_id is not None and not accepted
            episode_reward += result.reward
            rows.append(
                {
                    "episode": episode,
                    "step": global_step,
                    "sim_time": info.get("time"),
                    "reward": result.reward,
                    "episode_reward": episode_reward,
                    "loss": "" if loss is None else loss,
                    "epsilon": agent.current_epsilon,
                    "action_id": "" if action_id is None else action_id,
                    "accepted": accepted,
                    "rejected": rejected,
                    "reject_reason": "no_valid_action" if action_id is None else info.get("reject_reason", ""),
                    "valid_action_count": valid_action_count,
                    "masked_action_count": masked_action_count,
                    "noop_due_to_no_valid_action": action_id is None,
                    "mode_changes": info.get("mode_changes"),
                    "lo_cancellations": info.get("lo_cancellations"),
                    "deadline_misses": info.get("deadline_misses"),
                }
            )

            obs = result.observation
            done = result.done
            global_step += 1

        if args.checkpoint > 0 and (episode + 1) % args.checkpoint == 0:
            agent.save(checkpoint_dir / f"model_episode_{episode + 1:04d}.pt")

    train_log_path = args.output_dir / "train_log.csv"
    model_path = args.output_dir / "model_final.pt"
    config_path = args.output_dir / "config.json"

    fieldnames = [
        "episode",
        "step",
        "sim_time",
        "reward",
        "episode_reward",
        "loss",
        "epsilon",
        "action_id",
        "accepted",
        "rejected",
        "reject_reason",
        "valid_action_count",
        "masked_action_count",
        "noop_due_to_no_valid_action",
        "mode_changes",
        "lo_cancellations",
        "deadline_misses",
    ]
    with train_log_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    agent.save(model_path)
    config_payload = {
        "dqn_config": asdict(config),
        "taskset_seed": args.seed,
        "scenario": args.scenario,
        "scenario_name": bundle.scenario.name,
        "normalization_bounds": {
            task_name: {"min_cost": bound.min_cost, "max_cost": bound.max_cost}
            for task_name, bound in bundle.normalization_bounds.items()
        },
        "action_space_size": env.action_space_size,
        "observation_dim": len(initial_obs.state_vector),
        "tasks": _serialize_tasks(list(bundle.ordered_tasks)),
        "runtime_config": {
            "end_time": args.end_time,
            "agent_period": args.agent_period,
            "semantics": RuntimeSemantics.AMC_PLUS.value,
        },
    }
    with config_path.open("w", encoding="utf-8") as f:
        json.dump(config_payload, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
