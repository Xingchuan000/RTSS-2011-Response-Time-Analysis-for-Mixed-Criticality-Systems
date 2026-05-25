"""导出 action-aware 模型在评估轨迹上的 Q 排名诊断。"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch

from amc_py.dqn import DqnBudgetAgent, build_env_from_experiment_config, build_mc_fairgen_experiment_config
from amc_py.rl.actions import build_budget_action_space
from amc_py.rl.feature_config import FeatureConfig
from amc_py.runtime_models import RuntimeSemantics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--end-time", type=int, default=1_000_000)
    parser.add_argument("--agent-period", type=int, default=25000)
    args = parser.parse_args()

    exp = build_mc_fairgen_experiment_config(
        mode="paper_learnable_headroom",
        period_source="controlled_medium",
        period_scale=500,
        fixed_taskset_seed=args.seed,
    )
    env = build_env_from_experiment_config(
        exp,
        seed=args.seed,
        end_time=args.end_time,
        agent_period=args.agent_period,
        semantics=RuntimeSemantics.AMC_PLUS,
        reward_mode="interval_qos_v2",
        action_space="single",
        budget_increase_ratio=0.025,
        budget_decrease_ratio=0.015,
        include_explicit_noop=True,
        budget_floor_ratio=0.9,
        mask_detail_mode="minimal",
        feature_config=FeatureConfig(observation_mode="v11_full_10d"),
    )
    agent = DqnBudgetAgent.load(args.model)

    obs = env.reset(seed=args.seed)
    rows: list[dict[str, int | float | str]] = []
    done = False
    while not done:
        mask = env.valid_action_mask()
        if agent.q_network_type == "action_aware":
            features = env.get_action_feature_matrix(agent.action_feature_mode)
            names = env.get_action_feature_names(agent.action_feature_mode)
            agent.set_action_features(features, names)
        state = torch.tensor([obs.state_vector], dtype=torch.float32, device=agent.device)
        with torch.no_grad():
            q_values = agent._network_q_values(agent.policy_network, state)[0].detach().cpu().tolist()  # noqa: SLF001
        valid_ids = [idx for idx, ok in enumerate(mask) if ok]
        ranked = sorted(valid_ids, key=lambda idx: q_values[idx], reverse=True)
        action_id = agent.select_action_id(obs.state_vector, valid_action_mask=mask, training=False)
        q_top1 = q_values[ranked[0]] if ranked else 0.0
        q_top2 = q_values[ranked[1]] if len(ranked) > 1 else q_top1
        rows.append(
            {
                "time": int(env._engine.current_time if env._engine is not None else 0),  # noqa: SLF001
                "top1_action": int(ranked[0]) if ranked else -1,
                "top2_action": int(ranked[1]) if len(ranked) > 1 else -1,
                "top3_action": int(ranked[2]) if len(ranked) > 2 else -1,
                "q_top1": float(q_top1),
                "q_top2": float(q_top2),
                "q_margin_top1_top2": float(q_top1 - q_top2),
                "q_action7": float(q_values[7]) if len(q_values) > 7 else 0.0,
                "q_action8": float(q_values[8]) if len(q_values) > 8 else 0.0,
                "q_action9": float(q_values[9]) if len(q_values) > 9 else 0.0,
                "q_action10": float(q_values[10]) if len(q_values) > 10 else 0.0,
                "q_action11": float(q_values[11]) if len(q_values) > 11 else 0.0,
                "q_noop": float(q_values[0]) if q_values else 0.0,
                "rank_action7": int(ranked.index(7) + 1) if 7 in ranked else -1,
                "rank_noop": int(ranked.index(0) + 1) if 0 in ranked else -1,
                "valid_action_count": len(valid_ids),
                "selected_action": int(action_id) if action_id is not None else -1,
            }
        )
        result = env.step(action_id)
        obs = result.observation
        done = result.done

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["time"])
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
