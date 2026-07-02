"""VIPER 相关 CLI 共享的 mc_fairgen 参数 helper。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from amc_py.dqn import build_mc_fairgen_experiment_config


def add_mc_fairgen_args(parser: argparse.ArgumentParser) -> None:
    """向 CLI parser 注入完整的 mc_fairgen 参数集合。

    这里严格复用 DQN 训练/评估入口已经公开的参数命名与默认值，
    目的是让 VIPER dataset 采集、tree 训练、HOUT 评估三条链路始终使用同一套 workload 口径。
    """

    parser.add_argument("--mc-fairgen-mode", type=str, default="paper_learnable_headroom")
    parser.add_argument("--mc-fairgen-num-tasks", type=int, default=16)
    parser.add_argument("--mc-fairgen-hi-ratio", type=float, default=0.5)
    parser.add_argument("--mc-fairgen-period-source", type=str, default="automotive")
    parser.add_argument("--mc-fairgen-period-scale", type=int, default=100)
    parser.add_argument("--mc-fairgen-u-hi-lo-min", type=float, default=0.20)
    parser.add_argument("--mc-fairgen-u-hi-lo-max", type=float, default=0.35)
    parser.add_argument("--mc-fairgen-u-hi-hi-min", type=float, default=0.45)
    parser.add_argument("--mc-fairgen-u-hi-hi-max", type=float, default=0.70)
    parser.add_argument("--mc-fairgen-u-lo-lo-min", type=float, default=0.35)
    parser.add_argument("--mc-fairgen-u-lo-lo-max", type=float, default=0.60)
    parser.add_argument("--mc-fairgen-hi-budget-rho-min", type=float, default=0.55)
    parser.add_argument("--mc-fairgen-hi-budget-rho-max", type=float, default=0.75)
    parser.add_argument("--mc-fairgen-lo-budget-rho-min", type=float, default=0.05)
    parser.add_argument("--mc-fairgen-lo-budget-rho-max", type=float, default=0.25)
    parser.add_argument("--mc-fairgen-hi-overrun-prob", type=float, default=0.08)
    parser.add_argument("--mc-fairgen-lo-overrun-prob", type=float, default=0.40)
    parser.add_argument("--mc-fairgen-hi-overrun-factor-min", type=float, default=1.02)
    parser.add_argument("--mc-fairgen-hi-overrun-factor-max", type=float, default=1.25)
    parser.add_argument("--mc-fairgen-lo-overrun-factor-min", type=float, default=1.05)
    parser.add_argument("--mc-fairgen-lo-overrun-factor-max", type=float, default=1.80)


def mc_fairgen_args_to_dict(args: argparse.Namespace) -> dict[str, Any]:
    """把 CLI 中的 mc_fairgen 参数整理成稳定字典。

    统一的字典结构既用于 dataset manifest / run_config 落盘，
    也用于后续 preflight 一致性校验，避免每个入口各自手写一份字段列表。
    """

    return {
        "mc_fairgen_mode": args.mc_fairgen_mode,
        "mc_fairgen_num_tasks": args.mc_fairgen_num_tasks,
        "mc_fairgen_hi_ratio": args.mc_fairgen_hi_ratio,
        "mc_fairgen_period_source": args.mc_fairgen_period_source,
        "mc_fairgen_period_scale": args.mc_fairgen_period_scale,
        "mc_fairgen_u_hi_lo_min": args.mc_fairgen_u_hi_lo_min,
        "mc_fairgen_u_hi_lo_max": args.mc_fairgen_u_hi_lo_max,
        "mc_fairgen_u_hi_hi_min": args.mc_fairgen_u_hi_hi_min,
        "mc_fairgen_u_hi_hi_max": args.mc_fairgen_u_hi_hi_max,
        "mc_fairgen_u_lo_lo_min": args.mc_fairgen_u_lo_lo_min,
        "mc_fairgen_u_lo_lo_max": args.mc_fairgen_u_lo_lo_max,
        "mc_fairgen_hi_budget_rho_min": args.mc_fairgen_hi_budget_rho_min,
        "mc_fairgen_hi_budget_rho_max": args.mc_fairgen_hi_budget_rho_max,
        "mc_fairgen_lo_budget_rho_min": args.mc_fairgen_lo_budget_rho_min,
        "mc_fairgen_lo_budget_rho_max": args.mc_fairgen_lo_budget_rho_max,
        "mc_fairgen_hi_overrun_prob": args.mc_fairgen_hi_overrun_prob,
        "mc_fairgen_lo_overrun_prob": args.mc_fairgen_lo_overrun_prob,
        "mc_fairgen_hi_overrun_factor_min": args.mc_fairgen_hi_overrun_factor_min,
        "mc_fairgen_hi_overrun_factor_max": args.mc_fairgen_hi_overrun_factor_max,
        "mc_fairgen_lo_overrun_factor_min": args.mc_fairgen_lo_overrun_factor_min,
        "mc_fairgen_lo_overrun_factor_max": args.mc_fairgen_lo_overrun_factor_max,
    }


def build_mc_fairgen_config_from_args(args: argparse.Namespace):
    """从 argparse Namespace 构造 mc_fairgen experiment config。

    这里不引入任何计划外兜底逻辑，只把 CLI 已显式提供的字段逐个传给
    `build_mc_fairgen_experiment_config()`，从而保证最终 workload 分布完全可追踪。
    """

    return build_mc_fairgen_experiment_config(
        mode=args.mc_fairgen_mode,
        num_tasks=args.mc_fairgen_num_tasks,
        hi_ratio=args.mc_fairgen_hi_ratio,
        period_source=args.mc_fairgen_period_source,
        period_scale=args.mc_fairgen_period_scale,
        require_schedulable=args.require_schedulable,
        scenario_seed_offset=args.scenario_seed_offset,
        fixed_taskset_seed=args.fixed_taskset_seed,
        u_hi_lo_min=args.mc_fairgen_u_hi_lo_min,
        u_hi_lo_max=args.mc_fairgen_u_hi_lo_max,
        u_hi_hi_min=args.mc_fairgen_u_hi_hi_min,
        u_hi_hi_max=args.mc_fairgen_u_hi_hi_max,
        u_lo_lo_min=args.mc_fairgen_u_lo_lo_min,
        u_lo_lo_max=args.mc_fairgen_u_lo_lo_max,
        hi_budget_rho_min=args.mc_fairgen_hi_budget_rho_min,
        hi_budget_rho_max=args.mc_fairgen_hi_budget_rho_max,
        lo_budget_rho_min=args.mc_fairgen_lo_budget_rho_min,
        lo_budget_rho_max=args.mc_fairgen_lo_budget_rho_max,
        hi_overrun_prob=args.mc_fairgen_hi_overrun_prob,
        lo_overrun_prob=args.mc_fairgen_lo_overrun_prob,
        hi_overrun_factor_min=args.mc_fairgen_hi_overrun_factor_min,
        hi_overrun_factor_max=args.mc_fairgen_hi_overrun_factor_max,
        lo_overrun_factor_min=args.mc_fairgen_lo_overrun_factor_min,
        lo_overrun_factor_max=args.mc_fairgen_lo_overrun_factor_max,
    )


def build_workload_cli_config(args: argparse.Namespace) -> dict[str, Any]:
    """构造统一的 workload 参数落盘结构。

    这个结构被同时写入 dataset manifest 与 tree run_config，
    便于后续直接比对“teacher 采样分布”和“tree 训练/验证分布”是否一致。
    """

    workload_cli_config: dict[str, Any] = {
        "workload": args.workload,
        "scenario": args.scenario,
        "require_schedulable": args.require_schedulable,
        "scenario_seed_offset": args.scenario_seed_offset,
        "fixed_taskset_seed": args.fixed_taskset_seed,
    }
    if args.workload == "mc_fairgen":
        workload_cli_config.update(mc_fairgen_args_to_dict(args))
    return workload_cli_config


def assert_mc_fairgen_args_match_dataset(args: argparse.Namespace, dataset_dir: Path) -> None:
    """校验 tree 训练 CLI 与已有 dataset manifest 的 mc_fairgen 参数是否一致。

    该 preflight 只在 `mc_fairgen + initial_dataset` 路径上触发。
    一旦发现字段不一致，就立即报错，防止 teacher 数据采集、tree 训练、validation 选择
    在不同 workload 分布上悄悄混跑。
    """

    manifest_path = dataset_dir / "manifest.json"
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    dataset_config = manifest.get("workload_cli_config")
    if not isinstance(dataset_config, dict):
        raise ValueError(
            "initial_dataset 的 manifest.json 缺少 workload_cli_config，无法校验 mc_fairgen 参数一致性"
        )
    current_config = build_workload_cli_config(args)
    mismatches: list[str] = []
    for key, current_value in current_config.items():
        dataset_value = dataset_config.get(key)
        if dataset_value != current_value:
            mismatches.append(f"{key}: dataset={dataset_value!r}, current={current_value!r}")
    if mismatches:
        mismatch_text = "; ".join(mismatches)
        raise ValueError(f"mc_fairgen workload 参数与 initial_dataset 不一致: {mismatch_text}")
