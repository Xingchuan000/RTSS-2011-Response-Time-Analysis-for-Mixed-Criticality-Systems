"""正式 DQN 训练命令行入口。"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from pathlib import Path
from tempfile import TemporaryDirectory

import torch

from amc_py.dqn import (
    DqnBudgetAgent,
    DqnConfig,
    ExperimentConfig,
    Transition,
    build_automotive_experiment_config,
    build_mc_fairgen_experiment_config,
    build_env_from_experiment_config,
    build_rtss11_experiment_config,
    build_small_nominal_experiment_config,
    build_small_stress_experiment_config,
    resolve_experiment_bundle,
)
from amc_py.event_runtime import simulate_ordered_taskset_event_driven
from amc_py.models import Task
from amc_py.rl.feature_config import FeatureConfig
from amc_py.rl.reward_config import available_reward_modes, load_reward_mode_config
from amc_py.runtime_models import RuntimeConfig, RuntimeSemantics, SimulationResult


STEP_LOG_FIELDNAMES = [
    "episode",
    "step",
    "sim_time",
    "reward",
    "episode_reward",
    "total_reward",
    "loss",
    "epsilon",
    "action_id",
    "accepted",
    "rejected",
    "reject_reason",
    "valid_action_count",
    "masked_action_count",
    "noop_due_to_no_valid_action",
    "is_noop",
    "is_explicit_noop",
    "mode_changes",
    "lo_cancellations",
    "deadline_misses",
    "step_reward_total",
    "step_reward_job_start",
    "step_reward_lo_overrun",
    "step_reward_hi_overrun",
    "step_reward_mode_change",
    "step_reward_lo_cancellation",
    "step_reward_deadline_miss",
    "step_reward_invalid_action",
    "paper_reward",
    "noop_reward_bonus",
    "lo_overrun_rate",
    "hi_overrun_rate",
    "mode_change_per_job",
    "lo_cancellation_rate",
    "deadline_miss_rate",
    "invalid_action",
    "budget_change_norm",
    "budget_change_penalty_value",
    "budget_drift_mean",
    "budget_drift_penalty_value",
    "lo_pressure_mean",
    "lo_pressure_max",
    "lo_near_cancel_rate",
    "hi_mode_pressure_mean",
    "lo_pressure_penalty_value",
    "lo_pressure_max_penalty_value",
    "lo_near_cancel_penalty_value",
    "hi_mode_pressure_penalty_value",
    "reward_after_regularization",
    "workload",
    "total_util",
    "num_tasks",
    "cf",
    "cp",
    "taskset_seed",
    "scenario_seed",
    "require_schedulable",
    "observation_mode",
    "state_dim",
]

NOOP_Q_DIAGNOSTIC_FIELDNAMES = [
    "noop_q_mean",
    "noop_q_std",
    "noop_q_rank_mean",
    "noop_q_rank_median",
    "noop_q_rank_min",
    "noop_q_rank_max",
    "noop_q_margin_to_best_mean",
    "noop_q_is_best_rate",
    "noop_valid_rate",
    "noop_q_sample_count",
]


def _parse_hidden_layers(raw_value: str | None) -> tuple[int, ...] | None:
    """将逗号分隔的隐藏层字符串解析为整数元组。"""

    if raw_value is None or raw_value == "":
        return None
    return tuple(int(part.strip()) for part in raw_value.split(",") if part.strip())


def _parse_seed_spec(raw_value: str) -> list[int]:
    """解析 seed 规格，支持 `0:9` 与 `0,1,2`。"""

    text = raw_value.strip()
    if not text:
        return []
    seeds: list[int] = []
    for part in (item.strip() for item in text.split(",")):
        if not part:
            continue
        if ":" in part:
            begin_text, end_text = (token.strip() for token in part.split(":", maxsplit=1))
            begin = int(begin_text)
            end = int(end_text)
            if end < begin:
                raise ValueError(f"seed 区间必须满足 begin<=end，收到: {part}")
            seeds.extend(range(begin, end + 1))
        else:
            seeds.append(int(part))
    return seeds


def _build_episode_seed_schedule(
    *,
    episodes: int,
    seed: int,
    mode: str,
    train_seeds: list[int],
) -> list[int]:
    """构建每个 episode 使用的 seed 调度表。"""

    if episodes <= 0:
        raise ValueError("episodes 必须为正整数")
    if mode == "fixed":
        return [seed for _ in range(episodes)]
    if mode == "per-episode":
        return [seed + episode for episode in range(episodes)]
    if mode == "cycle":
        if not train_seeds:
            raise ValueError("train-seed-mode=cycle 时，--train-seeds 不能为空")
        return [train_seeds[episode % len(train_seeds)] for episode in range(episodes)]
    raise ValueError(f"不支持的 train-seed-mode: {mode}")


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


def _get_noop_action_id(env) -> int | None:
    """从环境动作空间中解析显式 noop 的 action_id。"""

    for action in env._actions:  # noqa: SLF001
        if bool(getattr(action, "is_noop", False)):
            return int(action.action_id)
    return None


def _noop_q_diagnostics_to_row(agent: DqnBudgetAgent, states: list[tuple[float, ...]], masks: list[tuple[bool, ...]]) -> dict[str, float | int | None]:
    """把采集到的 validation 决策状态转换为 CSV 可写字段。

    `states` 与 `masks` 只来自 agent 做 greedy 决策前的同一时刻：
    - state 表示 policy network 实际看到的 observation；
    - mask 表示同一 observation 下环境允许的合法动作集合。

    这里不对 Q 值结果做额外修正，只调用 `DqnBudgetAgent.compute_noop_q_diagnostics()`，
    保持文档要求的诊断口径集中在 agent 内部。
    """

    if states:
        state_tensor = torch.tensor(states, dtype=torch.float32, device=agent.device)
        mask_tensor = torch.tensor(masks, dtype=torch.bool, device=agent.device)
    else:
        state_tensor = torch.empty((0, agent.observation_dim), dtype=torch.float32, device=agent.device)
        mask_tensor = torch.empty((0, agent.action_dim), dtype=torch.bool, device=agent.device)
    diagnostics = agent.compute_noop_q_diagnostics(state_tensor, mask_tensor)
    return {
        "noop_q_mean": diagnostics.noop_q_mean,
        "noop_q_std": diagnostics.noop_q_std,
        "noop_q_rank_mean": diagnostics.noop_q_rank_mean,
        "noop_q_rank_median": diagnostics.noop_q_rank_median,
        "noop_q_rank_min": diagnostics.noop_q_rank_min,
        "noop_q_rank_max": diagnostics.noop_q_rank_max,
        "noop_q_margin_to_best_mean": diagnostics.noop_q_margin_to_best_mean,
        "noop_q_is_best_rate": diagnostics.noop_q_is_best_rate,
        "noop_valid_rate": diagnostics.noop_valid_rate,
        "noop_q_sample_count": diagnostics.sample_count,
    }


def _mean_optional_metric(rows: list[dict[str, int | float | None]], key: str) -> float | None:
    """对可能为空的 noop Q 诊断字段求均值。

    baseline 行没有 DQN Q 值，且没有显式 noop 或 noop 全部无效时相关字段会是 None。
    因此聚合时只对真实数值求均值；若所有 seed 都为空，则继续输出 None。
    """

    values = [float(row[key]) for row in rows if row.get(key) is not None]
    if not values:
        return None
    return sum(values) / len(values)


def _percentile(values: list[float], q: float) -> float:
    """计算分位数（线性插值），用于 episode 级特征诊断。

    这里不引入额外依赖，直接用稳定的线性插值实现：
    - q 取值范围是 [0, 1]；
    - 当样本为空时返回 0.0，保持 CSV 字段始终可写。
    """

    if not values:
        return 0.0
    if q <= 0.0:
        return float(min(values))
    if q >= 1.0:
        return float(max(values))
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * q
    lo = int(position)
    hi = min(lo + 1, len(ordered) - 1)
    weight = position - float(lo)
    return float(ordered[lo] * (1.0 - weight) + ordered[hi] * weight)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    """写入 jsonl 文件。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _trace_rows_from_runtime(result: SimulationResult) -> list[dict]:
    """将 runtime 结果转换为逐行 trace。"""

    rows: list[dict] = []
    for tick in result.trace:
        rows.append(
            {
                "event": "schedule_tick",
                "time": tick.time,
                "executing_task": tick.executing_task,
                "executing_release_index": tick.executing_release_index,
                "mode": tick.mode.name,
            }
        )
    rows.extend(result.debug_events)
    for miss in result.deadline_misses:
        rows.append(
            {
                "event": "deadline_miss",
                "task": miss.task,
                "release_index": miss.release_index,
                "release_time": miss.release_time,
                "absolute_deadline": miss.absolute_deadline,
                "mode_at_miss": miss.mode_at_miss.name,
                "executed_at_miss": miss.executed_at_miss,
            }
        )
    return rows


def _run_baseline_validation_seed_worker(args_tuple: tuple[ExperimentConfig, int, int]) -> dict[str, int]:
    """运行单个 validation seed 的 AMC+ baseline 仿真。

    这里必须使用模块顶层函数，而不能把逻辑写成 `_run_validation()` 的内部闭包。
    原因是 macOS 的 multiprocessing 默认采用 `spawn` 启动方式，
    子进程只能 pickle 并导入模块顶层对象；嵌套函数通常无法被子进程恢复。
    """

    experiment_config, seed, validation_end_time = args_tuple
    bundle = resolve_experiment_bundle(experiment_config, seed)
    baseline_result = simulate_ordered_taskset_event_driven(
        ordered_tasks=list(bundle.ordered_tasks),
        scenario=bundle.scenario,
        config=RuntimeConfig(
            end_time=validation_end_time,
            semantics=RuntimeSemantics.AMC_PLUS,
        ),
    )
    # worker 只返回单个 seed 的原始计数，主进程统一负责做均值聚合，
    # 这样可以保证串行路径和并行路径共用同一套聚合口径。
    return {
        "mode_changes": baseline_result.mode_change_count(),
        "lo_cancellations": baseline_result.lo_job_cancellation_count(),
    }


def _evaluate_agent_on_validation_seed(
    *,
    agent: DqnBudgetAgent,
    experiment_config: ExperimentConfig,
    seed: int,
    validation_end_time: int,
    agent_period: int,
    reward_mode: str,
    action_space: str,
    budget_increase_ratio: float,
    budget_decrease_ratio: float,
    include_explicit_noop: bool,
    budget_floor_ratio: float,
    forbid_decreasing_hi_budgets: bool,
    mask_detail_mode: str,
    feature_config: FeatureConfig,
    max_q_diagnostic_samples: int,
    constraint_guided_pair_top_k_risk: int,
    constraint_guided_pair_top_k_decrease: int,
    constraint_guided_pair_prefer_lo: bool,
    constraint_guided_pair_include_hi_risk_boost: bool,
    constraint_guided_pair_allow_increase_only_when_safe: bool,
) -> dict[str, int | float | None]:
    """评估一个 validation seed，并返回该 seed 的完整 DQN 指标。

    这个 helper 是串行路径与并行路径共享的唯一统计实现：
    - 串行路径直接复用主进程中的 `agent`；
    - 并行路径在 worker 中先从磁盘加载 CPU agent，再调用本函数。

    这样做的目的是让两条路径在动作选择、环境推进、指标累加和 rate 计算上
    保持严格一致，避免未来只改了一条路径导致 validation 口径漂移。
    """

    env = build_env_from_experiment_config(
        experiment_config,
        seed=seed,
        end_time=validation_end_time,
        agent_period=agent_period,
        semantics=RuntimeSemantics.AMC_PLUS,
        reward_mode=reward_mode,
        action_space=action_space,
        budget_increase_ratio=budget_increase_ratio,
        budget_decrease_ratio=budget_decrease_ratio,
        include_explicit_noop=include_explicit_noop,
        budget_floor_ratio=budget_floor_ratio,
        forbid_decreasing_hi_budgets=forbid_decreasing_hi_budgets,
        mask_detail_mode=mask_detail_mode,
        feature_config=feature_config,
        constraint_guided_pair_top_k_risk=constraint_guided_pair_top_k_risk,
        constraint_guided_pair_top_k_decrease=constraint_guided_pair_top_k_decrease,
        constraint_guided_pair_prefer_lo=constraint_guided_pair_prefer_lo,
        constraint_guided_pair_include_hi_risk_boost=constraint_guided_pair_include_hi_risk_boost,
        constraint_guided_pair_allow_increase_only_when_safe=constraint_guided_pair_allow_increase_only_when_safe,
    )
    obs = env.reset(seed=seed)
    done = False
    # 下面这批变量全部是“单个 validation seed 的原始计数器”，
    # 最终由 `_run_validation()` 跨 seed 做统一均值聚合。
    accepted_actions = 0
    rejected_actions = 0
    step_count = 0
    selected_action_count = 0
    noop_actions = 0
    explicit_noop_actions = 0
    total_reward = 0.0
    last_info: dict[str, int | float | str | bool | None] = {
        "mode_changes": 0,
        "lo_cancellations": 0,
        "deadline_misses": 0,
    }
    # 保存每次 validation 决策前的状态和合法动作 mask，用于结束后统一诊断 explicit noop Q 值。
    # 这里不改变 agent 的动作选择，只额外记录 policy network 当时实际看到的输入。
    diagnostic_states: list[tuple[float, ...]] = []
    diagnostic_valid_masks: list[tuple[bool, ...]] = []

    while not done:
        # step_count 明确定义为“完成了多少次 agent 决策循环”，
        # 因此所有 per-step rate 的分母都必须统一使用它。
        step_count += 1
        mask = env.valid_action_mask()
        if len(diagnostic_states) < max_q_diagnostic_samples:
            diagnostic_states.append(tuple(float(value) for value in obs.state_vector))
            diagnostic_valid_masks.append(tuple(bool(value) for value in mask))
        action_id = agent.select_action_id(
            obs.state_vector,
            valid_action_mask=mask,
            training=False,
        )
        # selected_action_count 只表示 agent 是否产生了离散动作编号，
        # 不表示该动作是否最终被环境接受。
        selected_action_count += int(action_id is not None)
        result = env.step(action_id)
        total_reward += result.reward

        is_noop = bool(result.info.get("is_noop", False))
        if is_noop:
            noop_actions += 1
            if bool(result.info.get("is_explicit_noop_action", False)):
                explicit_noop_actions += 1

        if action_id is not None:
            if bool(result.info.get("accepted")):
                accepted_actions += 1
            else:
                rejected_actions += 1

        obs = result.observation
        done = result.done
        last_info = result.info

    debug_stats = env.debug_statistics()
    row = {
        "mode_changes": int(last_info.get("mode_changes", 0)),
        "lo_cancellations": int(last_info.get("lo_cancellations", 0)),
        "deadline_misses": int(last_info.get("deadline_misses", 0)),
        "accepted_actions": accepted_actions,
        "rejected_actions": rejected_actions,
        "step_count": step_count,
        "selected_action_count": selected_action_count,
        "noop_actions": noop_actions,
        "explicit_noop_actions": explicit_noop_actions,
        # 所有 rate 一律只使用 step_count 做分母，避免一个 step 被重复计入多个类别后
        # 破坏概率解释。例如显式 noop 既可能是 accepted，也必须计入 noop。
        "noop_action_rate": noop_actions / step_count if step_count > 0 else 0.0,
        "explicit_noop_action_rate": explicit_noop_actions / step_count if step_count > 0 else 0.0,
        "accepted_action_rate": accepted_actions / step_count if step_count > 0 else 0.0,
        "rejected_action_rate": rejected_actions / step_count if step_count > 0 else 0.0,
        "valid_action_count_mean": float(debug_stats["valid_action_count_mean"]),
        "masked_action_count_mean": float(debug_stats["masked_action_count_mean"]),
        "no_safe_action_steps": int(debug_stats["no_safe_action_steps"]),
        "reward": float(total_reward),
        "observation_mode": str(last_info.get("observation_mode", feature_config.observation_mode)),
        "state_dim": int(last_info.get("state_dim", len(obs.state_vector))),
    }
    row.update(_noop_q_diagnostics_to_row(agent, diagnostic_states, diagnostic_valid_masks))
    return row


def _run_dqn_validation_seed_worker(
    args_tuple: tuple[
        str,
        ExperimentConfig,
        int,
        int,
        int,
        str,
        str,
        float,
        float,
        bool,
        float,
        bool,
        str,
        FeatureConfig,
        int,
        int,
        int,
        bool,
        bool,
        bool,
    ],
) -> dict[str, int | float | None]:
    """运行单个 validation seed 的 DQN policy evaluation。

    并行 validation 不直接把主进程里的 agent 实例发送给子进程，
    因为主进程中的 agent 可能绑定了 MPS/GPU 设备状态。这里严格按文档要求：
    先在主进程保存模型快照，再由子进程用 CPU 设备重新加载，只做推理与环境仿真。
    """

    (
        model_path,
        experiment_config,
        seed,
        validation_end_time,
        agent_period,
        reward_mode,
        action_space,
        budget_increase_ratio,
        budget_decrease_ratio,
        include_explicit_noop,
        budget_floor_ratio,
        forbid_decreasing_hi_budgets,
        mask_detail_mode,
        feature_config,
        max_q_diagnostic_samples,
        constraint_guided_pair_top_k_risk,
        constraint_guided_pair_top_k_decrease,
        constraint_guided_pair_prefer_lo,
        constraint_guided_pair_include_hi_risk_boost,
        constraint_guided_pair_allow_increase_only_when_safe,
    ) = args_tuple
    agent = DqnBudgetAgent.load(Path(model_path), device="cpu")
    return _evaluate_agent_on_validation_seed(
        agent=agent,
        experiment_config=experiment_config,
        seed=seed,
        validation_end_time=validation_end_time,
        agent_period=agent_period,
        reward_mode=reward_mode,
        action_space=action_space,
        budget_increase_ratio=budget_increase_ratio,
        budget_decrease_ratio=budget_decrease_ratio,
        include_explicit_noop=include_explicit_noop,
        budget_floor_ratio=budget_floor_ratio,
        forbid_decreasing_hi_budgets=forbid_decreasing_hi_budgets,
        mask_detail_mode=mask_detail_mode,
        feature_config=feature_config,
        max_q_diagnostic_samples=max_q_diagnostic_samples,
        constraint_guided_pair_top_k_risk=constraint_guided_pair_top_k_risk,
        constraint_guided_pair_top_k_decrease=constraint_guided_pair_top_k_decrease,
        constraint_guided_pair_prefer_lo=constraint_guided_pair_prefer_lo,
        constraint_guided_pair_include_hi_risk_boost=constraint_guided_pair_include_hi_risk_boost,
        constraint_guided_pair_allow_increase_only_when_safe=constraint_guided_pair_allow_increase_only_when_safe,
    )


def _build_experiment_config(args: argparse.Namespace) -> ExperimentConfig:
    """按 CLI 参数构建实验配置。"""

    if args.workload == "small":
        return (
            build_small_nominal_experiment_config()
            if args.scenario == "nominal"
            else build_small_stress_experiment_config()
        )
    if args.workload == "rtss11":
        return build_rtss11_experiment_config(
            total_util=args.total_util,
            num_tasks=args.num_tasks,
            cf=args.cf,
            cp=args.cp,
            require_schedulable=args.require_schedulable,
            hi_overrun_prob=args.hi_overrun_prob,
            lo_overrun_prob=args.lo_overrun_prob,
            lo_overrun_factor=args.lo_overrun_factor,
            scenario_seed_offset=args.scenario_seed_offset,
            fixed_taskset_seed=args.fixed_taskset_seed,
        )
    if args.workload == "mc_fairgen":
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
    if args.workload == "automotive":
        return build_automotive_experiment_config(
            num_runnables=args.automotive_num_runnables,
            mode=args.automotive_mode,
            require_schedulable=args.require_schedulable,
            scenario_seed_offset=args.scenario_seed_offset,
            fixed_taskset_seed=args.fixed_taskset_seed,
            learnable_target_budget_util_min=args.learnable_target_budget_util_min,
            learnable_target_budget_util_max=args.learnable_target_budget_util_max,
            learnable_hi_budget_rho_min=args.learnable_hi_budget_rho_min,
            learnable_hi_budget_rho_max=args.learnable_hi_budget_rho_max,
            learnable_lo_budget_rho_min=args.learnable_lo_budget_rho_min,
            learnable_lo_budget_rho_max=args.learnable_lo_budget_rho_max,
            budget_floor_ratio=args.budget_floor_ratio,
        )
    raise ValueError(f"unsupported workload: {args.workload}")


def _run_validation(
    *,
    agent: DqnBudgetAgent,
    experiment_config: ExperimentConfig,
    validation_seeds: list[int],
    validation_end_time: int,
    agent_period: int,
    reward_mode: str,
    action_space: str,
    budget_increase_ratio: float,
    budget_decrease_ratio: float,
    include_explicit_noop: bool,
    budget_floor_ratio: float,
    forbid_decreasing_hi_budgets: bool,
    mask_detail_mode: str,
    feature_config: FeatureConfig = FeatureConfig(),
    validation_workers: int = 1,
    baseline_cache: dict[str, float] | None = None,
    max_q_diagnostic_samples: int = 1000,
    constraint_guided_pair_top_k_risk: int = 3,
    constraint_guided_pair_top_k_decrease: int = 5,
    constraint_guided_pair_prefer_lo: bool = False,
    constraint_guided_pair_include_hi_risk_boost: bool = False,
    constraint_guided_pair_allow_increase_only_when_safe: bool = False,
) -> tuple[dict[str, int | float | None], dict[str, float], bool]:
    """在验证集上评估当前 agent，并返回聚合指标。"""

    used_baseline_cache = baseline_cache is not None

    if baseline_cache is None:
        # baseline 的输入参数只依赖 experiment_config、seed 和验证时长，
        # 因此最适合按 seed 拆成完全独立的 worker 任务。
        baseline_worker_args = [
            (experiment_config, seed, validation_end_time)
            for seed in validation_seeds
        ]
        if validation_workers == 1:
            baseline_rows = [
                _run_baseline_validation_seed_worker(item)
                for item in baseline_worker_args
            ]
        else:
            try:
                with ProcessPoolExecutor(max_workers=validation_workers) as executor:
                    baseline_rows = list(executor.map(_run_baseline_validation_seed_worker, baseline_worker_args))
            except PermissionError:
                # 某些受限执行环境（如沙箱）禁止创建进程信号量，这里回退到串行路径，
                # 保持统计口径不变，仅牺牲并行加速。
                baseline_rows = [_run_baseline_validation_seed_worker(item) for item in baseline_worker_args]
        baseline_cache = {
            "baseline_mode_changes_mean": sum(row["mode_changes"] for row in baseline_rows) / len(validation_seeds),
            "baseline_lo_cancellations_mean": (
                sum(row["lo_cancellations"] for row in baseline_rows) / len(validation_seeds)
            ),
        }

    if validation_workers == 1:
        # 串行路径直接复用当前主进程中的 agent，避免默认配置下每次 validation 都发生 save/load 开销。
        dqn_rows = [
            _evaluate_agent_on_validation_seed(
                agent=agent,
                experiment_config=experiment_config,
                seed=seed,
                validation_end_time=validation_end_time,
                agent_period=agent_period,
                reward_mode=reward_mode,
                action_space=action_space,
                budget_increase_ratio=budget_increase_ratio,
                budget_decrease_ratio=budget_decrease_ratio,
                include_explicit_noop=include_explicit_noop,
                budget_floor_ratio=budget_floor_ratio,
                forbid_decreasing_hi_budgets=forbid_decreasing_hi_budgets,
                mask_detail_mode=mask_detail_mode,
                feature_config=feature_config,
                max_q_diagnostic_samples=max_q_diagnostic_samples,
                constraint_guided_pair_top_k_risk=constraint_guided_pair_top_k_risk,
                constraint_guided_pair_top_k_decrease=constraint_guided_pair_top_k_decrease,
                constraint_guided_pair_prefer_lo=constraint_guided_pair_prefer_lo,
                constraint_guided_pair_include_hi_risk_boost=constraint_guided_pair_include_hi_risk_boost,
                constraint_guided_pair_allow_increase_only_when_safe=constraint_guided_pair_allow_increase_only_when_safe,
            )
            for seed in validation_seeds
        ]
    else:
        # 并行路径在 validation 开始时冻结一次模型快照，确保所有 worker 使用完全相同的策略参数。
        with TemporaryDirectory(prefix="dqn_validation_") as tmp_dir:
            model_path = Path(tmp_dir) / "policy_snapshot.pt"
            agent.save(model_path)
            dqn_worker_args = [
                (
                    str(model_path),
                    experiment_config,
                    seed,
                    validation_end_time,
                    agent_period,
                    reward_mode,
                    action_space,
                    budget_increase_ratio,
                    budget_decrease_ratio,
                    include_explicit_noop,
                    budget_floor_ratio,
                    forbid_decreasing_hi_budgets,
                    mask_detail_mode,
                    feature_config,
                    max_q_diagnostic_samples,
                    constraint_guided_pair_top_k_risk,
                    constraint_guided_pair_top_k_decrease,
                    constraint_guided_pair_prefer_lo,
                    constraint_guided_pair_include_hi_risk_boost,
                    constraint_guided_pair_allow_increase_only_when_safe,
                )
                for seed in validation_seeds
            ]
            try:
                with ProcessPoolExecutor(max_workers=validation_workers) as executor:
                    dqn_rows = list(executor.map(_run_dqn_validation_seed_worker, dqn_worker_args))
            except PermissionError:
                # 与 baseline 并行分支同理：受限环境下保持口径一致地退回串行。
                dqn_rows = [_run_dqn_validation_seed_worker(item) for item in dqn_worker_args]

    seed_count = len(validation_seeds)
    baseline_mode_changes_mean = float(baseline_cache["baseline_mode_changes_mean"])
    baseline_lo_cancellations_mean = float(baseline_cache["baseline_lo_cancellations_mean"])
    mode_delta_sum = sum(row["mode_changes"] for row in dqn_rows) - baseline_mode_changes_mean * seed_count
    cancel_delta_sum = sum(row["lo_cancellations"] for row in dqn_rows) - baseline_lo_cancellations_mean * seed_count
    validation_row = {
        "validation_seed_count": seed_count,
        "deadline_misses_sum": sum(int(row["deadline_misses"]) for row in dqn_rows),
        "mode_changes_mean": sum(row["mode_changes"] for row in dqn_rows) / seed_count,
        "lo_cancellations_mean": sum(row["lo_cancellations"] for row in dqn_rows) / seed_count,
        "baseline_mode_changes_mean": baseline_mode_changes_mean,
        "baseline_lo_cancellations_mean": baseline_lo_cancellations_mean,
        "dqn_mode_changes_delta_mean": mode_delta_sum / seed_count,
        "dqn_lo_cancellations_delta_mean": cancel_delta_sum / seed_count,
        "accepted_actions_mean": sum(row["accepted_actions"] for row in dqn_rows) / seed_count,
        "rejected_actions_mean": sum(row["rejected_actions"] for row in dqn_rows) / seed_count,
        "step_count_mean": sum(row["step_count"] for row in dqn_rows) / seed_count,
        "selected_action_count_mean": sum(row["selected_action_count"] for row in dqn_rows) / seed_count,
        "noop_actions_mean": sum(row["noop_actions"] for row in dqn_rows) / seed_count,
        "explicit_noop_actions_mean": sum(row["explicit_noop_actions"] for row in dqn_rows) / seed_count,
        "noop_action_rate_mean": sum(row["noop_action_rate"] for row in dqn_rows) / seed_count,
        "explicit_noop_action_rate_mean": sum(row["explicit_noop_action_rate"] for row in dqn_rows) / seed_count,
        "accepted_action_rate_mean": sum(row["accepted_action_rate"] for row in dqn_rows) / seed_count,
        "rejected_action_rate_mean": sum(row["rejected_action_rate"] for row in dqn_rows) / seed_count,
        "valid_action_count_mean": sum(row["valid_action_count_mean"] for row in dqn_rows) / seed_count,
        "masked_action_count_mean": sum(row["masked_action_count_mean"] for row in dqn_rows) / seed_count,
        "no_safe_action_steps_mean": sum(row["no_safe_action_steps"] for row in dqn_rows) / seed_count,
        "reward_mean": sum(row["reward"] for row in dqn_rows) / seed_count,
        "observation_mode": str(feature_config.observation_mode),
        "state_dim_mean": sum(row["state_dim"] for row in dqn_rows) / seed_count,
    }
    # validation 输出采用文档第 6 节的字段名，数值为各 validation seed 诊断结果的均值；
    # `noop_q_sample_count` 代表实际参与 Q 诊断的状态样本总数，便于核对采样是否达到上限。
    for fieldname in NOOP_Q_DIAGNOSTIC_FIELDNAMES:
        if fieldname == "noop_q_sample_count":
            validation_row[fieldname] = sum(int(row[fieldname] or 0) for row in dqn_rows)
        else:
            validation_row[fieldname] = _mean_optional_metric(dqn_rows, fieldname)
    return validation_row, baseline_cache, used_baseline_cache


def _is_better_validation_row(
    *,
    candidate_row: dict[str, int | float],
    best_row: dict[str, int | float] | None,
    save_best_by: str,
    relative_score_alpha: float = 1.0,
    require_better_than_baseline_for_best: bool = False,
) -> bool:
    """判断候选验证结果是否优于当前 best。"""

    if int(candidate_row["deadline_misses_sum"]) != 0:
        return False
    use_lo_cancellations_gate = save_best_by == "lo_cancellations"
    if best_row is None:
        if use_lo_cancellations_gate:
            return float(candidate_row["mode_changes_mean"]) <= float(
                candidate_row["baseline_mode_changes_mean"]
            )
        return True
    if int(best_row["deadline_misses_sum"]) != 0:
        return True
    if use_lo_cancellations_gate:
        candidate_mode_ok = float(candidate_row["mode_changes_mean"]) <= float(
            candidate_row["baseline_mode_changes_mean"]
        )
        if not candidate_mode_ok:
            return False
        best_mode_ok = float(best_row["mode_changes_mean"]) <= float(best_row["baseline_mode_changes_mean"])
        if not best_mode_ok:
            return True
        return float(candidate_row["lo_cancellations_mean"]) < float(best_row["lo_cancellations_mean"])
    if save_best_by == "reward":
        # 阶段 3：reward 指标方向应为“越大越好”，与 cancellation/mode-change 相反。
        return float(candidate_row["reward_mean"]) > float(best_row["reward_mean"])
    if save_best_by == "relative_score":
        # relative_score 越小越好，<0 表示综合优于 baseline。
        # 本轮要求：即便 relative_score>=0，也要保留“验证集里最好的那个”checkpoint。
        _ = (relative_score_alpha, require_better_than_baseline_for_best)
        candidate_score = float(candidate_row["relative_score"])
        return candidate_score < float(best_row["relative_score"])
    if save_best_by == "pareto_relative_score":
        # 更温和的 Pareto 风格选模分数：
        # 1) 保留原有 relative_score（越小越好）作为主项；
        # 2) 对“比 baseline 更差”的维度施加软惩罚，而不是硬性 +inf；
        # 3) 两个惩罚项都基于归一化差值，保持与 relative_score 同量纲。
        #
        # 公式：
        # score = relative_score
        #       + 10 * max(0, relative_delta_mode_changes)
        #       + 10 * max(0, relative_delta_lo_cancellations)
        #
        # 解释：
        # - 当候选同时不劣于 baseline（两个 delta 都 <=0）时，惩罚为 0，退化为 relative_score 比较；
        # - 当某一维劣于 baseline 时，仍允许进入比较，但会因软惩罚降低被选中概率；
        # - 这比 strict gate 更平滑，适合你要求的“温和版本”。
        _ = (relative_score_alpha, require_better_than_baseline_for_best)
        candidate_score = (
            float(candidate_row["relative_score"])
            + 10.0 * max(0.0, float(candidate_row["relative_delta_mode_changes"]))
            + 10.0 * max(0.0, float(candidate_row["relative_delta_lo_cancellations"]))
        )
        best_score = (
            float(best_row["relative_score"])
            + 10.0 * max(0.0, float(best_row["relative_delta_mode_changes"]))
            + 10.0 * max(0.0, float(best_row["relative_delta_lo_cancellations"]))
        )
        return candidate_score < best_score
    metric_field = {
        "lo_cancellations": "lo_cancellations_mean",
        "mode_changes": "mode_changes_mean",
    }[save_best_by]
    return float(candidate_row[metric_field]) < float(best_row[metric_field])


def _is_pareto_valid_checkpoint(row: dict[str, int | float | None]) -> bool:
    """判断一个 validation checkpoint 是否满足 Pareto-valid 条件。

    Pareto-valid 的定义严格按你给出的规则：
    1. deadline_misses_sum == 0（硬约束）；
    2. dqn_mode_changes_mean <= baseline_mode_changes_mean；
    3. dqn_lo_cancellations_mean <= baseline_lo_cancellations_mean。

    该函数只负责“是否通过筛选”，不负责在候选之间排序。
    """

    return (
        int(row["deadline_misses_sum"]) == 0
        and float(row["mode_changes_mean"]) <= float(row["baseline_mode_changes_mean"])
        and float(row["lo_cancellations_mean"]) <= float(row["baseline_lo_cancellations_mean"])
    )


def _build_validation_unified_summary_rows(
    validation_rows: list[dict[str, int | float]],
) -> list[dict[str, int | float | str]]:
    """把训练期 validation 指标转换为阶段 0/1 的统一 summary 口径。"""

    summary_rows: list[dict[str, int | float | str]] = []
    for row in validation_rows:
        baseline_mode_changes_mean = float(row["baseline_mode_changes_mean"])
        baseline_lo_cancellations_mean = float(row["baseline_lo_cancellations_mean"])
        dqn_mode_changes_mean = float(row["mode_changes_mean"])
        dqn_lo_cancellations_mean = float(row["lo_cancellations_mean"])
        mode_change_ratio: float | str
        lo_cancellation_ratio: float | str
        if dqn_mode_changes_mean == 0.0:
            mode_change_ratio = "inf" if baseline_mode_changes_mean > 0.0 else "nan"
        else:
            mode_change_ratio = baseline_mode_changes_mean / dqn_mode_changes_mean
        if dqn_lo_cancellations_mean == 0.0:
            lo_cancellation_ratio = "inf" if baseline_lo_cancellations_mean > 0.0 else "nan"
        else:
            lo_cancellation_ratio = baseline_lo_cancellations_mean / dqn_lo_cancellations_mean

        summary_rows.append(
            {
                "episode": int(row["episode"]),
                "validation_seed_count": int(row["validation_seed_count"]),
                "baseline_mode_changes_mean": baseline_mode_changes_mean,
                "baseline_lo_cancellations_mean": baseline_lo_cancellations_mean,
                "dqn_mode_changes_mean": dqn_mode_changes_mean,
                "dqn_lo_cancellations_mean": dqn_lo_cancellations_mean,
                # 这三列用于与文档要求一致地直观看到“相对 baseline 的差值与综合得分”。
                "relative_score": float(row["relative_score"]),
                "relative_score_alpha": float(row["relative_score_alpha"]),
                "relative_delta_mode_changes": float(row["relative_delta_mode_changes"]),
                "relative_delta_lo_cancellations": float(row["relative_delta_lo_cancellations"]),
                "raw_delta_mode_changes": float(
                    row.get("raw_delta_mode_changes", row["dqn_mode_changes_delta_mean"])
                ),
                "raw_delta_lo_cancellations": float(
                    row.get("raw_delta_lo_cancellations", row["dqn_lo_cancellations_delta_mean"])
                ),
                "is_better_than_baseline": bool(row["is_better_than_baseline"]),
                "mode_changes_delta_vs_baseline": float(row["dqn_mode_changes_delta_mean"]),
                "lo_cancellations_delta_vs_baseline": float(row["dqn_lo_cancellations_delta_mean"]),
                "mode_change_ratio": mode_change_ratio,
                "lo_cancellation_ratio": lo_cancellation_ratio,
                "accepted_action_count_mean": float(row["accepted_actions_mean"]),
                "rejected_action_count_mean": float(row["rejected_actions_mean"]),
                "noop_action_count_mean": float(row["noop_actions_mean"]),
                "noop_action_rate_mean": float(row["noop_action_rate_mean"]),
                "explicit_noop_action_rate_mean": float(row["explicit_noop_action_rate_mean"]),
                "accepted_action_rate_mean": float(row["accepted_action_rate_mean"]),
                "rejected_action_rate_mean": float(row["rejected_action_rate_mean"]),
                "masked_action_count_mean": float(row["masked_action_count_mean"]),
                "valid_action_count_mean": float(row["valid_action_count_mean"]),
                # explicit noop 的 Q 值诊断字段直接透传到 unified summary，
                # 这样验证趋势表可以同时观察动作实际选择频率与 Q 值排名变化。
                "noop_q_mean": row.get("noop_q_mean"),
                "noop_q_std": row.get("noop_q_std"),
                "noop_q_rank_mean": row.get("noop_q_rank_mean"),
                "noop_q_rank_median": row.get("noop_q_rank_median"),
                "noop_q_rank_min": row.get("noop_q_rank_min"),
                "noop_q_rank_max": row.get("noop_q_rank_max"),
                "noop_q_margin_to_best_mean": row.get("noop_q_margin_to_best_mean"),
                "noop_q_is_best_rate": row.get("noop_q_is_best_rate"),
                "noop_valid_rate": row.get("noop_valid_rate"),
                "noop_q_sample_count": row.get("noop_q_sample_count"),
            }
        )
    return summary_rows


def build_parser() -> argparse.ArgumentParser:
    """构建正式训练 CLI 的命令行参数解析器。"""

    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--end-time", type=int, default=100)
    parser.add_argument("--agent-period", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--network-seed", type=int, default=None)
    parser.add_argument("--exploration-seed", type=int, default=None)
    parser.add_argument("--replay-seed", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--replay-capacity", type=int, default=10000)
    parser.add_argument("--min-replay-size", type=int, default=500)
    parser.add_argument("--hidden-layers", type=str, default="128,128")
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--gamma", type=float, default=0.99)
    # 稳定性修改 C：默认 target network 更新频率改为 5（优化步为单位）。
    # 该值可继续通过 CLI 覆盖，用于后续对照实验。
    parser.add_argument("--target-update-freq", type=int, default=5)
    parser.add_argument("--target-update-frequency", type=int, default=None)
    parser.add_argument(
        "--grad-clip-norm",
        type=float,
        default=10.0,
        help="DQN 梯度裁剪阈值（L2 norm），用于限制反向传播后的梯度范数。",
    )
    parser.add_argument("--epsilon-start", type=float, default=1.0)
    parser.add_argument("--epsilon-end", type=float, default=0.05)
    parser.add_argument("--epsilon-decay-steps", type=int, default=5000)
    parser.add_argument(
        "--noop-exploration-prob",
        type=float,
        default=0.0,
        help=(
            "During epsilon exploration, if explicit noop is valid, choose noop "
            "with this probability before sampling other valid actions."
        ),
    )
    parser.add_argument(
        "--double-dqn",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use Double DQN target calculation: policy net selects next action, target net evaluates it.",
    )
    parser.add_argument(
        "--max-q-diagnostic-samples",
        type=int,
        default=1000,
        help="Maximum validation decision states sampled for noop Q diagnostics per validation seed.",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--checkpoint", type=int, default=0)
    parser.add_argument("--workload", choices=["small", "rtss11", "automotive", "mc_fairgen"], default="small")
    # automotive workload 允许从 CLI 显式切换 runnable 数量与 workload 语义模式，
    # 这样训练入口就不再把 automotive 固定写死为 150 + paper_like。
    parser.add_argument("--automotive-num-runnables", type=int, choices=[150, 250], default=150)
    parser.add_argument(
        "--automotive-mode",
        choices=["fast", "paper_like", "paper_exact", "paper_learnable_headroom"],
        default="paper_like",
    )
    parser.add_argument("--learnable-target-budget-util-min", type=float, default=0.62)
    parser.add_argument("--learnable-target-budget-util-max", type=float, default=0.78)
    parser.add_argument("--learnable-hi-budget-rho-min", type=float, default=0.45)
    parser.add_argument("--learnable-hi-budget-rho-max", type=float, default=0.65)
    parser.add_argument("--learnable-lo-budget-rho-min", type=float, default=0.35)
    parser.add_argument("--learnable-lo-budget-rho-max", type=float, default=0.60)
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
    parser.add_argument("--scenario", choices=["nominal", "stress"], default="stress")
    parser.add_argument("--total-util", type=float, default=0.65)
    parser.add_argument("--num-tasks", type=int, default=20)
    parser.add_argument("--cf", type=float, default=2.0)
    parser.add_argument("--cp", type=float, default=0.5)
    parser.add_argument("--require-schedulable", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--hi-overrun-prob", type=float, default=0.05)
    parser.add_argument("--lo-overrun-prob", type=float, default=0.10)
    parser.add_argument("--lo-overrun-factor", type=float, default=1.5)
    parser.add_argument("--log-train-metrics", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--trace-every", type=int, default=0)
    parser.add_argument("--trace-dir", type=Path, default=None)
    parser.add_argument("--train-seed-mode", choices=["fixed", "per-episode", "cycle"], default="fixed")
    parser.add_argument(
        "--fixed-taskset-seed",
        type=int,
        default=None,
        help=(
            "如果设置该参数，RTSS11 任务集生成固定使用该 seed；"
            "每个 episode 变化的 seed 仅用于 scenario 生成。"
        ),
    )
    parser.add_argument("--train-seeds", type=str, default="")
    parser.add_argument("--scenario-seed-offset", type=int, default=100000)
    parser.add_argument("--validation-seeds", type=str, default="100:129")
    parser.add_argument("--validate-every", type=int, default=50)
    parser.add_argument("--validation-end-time", type=int, default=10000)
    parser.add_argument(
        "--validation-workers",
        type=int,
        default=1,
        help="并行 validation 的进程数；1 表示保持串行。",
    )
    parser.add_argument(
        "--log-step-every",
        type=int,
        default=1,
        help="每隔多少个 global step 记录一行 step-level 日志；1 表示每步记录，0 表示关闭 step-level 日志。",
    )
    parser.add_argument(
        "--save-best-by",
        choices=["lo_cancellations", "mode_changes", "reward", "relative_score", "pareto_relative_score"],
        default="mode_changes",
    )
    parser.add_argument(
        "--reward-mode",
        choices=list(available_reward_modes()),
        default="mendes",
    )
    parser.add_argument(
        "--relative-score-alpha",
        type=float,
        default=1.0,
        help="relative_score 中 mode_changes 差值的权重 alpha。",
    )
    parser.add_argument(
        "--require-better-than-baseline-for-best",
        action="store_true",
        help="仅当 relative_score < 0 时才允许刷新 model_best.pt。",
    )
    parser.add_argument(
        "--action-space",
        choices=["triple", "pair", "single", "constraint_guided_pair", "constraint_guided_transfer"],
        default="triple",
    )
    parser.add_argument("--budget-increase-ratio", type=float, default=0.10)
    parser.add_argument("--budget-decrease-ratio", type=float, default=0.05)
    parser.add_argument("--constraint-guided-pair-top-k-risk", type=int, default=3)
    parser.add_argument("--constraint-guided-pair-top-k-decrease", type=int, default=5)
    parser.add_argument("--constraint-guided-pair-prefer-lo", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--constraint-guided-pair-include-hi-risk-boost", action="store_true")
    parser.add_argument(
        "--constraint-guided-pair-allow-increase-only-when-safe",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--include-explicit-noop", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--budget-floor-ratio",
        type=float,
        default=0.0,
        help=(
            "Reject budget actions that would reduce any task budget below "
            "initial_budget * this ratio. 0 disables the floor."
        ),
    )
    parser.add_argument(
        "--forbid-decreasing-hi-budgets",
        action="store_true",
        # 阶段 3：开启后会在 action mask 中屏蔽“decrease 命中 HI 任务”的动作。
        # 这是硬约束，不是软惩罚，不依赖 reward 权重调节。
        help="If set, action masks reject budget actions whose decrease tasks include any HI-criticality task.",
    )
    parser.add_argument("--mask-detail-mode", choices=["minimal", "full"], default="minimal")
    parser.add_argument("--observation-mode", choices=["v10_basic", "v11_full_10d"], default="v10_basic")
    parser.add_argument("--ema-alpha", type=float, default=0.2)
    parser.add_argument("--overrun-ema-alpha", type=float, default=0.1)
    parser.add_argument("--history-k", type=int, default=8)
    parser.add_argument("--event-window", type=int, default=10)
    parser.add_argument("--max-cost-weight", type=float, default=0.7)
    parser.add_argument("--risk-max-scale", type=float, default=3.0)
    parser.add_argument("--include-safety-margin", action=argparse.BooleanOptionalAction, default=True)
    return parser


def main() -> None:
    """运行正式 DQN 训练并产出完整目录结构。"""

    args = build_parser().parse_args()
    requested_action_space = args.action_space
    if requested_action_space == "constraint_guided_pair":
        # 兼容旧参数名：内部统一走 constraint_guided_transfer 语义。
        args.action_space = "constraint_guided_transfer"
    if args.trace_every < 0:
        raise ValueError("--trace-every 必须为非负整数")
    if args.trace_every > 0 and args.trace_dir is None:
        raise ValueError("设置 --trace-every 时必须同时提供 --trace-dir")
    if args.validate_every < 0:
        raise ValueError("--validate-every 必须为非负整数")
    if args.validation_workers < 1:
        raise ValueError("--validation-workers 必须为正整数")
    if args.log_step_every < 0:
        raise ValueError("--log-step-every 必须为非负整数")
    if args.max_q_diagnostic_samples < 0:
        raise ValueError("--max-q-diagnostic-samples 必须为非负整数")
    if args.target_update_frequency is not None:
        args.target_update_freq = int(args.target_update_frequency)
    if args.budget_floor_ratio < 0.0 or args.budget_floor_ratio > 1.0:
        raise ValueError("--budget-floor-ratio must be in [0, 1]")
    reward_mode_config = load_reward_mode_config(args.reward_mode)
    feature_config = FeatureConfig(
        observation_mode=args.observation_mode,
        ema_alpha=args.ema_alpha,
        overrun_ema_alpha=args.overrun_ema_alpha,
        history_k=args.history_k,
        event_window=args.event_window,
        max_cost_weight=args.max_cost_weight,
        risk_max_scale=args.risk_max_scale,
        include_safety_margin=args.include_safety_margin,
    )

    experiment_config = _build_experiment_config(args)
    train_seed_candidates = _parse_seed_spec(args.train_seeds)
    episode_seed_schedule = _build_episode_seed_schedule(
        episodes=args.episodes,
        seed=args.seed,
        mode=args.train_seed_mode,
        train_seeds=train_seed_candidates,
    )
    validation_seeds = _parse_seed_spec(args.validation_seeds)

    hidden_layers = _parse_hidden_layers(args.hidden_layers)
    # 阶段 0：显式拆分 DQN 侧随机源，保证同配置可复现且可追踪。
    # - network_seed:     只用于网络初始化；
    # - exploration_seed: 只用于 epsilon-greedy 探索；
    # - replay_seed:      只用于 replay 抽样。
    network_seed = args.seed if args.network_seed is None else int(args.network_seed)
    exploration_seed = args.seed if args.exploration_seed is None else int(args.exploration_seed)
    replay_seed = args.seed if args.replay_seed is None else int(args.replay_seed)
    config = DqnConfig(
        gamma=args.gamma,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        replay_capacity=args.replay_capacity,
        min_replay_size=args.min_replay_size,
        target_update_freq=args.target_update_freq,
        grad_clip_norm=args.grad_clip_norm,
        epsilon_start=args.epsilon_start,
        epsilon_end=args.epsilon_end,
        epsilon_decay_steps=args.epsilon_decay_steps,
        noop_exploration_prob=args.noop_exploration_prob,
        hidden_layers=hidden_layers,
        seed=args.seed,
        network_seed=network_seed,
        exploration_seed=exploration_seed,
        replay_seed=replay_seed,
    )

    initial_seed = episode_seed_schedule[0]
    initial_bundle = resolve_experiment_bundle(experiment_config, initial_seed)
    initial_env = build_env_from_experiment_config(
        experiment_config,
        seed=initial_seed,
        end_time=args.end_time,
        agent_period=args.agent_period,
        semantics=RuntimeSemantics.AMC_PLUS,
        reward_mode=args.reward_mode,
        action_space=args.action_space,
        budget_increase_ratio=args.budget_increase_ratio,
        budget_decrease_ratio=args.budget_decrease_ratio,
        include_explicit_noop=args.include_explicit_noop,
        budget_floor_ratio=args.budget_floor_ratio,
        forbid_decreasing_hi_budgets=args.forbid_decreasing_hi_budgets,
        mask_detail_mode=args.mask_detail_mode,
        feature_config=feature_config,
        constraint_guided_pair_top_k_risk=args.constraint_guided_pair_top_k_risk,
        constraint_guided_pair_top_k_decrease=args.constraint_guided_pair_top_k_decrease,
        constraint_guided_pair_prefer_lo=args.constraint_guided_pair_prefer_lo,
        constraint_guided_pair_include_hi_risk_boost=args.constraint_guided_pair_include_hi_risk_boost,
        constraint_guided_pair_allow_increase_only_when_safe=(
            args.constraint_guided_pair_allow_increase_only_when_safe
        ),
    )
    initial_obs = initial_env.reset(seed=initial_seed)
    agent = DqnBudgetAgent(
        observation_dim=len(initial_obs.state_vector),
        action_dim=initial_env.action_space_size,
        config=config,
        noop_action_id=_get_noop_action_id(initial_env),
        hidden_layers=hidden_layers,
        double_dqn=args.double_dqn,
    )

    if args.output_dir is None:
        output_dir = (
            Path(f"outputs/dqn_rtss11/u{int(round(args.total_util * 1000)):03d}_seed{args.seed}")
            if args.workload == "rtss11"
            else Path("outputs/dqn_amc")
        )
    else:
        output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_dir = output_dir / "checkpoints"
    if args.checkpoint > 0:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
    if args.trace_dir is not None:
        args.trace_dir.mkdir(parents=True, exist_ok=True)

    step_rows: list[dict[str, int | float | str | bool]] = []
    train_metric_rows: list[dict[str, int | float | str]] = []
    action_hist_rows: list[dict[str, int]] = []
    validation_rows: list[dict[str, int | float | None]] = []
    best_validation_row: dict[str, int | float | None] | None = None
    model_best_path = output_dir / "model_best.pt"
    best_model_metadata_path = output_dir / "best_model_metadata.json"
    best_model_saved = False
    baseline_validation_cache: dict[str, float] | None = None
    # `pareto_relative_score` 两阶段选模状态：
    # - False：尚未见到任何 Pareto-valid checkpoint，允许 fallback 到软分数；
    # - True ：已经见到至少一个 Pareto-valid checkpoint，后续仅在 Pareto-valid 集合内比较。
    pareto_valid_seen: bool = False

    global_step = 0
    for episode in range(args.episodes):
        episode_seed = episode_seed_schedule[episode]
        env = build_env_from_experiment_config(
            experiment_config,
            seed=episode_seed,
            end_time=args.end_time,
            agent_period=args.agent_period,
            semantics=RuntimeSemantics.AMC_PLUS,
            reward_mode=args.reward_mode,
            action_space=args.action_space,
            budget_increase_ratio=args.budget_increase_ratio,
            budget_decrease_ratio=args.budget_decrease_ratio,
            include_explicit_noop=args.include_explicit_noop,
            budget_floor_ratio=args.budget_floor_ratio,
            forbid_decreasing_hi_budgets=args.forbid_decreasing_hi_budgets,
            mask_detail_mode=args.mask_detail_mode,
            feature_config=feature_config,
            constraint_guided_pair_top_k_risk=args.constraint_guided_pair_top_k_risk,
            constraint_guided_pair_top_k_decrease=args.constraint_guided_pair_top_k_decrease,
            constraint_guided_pair_prefer_lo=args.constraint_guided_pair_prefer_lo,
            constraint_guided_pair_include_hi_risk_boost=args.constraint_guided_pair_include_hi_risk_boost,
            constraint_guided_pair_allow_increase_only_when_safe=(
                args.constraint_guided_pair_allow_increase_only_when_safe
            ),
        )
        bundle = resolve_experiment_bundle(experiment_config, episode_seed)
        obs = env.reset(seed=episode_seed)
        done = False
        episode_reward = 0.0
        episode_losses: list[float] = []
        episode_accepted_actions = 0
        episode_rejected_actions = 0
        episode_step_count = 0
        episode_selected_action_count = 0
        episode_noop_actions = 0
        episode_explicit_noop_actions = 0
        reward_job_start_sum = 0.0
        reward_lo_overrun_sum = 0.0
        reward_hi_overrun_sum = 0.0
        reward_mode_change_sum = 0.0
        reward_lo_cancellation_sum = 0.0
        reward_deadline_miss_sum = 0.0
        reward_paper_sum = 0.0
        reward_noop_bonus_sum = 0.0
        reward_budget_change_penalty_sum = 0.0
        reward_budget_change_norm_sum = 0.0
        reward_budget_drift_penalty_sum = 0.0
        reward_budget_drift_mean_sum = 0.0
        # v11 诊断：记录每个 step 的 safety margin，再聚合为 episode 级 mean/p05。
        feature_safety_margin_min_values: list[float] = []
        exploration_action_count_before = int(agent.exploration_action_count)
        exploration_noop_action_count_before = int(agent.exploration_noop_action_count)
        episode_action_hist: dict[int, dict[str, int]] = defaultdict(lambda: {"count": 0, "accepted": 0, "rejected": 0})
        last_info: dict[str, int | float | str | bool | None] = {
            "mode_changes": 0,
            "lo_cancellations": 0,
            "deadline_misses": 0,
        }

        while not done:
            # episode_step_count 是训练期指标的主分母，定义为“完成了多少个环境 step”。
            episode_step_count += 1
            mask = env.valid_action_mask()
            valid_action_count = sum(mask)
            masked_action_count = len(mask) - valid_action_count
            action_id = agent.select_action_id(
                obs.state_vector,
                valid_action_mask=mask,
                training=True,
            )
            # selected_action_count 只表示“agent 有没有给出 action_id”，
            # 与是否 accepted/noop 是正交维度。
            episode_selected_action_count += int(action_id is not None)
            result = env.step(action_id)
            if action_id is not None and not bool(result.info.get("selected_action_was_mask_valid", False)):
                raise RuntimeError(f"检测到动作选择与 mask 语义不一致，episode={episode}, action_id={action_id}")
            next_mask = (
                env.valid_action_mask()
                if not result.done
                else tuple(False for _ in range(env.action_space_size))
            )

            loss: float | None = None
            if action_id is not None:
                transition = Transition(
                    state=obs.state_vector,
                    action_id=action_id,
                    reward=result.reward,
                    next_state=result.observation.state_vector,
                    done=result.done,
                    valid_action_mask=tuple(mask),
                    next_valid_action_mask=tuple(next_mask),
                )
                agent.remember(transition)
                loss = agent.optimize_one_step()
                if loss is not None:
                    episode_losses.append(loss)
                episode_action_hist[action_id]["count"] += 1
                if bool(result.info.get("accepted")):
                    episode_action_hist[action_id]["accepted"] += 1
                else:
                    episode_action_hist[action_id]["rejected"] += 1

            accepted = bool(result.info.get("accepted"))
            rejected = action_id is not None and not accepted
            noop = bool(result.info.get("is_noop", False))
            explicit_noop = bool(result.info.get("is_explicit_noop_action", False))
            # 统计口径说明：
            # - explicit noop 既可能是 accepted，也必须计入 noop；
            # - 因此 rate 分母必须用 step_count，不能用 accepted+rejected+noop。
            if accepted:
                episode_accepted_actions += 1
            if rejected:
                episode_rejected_actions += 1
            if noop:
                episode_noop_actions += 1
            if explicit_noop:
                episode_explicit_noop_actions += 1

            episode_reward += result.reward
            reward_job_start_sum += float(result.info.get("step_reward_job_start", 0.0))
            reward_lo_overrun_sum += float(result.info.get("step_reward_lo_overrun", 0.0))
            reward_hi_overrun_sum += float(result.info.get("step_reward_hi_overrun", 0.0))
            reward_mode_change_sum += float(result.info.get("step_reward_mode_change", 0.0))
            reward_lo_cancellation_sum += float(result.info.get("step_reward_lo_cancellation", 0.0))
            reward_deadline_miss_sum += float(result.info.get("step_reward_deadline_miss", 0.0))
            reward_paper_sum += float(result.info.get("paper_reward", 0.0))
            reward_noop_bonus_sum += float(result.info.get("noop_reward_bonus", 0.0))
            reward_budget_change_penalty_sum += float(result.info.get("budget_change_penalty_value", 0.0))
            reward_budget_change_norm_sum += float(result.info.get("budget_change_norm", 0.0))
            reward_budget_drift_penalty_sum += float(result.info.get("budget_drift_penalty_value", 0.0))
            reward_budget_drift_mean_sum += float(result.info.get("budget_drift_mean", 0.0))
            feature_safety_margin_min_values.append(float(result.info.get("feature_safety_margin_min", 1.0)))
            # `global_step` 是跨 episode 的全局步号。
            # 文档要求只对 step-level 明细日志做采样，不改变任何训练统计与优化逻辑。
            should_log_step = args.log_step_every > 0 and global_step % args.log_step_every == 0
            if should_log_step:
                step_rows.append(
                    {
                        "episode": episode,
                        "step": global_step,
                        "sim_time": int(result.info.get("time", 0)),
                        "reward": float(result.reward),
                        "episode_reward": float(episode_reward),
                        "total_reward": float(episode_reward),
                        "loss": "" if loss is None else loss,
                        "epsilon": agent.current_epsilon,
                        "action_id": "" if action_id is None else action_id,
                        "accepted": accepted,
                        "rejected": rejected,
                        "reject_reason": (
                            "no_valid_action" if action_id is None else str(result.info.get("reject_reason", ""))
                        ),
                        "valid_action_count": valid_action_count,
                        "masked_action_count": masked_action_count,
                        "noop_due_to_no_valid_action": action_id is None,
                        "is_noop": noop,
                        "is_explicit_noop": explicit_noop,
                        "mode_changes": int(result.info.get("mode_changes", 0)),
                        "lo_cancellations": int(result.info.get("lo_cancellations", 0)),
                        "deadline_misses": int(result.info.get("deadline_misses", 0)),
                        "step_reward_total": float(result.info.get("step_reward_total", 0.0)),
                        "step_reward_job_start": float(result.info.get("step_reward_job_start", 0.0)),
                        "step_reward_lo_overrun": float(result.info.get("step_reward_lo_overrun", 0.0)),
                        "step_reward_hi_overrun": float(result.info.get("step_reward_hi_overrun", 0.0)),
                        "step_reward_mode_change": float(result.info.get("step_reward_mode_change", 0.0)),
                        "step_reward_lo_cancellation": float(result.info.get("step_reward_lo_cancellation", 0.0)),
                        "step_reward_deadline_miss": float(result.info.get("step_reward_deadline_miss", 0.0)),
                        "step_reward_invalid_action": float(result.info.get("step_reward_invalid_action", 0.0)),
                        "paper_reward": float(result.info.get("paper_reward", 0.0)),
                        "noop_reward_bonus": float(result.info.get("noop_reward_bonus", 0.0)),
                        # 新 reward 公式关键变量写入 step log，便于后续定位“哪一项在主导训练”。
                        "lo_overrun_rate": float(result.info.get("lo_overrun_rate", 0.0)),
                        "hi_overrun_rate": float(result.info.get("hi_overrun_rate", 0.0)),
                        "mode_change_per_job": float(result.info.get("mode_change_per_job", 0.0)),
                        "lo_cancellation_rate": float(result.info.get("lo_cancellation_rate", 0.0)),
                        "deadline_miss_rate": float(result.info.get("deadline_miss_rate", 0.0)),
                        "invalid_action": float(result.info.get("invalid_action", 0.0)),
                        "budget_change_norm": float(result.info.get("budget_change_norm", 0.0)),
                        "budget_change_penalty_value": float(result.info.get("budget_change_penalty_value", 0.0)),
                        "budget_drift_mean": float(result.info.get("budget_drift_mean", 0.0)),
                        "budget_drift_penalty_value": float(result.info.get("budget_drift_penalty_value", 0.0)),
                        "lo_pressure_mean": float(result.info.get("lo_pressure_mean", 0.0)),
                        "lo_pressure_max": float(result.info.get("lo_pressure_max", 0.0)),
                        "lo_near_cancel_rate": float(result.info.get("lo_near_cancel_rate", 0.0)),
                        "hi_mode_pressure_mean": float(result.info.get("hi_mode_pressure_mean", 0.0)),
                        "lo_pressure_penalty_value": float(result.info.get("lo_pressure_penalty_value", 0.0)),
                        "lo_pressure_max_penalty_value": float(
                            result.info.get("lo_pressure_max_penalty_value", 0.0)
                        ),
                        "lo_near_cancel_penalty_value": float(
                            result.info.get("lo_near_cancel_penalty_value", 0.0)
                        ),
                        "hi_mode_pressure_penalty_value": float(
                            result.info.get("hi_mode_pressure_penalty_value", 0.0)
                        ),
                        "reward_after_regularization": float(result.info.get("reward_after_regularization", 0.0)),
                        "workload": args.workload,
                        "total_util": args.total_util,
                        "num_tasks": len(bundle.ordered_tasks),
                        "cf": args.cf,
                        "cp": args.cp,
                        "taskset_seed": bundle.taskset_seed if bundle.taskset_seed is not None else episode_seed,
                        "scenario_seed": bundle.scenario_seed if bundle.scenario_seed is not None else episode_seed,
                        "require_schedulable": args.require_schedulable,
                        "observation_mode": str(result.info.get("observation_mode", args.observation_mode)),
                        "state_dim": int(result.info.get("state_dim", len(result.observation.state_vector))),
                    }
                )
            obs = result.observation
            done = result.done
            last_info = result.info
            global_step += 1

        debug_stats = env.debug_statistics()
        if int(debug_stats["selected_invalid_mask_actions"]) > 0:
            raise RuntimeError(
                f"selected_invalid_mask_actions 必须为 0，episode={episode}, value={debug_stats['selected_invalid_mask_actions']}"
            )

        loss_mean = sum(episode_losses) / len(episode_losses) if episode_losses else ""
        loss_last: float | str = episode_losses[-1] if episode_losses else ""
        taskset_seed = bundle.taskset_seed if bundle.taskset_seed is not None else episode_seed
        scenario_seed = bundle.scenario_seed if bundle.scenario_seed is not None else episode_seed
        train_metric_rows.append(
            {
                "episode": episode,
                "episode_seed": episode_seed,
                "taskset_seed": taskset_seed,
                "scenario_seed": scenario_seed,
                "network_seed": network_seed,
                "exploration_seed": exploration_seed,
                "replay_seed": replay_seed,
                "total_util": args.total_util,
                "num_tasks": len(bundle.ordered_tasks),
                "taskset_fingerprint": bundle.taskset_fingerprint or "",
                "steps": episode_accepted_actions + episode_rejected_actions + episode_noop_actions,
                # `steps` 保留旧字段以兼容历史脚本；其值可能与 step_count 不同（存在重复口径）。
                # 新分析请优先使用 `step_count`。
                "step_count": episode_step_count,
                "selected_action_count": episode_selected_action_count,
                "total_reward": episode_reward,
                "epsilon": agent.current_epsilon,
                "loss_mean": loss_mean,
                "loss_last": loss_last,
                "accepted_actions": episode_accepted_actions,
                "rejected_actions": episode_rejected_actions,
                "noop_actions": episode_noop_actions,
                "explicit_noop_actions": episode_explicit_noop_actions,
                "noop_action_rate": (
                    episode_noop_actions
                    / episode_step_count
                    if episode_step_count > 0
                    else 0.0
                ),
                # 显式 noop rate 单独输出，便于区分“主动不调预算”和“无动作可用”。
                "explicit_noop_action_rate": (
                    episode_explicit_noop_actions / episode_step_count if episode_step_count > 0 else 0.0
                ),
                "accepted_action_rate": (
                    episode_accepted_actions / episode_step_count if episode_step_count > 0 else 0.0
                ),
                "rejected_action_rate": (
                    episode_rejected_actions / episode_step_count if episode_step_count > 0 else 0.0
                ),
                "safety_checked_actions": int(debug_stats["safety_checked_actions"]),
                "selected_invalid_mask_actions": int(debug_stats["selected_invalid_mask_actions"]),
                "action_space_type": str(debug_stats["action_space_type"]),
                "action_count": int(debug_stats["action_count"]),
                "budget_increase_ratio": float(debug_stats["budget_increase_ratio"]),
                "budget_decrease_ratio": float(debug_stats["budget_decrease_ratio"]),
                "budget_floor_ratio": float(debug_stats["budget_floor_ratio"]),
                "valid_action_count_mean": float(debug_stats["valid_action_count_mean"]),
                "masked_action_count_mean": float(debug_stats["masked_action_count_mean"]),
                "masked_decrease_hi_forbidden_count": int(debug_stats["masked_decrease_hi_forbidden_count"]),
                "masked_decrease_hi_forbidden_rate": float(debug_stats["masked_decrease_hi_forbidden_rate"]),
                "masked_budget_floor_violation_count": int(debug_stats["masked_budget_floor_violation_count"]),
                "masked_budget_floor_violation_rate": float(debug_stats["masked_budget_floor_violation_rate"]),
                "no_safe_action_steps": int(debug_stats["no_safe_action_steps"]),
                "selected_explicit_noop_actions": int(debug_stats["selected_explicit_noop_actions"]),
                "selected_explicit_noop_rate": float(debug_stats["selected_explicit_noop_rate"]),
                "mode_changes": int(last_info.get("mode_changes", 0)),
                "lo_cancellations": int(last_info.get("lo_cancellations", 0)),
                "deadline_misses": int(last_info.get("deadline_misses", 0)),
                "job_starts": int(env._monitor.job_start_count),
                "lo_overruns": int(env._monitor.lo_overrun_count),
                "hi_overruns": int(env._monitor.hi_overrun_count),
                "reward_job_start_sum": reward_job_start_sum,
                "reward_lo_overrun_sum": reward_lo_overrun_sum,
                "reward_hi_overrun_sum": reward_hi_overrun_sum,
                "reward_mode_change_sum": reward_mode_change_sum,
                "reward_lo_cancellation_sum": reward_lo_cancellation_sum,
                "reward_deadline_miss_sum": reward_deadline_miss_sum,
                "reward_paper_sum": reward_paper_sum,
                "reward_noop_bonus_sum": reward_noop_bonus_sum,
                "reward_budget_change_penalty_sum": reward_budget_change_penalty_sum,
                "reward_budget_change_norm_sum": reward_budget_change_norm_sum,
                "reward_budget_drift_penalty_sum": reward_budget_drift_penalty_sum,
                "reward_budget_drift_mean_sum": reward_budget_drift_mean_sum,
                "noop_exploration_prob": args.noop_exploration_prob,
                "exploration_action_count": int(agent.exploration_action_count - exploration_action_count_before),
                "exploration_noop_action_count": int(
                    agent.exploration_noop_action_count - exploration_noop_action_count_before
                ),
                "exploration_noop_action_rate": (
                    float(agent.exploration_noop_action_count - exploration_noop_action_count_before)
                    / float(agent.exploration_action_count - exploration_action_count_before)
                    if int(agent.exploration_action_count - exploration_action_count_before) > 0
                    else 0.0
                ),
                "observation_mode": str(last_info.get("observation_mode", args.observation_mode)),
                "state_dim": int(last_info.get("state_dim", len(obs.state_vector))),
                "feature_safety_margin_min_mean": (
                    sum(feature_safety_margin_min_values) / len(feature_safety_margin_min_values)
                    if feature_safety_margin_min_values
                    else 0.0
                ),
                "feature_safety_margin_min_p05": _percentile(feature_safety_margin_min_values, 0.05),
            }
        )
        for action_id in sorted(episode_action_hist):
            action_hist = episode_action_hist[action_id]
            action_hist_rows.append(
                {
                    "episode": episode,
                    "action_id": action_id,
                    "count": action_hist["count"],
                    "accepted_count": action_hist["accepted"],
                    "rejected_count": action_hist["rejected"],
                }
            )

        if args.trace_every > 0 and (episode + 1) % args.trace_every == 0 and args.trace_dir is not None:
            runtime_result = env._engine.finish() if env._engine is not None else SimulationResult()
            _write_jsonl(args.trace_dir / f"episode_{episode + 1:04d}_action_log.jsonl", env.action_log)
            _write_jsonl(args.trace_dir / f"episode_{episode + 1:04d}_mask_log.jsonl", env.mask_log)
            _write_jsonl(
                args.trace_dir / f"episode_{episode + 1:04d}_runtime_trace.jsonl",
                _trace_rows_from_runtime(runtime_result),
            )

        if args.checkpoint > 0 and (episode + 1) % args.checkpoint == 0:
            agent.save(checkpoint_dir / f"model_episode_{episode + 1:04d}.pt")

        if args.validate_every > 0 and validation_seeds and (episode + 1) % args.validate_every == 0:
            validation_row, baseline_validation_cache, used_baseline_cache = _run_validation(
                agent=agent,
                experiment_config=experiment_config,
                validation_seeds=validation_seeds,
                validation_end_time=args.validation_end_time,
                agent_period=args.agent_period,
                reward_mode=args.reward_mode,
                action_space=args.action_space,
                budget_increase_ratio=args.budget_increase_ratio,
                budget_decrease_ratio=args.budget_decrease_ratio,
                include_explicit_noop=args.include_explicit_noop,
                budget_floor_ratio=args.budget_floor_ratio,
                forbid_decreasing_hi_budgets=args.forbid_decreasing_hi_budgets,
                mask_detail_mode=args.mask_detail_mode,
                feature_config=feature_config,
                validation_workers=args.validation_workers,
                baseline_cache=baseline_validation_cache,
                max_q_diagnostic_samples=args.max_q_diagnostic_samples,
                constraint_guided_pair_top_k_risk=args.constraint_guided_pair_top_k_risk,
                constraint_guided_pair_top_k_decrease=args.constraint_guided_pair_top_k_decrease,
                constraint_guided_pair_prefer_lo=args.constraint_guided_pair_prefer_lo,
                constraint_guided_pair_include_hi_risk_boost=args.constraint_guided_pair_include_hi_risk_boost,
                constraint_guided_pair_allow_increase_only_when_safe=(
                    args.constraint_guided_pair_allow_increase_only_when_safe
                ),
            )
            if used_baseline_cache:
                print("Using cached baseline validation metrics")
            validation_row["episode"] = episode + 1
            # 阶段 2：显式记录相对 baseline 的两类 delta，并按 alpha 计算综合分数。
            # 当前版本采用“归一化 validation 口径”，使 validation 选模尺度与 interval reward 更一致：
            # - raw_delta_lo   = dqn_lo_cancellations_mean - baseline_lo_cancellations_mean
            # - raw_delta_mode = dqn_mode_changes_mean    - baseline_mode_changes_mean
            # - normalized_delta_lo   = raw_delta_lo   / max(1, baseline_lo_cancellations_mean)
            # - normalized_delta_mode = raw_delta_mode / max(1, baseline_mode_changes_mean)
            # 公式：relative_score = normalized_delta_lo + alpha * normalized_delta_mode
            # 因此 relative_score 越小越好，<0 表示按归一化综合指标优于 baseline。
            # raw delta 仍写入额外字段，便于人工查看实际 count 差异。
            raw_delta_lo = (
                float(validation_row["lo_cancellations_mean"])
                - float(validation_row["baseline_lo_cancellations_mean"])
            )
            raw_delta_mode = float(validation_row["mode_changes_mean"]) - float(
                validation_row["baseline_mode_changes_mean"]
            )

            baseline_lo_denominator = max(
                1.0,
                float(validation_row["baseline_lo_cancellations_mean"]),
            )
            baseline_mode_denominator = max(
                1.0,
                float(validation_row["baseline_mode_changes_mean"]),
            )

            normalized_delta_lo = raw_delta_lo / baseline_lo_denominator
            normalized_delta_mode = raw_delta_mode / baseline_mode_denominator

            validation_row["relative_score_alpha"] = args.relative_score_alpha
            validation_row["raw_delta_lo_cancellations"] = raw_delta_lo
            validation_row["raw_delta_mode_changes"] = raw_delta_mode
            validation_row["relative_delta_lo_cancellations"] = normalized_delta_lo
            validation_row["relative_delta_mode_changes"] = normalized_delta_mode
            validation_row["relative_score"] = (
                normalized_delta_lo
                + args.relative_score_alpha * normalized_delta_mode
            )
            validation_row["is_better_than_baseline"] = float(validation_row["relative_score"]) < 0.0
            # 标记当前候选是否满足 Pareto-valid，用于“先筛选，再排序”的两阶段策略。
            validation_row["is_pareto_valid"] = _is_pareto_valid_checkpoint(validation_row)
            validation_rows.append(validation_row)

            should_update_best = False
            if args.save_best_by == "pareto_relative_score":
                # 方向 2：Pareto 选模改为“先筛选，再排序”
                # ------------------------------------------
                # 阶段一（筛选）：
                # - 先判断当前候选是否 Pareto-valid。
                # - 一旦历史中出现过 Pareto-valid，后续只允许在 Pareto-valid 集合中竞争 best。
                #
                # 阶段二（排序）：
                # - 在 Pareto-valid 集合内，按 relative_score（越小越好）排序。
                #
                # fallback（仅在阶段一尚未出现任何 Pareto-valid 时）：
                # - 临时使用温和软分数 `pareto_relative_score` 维持“有 best 可保存”的行为。
                is_pareto_valid = bool(validation_row["is_pareto_valid"])
                if is_pareto_valid:
                    if not pareto_valid_seen:
                        # 首次出现 Pareto-valid：立刻切换到“只看 Pareto-valid”阶段，
                        # 并直接把当前候选设为新的 best 基准。
                        pareto_valid_seen = True
                        should_update_best = True
                    else:
                        # 已进入 Pareto-only 阶段：
                        # - 如果当前 best 不是 Pareto-valid（理论上只会在切换边界出现），直接替换；
                        # - 否则只比较 relative_score。
                        if best_validation_row is None or not _is_pareto_valid_checkpoint(best_validation_row):
                            should_update_best = True
                        else:
                            should_update_best = float(validation_row["relative_score"]) < float(
                                best_validation_row["relative_score"]
                            )
                else:
                    if not pareto_valid_seen:
                        # 还没见到任何 Pareto-valid，允许 fallback 到温和软分数排序。
                        should_update_best = _is_better_validation_row(
                            candidate_row=validation_row,
                            best_row=best_validation_row,
                            save_best_by="pareto_relative_score",
                            relative_score_alpha=args.relative_score_alpha,
                            require_better_than_baseline_for_best=args.require_better_than_baseline_for_best,
                        )
                    else:
                        # 已见到 Pareto-valid 后，非 Pareto-valid 候选直接丢弃，不参与 best 竞争。
                        should_update_best = False
            else:
                should_update_best = _is_better_validation_row(
                    candidate_row=validation_row,
                    best_row=best_validation_row,
                    save_best_by=args.save_best_by,
                    relative_score_alpha=args.relative_score_alpha,
                    require_better_than_baseline_for_best=args.require_better_than_baseline_for_best,
                )

            if should_update_best:
                best_validation_row = validation_row
                agent.save(model_best_path)
                best_model_saved = True
            print(
                {
                    "episode": episode + 1,
                    "validation_reward_mean": float(validation_row["reward_mean"]),
                    "train_loss_last": loss_last,
                }
            )

    train_log_path = output_dir / "train_log.csv"
    train_metrics_path = output_dir / "train_metrics.csv"
    action_hist_path = output_dir / "train_action_histogram.csv"
    validation_metrics_path = output_dir / "validation_metrics.csv"
    validation_unified_summary_path = output_dir / "validation_unified_summary.csv"
    model_path = output_dir / "model_final.pt"
    config_path = output_dir / "config.json"

    with train_log_path.open("w", encoding="utf-8", newline="") as f:
        # 即使 `--log-step-every 0` 让 step_rows 为空，也仍然写出固定 header，
        # 这样下游脚本可以明确识别“文件存在但没有逐步日志”。
        writer = csv.DictWriter(f, fieldnames=STEP_LOG_FIELDNAMES)
        writer.writeheader()
        writer.writerows(step_rows)

    if args.log_train_metrics:
        metric_fieldnames = [
            "episode",
            "episode_seed",
            "taskset_seed",
            "scenario_seed",
            "network_seed",
            "exploration_seed",
            "replay_seed",
            "total_util",
            "num_tasks",
            "taskset_fingerprint",
            "steps",
            "step_count",
            "selected_action_count",
            "total_reward",
            "epsilon",
            "loss_mean",
            "loss_last",
            "accepted_actions",
            "rejected_actions",
            "noop_actions",
            "explicit_noop_actions",
            "noop_action_rate",
            "explicit_noop_action_rate",
            "accepted_action_rate",
            "rejected_action_rate",
            "safety_checked_actions",
            "selected_invalid_mask_actions",
            "action_space_type",
            "action_count",
            "budget_increase_ratio",
            "budget_decrease_ratio",
            "budget_floor_ratio",
            "valid_action_count_mean",
            "masked_action_count_mean",
            "masked_decrease_hi_forbidden_count",
            "masked_decrease_hi_forbidden_rate",
            "masked_budget_floor_violation_count",
            "masked_budget_floor_violation_rate",
            "no_safe_action_steps",
            "selected_explicit_noop_actions",
            "selected_explicit_noop_rate",
            "mode_changes",
            "lo_cancellations",
            "deadline_misses",
            "job_starts",
            "lo_overruns",
            "hi_overruns",
            "reward_job_start_sum",
            "reward_lo_overrun_sum",
            "reward_hi_overrun_sum",
            "reward_mode_change_sum",
            "reward_lo_cancellation_sum",
            "reward_deadline_miss_sum",
            "reward_paper_sum",
            "reward_noop_bonus_sum",
            "reward_budget_change_penalty_sum",
            "reward_budget_change_norm_sum",
            "reward_budget_drift_penalty_sum",
            "reward_budget_drift_mean_sum",
            "noop_exploration_prob",
            "exploration_action_count",
            "exploration_noop_action_count",
            "exploration_noop_action_rate",
            "observation_mode",
            "state_dim",
            "feature_safety_margin_min_mean",
            "feature_safety_margin_min_p05",
        ]
        with train_metrics_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=metric_fieldnames)
            writer.writeheader()
            writer.writerows(train_metric_rows)

    action_hist_fieldnames = ["episode", "action_id", "count", "accepted_count", "rejected_count"]
    with action_hist_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=action_hist_fieldnames)
        writer.writeheader()
        writer.writerows(action_hist_rows)

    if validation_rows:
        validation_fieldnames = [
            "episode",
            "validation_seed_count",
            "deadline_misses_sum",
            "mode_changes_mean",
            "lo_cancellations_mean",
            "baseline_mode_changes_mean",
            "baseline_lo_cancellations_mean",
            "dqn_mode_changes_delta_mean",
            "dqn_lo_cancellations_delta_mean",
            "accepted_actions_mean",
            "rejected_actions_mean",
            "step_count_mean",
            "selected_action_count_mean",
            "noop_actions_mean",
            "explicit_noop_actions_mean",
            "noop_action_rate_mean",
            "explicit_noop_action_rate_mean",
            "accepted_action_rate_mean",
            "rejected_action_rate_mean",
            "valid_action_count_mean",
            "masked_action_count_mean",
            "no_safe_action_steps_mean",
            "reward_mean",
            "observation_mode",
            "state_dim_mean",
            "relative_score_alpha",
            "raw_delta_lo_cancellations",
            "raw_delta_mode_changes",
            "relative_delta_lo_cancellations",
            "relative_delta_mode_changes",
            "relative_score",
            "is_better_than_baseline",
            "is_pareto_valid",
        ]
        validation_fieldnames.extend(NOOP_Q_DIAGNOSTIC_FIELDNAMES)
        with validation_metrics_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=validation_fieldnames)
            writer.writeheader()
            writer.writerows(validation_rows)
        # 阶段 0/1 统一 summary：保持与评估脚本一致的核心字段命名，便于直接横向对比。
        validation_unified_summary_rows = _build_validation_unified_summary_rows(validation_rows)
        validation_unified_summary_fields = [
            "episode",
            "validation_seed_count",
            "baseline_mode_changes_mean",
            "baseline_lo_cancellations_mean",
            "dqn_mode_changes_mean",
            "dqn_lo_cancellations_mean",
            "relative_score",
            "relative_score_alpha",
            "relative_delta_mode_changes",
            "relative_delta_lo_cancellations",
            "raw_delta_mode_changes",
            "raw_delta_lo_cancellations",
            "is_better_than_baseline",
            "mode_changes_delta_vs_baseline",
            "lo_cancellations_delta_vs_baseline",
            "mode_change_ratio",
            "lo_cancellation_ratio",
            "accepted_action_count_mean",
            "rejected_action_count_mean",
            "noop_action_count_mean",
            "noop_action_rate_mean",
            "explicit_noop_action_rate_mean",
            "accepted_action_rate_mean",
            "rejected_action_rate_mean",
            "masked_action_count_mean",
            "valid_action_count_mean",
        ]
        validation_unified_summary_fields.extend(NOOP_Q_DIAGNOSTIC_FIELDNAMES)
        with validation_unified_summary_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=validation_unified_summary_fields)
            writer.writeheader()
            writer.writerows(validation_unified_summary_rows)

    agent.save(model_path)
    if validation_rows:
        if best_validation_row is None:
            best_validation_row = validation_rows[-1]
        best_relative_score = float(best_validation_row.get("relative_score", 0.0))
        best_delta_mode = float(best_validation_row.get("relative_delta_mode_changes", 0.0))
        best_delta_lo = float(best_validation_row.get("relative_delta_lo_cancellations", 0.0))
        best_raw_delta_mode = float(
            best_validation_row.get(
                "raw_delta_mode_changes",
                best_validation_row.get("dqn_mode_changes_delta_mean", 0.0),
            )
        )
        best_raw_delta_lo = float(
            best_validation_row.get(
                "raw_delta_lo_cancellations",
                best_validation_row.get("dqn_lo_cancellations_delta_mean", 0.0),
            )
        )
        best_model_metadata = {
            "save_best_by": args.save_best_by,
            "selection_metric": (
                "relative_score"
                if args.save_best_by == "relative_score"
                else (
                    "pareto_relative_score"
                    if args.save_best_by == "pareto_relative_score"
                    else args.save_best_by
                )
            ),
            "relative_score_alpha": args.relative_score_alpha,
            "require_better_than_baseline_for_best": args.require_better_than_baseline_for_best,
            "best_validation_episode": int(best_validation_row["episode"]),
            "best_relative_score": best_relative_score,
            "dqn_lo_cancellations_mean": float(best_validation_row["lo_cancellations_mean"]),
            "baseline_lo_cancellations_mean": float(best_validation_row["baseline_lo_cancellations_mean"]),
            "dqn_mode_changes_mean": float(best_validation_row["mode_changes_mean"]),
            "baseline_mode_changes_mean": float(best_validation_row["baseline_mode_changes_mean"]),
            "normalized_delta_lo_cancellations": best_delta_lo,
            "normalized_delta_mode_changes": best_delta_mode,
            "raw_delta_lo_cancellations_mean": best_raw_delta_lo,
            "raw_delta_mode_changes_mean": best_raw_delta_mode,
            "best_model_is_better_than_baseline": best_relative_score < 0.0,
            "reward_mode": args.reward_mode,
            "reward_definition": reward_mode_config.describe(),
            "double_dqn": args.double_dqn,
            "note": (
                "model_best.pt is the best available checkpoint on validation, but it may still be worse than baseline."
            ),
            "selection_reason": (
                "Best checkpoint selected by configured criterion. "
                "For relative_score / pareto_relative_score, validation uses normalized deltas against baseline."
            ),
        }
        with best_model_metadata_path.open("w", encoding="utf-8") as f:
            json.dump(best_model_metadata, f, ensure_ascii=False, indent=2)
        if best_relative_score >= 0.0:
            # 文档要求：即使当前 best 仍劣于 baseline，也必须保留 best checkpoint。
            # 因此这里只打印提示，不中断、不回滚、不跳过保存。
            print("WARNING: Best available checkpoint is still worse than baseline on validation.")
            print("Saved anyway for trend analysis: model_best.pt")
    config_payload = {
        "dqn_config": asdict(config),
        "workload": args.workload,
        "scenario": args.scenario,
        "scenario_name": initial_bundle.scenario.name,
        "total_util": args.total_util,
        "num_tasks": (
            args.mc_fairgen_num_tasks if args.workload == "mc_fairgen" else args.num_tasks
        ),
        "cf": args.cf,
        "cp": args.cp,
        "require_schedulable": args.require_schedulable,
        "hi_overrun_prob": args.hi_overrun_prob,
        "lo_overrun_prob": args.lo_overrun_prob,
        "lo_overrun_factor": args.lo_overrun_factor,
        "train_seed_mode": args.train_seed_mode,
        "train_seeds": episode_seed_schedule,
        "scenario_seed_offset": args.scenario_seed_offset,
        "fixed_taskset_seed": args.fixed_taskset_seed,
        "mc_fairgen_mode": args.mc_fairgen_mode,
        "mc_fairgen_num_tasks": args.mc_fairgen_num_tasks,
        "mc_fairgen_hi_ratio": args.mc_fairgen_hi_ratio,
        "mc_fairgen_period_source": args.mc_fairgen_period_source,
        "mc_fairgen_period_scale": args.mc_fairgen_period_scale,
        "mc_fairgen_tick_ns": 10,
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
        "validation_seeds": validation_seeds,
        "validate_every": args.validate_every,
        "validation_end_time": args.validation_end_time,
        "validation_workers": args.validation_workers,
        "max_q_diagnostic_samples": args.max_q_diagnostic_samples,
        "save_best_by": args.save_best_by,
        "relative_score_alpha": args.relative_score_alpha,
        "relative_score_normalization": "baseline_mean_denominator",
        "reward_mode": args.reward_mode,
        "reward_definition": reward_mode_config.describe(),
        "double_dqn": args.double_dqn,
        "requested_action_space": requested_action_space,
        "action_space": args.action_space,
        "constraint_guided_transfer_top_k_risk": args.constraint_guided_pair_top_k_risk,
        "constraint_guided_transfer_top_k_decrease": args.constraint_guided_pair_top_k_decrease,
        "constraint_guided_transfer_prefer_lo": args.constraint_guided_pair_prefer_lo,
        "constraint_guided_transfer_include_hi_risk_boost": args.constraint_guided_pair_include_hi_risk_boost,
        "constraint_guided_pair_top_k_risk": args.constraint_guided_pair_top_k_risk,
        "constraint_guided_pair_top_k_decrease": args.constraint_guided_pair_top_k_decrease,
        "constraint_guided_pair_prefer_lo": args.constraint_guided_pair_prefer_lo,
        "constraint_guided_pair_include_hi_risk_boost": args.constraint_guided_pair_include_hi_risk_boost,
        "constraint_guided_pair_allow_increase_only_when_safe": (
            args.constraint_guided_pair_allow_increase_only_when_safe
        ),
        "budget_increase_ratio": args.budget_increase_ratio,
        "budget_decrease_ratio": args.budget_decrease_ratio,
        "budget_floor_ratio": args.budget_floor_ratio,
        "include_explicit_noop": args.include_explicit_noop,
        "forbid_decreasing_hi_budgets": args.forbid_decreasing_hi_budgets,
        "mask_detail_mode": args.mask_detail_mode,
        "observation_mode": args.observation_mode,
        "feature_config": {
            "ema_alpha": args.ema_alpha,
            "overrun_ema_alpha": args.overrun_ema_alpha,
            "history_k": args.history_k,
            "event_window": args.event_window,
            "max_cost_weight": args.max_cost_weight,
            "risk_max_scale": args.risk_max_scale,
            "include_safety_margin": args.include_safety_margin,
        },
        "log_step_every": args.log_step_every,
        "log_train_metrics": args.log_train_metrics,
        "trace_every": args.trace_every,
        "seed_metadata": {
            "base_seed": args.seed,
            "network_seed": network_seed,
            "exploration_seed": exploration_seed,
            "replay_seed": replay_seed,
        },
        "normalization_bounds": {
            task_name: {"min_cost": bound.min_cost, "max_cost": bound.max_cost}
            for task_name, bound in initial_bundle.normalization_bounds.items()
        },
        "action_space_size": initial_env.action_space_size,
        "observation_dim": len(initial_obs.state_vector),
        "tasks": _serialize_tasks(list(initial_bundle.ordered_tasks)),
        "runtime_config": {
            "end_time": args.end_time,
            "agent_period": args.agent_period,
            "semantics": RuntimeSemantics.AMC_PLUS.value,
        },
        "effective_taskset_seed": (
            (initial_bundle.metadata or {}).get("workload_metadata", {}).get("effective_taskset_seed")
            if isinstance((initial_bundle.metadata or {}).get("workload_metadata", {}), dict)
            else None
        ),
        "provider_attempts": (
            (initial_bundle.metadata or {}).get("workload_metadata", {}).get("attempts")
            if isinstance((initial_bundle.metadata or {}).get("workload_metadata", {}), dict)
            else None
        ),
    }
    with config_path.open("w", encoding="utf-8") as f:
        json.dump(config_payload, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
