"""teacher registry 行构造与校验。"""

from __future__ import annotations

from pathlib import Path

from amc_py.dqn import DqnBudgetAgent


def build_teacher_registry_row(
    *,
    teacher_id: str,
    taskset_seed: int | None,
    model_path: Path,
    config_path: Path | None,
    train_output_dir: Path | None,
    runtime_semantics: str,
    c_amc_sem_xf: float,
    reward_mode: str,
    action_space: str,
    observation_mode: str,
    agent_period: int,
    budget_increase_ratio: float,
    budget_decrease_ratio: float,
    budget_floor_ratio: float,
    forbid_decreasing_hi_budgets: bool,
    enable_deploy_cap_mask: bool,
    deploy_cap_mask_ratio: float,
    deploy_cap_mask_criticality: str,
    checkpoint_kind: str,
) -> dict[str, object]:
    """读取 checkpoint 元信息并构造 registry row。"""

    if not model_path.exists():
        raise FileNotFoundError(f"teacher model 不存在: {model_path}")
    agent = DqnBudgetAgent.load(model_path)
    return {
        "teacher_id": teacher_id,
        "taskset_seed": taskset_seed,
        "model_path": str(model_path),
        "config_path": (None if config_path is None else str(config_path)),
        "train_output_dir": (None if train_output_dir is None else str(train_output_dir)),
        "runtime_semantics": runtime_semantics,
        "c_amc_sem_xf": c_amc_sem_xf,
        "reward_mode": reward_mode,
        "action_space": action_space,
        "observation_mode": observation_mode,
        "state_dim": int(agent.observation_dim),
        "action_dim": int(agent.action_dim),
        "agent_period": int(agent_period),
        "budget_increase_ratio": float(budget_increase_ratio),
        "budget_decrease_ratio": float(budget_decrease_ratio),
        "budget_floor_ratio": float(budget_floor_ratio),
        "forbid_decreasing_hi_budgets": bool(forbid_decreasing_hi_budgets),
        "enable_deploy_cap_mask": bool(enable_deploy_cap_mask),
        "deploy_cap_mask_ratio": float(deploy_cap_mask_ratio),
        "deploy_cap_mask_criticality": deploy_cap_mask_criticality,
        "checkpoint_kind": checkpoint_kind,
    }
