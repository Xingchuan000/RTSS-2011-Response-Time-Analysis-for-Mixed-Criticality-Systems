"""v11 观测特征配置与状态定义。

本文件只负责两件事：
1. 定义 observation_mode 与特征超参数配置（对应实现计划“阶段一”）；
2. 固化 v11_full_10d 的状态维度定义（每任务 10 维 + 全局 8 维）。

注意：
- 这里不实现运行时特征更新逻辑（EMA/history/event window），那属于后续阶段；
- 默认模式必须保持 v10_basic，确保旧实验与旧脚本行为不变。
"""

from __future__ import annotations

from dataclasses import dataclass

# 旧版本观测模式常量：
# v10_basic 的状态向量按每任务 2 维构造（budget_norm, recent_cost_norm）。
OBSERVATION_MODE_V10_BASIC = "v10_basic"

# 新版本观测模式常量：
# v11_full_10d 的状态向量为“每任务 10 维 + 全局 8 维”。
OBSERVATION_MODE_V11_FULL_10D = "v11_full_10d"
OBSERVATION_MODE_V12_FULL_14D = "v12_full_14d"

# v10 基础模式下，每个任务固定 2 个输入维度。
V10_PER_TASK_FEATURE_DIM = 2

# v11 全量模式下，每个任务固定 10 个输入维度。
V11_PER_TASK_FEATURE_DIM = 10

# v11 全量模式下，全局附加 8 个输入维度。
V11_GLOBAL_FEATURE_DIM = 8
V12_PER_TASK_FEATURE_DIM = 14
V12_GLOBAL_FEATURE_DIM = 8

# v11 每任务 10 维的特征名称，顺序必须与设计文档一致。
# 后续真正拼接 state_vector 时必须严格按该顺序输出，避免训练输入语义漂移。
V11_PER_TASK_FEATURE_NAMES: tuple[str, ...] = (
    "budget_norm",
    "recent_cost_norm",
    "ema_cost_norm",
    "max_cost_k_norm",
    "overrun_ema",
    "risk",
    "surplus",
    "criticality",
    "priority_norm",
    "util_budget",
)

# v11 全局 8 维的特征名称，顺序必须与设计文档一致。
V11_GLOBAL_FEATURE_NAMES: tuple[str, ...] = (
    "total_budget_util",
    "hi_budget_util",
    "lo_budget_util",
    "recent_mode_change_rate",
    "recent_lo_cancel_rate",
    "recent_hi_overrun_rate",
    "recent_lo_overrun_rate",
    "safety_margin_min",
)

# v12 每任务 14 维特征名称（前 10 维与 v11 保持一致，后 4 维为新增特征）。
V12_PER_TASK_FEATURE_NAMES: tuple[str, ...] = (
    "budget_norm",
    "recent_cost_norm",
    "ema_cost_norm",
    "max_cost_k_norm",
    "overrun_ema",
    "risk",
    "surplus",
    "criticality",
    "priority_norm",
    "util_budget",
    "positive_budget_drift",
    "negative_budget_drift",
    "task_cancel_ema",
    "safe_inc_possible",
)

# v12 全局维度定义沿用 v11 的 8 维。
V12_GLOBAL_FEATURE_NAMES: tuple[str, ...] = V11_GLOBAL_FEATURE_NAMES


@dataclass(frozen=True, slots=True)
class FeatureConfig:
    """v11 observation feature 配置。

    字段含义与设计文档保持一一对应：
    - observation_mode：当前使用的观测模式，默认 v10_basic（兼容旧行为）；
    - ema_alpha：cost EMA 更新系数；
    - overrun_ema_alpha：overrun 事件 EMA 更新系数；
    - history_k：max-k 特征使用的历史窗口长度；
    - event_window：全局事件率统计窗口长度；
    - max_cost_weight：risk/surplus 中 max_cost_k 的权重；
    - risk_max_scale：raw_risk 到 [0,1] 的缩放系数；
    - include_safety_margin：是否将安全裕量作为全局特征接入。
    """

    # 观测模式默认值必须是 v10_basic，确保不影响历史实验。
    observation_mode: str = OBSERVATION_MODE_V10_BASIC

    # ema_cost 的指数滑动平均系数（alpha）。
    ema_alpha: float = 0.2
    # overrun_ema 的指数滑动平均系数（alpha_o）。
    overrun_ema_alpha: float = 0.1
    # max_cost_k 特征的历史长度 k。
    history_k: int = 8
    # 全局事件率（mode change / cancel / overrun）的滑动窗口长度。
    event_window: int = 10

    # pred_cost = max(recent_cost, ema_cost, max_cost_weight * max_cost_k) 中的系数。
    max_cost_weight: float = 0.7
    # risk = clip(raw_risk / risk_max_scale, 0, 1) 中的缩放系数。
    risk_max_scale: float = 3.0

    # 是否在全局特征中纳入 safety_margin_min。
    include_safety_margin: bool = True

    def expected_state_dim(self, task_count: int) -> int:
        """根据当前 observation_mode 计算理论状态维度。

        该方法只做维度定义层面的纯计算，不涉及任何运行时环境状态：
        - v10_basic：2 * n_tasks；
        - v11_full_10d：10 * n_tasks + 8。
        """

        if task_count < 0:
            raise ValueError("task_count 必须为非负整数")
        if self.observation_mode == OBSERVATION_MODE_V10_BASIC:
            return V10_PER_TASK_FEATURE_DIM * task_count
        if self.observation_mode == OBSERVATION_MODE_V11_FULL_10D:
            return V11_PER_TASK_FEATURE_DIM * task_count + V11_GLOBAL_FEATURE_DIM
        if self.observation_mode == OBSERVATION_MODE_V12_FULL_14D:
            return V12_PER_TASK_FEATURE_DIM * task_count + V12_GLOBAL_FEATURE_DIM
        raise ValueError(f"不支持的 observation_mode: {self.observation_mode}")
