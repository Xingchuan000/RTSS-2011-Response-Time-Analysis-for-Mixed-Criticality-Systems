"""运行时监控器：统计最近执行量与奖励累计。"""

from __future__ import annotations

from dataclasses import dataclass, field

from amc_py.rl.reward_config import RewardWeights, load_reward_weights


@dataclass(slots=True)
class RuntimeMonitor:
    """用于 RL 预集成阶段的轻量运行时统计。"""

    # 奖励模式名称。运行时会据此从 `configs/reward_modes/<mode>.json` 加载参数。
    reward_mode: str = "mendes"
    # 解析后的奖励参数，统一供 record_* 方法累计奖励时使用。
    reward_weights: RewardWeights = field(init=False)
    recent_execution: dict[str, int] = field(default_factory=dict)
    # 每任务“新样本版本号”计数器：
    # 只要该任务在运行时产生了一个新的 executed_time 样本（正常完成或 overrun 截断），
    # 对应计数就 +1。后续 feature_state 会用它判断“本 step 是否出现新样本”，
    # 避免把旧 recent_execution 重复当作新观测去更新 EMA/history。
    completed_job_count_by_task: dict[str, int] = field(default_factory=dict)
    # 每任务 overrun 事件计数器：
    # 该字段用于保留 overrun 事件的显式统计事实，不直接作为 EMA 更新门控，
    # 但可用于后续诊断“某任务最近是否在持续触发超预算”。
    overrun_count_by_task: dict[str, int] = field(default_factory=dict)
    reward_since_last_agent: float = 0.0
    job_start_count: int = 0
    lo_overrun_count: int = 0
    hi_overrun_count: int = 0

    def __post_init__(self) -> None:
        """根据 reward_mode 读取奖励配置文件。"""

        self.reward_weights = load_reward_weights(self.reward_mode)

    def ensure_tasks(self, task_names: list[str]) -> None:
        """确保每个任务在 recent_execution 中都有键位。"""

        for name in task_names:
            self.recent_execution.setdefault(name, 0)
            self.completed_job_count_by_task.setdefault(name, 0)
            self.overrun_count_by_task.setdefault(name, 0)

    def record_job_start(self, task_name: str) -> None:
        """记录一次 job 首次开始执行并累计奖励。"""

        self.job_start_count += 1
        self.reward_since_last_agent += self.reward_weights.job_start
        self.recent_execution.setdefault(task_name, 0)

    def record_job_completion(self, task_name: str, executed_time: int) -> None:
        """记录一次 job 完成时的最终执行量。"""

        self.recent_execution[task_name] = executed_time
        # 正常完成会产生一个新的 executed_time 样本，因此样本版本号 +1。
        self.completed_job_count_by_task[task_name] = self.completed_job_count_by_task.get(task_name, 0) + 1

    def record_lo_budget_overrun(self, task_name: str, executed_time: int) -> None:
        """记录 LO 预算超限事件并累计负奖励。"""

        self.lo_overrun_count += 1
        self.reward_since_last_agent += self.reward_weights.lo_overrun
        self.recent_execution[task_name] = executed_time
        # overrun 同样会产出新的 executed_time 样本（被截断/终止时的执行量），
        # 必须计入 completed_job_count_by_task，确保特征更新能看到这次新样本。
        self.completed_job_count_by_task[task_name] = self.completed_job_count_by_task.get(task_name, 0) + 1
        # 同时累计 overrun 次数，供独立诊断使用。
        self.overrun_count_by_task[task_name] = self.overrun_count_by_task.get(task_name, 0) + 1

    def record_hi_budget_overrun(self, task_name: str, executed_time: int) -> None:
        """记录 HI 预算超限事件并累计更高负奖励。"""

        self.hi_overrun_count += 1
        self.reward_since_last_agent += self.reward_weights.hi_overrun
        self.recent_execution[task_name] = executed_time
        # HI overrun 也属于新样本，逻辑与 LO overrun 一致。
        self.completed_job_count_by_task[task_name] = self.completed_job_count_by_task.get(task_name, 0) + 1
        self.overrun_count_by_task[task_name] = self.overrun_count_by_task.get(task_name, 0) + 1

    def consume_reward(self) -> float:
        """取出并清空自上次 agent 读取后的累计奖励。"""

        reward = self.reward_since_last_agent
        self.reward_since_last_agent = 0.0
        return reward
