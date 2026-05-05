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

    def record_job_start(self, task_name: str) -> None:
        """记录一次 job 首次开始执行并累计奖励。"""

        self.job_start_count += 1
        self.reward_since_last_agent += self.reward_weights.job_start
        self.recent_execution.setdefault(task_name, 0)

    def record_job_completion(self, task_name: str, executed_time: int) -> None:
        """记录一次 job 完成时的最终执行量。"""

        self.recent_execution[task_name] = executed_time

    def record_lo_budget_overrun(self, task_name: str, executed_time: int) -> None:
        """记录 LO 预算超限事件并累计负奖励。"""

        self.lo_overrun_count += 1
        self.reward_since_last_agent += self.reward_weights.lo_overrun
        self.recent_execution[task_name] = executed_time

    def record_hi_budget_overrun(self, task_name: str, executed_time: int) -> None:
        """记录 HI 预算超限事件并累计更高负奖励。"""

        self.hi_overrun_count += 1
        self.reward_since_last_agent += self.reward_weights.hi_overrun
        self.recent_execution[task_name] = executed_time

    def consume_reward(self) -> float:
        """取出并清空自上次 agent 读取后的累计奖励。"""

        reward = self.reward_since_last_agent
        self.reward_since_last_agent = 0.0
        return reward
