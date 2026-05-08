"""v11 观测特征的运行时状态缓存。

本模块对应实现计划“阶段 2”，职责是维护会跨 step 变化的统计量：
1. 每任务 cost EMA；
2. 每任务 overrun EMA；
3. 每任务最近 k 次 cost 历史（用于 max-k 特征）；
4. 全局事件窗口（mode-change / cancellation / overrun / job-start）。

注意：
- 本文件只负责“存储与窗口维护”，不负责 reward/safety/action 语义；
- dataclass 不可 frozen，因为训练过程中需要原位更新。
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field


@dataclass(slots=True)
class RuntimeFeatureState:
    """v11 observation 运行时特征缓存。

    字段说明：
    - history_k：每个任务 cost_history 的最大长度；
    - event_window：全局事件窗口的最大长度（按“step 事件块”计数）。
    """

    # 每任务成本历史窗口长度（用于 max-cost-k 特征）。
    history_k: int = 8
    # 全局事件率窗口长度（用于近端事件率特征）。
    event_window: int = 10

    # 每任务执行成本 EMA：key=task_name, value=ema_cost。
    ema_cost: dict[str, float] = field(default_factory=dict)
    # 每任务超预算事件 EMA：key=task_name, value=overrun_ema。
    overrun_ema: dict[str, float] = field(default_factory=dict)
    # 每任务最近 k 次执行成本历史。
    # 使用 deque(maxlen=history_k) 自动限制长度，不需要手工裁剪。
    cost_history: dict[str, deque[float]] = field(default_factory=dict)
    # 每任务“上次已消费的新样本计数”：
    # 该值与 RuntimeMonitor.completed_job_count_by_task 对齐使用。
    # 只有当 monitor 的当前计数 > 这里记录的 last_seen 计数时，
    # 才说明该任务在当前 step 区间内出现了新样本，允许更新 EMA/history。
    last_seen_completion_count: dict[str, int] = field(default_factory=dict)

    # 以下五个窗口按 step 同步追加：同一索引代表同一时间片的统计。
    window_mode_changes: deque[int] = field(default_factory=deque)
    window_lo_cancellations: deque[int] = field(default_factory=deque)
    window_hi_overruns: deque[int] = field(default_factory=deque)
    window_lo_overruns: deque[int] = field(default_factory=deque)
    window_job_starts: deque[int] = field(default_factory=deque)

    def init_task(self, task_name: str, init_cost: float) -> None:
        """按需初始化单个任务的特征缓存。

        设计要点：
        - 只在 key 不存在时初始化，避免覆盖已学习到的 EMA/历史；
        - ema_cost 初值优先由调用方传入（计划建议使用 task.c_lo）；
        - overrun_ema 初值固定为 0.0；
        - cost_history 使用 deque(maxlen=history_k) 严格限制长度。
        """

        if task_name not in self.ema_cost:
            self.ema_cost[task_name] = float(init_cost)
        if task_name not in self.overrun_ema:
            self.overrun_ema[task_name] = 0.0
        if task_name not in self.cost_history:
            self.cost_history[task_name] = deque(maxlen=self.history_k)
        if task_name not in self.last_seen_completion_count:
            self.last_seen_completion_count[task_name] = 0

    def append_event_window(
        self,
        *,
        mode_changes: int,
        lo_cancellations: int,
        hi_overruns: int,
        lo_overruns: int,
        job_starts: int,
    ) -> None:
        """追加一组“同一 step”的全局事件计数，并在末尾统一裁剪窗口长度。"""

        self.window_mode_changes.append(int(mode_changes))
        self.window_lo_cancellations.append(int(lo_cancellations))
        self.window_hi_overruns.append(int(hi_overruns))
        self.window_lo_overruns.append(int(lo_overruns))
        self.window_job_starts.append(int(job_starts))
        self._trim_windows()

    def _trim_windows(self) -> None:
        """将事件窗口裁剪到 event_window 内。

        这里以 window_job_starts 长度作为统一基准，保证五个窗口严格等长。
        每次超长时同步 popleft，避免索引错位导致 rate 统计污染。
        """

        while len(self.window_job_starts) > self.event_window:
            self.window_mode_changes.popleft()
            self.window_lo_cancellations.popleft()
            self.window_hi_overruns.popleft()
            self.window_lo_overruns.popleft()
            self.window_job_starts.popleft()

    def window_denominator_jobs(self) -> int:
        """返回事件率分母（窗口内 job_starts 总和），并保证分母至少为 1。"""

        return max(1, int(sum(self.window_job_starts)))

    def rate(self, window: deque[int]) -> float:
        """按统一分母计算窗口事件率，避免分母为 0。"""

        return float(sum(window)) / float(self.window_denominator_jobs())
