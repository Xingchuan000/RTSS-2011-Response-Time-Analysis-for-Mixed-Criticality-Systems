"""运行时仿真的执行场景层（阶段运行时模拟 · 第 2 轮）。

本模块描述“每个 job 的实际执行时间如何被决定”这件事，它是 runtime
仿真器与具体实验假设之间的接口层。AMC 语义中，任务的 `c_lo`/`c_hi`
只是“设计时参数与分析上界”，而 job 真正会跑多久取决于实验设定：

- **nominal**：所有 job 都跑 `c_lo`，不会触发 HI 切换；
- **single HI overrun**：只有某个 HI 任务的指定 release 跑到 `c_hi`
  或 `c_lo + 1`，其它 job 跑 `c_lo`，可复现一次 HI 切换；
- **all HI jobs HI budget**：所有（或指定的）HI 任务每次都跑 `c_hi`，
  用来复现“最坏情况”压力；
- **table**：通过一张 (task_name, release_index) → actual_cost 的表
  显式指定，方便写针对性单测。

统一入口是 `ExecutionScenario`，runtime 仿真器在释放 job 时调用
`scenario.actual_cost_for(task, release_index)` 取实际执行时间，并由
本模块的 `_validate_actual_cost` 保证基础合法性约束：

- HI 任务：`1 <= actual_cost <= task.c_hi`
- LO 任务：`actual_cost >= 1`（允许超过 `c_lo`，由 runtime 在预算层决定是否取消）
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
import math
import random
from numbers import Integral

from .models import Criticality, Task

# ---------------------------------------------------------------------------
# 类型别名
# ---------------------------------------------------------------------------

# Scenario 的“解析器”签名：给定任务与 release 索引，返回该 job 的实际执行时间。
# 这里刻意不把 release_time 也传进来，因为它可以由 `release_index * task.period`
# 算出；保持参数最少，避免 scenario 的使用者被迫关心 runtime 内部时序。
ActualCostResolver = Callable[[Task, int], int]

# 合法的“单次 HI 超限”目标预算策略。
# - "c_hi":         直接跑到 C(HI)，最常见的复现方式；
# - "c_lo_plus_one": 只跑到 C(LO) + 1，用于精确测试“刚好越过 LO 边界”的临界场景。
_SINGLE_HI_OVERRUN_TARGETS = {"c_hi", "c_lo_plus_one"}

# table scenario 中 HI 任务默认目标预算的合法取值。
_DEFAULT_HI_TARGETS = {"c_lo", "c_hi"}

# table scenario 中 LO 任务默认目标预算的合法取值。
# 目前仅允许 "c_lo"；保留该集合的形式是为了后续扩展（例如若未来支持
# 随机抽取 LO job 的执行时间，可以在此扩充关键字）。
_DEFAULT_LO_TARGETS = {"c_lo"}


def _validate_actual_cost(task: Task, actual_cost: int) -> None:
    """统一校验某个 job 的 `actual_cost` 是否满足关键级约束。

    约束：
    - 必须是正整数（至少 1 个 tick），否则仿真器无法排程；
    - HI 任务：`actual_cost` 不得超过 `task.c_hi`；
    - LO 任务：允许超过 `task.c_lo`，因为 AMC+ 运行时会在预算检查时
      对超预算 LO job 进行局部取消。

    该函数在 scenario 层集中执行，这样无论是内置工厂还是用户自定义
    resolver，只要经过 `ExecutionScenario.actual_cost_for` 都能被拦下。
    """

    if not isinstance(actual_cost, int):
        raise TypeError(
            f"任务 {task.name} 的 actual_cost 必须是 int，收到 {type(actual_cost).__name__}"
        )

    if actual_cost < 1:
        raise ValueError(
            f"任务 {task.name} 的 actual_cost={actual_cost} 非法，必须 >= 1"
        )

    if task.criticality is Criticality.HI:
        if actual_cost > task.c_hi:
            raise ValueError(
                f"HI 任务 {task.name} 的 actual_cost={actual_cost} 超过 c_hi={task.c_hi}"
            )
    else:
        # AMC+ 下 LO job 允许超过 c_lo，是否继续执行由 runtime budget 机制决定。
        pass


@dataclass(frozen=True, slots=True)
class ExecutionScenario:
    """运行时场景：把“谁跑多久”的实验假设封装成一个对象。

    字段：
    - `name`:     可读的场景名（只用于调试/日志/结果标注）；
    - `resolver`: 核心回调，签名见 `ActualCostResolver`。

    约定：调用方永远不直接使用 `resolver`，而是通过 `actual_cost_for()`
    方法去拿实际执行时间 —— 这样可以保证所有路径都会经过
    `_validate_actual_cost` 的检查。
    """

    name: str
    resolver: ActualCostResolver

    def actual_cost_for(self, task: Task, release_index: int) -> int:
        """返回某个 job 的实际执行时间，并执行关键级约束校验。

        - 不允许 resolver 返回负值/零/超上界的数；
        - 只接受“整数类型”返回值（例如内建 int、numpy integer）；
        - 会拒绝 float / str / bool 等非整数类型，避免出现静默截断或隐式转换。
        """

        if release_index < 0:
            raise ValueError(
                f"release_index={release_index} 非法，必须 >= 0"
            )

        raw_value = self.resolver(task, release_index)

        # 这里必须执行“严格类型校验”，避免发生以下静默错误：
        # - 1.9 被 int(...) 截断成 1；
        # - "2" 被 int(...) 解析成 2；
        # - True 被当成 1（因为 bool 是 int 的子类）。
        # 这些情况都会污染实验含义，所以一律 fail-fast。
        if isinstance(raw_value, bool) or not isinstance(raw_value, Integral):
            raise TypeError(
                f"任务 {task.name} 的 actual_cost 必须是整数类型，收到 {type(raw_value).__name__}"
            )

        # 经过严格类型检查后，再统一规范化为内建 int，便于后续预算边界校验。
        coerced = int(raw_value)
        _validate_actual_cost(task, coerced)
        return coerced


# ---------------------------------------------------------------------------
# 场景工厂
# ---------------------------------------------------------------------------


def make_nominal_scenario() -> ExecutionScenario:
    """构造“所有 job 都按 C(LO) 执行”的标称场景。

    语义：
    - HI 任务每次执行时间 = `task.c_lo`；
    - LO 任务每次执行时间 = `task.c_lo`。

    这是最常用的基准场景，运行时仿真在此场景下应当完全不触发 HI 切换。
    """

    def resolver(task: Task, release_index: int) -> int:  # noqa: ARG001 (release_index 故意保留在签名中)
        # nominal 场景：不论 HI 还是 LO，都按 C(LO) 作为实际执行时间。
        return task.c_lo

    return ExecutionScenario(name="nominal", resolver=resolver)


def make_single_hi_overrun_scenario(
    task_name: str,
    release_index: int = 0,
    overrun_to: str = "c_hi",
) -> ExecutionScenario:
    """构造“单次 HI 超限”的场景：指定任务的指定一次释放跑超预算。

    参数：
    - `task_name`:     触发超限的 HI 任务名；
    - `release_index`: 该任务第几次 release 超限（默认第 0 次）；
    - `overrun_to`:    `c_hi` 或 `c_lo_plus_one`，决定该 job 实际跑到多少。

    约束（call-time 才能校验，因为此时拿不到 Task 对象）：
    - 被指定的任务必须是 HI 任务；
    - 若 `overrun_to == "c_lo_plus_one"`，该 HI 任务必须满足 `c_hi > c_lo`；
    - 其他任意 job 一律跑 `c_lo`，不触发额外的 HI 切换。

    如果调用方传入了未知的 `overrun_to`，工厂直接抛 ValueError。
    """

    if overrun_to not in _SINGLE_HI_OVERRUN_TARGETS:
        supported = ", ".join(sorted(_SINGLE_HI_OVERRUN_TARGETS))
        raise ValueError(
            f"不支持的 overrun_to={overrun_to}，可选值：{supported}"
        )

    if release_index < 0:
        raise ValueError(f"release_index={release_index} 非法，必须 >= 0")

    def resolver(task: Task, current_release_index: int) -> int:
        # 命中被“钦定”超限的那一次 release。
        if task.name == task_name and current_release_index == release_index:
            # 仅允许 HI 任务执行超限；LO 任务原则上不应该被“故意超限”。
            if task.criticality is not Criticality.HI:
                raise ValueError(
                    f"single HI overrun 场景仅支持 HI 任务，但 {task.name} 为 LO"
                )

            if overrun_to == "c_hi":
                # 跑满 c_hi —— 常见复现方式。
                return task.c_hi

            # overrun_to == "c_lo_plus_one"：需要 c_hi 严格大于 c_lo。
            if task.c_hi <= task.c_lo:
                raise ValueError(
                    f"任务 {task.name} 的 c_hi={task.c_hi} 不大于 c_lo={task.c_lo}，"
                    "无法应用 c_lo_plus_one 场景"
                )
            return task.c_lo + 1

        # 其余 job 一律按标称执行 —— 这是“single HI overrun”的核心语义。
        return task.c_lo

    scenario_name = f"single_hi_overrun[{task_name}@{release_index}->{overrun_to}]"
    return ExecutionScenario(name=scenario_name, resolver=resolver)


def make_single_lo_overrun_scenario(
    task_name: str,
    release_index: int = 0,
    actual_cost: int | None = None,
) -> ExecutionScenario:
    """构造“单次 LO overrun”的场景。

    语义：
    - 命中指定任务和 release 时，返回 `actual_cost`（默认 `task.c_lo + 1`）；
    - 命中的任务必须为 LO 任务；
    - 其它所有 job 返回 `task.c_lo`。
    """

    if release_index < 0:
        raise ValueError(f"release_index={release_index} 非法，必须 >= 0")

    def resolver(task: Task, current_release_index: int) -> int:
        if task.name == task_name and current_release_index == release_index:
            if task.criticality is not Criticality.LO:
                raise ValueError("single LO overrun scenario only supports LO tasks")
            return actual_cost if actual_cost is not None else task.c_lo + 1
        return task.c_lo

    scenario_name = f"single_lo_overrun[{task_name}@{release_index}]"
    return ExecutionScenario(name=scenario_name, resolver=resolver)


def make_all_hi_jobs_hi_budget_scenario(
    task_names: Iterable[str] | None = None,
) -> ExecutionScenario:
    """构造“所有 HI 任务每次都跑 c_hi”的最坏情况场景。

    参数：
    - `task_names`: 可选过滤集合；
        * None：对“所有”HI 任务生效（最严格）；
        * 具体集合：只有名字命中该集合的 HI 任务才跑 c_hi，其余（包括
          未命中的 HI 任务与所有 LO 任务）一律跑 c_lo。

    注意：如果 `task_names` 中包含了 LO 任务名，该任务的 job 仍然会被
    强制跑 c_lo —— 我们不会为了“配合实验”而违反关键级约束。
    """

    # 归一化过滤集合；None 表示“不过滤”。
    filter_set: frozenset[str] | None
    filter_set = None if task_names is None else frozenset(task_names)

    def resolver(task: Task, release_index: int) -> int:  # noqa: ARG001
        if task.criticality is Criticality.HI:
            if filter_set is None or task.name in filter_set:
                # 被选中的 HI 任务：跑满 c_hi。
                return task.c_hi
        # LO 任务或未被选中的 HI 任务：标称执行。
        return task.c_lo

    scenario_name = (
        "all_hi_jobs_hi_budget"
        if filter_set is None
        else f"all_hi_jobs_hi_budget[{sorted(filter_set)}]"
    )
    return ExecutionScenario(name=scenario_name, resolver=resolver)


def make_table_scenario(
    actual_costs: Mapping[tuple[str, int], int],
    default_hi: str = "c_lo",
    default_lo: str = "c_lo",
) -> ExecutionScenario:
    """构造“按表显式指定 job 实际执行时间”的场景。

    参数：
    - `actual_costs`: 字典，键为 `(task_name, release_index)`，值为该 job
      期望的实际执行时间（具体是否合法仍会在 `actual_cost_for` 被调用
      时由关键级校验兜底）；
    - `default_hi`:   未命中表的 HI 任务的默认预算，取 `c_lo` 或 `c_hi`；
    - `default_lo`:   未命中表的 LO 任务的默认预算，目前仅支持 `c_lo`。

    使用方式：
    - 先在测试中列出所有“需要特别指定”的 job；
    - 未列出的 job 则根据其关键级按 default 参数回退。

    校验：
    - `default_hi` / `default_lo` 必须在受支持集合中，否则立刻抛 ValueError
      （fail-fast）；
    - 表中的每个值都是 int；
    - 表中每个 key 必须是 `(str, int)` 形式，release_index >= 0。
    """

    if default_hi not in _DEFAULT_HI_TARGETS:
        supported = ", ".join(sorted(_DEFAULT_HI_TARGETS))
        raise ValueError(
            f"不支持的 default_hi={default_hi}，可选值：{supported}"
        )
    if default_lo not in _DEFAULT_LO_TARGETS:
        supported = ", ".join(sorted(_DEFAULT_LO_TARGETS))
        raise ValueError(
            f"不支持的 default_lo={default_lo}，可选值：{supported}"
        )

    # 对入参做最基本的结构性校验，避免把非法键藏到仿真运行时才暴露。
    normalized: dict[tuple[str, int], int] = {}
    for key, value in actual_costs.items():
        if (
            not isinstance(key, tuple)
            or len(key) != 2
            or not isinstance(key[0], str)
            or not isinstance(key[1], int)
        ):
            raise ValueError(
                f"actual_costs 的键必须是 (task_name: str, release_index: int)，收到 {key!r}"
            )
        task_name, release_index = key
        if release_index < 0:
            raise ValueError(
                f"actual_costs 中 release_index={release_index} 非法，必须 >= 0"
            )
        if not isinstance(value, int):
            raise TypeError(
                f"actual_costs[{key!r}] 必须是 int，收到 {type(value).__name__}"
            )
        normalized[(task_name, release_index)] = value

    def resolver(task: Task, release_index: int) -> int:
        key = (task.name, release_index)
        if key in normalized:
            # 命中表：直接返回用户指定的值，合法性交由 _validate_actual_cost 兜底。
            return normalized[key]

        # 未命中表：按默认策略回退。
        if task.criticality is Criticality.HI:
            return task.c_lo if default_hi == "c_lo" else task.c_hi
        # LO 默认只能是 c_lo（见 _DEFAULT_LO_TARGETS 的注释）。
        return task.c_lo

    scenario_name = f"table[default_hi={default_hi},default_lo={default_lo}]"
    return ExecutionScenario(name=scenario_name, resolver=resolver)


def make_rtss11_random_scenario(
    tasks: Iterable[Task],
    seed: int,
    hi_overrun_prob: float = 0.05,
    lo_overrun_prob: float = 0.10,
    lo_overrun_factor: float = 1.5,
) -> ExecutionScenario:
    """构造 RTSS2011 风格随机执行场景。

    采样规则：
    - HI 任务：
      1. 以 `1 - hi_overrun_prob` 的概率采样 `[0.5*C_LO, C_LO]`；
      2. 以 `hi_overrun_prob` 的概率采样 `(C_LO, C_HI]`。
    - LO 任务：
      1. 以 `1 - lo_overrun_prob` 的概率采样 `[0.5*C_LO, C_LO]`；
      2. 以 `lo_overrun_prob` 的概率采样 `(C_LO, lo_overrun_factor*C_LO]`。

    设计说明：
    - 使用 `(seed, task_name, release_index)` 构造局部 RNG，确保同一 taskset + seed
      在任意调用顺序下都可复现同样的样本；
    - HI 任务采样上界严格受 `C_HI` 约束，永不超过设计时 HI 预算。
    """

    if not (0.0 <= hi_overrun_prob <= 1.0):
        raise ValueError("hi_overrun_prob 必须在 [0,1] 区间内")
    if not (0.0 <= lo_overrun_prob <= 1.0):
        raise ValueError("lo_overrun_prob 必须在 [0,1] 区间内")
    if lo_overrun_factor <= 1.0:
        raise ValueError("lo_overrun_factor 必须 > 1")

    task_list = list(tasks)
    if not task_list:
        raise ValueError("tasks 不能为空")

    # 这里不对每个 LO 任务做“可形成整数 overrun 区间”的前置拒绝。
    # 原因：RTSS11 任务集中允许出现 c_lo=1 的小预算任务，此时连续区间
    # (C_LO, factor*C_LO] 可能没有整数点。该情况应在采样语义中处理，而不是抛错。

    def _task_name_code(name: str) -> int:
        """把任务名编码为稳定整数，避免依赖 Python 随机哈希。"""

        acc = 0
        for idx, ch in enumerate(name):
            acc += (idx + 1) * ord(ch)
        return acc

    def _job_rng(task: Task, release_index: int) -> random.Random:
        """为每个 job 构造独立 RNG，保证采样与调用顺序无关。"""

        mixed_seed = (
            seed * 1_000_003
            + _task_name_code(task.name) * 97
            + release_index * 9_973
        )
        return random.Random(mixed_seed)

    def resolver(task: Task, release_index: int) -> int:
        rng = _job_rng(task, release_index)
        nominal_low = max(1, math.ceil(0.5 * task.c_lo))
        nominal_high = task.c_lo

        if task.criticality is Criticality.HI:
            # HI 任务 overrun 仅在存在严格上界空间时触发，确保始终满足 actual_cost <= C_HI。
            if rng.random() < hi_overrun_prob and task.c_hi > task.c_lo:
                return rng.randint(task.c_lo + 1, task.c_hi)
            return rng.randint(nominal_low, nominal_high)

        if rng.random() < lo_overrun_prob:
            # 对小预算 LO 任务，使用 “至少 c_lo+1” 与 “ceil(factor*c_lo)” 取最大值，
            # 保证最小 1 tick overrun 可采样，不会因整数区间为空而崩溃。
            lo_overrun_upper = max(
                task.c_lo + 1,
                int(math.ceil(lo_overrun_factor * task.c_lo)),
            )
            return rng.randint(task.c_lo + 1, lo_overrun_upper)
        return rng.randint(nominal_low, nominal_high)

    scenario_name = (
        "rtss11_random"
        f"[seed={seed},hi_p={hi_overrun_prob},lo_p={lo_overrun_prob},lo_f={lo_overrun_factor}]"
    )
    return ExecutionScenario(name=scenario_name, resolver=resolver)


__all__ = [
    "ActualCostResolver",
    "ExecutionScenario",
    "make_nominal_scenario",
    "make_single_hi_overrun_scenario",
    "make_single_lo_overrun_scenario",
    "make_all_hi_jobs_hi_budget_scenario",
    "make_table_scenario",
    "make_rtss11_random_scenario",
]
