"""任务集生成器。

本模块负责论文复现实验中的任务生成语义，重点能力：
1. 通过 `time_scale` 提升内部时间分辨率，降低利用率离散化误差；
2. 通过 `deadline_mode` 显式控制截止期生成策略；
3. 通过 `lo_hi_budget_policy` 控制 LO 任务是否拥有独立分析用 `c_hi`；
4. 支持 fixed_count / bernoulli 两种关键级分配策略；
5. 支持从 YAML 加载配置并构建 `fast/paper` 预设。
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Sequence

from .models import Criticality, Task

CriticalityAssignment = Literal["fixed_count", "bernoulli"]
GeneratorMode = Literal["fast", "paper"]
DeadlineMode = Literal["implicit", "ratio_uniform", "arbitrary_paper"]
LoHiBudgetPolicy = Literal["equal_lo", "scaled_by_cf"]


@dataclass(frozen=True, slots=True)
class GenerationConfig:
    """任务集生成配置。

    字段语义说明（与论文复现相关）：
    - min_period / max_period：论文单位下的周期范围（例如 ms）；
    - time_scale：内部离散 tick 缩放因子，真实参与计算的是 `period * time_scale`；
    - deadline_mode：截止期采样策略，避免仅靠布尔参数表达多种语义；
    - lo_hi_budget_policy：LO 任务是否拥有独立分析预算 c_hi（用于区分 SMC 与 SMC-NO）。
    """

    num_tasks: int
    total_util: float
    min_period: int = 10
    max_period: int = 1000
    time_scale: int = 100
    cf: float = 2.0
    cp: float = 0.5
    deadline_mode: DeadlineMode = "implicit"
    deadline_ratio_min: float = 0.5
    criticality_assignment: CriticalityAssignment = "fixed_count"
    lo_hi_budget_policy: LoHiBudgetPolicy = "scaled_by_cf"

    def __post_init__(self) -> None:
        """统一配置合法性校验。"""

        if self.num_tasks <= 0:
            raise ValueError("num_tasks 必须为正整数")
        if self.total_util <= 0:
            raise ValueError("total_util 必须为正数")
        if self.min_period <= 0 or self.max_period <= 0:
            raise ValueError("min_period / max_period 必须为正整数")
        if self.min_period > self.max_period:
            raise ValueError("min_period 不能大于 max_period")
        if self.time_scale <= 0:
            raise ValueError("time_scale 必须为正整数")
        if self.cf < 1.0:
            raise ValueError("cf 必须 >= 1.0")
        if not (0.0 <= self.cp <= 1.0):
            raise ValueError("cp 必须在 [0,1] 区间内")
        if self.deadline_mode not in {"implicit", "ratio_uniform", "arbitrary_paper"}:
            raise ValueError("deadline_mode 仅支持 implicit / ratio_uniform / arbitrary_paper")
        if not (0.0 < self.deadline_ratio_min <= 1.0):
            raise ValueError("deadline_ratio_min 必须在 (0,1] 区间内")
        if self.criticality_assignment not in {"fixed_count", "bernoulli"}:
            raise ValueError("criticality_assignment 仅支持 fixed_count / bernoulli")
        if self.lo_hi_budget_policy not in {"equal_lo", "scaled_by_cf"}:
            raise ValueError("lo_hi_budget_policy 仅支持 equal_lo / scaled_by_cf")


# 两套预设模式：
# - fast：本地快速验证。
# - paper：更贴近论文复现实验配置。
GENERATOR_MODE_DEFAULTS: dict[GeneratorMode, dict[str, Any]] = {
    "fast": {
        "num_tasks": 8,
        "total_util": 0.7,
        "min_period": 10,
        "max_period": 200,
        "time_scale": 10,
        "cf": 2.0,
        "cp": 0.5,
        "deadline_mode": "implicit",
        "deadline_ratio_min": 0.5,
        "criticality_assignment": "fixed_count",
        "lo_hi_budget_policy": "scaled_by_cf",
    },
    "paper": {
        "num_tasks": 20,
        "total_util": 0.8,
        "min_period": 10,
        "max_period": 1000,
        "time_scale": 100,
        "cf": 2.0,
        "cp": 0.5,
        "deadline_mode": "implicit",
        "deadline_ratio_min": 0.5,
        "criticality_assignment": "bernoulli",
        "lo_hi_budget_policy": "scaled_by_cf",
    },
}


def uunifast(num_tasks: int, total_util: float, rng: random.Random | None = None) -> list[float]:
    """使用 UUniFast 生成利用率向量。

    UUniFast 的关键性质是：返回向量元素均为正，且总和严格等于 total_util（忽略浮点误差）。
    """

    if num_tasks <= 0:
        raise ValueError("num_tasks 必须为正整数")
    if total_util <= 0:
        raise ValueError("total_util 必须为正数")

    local_rng = rng or random.Random()
    utils: list[float] = []
    sum_u = total_util

    for idx in range(1, num_tasks):
        next_sum_u = sum_u * (local_rng.random() ** (1 / (num_tasks - idx)))
        utils.append(sum_u - next_sum_u)
        sum_u = next_sum_u

    utils.append(sum_u)
    return utils


def sample_period_log_uniform(min_period: int, max_period: int, rng: random.Random | None = None) -> int:
    """在 [min_period, max_period] 上按 log-uniform 采样“论文单位”周期。

    论文常用对数均匀采样周期，目的是避免大量样本集中在大周期或小周期端。
    """

    if min_period <= 0 or max_period <= 0:
        raise ValueError("period 边界必须为正整数")
    if min_period > max_period:
        raise ValueError("min_period 不能大于 max_period")

    local_rng = rng or random.Random()
    sampled = math.exp(local_rng.uniform(math.log(min_period), math.log(max_period)))
    return int(round(sampled))


def scale_period(period: int, time_scale: int) -> int:
    """把论文单位周期缩放为内部 tick。"""

    if period <= 0:
        raise ValueError("period 必须为正整数")
    if time_scale <= 0:
        raise ValueError("time_scale 必须为正整数")
    return period * time_scale


def compute_task_budgets(
    util: float,
    period: int,
    criticality: Criticality,
    cf: float,
    lo_hi_budget_policy: LoHiBudgetPolicy,
) -> tuple[int, int]:
    """根据 util/period/关键级计算 (c_lo, c_hi)。

    说明：
    - HI 任务始终按 `c_hi = max(c_lo, round(c_lo * cf))`。
    - LO 任务由 `lo_hi_budget_policy` 决定是否拥有独立分析预算：
      - equal_lo: c_hi = c_lo
      - scaled_by_cf: c_hi = max(c_lo, round(c_lo * cf))
    """

    if util <= 0:
        raise ValueError("util 必须为正数")
    if period <= 0:
        raise ValueError("period 必须为正整数")
    if cf < 1.0:
        raise ValueError("cf 必须 >= 1.0")

    # 执行预算离散化到整数 tick。最小为 1 tick，避免“零执行时间任务”。
    c_lo = max(1, int(round(util * period)))

    if criticality is Criticality.HI:
        return c_lo, max(c_lo, int(round(c_lo * cf)))

    if lo_hi_budget_policy == "equal_lo":
        return c_lo, c_lo
    if lo_hi_budget_policy == "scaled_by_cf":
        return c_lo, max(c_lo, int(round(c_lo * cf)))

    raise ValueError("unsupported lo_hi_budget_policy")


def sample_deadline(
    period: int,
    criticality: Criticality,
    c_lo: int,
    c_hi: int,
    deadline_mode: DeadlineMode,
    rng: random.Random,
    deadline_ratio_min: float = 0.5,
) -> int:
    """按 deadline_mode 采样截止期。

    三种模式：
    - implicit：D = T；
    - ratio_uniform：D 从比例下界到 T 均匀采样；
    - arbitrary_paper：严格按论文语义，HI 任务从 [c_hi, T]，LO 任务从 [c_lo, T] 采样。
    """

    if period <= 0:
        raise ValueError("period 必须为正整数")

    if deadline_mode == "implicit":
        return period

    if deadline_mode == "ratio_uniform":
        min_by_ratio = int(math.ceil(period * deadline_ratio_min))
        c_required = c_hi if criticality is Criticality.HI else c_lo
        min_deadline = max(min_by_ratio, c_required)
        if min_deadline > period:
            raise ValueError("ratio_uniform 下无合法 deadline 区间，请检查参数")
        return rng.randint(min_deadline, period)

    if deadline_mode == "arbitrary_paper":
        min_deadline = c_hi if criticality is Criticality.HI else c_lo
        if min_deadline > period:
            raise ValueError("arbitrary_paper 下无合法 deadline 区间，请检查参数")
        return rng.randint(min_deadline, period)

    raise ValueError("unsupported deadline_mode")


def generate_task(
    task_id: int,
    util: float,
    period: int,
    criticality: Criticality,
    cf: float,
    rng: random.Random | None = None,
    deadline_mode: DeadlineMode = "implicit",
    deadline_ratio_min: float = 0.5,
    lo_hi_budget_policy: LoHiBudgetPolicy = "scaled_by_cf",
) -> Task:
    """根据参数构造单个任务。

    先预算、后截止期的顺序非常关键：
    - 截止期下界依赖 c_lo/c_hi；
    - 避免“先定 deadline 再裁剪预算”导致模型语义被悄悄改写。
    """

    local_rng = rng or random.Random()

    # 先生成预算，再生成 deadline，避免先定 deadline 后反向裁剪预算。
    c_lo, c_hi = compute_task_budgets(
        util=util,
        period=period,
        criticality=criticality,
        cf=cf,
        lo_hi_budget_policy=lo_hi_budget_policy,
    )

    deadline = sample_deadline(
        period=period,
        criticality=criticality,
        c_lo=c_lo,
        c_hi=c_hi,
        deadline_mode=deadline_mode,
        rng=local_rng,
        deadline_ratio_min=deadline_ratio_min,
    )

    if deadline < c_lo:
        raise ValueError("deadline 小于 c_lo，生成逻辑不合法")

    return Task(
        name=f"tau_{task_id}",
        period=period,
        deadline=deadline,
        c_lo=c_lo,
        c_hi=c_hi,
        criticality=criticality,
    )


def _choose_hi_indices_fixed_count(num_tasks: int, cp: float, rng: random.Random) -> set[int]:
    """按 cp 比例固定抽取 HI 任务（fixed_count）。"""

    if not (0.0 <= cp <= 1.0):
        raise ValueError("cp 必须在 [0,1] 区间内")

    hi_count = int(round(num_tasks * cp))
    indices = list(range(num_tasks))
    rng.shuffle(indices)
    return set(indices[:hi_count])


def _choose_hi_indices_bernoulli(num_tasks: int, cp: float, rng: random.Random) -> set[int]:
    """按伯努利采样选择 HI 任务索引集合。"""

    if not (0.0 <= cp <= 1.0):
        raise ValueError("cp 必须在 [0,1] 区间内")

    return {idx for idx in range(num_tasks) if rng.random() < cp}


def choose_hi_indices(
    num_tasks: int,
    cp: float,
    rng: random.Random,
    strategy: CriticalityAssignment = "fixed_count",
) -> set[int]:
    """按指定策略选取 HI 任务索引。"""

    if strategy == "fixed_count":
        return _choose_hi_indices_fixed_count(num_tasks, cp, rng)
    if strategy == "bernoulli":
        return _choose_hi_indices_bernoulli(num_tasks, cp, rng)
    raise ValueError("strategy 仅支持 fixed_count / bernoulli")


def make_generation_config(mode: GeneratorMode, **overrides: Any) -> GenerationConfig:
    """根据模式创建配置对象，并允许覆盖局部参数。"""

    base = dict(GENERATOR_MODE_DEFAULTS[mode])
    base.update(overrides)
    return GenerationConfig(**base)


def _parse_scalar(value: str) -> Any:
    """把 YAML 简单标量字符串解析为 Python 值。"""

    raw = value.strip()
    lower = raw.lower()
    if lower in {"true", "false"}:
        return lower == "true"

    try:
        if "." in raw:
            return float(raw)
        return int(raw)
    except ValueError:
        return raw


def load_generation_config(path: str | Path) -> GenerationConfig:
    """从简化 YAML 文件加载生成配置。"""

    config_path = Path(path)
    content = config_path.read_text(encoding="utf-8")
    parsed: dict[str, Any] = {}

    for line in content.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if "#" in stripped:
            stripped = stripped.split("#", 1)[0].strip()
        if not stripped:
            continue
        if ":" not in stripped:
            raise ValueError(f"无效配置行（缺少冒号）: {line}")
        key, value = stripped.split(":", 1)
        parsed[key.strip()] = _parse_scalar(value)

    # 兼容旧字段：deadline_equals_period -> deadline_mode
    if "deadline_mode" not in parsed and "deadline_equals_period" in parsed:
        parsed["deadline_mode"] = "implicit" if parsed["deadline_equals_period"] else "ratio_uniform"
        del parsed["deadline_equals_period"]

    return GenerationConfig(**parsed)


def _resolve_deadline_mode(
    deadline_mode: DeadlineMode | None,
    deadline_equals_period: bool | None,
) -> DeadlineMode:
    """兼容旧参数：若调用方仍传 bool，则映射为新枚举模式。

    说明：该函数只用于兼容历史调用；新代码应直接传 deadline_mode。
    """

    if deadline_mode is not None:
        return deadline_mode
    if deadline_equals_period is None:
        return "implicit"
    return "implicit" if deadline_equals_period else "ratio_uniform"


def generate_taskset(
    num_tasks: int,
    total_util: float,
    min_period: int = 10,
    max_period: int = 1000,
    time_scale: int = 100,
    cf: float = 2.0,
    cp: float = 0.5,
    seed: int | None = None,
    deadline_mode: DeadlineMode | None = None,
    deadline_ratio_min: float = 0.5,
    criticality_assignment: CriticalityAssignment = "fixed_count",
    lo_hi_budget_policy: LoHiBudgetPolicy = "scaled_by_cf",
    # 兼容旧调用：仍支持传 deadline_equals_period。
    deadline_equals_period: bool | None = None,
) -> list[Task]:
    """生成一个混合关键级任务集。

    重要实现点：
    1. 先用 UUniFast 固定总利用率；
    2. 对每个任务独立采样周期并缩放为 tick；
    3. 如果某次 period 无法满足预算/截止期约束，重采样 period，
       而不是静默修改 c_lo/c_hi，确保预算语义稳定可解释。
    """

    resolved_deadline_mode = _resolve_deadline_mode(deadline_mode, deadline_equals_period)

    config = GenerationConfig(
        num_tasks=num_tasks,
        total_util=total_util,
        min_period=min_period,
        max_period=max_period,
        time_scale=time_scale,
        cf=cf,
        cp=cp,
        deadline_mode=resolved_deadline_mode,
        deadline_ratio_min=deadline_ratio_min,
        criticality_assignment=criticality_assignment,
        lo_hi_budget_policy=lo_hi_budget_policy,
    )

    rng = random.Random(seed)
    utils = uunifast(config.num_tasks, config.total_util, rng=rng)
    hi_indices = choose_hi_indices(config.num_tasks, config.cp, rng, strategy=config.criticality_assignment)

    taskset: list[Task] = []
    for idx, util in enumerate(utils, start=1):
        criticality = Criticality.HI if (idx - 1) in hi_indices else Criticality.LO
        last_error: ValueError | None = None

        # 当当前 period 无法满足预算/截止期约束时，重采样 period。
        # 这样可以避免静默修改 c_lo/c_hi，保持预算语义与配置一致。
        # 上限 256 次用于防止极端参数下出现无限循环。
        for _ in range(256):
            period_base = sample_period_log_uniform(config.min_period, config.max_period, rng=rng)
            period = scale_period(period_base, config.time_scale)
            try:
                taskset.append(
                    generate_task(
                        task_id=idx,
                        util=util,
                        period=period,
                        criticality=criticality,
                        cf=config.cf,
                        rng=rng,
                        deadline_mode=config.deadline_mode,
                        deadline_ratio_min=config.deadline_ratio_min,
                        lo_hi_budget_policy=config.lo_hi_budget_policy,
                    )
                )
                break
            except ValueError as exc:
                last_error = exc
        else:
            raise ValueError("连续重采样 period 仍无法生成合法任务，请检查配置参数") from last_error

    return taskset


def taskset_total_util(taskset: Sequence[Task], mode: Criticality = Criticality.LO) -> float:
    """计算任务集总利用率。

    注意：HI 模式统计仅累计 HI 任务，这与论文实验的口径一致。
    """

    if mode is Criticality.LO:
        return sum(task.c_lo / task.period for task in taskset)
    return sum(task.c_hi / task.period for task in taskset if task.criticality is Criticality.HI)
