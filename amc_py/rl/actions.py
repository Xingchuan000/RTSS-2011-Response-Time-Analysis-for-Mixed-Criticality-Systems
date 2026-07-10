"""离散预算动作空间与候选预算更新构建。"""

from __future__ import annotations

import math
from dataclasses import dataclass
from decimal import Decimal
from fractions import Fraction
from itertools import combinations, permutations
from collections.abc import Sequence

from amc_py.budget_runtime import BudgetState
from amc_py.models import Criticality, Task


@dataclass(frozen=True, slots=True)
class BudgetRatio:
    """预算比例的规范有理数表示，避免正式路径依赖二进制浮点边界。"""

    numerator: int
    denominator: int

    def __post_init__(self) -> None:
        if self.denominator <= 0 or self.numerator < 0 or self.numerator >= self.denominator:
            raise ValueError("BudgetRatio 必须位于 [0, 1)")

    @classmethod
    def from_decimal_string(cls, value: str) -> "BudgetRatio":
        fraction = Fraction(Decimal(value))
        return cls(fraction.numerator, fraction.denominator)

    @classmethod
    def from_float_via_string(cls, value: float) -> "BudgetRatio":
        return cls.from_decimal_string(str(value))


def ceil_multiply_ratio(base: int, numerator: int, denominator: int) -> int:
    if base < 0 or numerator < 0 or denominator <= 0:
        raise ValueError("整数比例参数非法")
    return (base * numerator + denominator - 1) // denominator


def floor_multiply_ratio(base: int, numerator: int, denominator: int) -> int:
    if base < 0 or numerator < 0 or denominator <= 0:
        raise ValueError("整数比例参数非法")
    return (base * numerator) // denominator


def compute_increase_candidate(base: int, increase_ratio: BudgetRatio) -> int:
    return base + ceil_multiply_ratio(base, increase_ratio.numerator, increase_ratio.denominator)


def compute_decrease_candidate(base: int, decrease_ratio: BudgetRatio) -> int:
    # Dec(B)=floor(B*(1-r))。减少量必须使用 ceil(B*r)，否则 8*5% 会被
    # 舍成 0 而错误地变成 no-op。
    decrease_amount = ceil_multiply_ratio(base, decrease_ratio.numerator, decrease_ratio.denominator)
    return base - decrease_amount


@dataclass(frozen=True, slots=True)
class BudgetAction:
    """单个离散预算动作。"""

    action_id: int
    increase_task: str | None
    decrease_tasks: tuple[str, ...]
    increase_idx: int | None = None
    decrease_indices: tuple[int, ...] = ()
    increase_ratio: float = 0.10
    decrease_ratio: float = 0.05
    increase_ratio_num: int | None = None
    increase_ratio_den: int | None = None
    decrease_ratio_num: int | None = None
    decrease_ratio_den: int | None = None
    action_space_type: str = "triple"
    is_noop: bool = False
    # constraint-guided transfer 固定槽位标记：
    # 该类动作不会在动作空间阶段绑定具体任务索引，而是在环境中按当前状态动态解析。
    is_constraint_guided_pair: bool = False
    constraint_guided_increase_rank: int | None = None
    # residual_ranked 固定槽位标记：
    # - 动作空间只定义语义槽位，不在构建时绑定具体 task；
    # - 具体目标任务在 env.step() 中按当前风险排序动态解析。
    is_residual_ranked: bool = False
    residual_action_type: str | None = None
    residual_rank: int | None = None
    # residual transfer 动作的 decrease 端参数：
    # - residual_decrease_rank：decrease 池内起始 rank；
    # - residual_decrease_count：一次动作要降低多少个任务预算；
    # - residual_decrease_pool：decrease 目标池（如 global_low_risk / lo_low_risk）。
    residual_decrease_rank: int | None = None
    residual_decrease_count: int = 1
    residual_decrease_pool: str | None = None

    def __post_init__(self) -> None:
        """把历史 float ratio 规范化为 action 自带的整数有理数元数据。"""
        if self.increase_ratio_num is None or self.increase_ratio_den is None:
            ratio = BudgetRatio.from_float_via_string(self.increase_ratio)
            object.__setattr__(self, "increase_ratio_num", ratio.numerator)
            object.__setattr__(self, "increase_ratio_den", ratio.denominator)
        if self.decrease_ratio_num is None or self.decrease_ratio_den is None:
            ratio = BudgetRatio.from_float_via_string(self.decrease_ratio)
            object.__setattr__(self, "decrease_ratio_num", ratio.numerator)
            object.__setattr__(self, "decrease_ratio_den", ratio.denominator)


def action_violates_hi_decrease_guard(
    action: BudgetAction | None,
    ordered_tasks: Sequence[Task],
    forbid_decreasing_hi_budgets: bool,
) -> bool:
    """判断动作是否违反“禁止降低 HI 任务预算”约束。

    约束语义严格按文档执行：
    1. 配置关闭时，任何动作都不触发该约束；
    2. `None` 或显式 `noop` 都不触发该约束；
    3. 只要 decrease 集合中出现任意 HI 任务，就判定为违规。

    设计说明：
    - 这里不做任何“容错式放行”，例如“只降低一点 HI 预算也允许”等；
      只要命中 HI decrease 就一票否决，确保行为和文档完全一致。
    - 函数被 env 与 runtime_wrapper 共用，目的是保证训练路径、评估路径、
      以及 baseline wrapper 路径使用同一条判定逻辑，避免两套实现出现漂移。
    """

    if not forbid_decreasing_hi_budgets:
        return False
    if action is None or action.is_noop:
        return False
    return any(ordered_tasks[idx].criticality is Criticality.HI for idx in action.decrease_indices)


def build_budget_action_space(
    ordered_tasks: Sequence[Task],
    *,
    action_space: str = "triple",
    budget_increase_ratio: float = 0.10,
    budget_decrease_ratio: float = 0.05,
    include_explicit_noop: bool = False,
    constraint_guided_pair_top_k_risk: int = 3,
    constraint_guided_pair_top_k_decrease: int = 5,
    constraint_guided_pair_include_hi_risk_boost: bool = False,
) -> tuple[BudgetAction, ...]:
    """构建确定性动作空间。

    - `triple`: 增加 1 个任务预算，降低 2 个任务预算；
    - `pair`: 增加 1 个任务预算，降低 1 个任务预算；
    - `single`: 只增加或只降低 1 个任务预算；
    - `residual_ranked`: 固定 residual action 槽位，运行时按风险排序动态解析目标任务；
    - `residual_safe_ranked`: safe residual 槽位，运行时先过滤安全候选再按 rank 选择；
    - `residual_anchor_mc_lo_2`: 在 residual_safe_ranked 基础上加入 direct safe increase mc_lo_2 anchor；
    - `residual_safe_adjust_15a`: 解耦的 safe increase/decrease 槽位；
      0 noop，1-8 safe increase utility rank，9-14 safe decrease redundant rank；
    - 可选 `include_explicit_noop` 追加显式 NoOp 动作（`residual_ranked` 内部已自带 noop）。
    """

    if action_space == "constraint_guided_pair":
        # 兼容 alias：外部仍可传旧名称，内部统一为 transfer 语义。
        action_space = "constraint_guided_transfer"
    if action_space not in {
        "triple",
        "pair",
        "single",
        "constraint_guided_transfer",
        "residual_ranked",
        "residual_safe_ranked",
        "residual_anchor_mc_lo_2",
        "residual_safe_adjust_15a",
    }:
        raise ValueError(f"不支持的 action_space: {action_space}")
    if budget_increase_ratio <= 0.0:
        raise ValueError("budget_increase_ratio 必须为正数")
    if budget_decrease_ratio <= 0.0 or budget_decrease_ratio >= 1.0:
        raise ValueError("budget_decrease_ratio 必须在 (0, 1) 区间内")

    names = [task.name for task in ordered_tasks]
    name_to_index = {name: idx for idx, name in enumerate(names)}
    actions: list[BudgetAction] = []
    action_id = 0

    if action_space == "triple":
        for increase_name in names:
            candidates = [name for name in names if name != increase_name]
            for dec_a, dec_b in combinations(candidates, 2):
                actions.append(
                    BudgetAction(
                        action_id=action_id,
                        increase_task=increase_name,
                        decrease_tasks=(dec_a, dec_b),
                        increase_idx=name_to_index[increase_name],
                        decrease_indices=(name_to_index[dec_a], name_to_index[dec_b]),
                        increase_ratio=budget_increase_ratio,
                        decrease_ratio=budget_decrease_ratio,
                        action_space_type=action_space,
                    )
                )
                action_id += 1
    elif action_space == "pair":
        for increase_name, decrease_name in permutations(names, 2):
            actions.append(
                BudgetAction(
                    action_id=action_id,
                    increase_task=increase_name,
                    decrease_tasks=(decrease_name,),
                    increase_idx=name_to_index[increase_name],
                    decrease_indices=(name_to_index[decrease_name],),
                    increase_ratio=budget_increase_ratio,
                    decrease_ratio=budget_decrease_ratio,
                    action_space_type=action_space,
                )
            )
            action_id += 1
    elif action_space == "single":
        for increase_name in names:
            actions.append(
                BudgetAction(
                    action_id=action_id,
                    increase_task=increase_name,
                    decrease_tasks=(),
                    increase_idx=name_to_index[increase_name],
                    decrease_indices=(),
                    increase_ratio=budget_increase_ratio,
                    decrease_ratio=budget_decrease_ratio,
                    action_space_type=action_space,
                )
            )
            action_id += 1
        for decrease_name in names:
            actions.append(
                BudgetAction(
                    action_id=action_id,
                    increase_task=None,
                    decrease_tasks=(decrease_name,),
                    increase_idx=None,
                    decrease_indices=(name_to_index[decrease_name],),
                    increase_ratio=budget_increase_ratio,
                    decrease_ratio=budget_decrease_ratio,
                    action_space_type=action_space,
                )
            )
            action_id += 1
    elif action_space == "constraint_guided_transfer":
        # constraint-guided transfer 采用“固定动作槽位 + 运行时动态映射”：
        # - 动作空间中只记录 increase_rank（每个槽位对应一个 bundled transfer）；
        # - decrease 目标集合由共享枚举器在当前观测下诊断得到；
        # - 动作维度固定为 1(noop) + dynamic_slots，默认 dynamic_slots=top_k_risk。
        dynamic_slots = (
            2 * constraint_guided_pair_top_k_risk
            if constraint_guided_pair_include_hi_risk_boost
            else constraint_guided_pair_top_k_risk
        )
        for inc_rank in range(dynamic_slots):
            actions.append(
                BudgetAction(
                    action_id=action_id,
                    increase_task=None,
                    decrease_tasks=(),
                    increase_idx=None,
                    decrease_indices=(),
                    increase_ratio=budget_increase_ratio,
                    decrease_ratio=budget_decrease_ratio,
                    action_space_type=action_space,
                    is_constraint_guided_pair=True,
                    constraint_guided_increase_rank=inc_rank,
                )
            )
            action_id += 1

    elif action_space == "residual_ranked":
        # residual_ranked 固定 15 个语义槽位：
        # - 目标是优先增强 LO 风险任务的可调性，并提供可学习的 transfer 语义；
        # - 槽位不绑定具体任务，具体 increase/decrease 目标在 env 中按实时风险解析。
        def _append_residual_action(
            *,
            residual_action_type: str,
            residual_rank: int | None = None,
            residual_decrease_rank: int | None = None,
            residual_decrease_count: int = 1,
            residual_decrease_pool: str | None = None,
            is_noop: bool = False,
        ) -> None:
            nonlocal action_id
            actions.append(
                BudgetAction(
                    action_id=action_id,
                    increase_task=None,
                    decrease_tasks=(),
                    increase_idx=None,
                    decrease_indices=(),
                    increase_ratio=budget_increase_ratio,
                    decrease_ratio=budget_decrease_ratio,
                    action_space_type=action_space,
                    is_noop=is_noop,
                    is_residual_ranked=True,
                    residual_action_type=residual_action_type,
                    residual_rank=residual_rank,
                    residual_decrease_rank=residual_decrease_rank,
                    residual_decrease_count=residual_decrease_count,
                    residual_decrease_pool=residual_decrease_pool,
                )
            )
            action_id += 1

        _append_residual_action(residual_action_type="noop", is_noop=True)

        for rank in range(4):
            _append_residual_action(
                residual_action_type="increase_lo_risk",
                residual_rank=rank,
            )

        for rank in range(2):
            _append_residual_action(
                residual_action_type="decrease_lowest_risk",
                residual_rank=rank,
            )

        for rank in range(2):
            _append_residual_action(
                residual_action_type="decrease_lo_lowest_risk",
                residual_rank=rank,
            )

        for rank in range(3):
            _append_residual_action(
                residual_action_type="transfer_to_lo_risk_from_global_low",
                residual_rank=rank,
                residual_decrease_rank=0,
                residual_decrease_count=1,
                residual_decrease_pool="global_low_risk",
            )

        for rank in range(2):
            _append_residual_action(
                residual_action_type="transfer_to_lo_risk_from_lo_low",
                residual_rank=rank,
                residual_decrease_rank=0,
                residual_decrease_count=1,
                residual_decrease_pool="lo_low_risk",
            )

        _append_residual_action(
            residual_action_type="transfer_to_lo_risk_from_global_low2",
            residual_rank=0,
            residual_decrease_rank=0,
            residual_decrease_count=2,
            residual_decrease_pool="global_low_risk",
        )
    elif action_space in {"residual_safe_ranked", "residual_anchor_mc_lo_2"}:
        # residual_safe_ranked 固定 15 个 safe 槽位：
        # - 移除 pure decrease 动作；
        # - increase/transfer 的具体目标在 env 中通过“安全候选过滤”后再按 rank 选择。
        # residual_anchor_mc_lo_2 采用最小 5 槽位：
        # 0 noop
        # 1 direct_safe_increase_anchor(mc_lo_2)
        # 2..4 safe_increase_lo_risk(rank=0..2)
        def _append_residual_safe_action(
            *,
            residual_action_type: str,
            residual_rank: int | None = None,
            residual_decrease_rank: int | None = None,
            residual_decrease_count: int = 1,
            residual_decrease_pool: str | None = None,
            increase_task: str | None = None,
            increase_idx: int | None = None,
            is_noop: bool = False,
        ) -> None:
            nonlocal action_id
            actions.append(
                BudgetAction(
                    action_id=action_id,
                    increase_task=increase_task,
                    decrease_tasks=(),
                    increase_idx=increase_idx,
                    decrease_indices=(),
                    increase_ratio=budget_increase_ratio,
                    decrease_ratio=budget_decrease_ratio,
                    action_space_type=action_space,
                    is_noop=is_noop,
                    is_residual_ranked=True,
                    residual_action_type=residual_action_type,
                    residual_rank=residual_rank,
                    residual_decrease_rank=residual_decrease_rank,
                    residual_decrease_count=residual_decrease_count,
                    residual_decrease_pool=residual_decrease_pool,
                )
            )
            action_id += 1

        _append_residual_safe_action(residual_action_type="noop", is_noop=True)

        if action_space == "residual_anchor_mc_lo_2":
            anchor_name = "mc_lo_2"
            if anchor_name not in name_to_index:
                raise ValueError(
                    "residual_anchor_mc_lo_2 requires task named 'mc_lo_2', "
                    f"but available tasks are: {names}"
                )
            _append_residual_safe_action(
                residual_action_type="direct_safe_increase_anchor",
                residual_rank=None,
                increase_task=anchor_name,
                increase_idx=name_to_index[anchor_name],
            )
            for rank in range(3):
                _append_residual_safe_action(
                    residual_action_type="safe_increase_lo_risk",
                    residual_rank=rank,
                )
            assert len(actions) == 5
            assert actions[0].is_noop
            return tuple(actions)
        else:
            safe_increase_ranks = range(4)

        for rank in safe_increase_ranks:
            _append_residual_safe_action(
                residual_action_type="safe_increase_lo_risk",
                residual_rank=rank,
            )

        for rank in range(4):
            _append_residual_safe_action(
                residual_action_type="safe_transfer_global_low_to_lo_risk",
                residual_rank=rank,
                residual_decrease_rank=0,
                residual_decrease_count=1,
                residual_decrease_pool="global_low_risk",
            )

        for rank in range(4):
            _append_residual_safe_action(
                residual_action_type="safe_transfer_lo_low_to_lo_risk",
                residual_rank=rank,
                residual_decrease_rank=0,
                residual_decrease_count=1,
                residual_decrease_pool="lo_low_risk",
            )

        for rank in range(2):
            _append_residual_safe_action(
                residual_action_type="safe_transfer_global_low2_to_lo_risk",
                residual_rank=rank,
                residual_decrease_rank=0,
                residual_decrease_count=2,
                residual_decrease_pool="global_low_risk",
            )
        assert len(actions) == 15
        assert actions[0].is_noop
    elif action_space == "residual_safe_adjust_15a":
        # residual_safe_adjust_15a 固定 15 个语义槽位：
        # 0 noop；
        # 1..8  安全增加：按 utility 排序选择第 k 个 LO 任务；
        # 9..14 安全降低：按 redundant 排序选择第 k 个 LO 任务。
        # 该空间不包含 transfer，不绑定固定 task-name anchor。
        def _append_adjust_action(
            *,
            residual_action_type: str,
            residual_rank: int | None = None,
            is_noop: bool = False,
        ) -> None:
            nonlocal action_id
            actions.append(
                BudgetAction(
                    action_id=action_id,
                    increase_task=None,
                    decrease_tasks=(),
                    increase_idx=None,
                    decrease_indices=(),
                    increase_ratio=budget_increase_ratio,
                    decrease_ratio=budget_decrease_ratio,
                    action_space_type=action_space,
                    is_noop=is_noop,
                    is_residual_ranked=True,
                    residual_action_type=residual_action_type,
                    residual_rank=residual_rank,
                )
            )
            action_id += 1

        _append_adjust_action(residual_action_type="noop", is_noop=True)

        for rank in range(8):
            _append_adjust_action(
                residual_action_type="safe_increase_lo_utility",
                residual_rank=rank,
            )

        for rank in range(6):
            _append_adjust_action(
                residual_action_type="safe_decrease_lo_redundant",
                residual_rank=rank,
            )

        assert len(actions) == 15
        assert actions[0].is_noop

    if include_explicit_noop and action_space not in {
        "residual_ranked",
        "residual_safe_ranked",
        "residual_anchor_mc_lo_2",
        "residual_safe_adjust_15a",
    }:
        actions.append(
            BudgetAction(
                action_id=action_id,
                increase_task=None,
                decrease_tasks=(),
                increase_idx=None,
                decrease_indices=(),
                increase_ratio=budget_increase_ratio,
                decrease_ratio=budget_decrease_ratio,
                action_space_type=action_space,
                is_noop=True,
            )
        )

    return tuple(actions)


def apply_budget_action_candidate(
    *,
    action: BudgetAction,
    budget_state: BudgetState,
    ordered_tasks: Sequence[Task],
) -> dict[str, int]:
    """将动作转换为候选更新，不直接改写原 BudgetState。"""

    if action.is_constraint_guided_pair:
        raise ValueError("constraint_guided_pair action must be resolved by AmcBudgetEnv before applying")

    if action.is_noop:
        return {}

    task_names = [task.name for task in ordered_tasks]
    candidate: dict[str, int] = {}

    if action.increase_idx is not None:
        inc_name = task_names[action.increase_idx]
        old_inc = budget_state.budgets[inc_name]
        inc_task = ordered_tasks[action.increase_idx]
        inc_value = compute_increase_candidate(old_inc, BudgetRatio(action.increase_ratio_num, action.increase_ratio_den))

        if inc_task.criticality is Criticality.HI:
            upper_bound = inc_task.c_hi if inc_task.c_hi > 0 else inc_task.deadline
            inc_value = min(inc_value, upper_bound)
        else:
            inc_value = min(inc_value, inc_task.deadline)

        candidate[inc_name] = max(1, inc_value)

    for dec_idx in action.decrease_indices:
        dec_name = task_names[dec_idx]
        old_dec = budget_state.budgets[dec_name]
        dec_value = compute_decrease_candidate(old_dec, BudgetRatio(action.decrease_ratio_num, action.decrease_ratio_den))
        candidate[dec_name] = max(1, dec_value)

    return candidate


def describe_budget_action(action: BudgetAction) -> str:
    """把离散预算动作转换为可读字符串，便于写入分析型 CSV。

    设计目标：
    1. 输出稳定、可机读：同一动作在不同运行中描述字符串保持一致；
    2. residual_ranked 动作要包含关键槽位参数，便于离线分析“策略到底用了哪一类动作”；
    3. 不引入额外兜底推断逻辑，只按动作对象中已有字段直接拼接。
    """

    if action.is_noop:
        return "noop"

    if action.action_space_type in {
        "residual_ranked",
        "residual_safe_ranked",
        "residual_anchor_mc_lo_2",
        "residual_safe_adjust_15a",
    }:
        parts: list[str] = [action.residual_action_type or "unknown"]
        if action.increase_task is not None:
            parts.append(f"anchor={action.increase_task}")
        if action.residual_rank is not None:
            parts.append(f"inc_rank={action.residual_rank}")
        if action.residual_decrease_pool is not None:
            parts.append(f"dec_pool={action.residual_decrease_pool}")
        if action.residual_decrease_rank is not None:
            parts.append(f"dec_rank={action.residual_decrease_rank}")
        if action.residual_decrease_count is not None:
            parts.append(f"dec_count={action.residual_decrease_count}")
        return "|".join(parts)

    return action.action_space_type
