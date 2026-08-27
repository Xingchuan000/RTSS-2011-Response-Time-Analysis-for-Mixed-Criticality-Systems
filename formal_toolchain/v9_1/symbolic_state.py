"""Finite symbolic state for the V9.1 Policy--Timing Kernel.

The state is deliberately finite only at the proof boundary: a caller chooses
the number of job slots needed by a finite window.  It is not a protected-job
filter and it never stores a fixed execution scenario.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Any, Mapping

import z3


JobSlotKey = tuple[str, int]


@dataclass(frozen=True, slots=True)
class TaskBound:
    name: str
    priority: int
    period: int
    deadline: int
    criticality: str
    c_lo: int
    c_hi: int
    initial_budget: int
    budget_floor: int = 1
    action_hard_upper: int | None = None
    degraded_cost: int | None = None

    def __post_init__(self) -> None:
        if self.period <= 0 or self.deadline <= 0:
            raise ValueError("task period/deadline must be positive")
        if self.criticality not in {"HI", "LO"}:
            raise ValueError("task criticality must be HI or LO")
        if not (1 <= self.c_lo <= self.c_hi):
            raise ValueError("task demand bounds are invalid")
        upper = self.action_hard_upper if self.action_hard_upper is not None else self.c_hi
        if self.degraded_cost is not None and not (1 <= self.degraded_cost <= self.c_lo):
            raise ValueError("task degraded cost is invalid")
        if not (self.budget_floor <= self.initial_budget <= upper):
            raise ValueError("task budget bounds are invalid")

    @property
    def budget_upper(self) -> int:
        return self.action_hard_upper if self.action_hard_upper is not None else self.c_hi


@dataclass(frozen=True, slots=True)
class BoundModel:
    """All finite source bindings consumed by the symbolic encoders."""

    tasks: tuple[TaskBound, ...]
    agent_period: int
    action_dim: int = 0
    noop_id: int | None = None
    feature_names: tuple[str, ...] = ()
    fixed_point_config: Any = None
    tree: Any = None
    action_definitions: tuple[Mapping[str, Any], ...] = ()
    feature_config: Mapping[str, Any] = field(default_factory=dict)
    rounding_mode: str = "ceil_floor"
    min_budget_delta: int = 1
    forbid_decreasing_hi_budgets: bool = False
    enable_deploy_cap_mask: bool = False
    deploy_cap_mask_ratio: float = 1.0
    deploy_cap_mask_criticality: str = "lo"
    check_safety: bool = False
    safety_constraints: tuple[Mapping[str, Any], ...] = ()
    history_k: int = 8
    event_window: int = 10
    max_jobs_per_task: int = 2

    def __post_init__(self) -> None:
        if not self.tasks:
            raise ValueError("BoundModel requires at least one task")
        if self.agent_period <= 0 or self.max_jobs_per_task <= 0:
            raise ValueError("BoundModel finite dimensions must be positive")
        names = [task.name for task in self.tasks]
        if len(set(names)) != len(names):
            raise ValueError("BoundModel task names must be unique")
        if tuple(sorted(self.tasks, key=lambda task: task.priority)) != self.tasks:
            raise ValueError("BoundModel.tasks must use canonical priority order")
        if self.action_dim < 0:
            raise ValueError("action_dim must be non-negative")
        if self.action_dim and self.noop_id is None:
            raise ValueError("an action model must identify its explicit noop")
        if self.noop_id is not None and not (0 <= self.noop_id < self.action_dim):
            raise ValueError("noop_id is outside the action alphabet")
        if self.rounding_mode not in {"ceil_floor", "nearest"}:
            raise ValueError("unsupported P5 budget rounding mode")
        if self.min_budget_delta <= 0:
            raise ValueError("P5 min_budget_delta must be positive")
        if self.deploy_cap_mask_criticality not in {"lo", "all"}:
            raise ValueError("unsupported deploy-cap criticality")
        if self.check_safety and not self.safety_constraints:
            raise ValueError("P5 safety checker enabled without frozen linear constraints")

    @property
    def task_by_name(self) -> dict[str, TaskBound]:
        return {task.name: task for task in self.tasks}

    @property
    def hi_tasks(self) -> tuple[TaskBound, ...]:
        return tuple(task for task in self.tasks if task.criticality == "HI")

    @classmethod
    def from_bindings(cls, bindings: Mapping[str, Any], *, max_jobs_per_task: int = 2) -> "BoundModel":
        rows = bindings.get("taskset", {}).get("ordered_tasks", ())
        if not isinstance(rows, (list, tuple)) or not rows:
            raise ValueError("bindings do not contain canonical taskset rows")
        tasks = tuple(
            TaskBound(
                name=str(row["name"]),
                priority=int(row["priority_index"]),
                period=int(row["period"]),
                deadline=int(row["deadline"]),
                criticality=str(row["criticality"]),
                c_lo=int(row["code_c_lo"]),
                c_hi=int(row["code_c_hi"]),
                initial_budget=int(row["initial_runtime_budget"]),
                budget_floor=int(row.get("budget_floor", 1)),
                action_hard_upper=int(row.get("action_hard_upper", row["code_c_hi"])),
                degraded_cost=int(
                    bindings.get("environment_binding", {}).get("degraded_cost_by_task", {}).get(
                        str(row["name"]), row["code_c_lo"]
                    )
                ),
            )
            for row in rows
        )
        action = bindings.get("policy_action_binding", {})
        alphabet = action.get("action_alphabet", ())
        noop_ids = [int(row["action_id"]) for row in alphabet if row.get("is_noop") is True]
        if len(noop_ids) != 1:
            raise ValueError("V9.1 requires exactly one explicit noop in bindings")
        numeric = bindings.get("numeric_observation_binding", {})
        quant = numeric.get("quantization", {})
        feature_config = dict(numeric.get("feature_config", {}))
        action_cfg = dict(action.get("execution_config", {}))
        tree_data = bindings.get("tree_identity", {}).get("integer_tree")
        if not isinstance(tree_data, Mapping):
            raise ValueError("bindings do not contain the deployed integer tree")
        from formal_toolchain.policy.tree_io import integer_tree_from_dict
        tree = integer_tree_from_dict(tree_data)
        trigger = str(bindings.get("p0_event_order_binding", {}).get("controller_trigger_predicate", ""))
        match = re.search(r"mod\s+(\d+)", trigger)
        agent_period = int(match.group(1)) if match else int(bindings.get("agent_period", 1))
        return cls(
            tasks=tasks,
            agent_period=agent_period,
            action_dim=len(alphabet),
            noop_id=noop_ids[0],
            feature_names=tuple(str(value) for value in numeric.get("feature_names", ())),
            fixed_point_config=quant,
            tree=tree,
            action_definitions=tuple(dict(row) for row in alphabet),
            feature_config=feature_config,
            rounding_mode=str(action_cfg.get("rounding_mode", "ceil_floor")),
            min_budget_delta=int(action_cfg.get("min_budget_delta", 1)),
            forbid_decreasing_hi_budgets=bool(action_cfg.get("forbid_decreasing_hi_budgets", False)),
            enable_deploy_cap_mask=bool(action_cfg.get("enable_deploy_cap_mask", False)),
            deploy_cap_mask_ratio=float(action_cfg.get("deploy_cap_mask_ratio", 1.0)),
            deploy_cap_mask_criticality=str(action_cfg.get("deploy_cap_mask_criticality", "lo")),
            check_safety=bool(action_cfg.get("check_safety", False)),
            safety_constraints=tuple(dict(row) for row in action_cfg.get("safety_constraints", ())),
            history_k=int(feature_config.get("history_k", 8)),
            event_window=int(feature_config.get("event_window", 10)),
            max_jobs_per_task=max_jobs_per_task,
        )


@dataclass(frozen=True, slots=True)
class SymbolicJob:
    present: z3.BoolRef
    task_id: int
    release_index: z3.ArithRef
    release_time: z3.ArithRef
    absolute_deadline: z3.ArithRef
    priority: int
    tie_break: z3.ArithRef
    release_entry_mode_hi: z3.BoolRef
    classification_abnormal: z3.BoolRef
    budget_at_release: z3.ArithRef
    actual_demand: z3.ArithRef
    effective_demand: z3.ArithRef
    executed_service: z3.ArithRef
    removed: z3.BoolRef
    ready: z3.BoolRef

    @property
    def remaining(self) -> z3.ArithRef:
        return z3.If(self.effective_demand > self.executed_service,
                     self.effective_demand - self.executed_service, 0)


@dataclass(frozen=True, slots=True)
class SymbolicFrontier:
    selected_slot: z3.IntNumRef | z3.ArithRef
    running: z3.BoolRef


@dataclass(frozen=True, slots=True)
class SymbolicPolicyHistory:
    # Runtime observation history contains binary64-valued EMA signals.  Use
    # mathematical Reals at the symbolic state boundary; the exact binary64 ->
    # Decimal(str(float)) quantization relation remains a separate proof
    # obligation and therefore keeps the global V9.1 gate closed.
    recent_cost: dict[str, z3.ArithRef] = field(default_factory=dict)
    ema_cost: dict[str, z3.ArithRef] = field(default_factory=dict)
    overrun_ema: dict[str, z3.ArithRef] = field(default_factory=dict)
    max_cost_k: dict[str, z3.ArithRef] = field(default_factory=dict)
    mode_change_window: tuple[z3.ArithRef, ...] = ()
    lo_cancel_window: tuple[z3.ArithRef, ...] = ()
    hi_overrun_window: tuple[z3.ArithRef, ...] = ()
    lo_overrun_window: tuple[z3.ArithRef, ...] = ()
    job_start_window: tuple[z3.ArithRef, ...] = ()


@dataclass(frozen=True, slots=True)
class SymbolicKernelState:
    t: z3.ArithRef
    p: z3.ArithRef
    mode_hi: z3.BoolRef
    budgets: dict[str, z3.ArithRef]
    eta: dict[str, z3.ArithRef]
    jobs: dict[JobSlotKey, SymbolicJob]
    frontier: SymbolicFrontier
    hi_miss_ledger: z3.ArithRef
    chi: SymbolicPolicyHistory


def _int(name: str) -> z3.ArithRef:
    return z3.Int(name)


def _real(name: str) -> z3.ArithRef:
    return z3.Real(name)


def declare_state(prefix: str, model: BoundModel) -> SymbolicKernelState:
    """Declare one complete finite state using the canonical kernel names."""

    tasks = model.tasks
    jobs: dict[JobSlotKey, SymbolicJob] = {}
    for task_id, task in enumerate(tasks):
        for slot in range(model.max_jobs_per_task):
            key = (task.name, slot)
            base = f"{prefix}.J.{task_id}.{slot}"
            jobs[key] = SymbolicJob(
                present=z3.Bool(f"{base}.present"),
                task_id=task_id,
                release_index=_int(f"{base}.release_index"),
                release_time=_int(f"{base}.release_time"),
                absolute_deadline=_int(f"{base}.deadline"),
                priority=task.priority,
                tie_break=_int(f"{base}.tie_break"),
                release_entry_mode_hi=z3.Bool(f"{base}.entry_hi"),
                classification_abnormal=z3.Bool(f"{base}.abnormal"),
                budget_at_release=_int(f"{base}.B_rel"),
                actual_demand=_int(f"{base}.A"),
                effective_demand=_int(f"{base}.E"),
                executed_service=_int(f"{base}.service"),
                removed=z3.Bool(f"{base}.removed"),
                ready=z3.Bool(f"{base}.ready"),
            )
    history = SymbolicPolicyHistory(
        recent_cost={task.name: _real(f"{prefix}.chi.recent.{task.name}") for task in tasks},
        ema_cost={task.name: _real(f"{prefix}.chi.ema.{task.name}") for task in tasks},
        overrun_ema={task.name: _real(f"{prefix}.chi.overrun_ema.{task.name}") for task in tasks},
        max_cost_k={task.name: _real(f"{prefix}.chi.maxk.{task.name}") for task in tasks},
        mode_change_window=tuple(_int(f"{prefix}.chi.mode.{i}") for i in range(model.event_window)),
        lo_cancel_window=tuple(_int(f"{prefix}.chi.locancel.{i}") for i in range(model.event_window)),
        hi_overrun_window=tuple(_int(f"{prefix}.chi.hioverrun.{i}") for i in range(model.event_window)),
        lo_overrun_window=tuple(_int(f"{prefix}.chi.looverrun.{i}") for i in range(model.event_window)),
        job_start_window=tuple(_int(f"{prefix}.chi.starts.{i}") for i in range(model.event_window)),
    )
    return SymbolicKernelState(
        t=_int(f"{prefix}.t"),
        p=_int(f"{prefix}.p"),
        mode_hi=z3.Bool(f"{prefix}.mode_hi"),
        budgets={task.name: _int(f"{prefix}.B.{task.name}") for task in tasks},
        eta={task.name: _int(f"{prefix}.eta.{task.name}") for task in tasks},
        jobs=jobs,
        frontier=SymbolicFrontier(_int(f"{prefix}.F.selected"), z3.Bool(f"{prefix}.F.running")),
        hi_miss_ledger=_int(f"{prefix}.M"),
        chi=history,
    )


def well_formed(state: SymbolicKernelState, model: BoundModel) -> z3.BoolRef:
    """Return the structural state contract; no safety claim is hidden here."""

    clauses: list[z3.BoolRef] = [state.t >= 0, z3.And(state.p >= 0, state.p <= 7), state.hi_miss_ledger >= 0]
    for task in model.tasks:
        budget = state.budgets[task.name]
        eta = state.eta[task.name]
        clauses.extend((budget >= task.budget_floor, budget <= task.budget_upper,
                        eta >= 0, eta <= task.period))
        for slot in range(model.max_jobs_per_task):
            job = state.jobs[(task.name, slot)]
            lo_aggregate = task.criticality == "LO" and slot == 0
            unused = (task.criticality == "HI" and slot != 0) or (
                task.criticality == "LO" and slot not in {0, 1}
            )
            if unused:
                clauses.extend((z3.Not(job.present), z3.Not(job.ready)))
                continue
            if lo_aggregate:
                clauses.extend((
                    z3.Implies(job.present, job.release_index == -1),
                    z3.Implies(job.present, job.release_time <= state.t),
                    z3.Implies(job.present, job.tie_break == -1),
                    z3.Implies(job.present, job.actual_demand == job.effective_demand),
                    z3.Implies(job.present, job.effective_demand >= 1),
                ))
            else:
                clauses.extend((
                    z3.Implies(job.present, job.release_index >= 0),
                    z3.Implies(job.present, job.release_time >= 0),
                    z3.Implies(job.present, job.absolute_deadline == job.release_time + task.deadline),
                    z3.Implies(job.present, job.actual_demand >= 1),
                    z3.Implies(job.present, job.actual_demand <= (task.c_hi if task.criticality == "HI" else task.c_lo)),
                ))
            clauses.extend((
                z3.Implies(job.present, job.budget_at_release >= task.budget_floor),
                z3.Implies(job.present, job.budget_at_release <= task.budget_upper),
                z3.Implies(job.present, job.effective_demand >= 1),
                job.executed_service >= 0,
                z3.Implies(job.present, job.executed_service <= job.effective_demand),
                z3.Implies(job.removed, z3.Not(job.ready)),
                z3.Implies(job.present, z3.Not(job.removed)),
            ))
    if model.action_dim:
        clauses.extend((state.frontier.selected_slot >= -1,
                        state.frontier.selected_slot < len(state.jobs)))
    for values in (
        state.chi.recent_cost.values(), state.chi.ema_cost.values(), state.chi.overrun_ema.values(),
        state.chi.max_cost_k.values(),
        state.chi.mode_change_window, state.chi.lo_cancel_window,
        state.chi.hi_overrun_window, state.chi.lo_overrun_window, state.chi.job_start_window,
    ):
        clauses.extend(value >= 0 for value in values)
    return z3.And(*clauses)


__all__ = [
    "BoundModel", "JobSlotKey", "SymbolicFrontier", "SymbolicJob", "SymbolicKernelState",
    "SymbolicPolicyHistory", "TaskBound", "declare_state", "well_formed",
]
