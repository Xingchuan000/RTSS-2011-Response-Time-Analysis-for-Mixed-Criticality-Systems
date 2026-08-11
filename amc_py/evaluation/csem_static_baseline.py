"""C-AMC-sem validation-tuned static budget baseline."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import json
import math
from pathlib import Path

from amc_py.amc import build_design_r_lo_map
from amc_py.budget_runtime import BudgetState
from amc_py.event_runtime import simulate_ordered_taskset_event_driven
from amc_py.models import Criticality, Task
from amc_py.rl.safety import RuntimeBudgetSafetyChecker, RuntimeSafetyReport
from amc_py.runtime_models import RuntimeConfig, SimulationResult
from amc_py.runtime_scenarios import ExecutionScenario


@dataclass(frozen=True, slots=True)
class StaticBudgetSelection:
    """Validation-selected scalar alpha and its audit metadata."""

    taskset_seed: int
    selection_seeds: str
    selection_end_time: int
    alpha: float
    selection_metric: str
    selection_metric_value: float
    validation_lo_zero_service_ratio: float
    validation_lo_equiv_jne: float
    budgets: dict[str, int]


@dataclass(frozen=True, slots=True)
class StaticBudgetRun:
    """Result of one fixed-budget C-AMC-sem rollout."""

    runtime_result: SimulationResult
    alpha: float
    budgets: dict[str, int]
    safety_report: RuntimeSafetyReport


def build_static_lo_budgets(
    ordered_tasks: Sequence[Task],
    *,
    alpha: float,
) -> dict[str, int]:
    """Build the fixed budget vector for a scalar LO budget multiplier."""

    if alpha < 1.0:
        raise ValueError("alpha must be >= 1.0")

    budgets: dict[str, int] = {}
    for task in ordered_tasks:
        if task.criticality is Criticality.HI:
            budgets[task.name] = int(task.c_lo)
            continue

        candidate = int(math.ceil(float(task.c_lo) * float(alpha)))
        budgets[task.name] = min(int(task.deadline), candidate)
    return budgets


def check_static_budget_feasibility(
    ordered_tasks: Sequence[Task],
    budgets: dict[str, int],
) -> RuntimeSafetyReport:
    """Check a complete static budget vector with the existing safety checker."""

    checker = RuntimeBudgetSafetyChecker(
        ordered_tasks=ordered_tasks,
        design_r_lo=build_design_r_lo_map(ordered_tasks),
    )
    return checker.validate_candidate(budgets)


def run_static_budget_baseline(
    *,
    ordered_tasks: Sequence[Task],
    scenario: ExecutionScenario,
    runtime_config: RuntimeConfig,
    alpha: float,
) -> StaticBudgetRun:
    """Run C-AMC-sem with a fixed budget state from simulation time zero."""

    budgets = build_static_lo_budgets(ordered_tasks, alpha=alpha)
    safety_report = check_static_budget_feasibility(ordered_tasks, budgets)
    if not safety_report.accepted:
        raise ValueError(f"STATIC_BUDGET_INFEASIBLE:{safety_report.reason}")

    original = {task.name: int(task.c_lo) for task in ordered_tasks}
    budget_state = BudgetState(
        budgets=dict(budgets),
        initial_budgets=original,
    )
    result = simulate_ordered_taskset_event_driven(
        ordered_tasks=ordered_tasks,
        scenario=scenario,
        config=runtime_config,
        budget_state=budget_state,
    )
    return StaticBudgetRun(
        runtime_result=result,
        alpha=float(alpha),
        budgets=dict(budgets),
        safety_report=safety_report,
    )


def save_static_budget_selection(
    selection: StaticBudgetSelection,
    path: Path,
) -> None:
    """Write the selection record using the Plan 01 JSON schema."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "taskset_seed": selection.taskset_seed,
                "selection_seeds": selection.selection_seeds,
                "selection_end_time": selection.selection_end_time,
                "alpha": selection.alpha,
                "selection_metric": selection.selection_metric,
                "selection_metric_value": selection.selection_metric_value,
                "validation_lo_zero_service_ratio": selection.validation_lo_zero_service_ratio,
                "validation_lo_equiv_jne": selection.validation_lo_equiv_jne,
                "budgets": selection.budgets,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )


def load_static_budget_selection(path: Path) -> StaticBudgetSelection:
    """Load a Plan 01 static budget selection record."""

    data = json.loads(path.read_text(encoding="utf-8"))
    return StaticBudgetSelection(
        taskset_seed=int(data["taskset_seed"]),
        selection_seeds=str(data["selection_seeds"]),
        selection_end_time=int(data["selection_end_time"]),
        alpha=float(data["alpha"]),
        selection_metric=str(data["selection_metric"]),
        selection_metric_value=float(data["selection_metric_value"]),
        validation_lo_zero_service_ratio=float(data["validation_lo_zero_service_ratio"]),
        validation_lo_equiv_jne=float(data["validation_lo_equiv_jne"]),
        budgets={str(k): int(v) for k, v in data["budgets"].items()},
    )
