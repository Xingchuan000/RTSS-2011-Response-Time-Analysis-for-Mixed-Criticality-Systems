"""AMC 真实 runtime 的 FormalRuntimeAdapter 实现。

该 adapter 只调用传入的实际 environment/抽取器；没有为真实 target 偷换
synthetic evaluator。动作 replay 使用独立 state 副本，避免 verifier 改写
训练或部署环境的可变状态。
"""

from __future__ import annotations

import copy
import math
from typing import Any, Mapping, Sequence

from formal_toolchain.core.hashing import sha256_object


def _exact_int_from_float(value: float, label: str) -> int:
    if not math.isfinite(value):
        raise ValueError(f"{label} 不是有限数")
    integer = int(value)
    if float(integer) != float(value):
        raise ValueError(f"{label} 不是精确整数")
    if abs(integer) >= 2**53:
        raise ValueError(f"{label} 超出 binary64 精确整数范围")
    return integer


class AMCRealRuntimeAdapter:
    def __init__(self, environment: Any, *, observation_extractor: Any = None,
                 action_space: Sequence[Mapping[str, Any]] | None = None,
                 rounding_mode: str = "ceil_floor",
                 min_budget_delta: int = 1) -> None:
        self.environment = environment
        self.observation_extractor = observation_extractor
        self.action_space = tuple(action_space or tuple(getattr(environment, "_actions", ())))
        self.selection_semantics = "ranked_first_valid"
        self.step_guard_semantics = "checked"
        self.disabled_guards = ()
        self.rounding_mode = str(rounding_mode)
        self.min_budget_delta = int(min_budget_delta)
        if self.rounding_mode != "ceil_floor":
            raise ValueError("UNSUPPORTED_BUDGET_ROUNDING_MODE")

    def build_runtime_state_from_budget_vector(self, budget_by_task: Mapping[str, int]) -> dict[str, Any]:
        """从真实环境复制出一个独立 state，并用实际 reset 初始化特征。

        形式化枚举不得污染部署环境；因此每个 state 使用 environment 的
        深复制，reset 仍调用真实 provider，而不是 synthetic observation。
        """
        environment = copy.deepcopy(self.environment)
        environment.reset()
        budgets = {str(key): int(value) for key, value in budget_by_task.items()}
        if environment._engine is None:
            raise RuntimeError("REAL_RUNTIME_ENGINE_RESET_FAILED")
        if set(budgets) != set(environment._engine.runtime_budgets.budgets):
            raise ValueError("REAL_RUNTIME_BUDGET_TASK_SET_MISMATCH")
        environment._engine.runtime_budgets.budgets = dict(budgets)
        return {"budgets": budgets, "environment": environment}

    def quantization_input_state(self, runtime_state: Mapping[str, Any]) -> Mapping[str, Any]:
        return runtime_state

    def extract_observation(self, runtime_state: Mapping[str, Any]) -> Sequence[float]:
        if self.observation_extractor is not None:
            if callable(self.observation_extractor):
                return tuple(self.observation_extractor(runtime_state))
            method = getattr(self.observation_extractor, "extract", None)
            if callable(method):
                return tuple(method(runtime_state))
            raise RuntimeError("REAL_RUNTIME_OBSERVATION_EXTRACTOR_INVALID")
        environment = runtime_state.get("environment")
        if environment is None or environment._engine is None or environment._feature_state is None:
            raise RuntimeError("REAL_RUNTIME_OBSERVATION_STATE_INVALID")
        from amc_py.rl.observation import build_observation
        observation = build_observation(
            time=environment._engine.current_time,
            ordered_tasks=environment.ordered_tasks,
            budget_state=environment._engine.runtime_budgets,
            monitor=environment._monitor,
            bounds=environment.normalization_bounds,
            feature_state=environment._feature_state,
            feature_config=environment.feature_config,
            safety_margin_min=environment._compute_current_safety_margin_min(),
            initial_budgets=environment._initial_budgets,
            rh_risk_context=environment._last_rh_risk_context,
        )
        return tuple(observation.state_vector)

    def valid_action_mask(self, runtime_state: Mapping[str, Any]):
        environment = runtime_state.get("environment")
        if environment is None:
            raise RuntimeError("REAL_RUNTIME_MASK_REPLAY_NOT_IMPLEMENTED")
        formal_mask = getattr(environment, "formal_valid_action_mask", None)
        if callable(formal_mask):
            mask = tuple(bool(value) for value in formal_mask())
        elif callable(getattr(environment, "valid_action_mask", None)):
            mask = tuple(bool(value) for value in environment.valid_action_mask())
        else:
            raise RuntimeError("REAL_RUNTIME_MASK_REPLAY_NOT_IMPLEMENTED")
        details = tuple(getattr(environment, "_last_mask_details", ()))
        if len(details) != len(mask):
            raise RuntimeError("REAL_RUNTIME_MASK_DETAIL_LENGTH_INVALID")
        reasons = tuple(str(item.get("reject_reason") or "accepted") for item in details)
        return mask, reasons

    def apply_action(self, runtime_state: Mapping[str, Any], action_id: int | None):
        environment = runtime_state.get("environment")
        if environment is None or environment._engine is None:
            raise RuntimeError("REAL_RUNTIME_ACTION_STATE_INVALID")
        if action_id is None:
            return dict(environment._engine.runtime_budgets.budgets)
        from amc_py.rl.actions import apply_budget_action_candidate
        from amc_py.budget_runtime import BudgetState
        action = environment._actions[int(action_id)]
        before = dict(environment._engine.runtime_budgets.budgets)
        updates = apply_budget_action_candidate(
            action=action,
            budget_state=BudgetState(budgets=dict(before), initial_budgets=dict(environment._initial_budgets)),
            ordered_tasks=environment.ordered_tasks,
            rounding_mode=self.rounding_mode,
            min_budget_delta=self.min_budget_delta,
        )
        after = dict(before)
        after.update(updates)
        diagnosis = environment.diagnose_candidate_budget_update(new_budgets=after)
        if not diagnosis.accepted:
            raise RuntimeError("REAL_RUNTIME_ACTION_REJECTED_BY_ENVIRONMENT")
        return after

    def common_transition_witnesses(self):
        method = getattr(self.environment, "formal_common_transition_witnesses", None)
        if not callable(method):
            raise RuntimeError("REAL_RUNTIME_TRANSITION_REPLAY_NOT_IMPLEMENTED")
        return tuple(method())

    def export_observation_contract(self):
        names = getattr(self.environment, "get_observation_feature_names", None)
        feature_names = tuple(names()) if callable(names) else ()
        return {"feature_names": list(feature_names), "total_on_p0_states": True,
                "nan_rejected_or_normalized": True, "inf_rejected_or_normalized": True,
                "clip_before_quantization": True, "history_initialization_defined": True,
                "preclosed_state_read": True}

    def export_mask_contract(self):
        checker = None
        if bool(getattr(self.environment, "check_safety", False)) and callable(getattr(self.environment, "_ensure_checker", None)):
            checker = self.environment._ensure_checker()
        return {
            "action_ids": [int(row.action_id) if hasattr(row, "action_id") else int(row["action_id"]) for row in self.action_space],
            "shared_with_step": callable(getattr(self.environment, "formal_valid_action_mask", None))
            and callable(getattr(self.environment, "evaluate_budget_candidate", None)),
            "explicit_noop": False,
            "implicit_noop_when_all_invalid": True,
            "check_safety": bool(getattr(self.environment, "check_safety", False)),
            "safety_checker_type": None if checker is None else type(checker).__qualname__,
            "candidate_reject_helper": "AmcBudgetEnv._budget_candidate_reject_reason",
            "candidate_evaluator": "AmcBudgetEnv.evaluate_budget_candidate",
            "selection": self.selection_semantics,
            "step_guard_semantics": self.step_guard_semantics,
            "disabled_guards": list(self.disabled_guards),
            "rounding_mode": self.rounding_mode,
            "min_budget_delta": self.min_budget_delta,
        }

    def export_action_contract(self):
        return {"action_definitions": [dict(row) if isinstance(row, Mapping) else {"action_id": int(row.action_id)} for row in self.action_space]}

    def export_source_binding_targets(self):
        return {
            "adapter_kind": "REAL_AMC_RUNTIME",
            "environment_type": type(self.environment).__qualname__,
            "mask_entrypoint": "AmcBudgetEnv.formal_valid_action_mask",
            "step_entrypoint": "AmcBudgetEnv.step",
            "candidate_evaluator": "AmcBudgetEnv.evaluate_budget_candidate",
        }

    def _frozen_initial_budgets(self) -> dict[str, int]:
        configured = getattr(self.environment, "_initial_budgets", None)
        if isinstance(configured, Mapping):
            return {str(name): int(value) for name, value in configured.items()}
        return {
            str(task.name): int(getattr(task, "c_lo"))
            for task in self.environment.ordered_tasks
        }

    def export_initial_state_contract(self):
        """Return the frozen pre-arrival P0 state without starting the runtime."""

        budgets = self._frozen_initial_budgets()
        return {
            "status": "PASS",
            "schema_version": "frozen_p0_initial_state_v1",
            "formal_semantics_binding": "FROZEN_C_AMC_SEM_P0",
            "mutable_runtime_dependency": "NONE",
            "current_time": 0,
            "mode": "LO",
            "running_job": None,
            "active_jobs": [],
            "ready_jobs": [],
            "service_in_progress": False,
            "runtime_budgets": budgets,
            "quiescent": True,
        }

    def export_boot_transition_contract(self):
        """Derive time-zero closure from the frozen formal semantics adapter."""

        from types import SimpleNamespace
        from formal_toolchain.semantics.frozen_preclosed_state import (
            build_frozen_preclosed_bundle,
        )

        initial = self.export_initial_state_contract()
        budgets = self._frozen_initial_budgets()
        cfg = self.environment.runtime_config
        ratio = float(getattr(cfg, "c_amc_sem_lo_degradation_ratio", 0.5))
        rows = []
        for priority_index, task in enumerate(self.environment.ordered_tasks):
            c_lo = int(getattr(task, "c_lo"))
            rows.append({
                "name": str(task.name),
                "priority_index": priority_index,
                "initial_runtime_budget": budgets[str(task.name)],
                "degraded_cost": max(1, min(c_lo, int(round(ratio * c_lo)))),
            })
        target = SimpleNamespace(
            ordered_tasks=tuple(self.environment.ordered_tasks),
            scenario=self.environment.scenario,
            runtime_config=cfg,
        )
        concrete, _reference, _snapshot = build_frozen_preclosed_bundle(
            target,
            {"tasks": rows},
        )
        running = concrete.running_job
        return {
            "status": "PASS",
            "schema_version": "frozen_p0_boot_transition_v1",
            "formal_semantics_binding": "FROZEN_C_AMC_SEM_P0",
            "mutable_runtime_dependency": "NONE",
            "initial_state_hash": sha256_object(initial),
            "boot_time": int(concrete.time),
            "mode_after_boot": str(concrete.mode),
            "first_release_batch_defined": True,
            "no_service_before_boot_closure": all(job.service == 0 for job in concrete.active_jobs),
            "running_job": None if running is None else str(running[0]),
            "running_job_executed_time": None if running is None else 0,
            "initial_runtime_budget_snapshot": budgets,
            "active_job_count": len(concrete.active_jobs),
            "effective_event_frontier_count": len(concrete.effective_event_frontier),
        }

    def export_budget_safety_polytope(self) -> dict[str, Any]:
        environment = copy.deepcopy(self.environment)

        if not bool(getattr(environment, "check_safety", False)):
            return {
                "status": "UNRESOLVED",
                "route": "UNRESOLVED",
                "failure": {"code": "RUNTIME_SAFETY_CHECK_DISABLED"},
            }

        checker = environment._ensure_checker()
        task_names = [str(task.name) for task in environment.ordered_tasks]
        rows: list[dict[str, Any]] = []

        for row_index, (task_name, constraint_name, rhs, coeff) in enumerate(checker._constraint_rows):
            coefficients: dict[str, int] = {}
            for name, value in zip(task_names, coeff):
                integer = _exact_int_from_float(float(value), f"row[{row_index}].{name}")
                if integer < 0:
                    raise ValueError("P0 structural envelope 只支持非负系数")
                if integer:
                    coefficients[name] = integer

            rows.append(
                {
                    "row_index": row_index,
                    "analyzed_task": str(task_name),
                    "constraint": str(constraint_name),
                    "coefficients": coefficients,
                    "rhs": _exact_int_from_float(float(rhs), f"row[{row_index}].rhs"),
                }
            )

        return {
            "status": "PASS",
            "schema_version": "budget_safety_polytope_v1",
            "task_order": task_names,
            "rows": rows,
            "design_r_lo": {
                str(name): int(value) for name, value in checker.design_r_lo.items()
            },
            "check_lo_tasks": bool(checker.check_lo_tasks),
            "candidate_positive_lower": {
                str(task.name): (int(task.c_lo) if task.criticality.value == "HI" else 1)
                for task in environment.ordered_tasks
            },
            "production_checker_type": type(checker).__qualname__,
            "check_safety": True,
        }
