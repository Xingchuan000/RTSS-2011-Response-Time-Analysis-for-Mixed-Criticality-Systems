from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

from formal_toolchain.core.hashing import sha256_object


def _criticality(task: Any) -> str:
    return str(getattr(task.criticality, "value", task.criticality))


def rebuild_expected_rows(
    tasks: Sequence[Any],
    *,
    design_r_lo: Mapping[str, int],
    check_lo_tasks: bool,
) -> list[dict[str, Any]]:
    """用精确整数独立重建 RuntimeBudgetSafetyChecker 的约束。"""
    rows: list[dict[str, Any]] = []

    for idx, task_i in enumerate(tasks):
        hp = tasks[:idx]
        crit = _criticality(task_i)

        if crit == "HI":
            r_lo_i = int(design_r_lo[str(task_i.name)])
            coeff = {str(task_i.name): 1}
            for task_j in hp:
                name = str(task_j.name)
                coeff[name] = coeff.get(name, 0) + math.ceil(r_lo_i / int(task_j.period))
            rows.append({
                "analyzed_task": str(task_i.name),
                "constraint": "hi_lo_mode",
                "coefficients": coeff,
                "rhs": r_lo_i,
            })

            coeff = {}
            rhs = int(task_i.deadline) - int(task_i.c_hi)
            for task_j in hp:
                name = str(task_j.name)
                if _criticality(task_j) == "LO":
                    coeff[name] = coeff.get(name, 0) + math.ceil(
                        r_lo_i / int(task_j.period)
                    )
                else:
                    rhs -= math.ceil(
                        int(task_i.deadline) / int(task_j.period)
                    ) * int(task_j.c_hi)
            rows.append({
                "analyzed_task": str(task_i.name),
                "constraint": "hi_mode_switch",
                "coefficients": coeff,
                "rhs": rhs,
            })

        if check_lo_tasks and crit == "LO":
            coeff = {str(task_i.name): 1}
            for task_j in hp:
                name = str(task_j.name)
                coeff[name] = coeff.get(name, 0) + math.ceil(
                    int(task_i.deadline) / int(task_j.period)
                )
            rows.append({
                "analyzed_task": str(task_i.name),
                "constraint": "lo_deadline_bound",
                "coefficients": coeff,
                "rhs": int(task_i.deadline),
            })

    for row_index, row in enumerate(rows):
        row["row_index"] = row_index
    return rows


def normalize_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    normalized = []
    for idx, row in enumerate(rows):
        coeff = {
            str(name): int(value)
            for name, value in row.get("coefficients", {}).items()
            if int(value) != 0
        }
        if any(value < 0 for value in coeff.values()):
            raise ValueError("STRUCTURAL_ENVELOPE_NEGATIVE_COEFFICIENT")
        normalized.append({
            "row_index": idx,
            "analyzed_task": str(row["analyzed_task"]),
            "constraint": str(row["constraint"]),
            "coefficients": coeff,
            "rhs": int(row["rhs"]),
        })
    return normalized


def verify_production_rows(
    production: Mapping[str, Any],
    tasks: Sequence[Any],
) -> dict[str, Any]:
    if production.get("status") != "PASS":
        return dict(production)
    expected = rebuild_expected_rows(
        tasks,
        design_r_lo={
            str(name): int(value)
            for name, value in production["design_r_lo"].items()
        },
        check_lo_tasks=bool(production["check_lo_tasks"]),
    )
    actual = normalize_rows(production["rows"])
    expected = normalize_rows(expected)
    if actual != expected:
        return {
            "status": "FAIL",
            "route": "MODEL_CONFORMANCE_FAILED",
            "failure": {"code": "SAFETY_POLYTOPE_PRODUCTION_MISMATCH"},
            "expected": expected,
            "actual": actual,
        }
    return {
        "status": "PASS",
        "schema_version": "verified_budget_safety_polytope_v1",
        "rows": actual,
        "row_hash": sha256_object(actual),
        "candidate_positive_lower": dict(production["candidate_positive_lower"]),
    }


def derive_componentwise_upper(
    *,
    rows: Sequence[Mapping[str, Any]],
    task_order: Sequence[str],
    candidate_lower: Mapping[str, int],
    action_hard_upper: Mapping[str, int],
) -> dict[str, Any]:
    """计算 safety polytope 在每一坐标上的精确整数最大值。"""
    upper: dict[str, int] = {}
    witnesses: dict[str, Any] = {}

    for task_name in task_order:
        best = int(action_hard_upper[task_name])
        limiting_rows: list[dict[str, Any]] = []

        for row in rows:
            coeff = row["coefficients"]
            ai = int(coeff.get(task_name, 0))
            if ai <= 0:
                continue

            residual = int(row["rhs"])
            for other in task_order:
                if other == task_name:
                    continue
                residual -= int(coeff.get(other, 0)) * int(candidate_lower[other])

            row_upper = residual // ai
            if row_upper < best:
                best = row_upper
                limiting_rows = [{
                    "row_index": int(row["row_index"]),
                    "constraint": str(row["constraint"]),
                    "rhs": int(row["rhs"]),
                    "coefficient": ai,
                    "residual_at_other_lowers": residual,
                    "row_upper": row_upper,
                }]
            elif row_upper == best:
                limiting_rows.append({
                    "row_index": int(row["row_index"]),
                    "constraint": str(row["constraint"]),
                    "rhs": int(row["rhs"]),
                    "coefficient": ai,
                    "residual_at_other_lowers": residual,
                    "row_upper": row_upper,
                })

        if best < int(candidate_lower[task_name]):
            return {
                "status": "FAIL",
                "route": "POLICY_CONTRACT_VIOLATION",
                "failure": {
                    "code": "SAFETY_POLYTOPE_EMPTY_COORDINATE",
                    "task": task_name,
                    "lower": int(candidate_lower[task_name]),
                    "upper": best,
                },
            }

        upper[task_name] = best
        witnesses[task_name] = {
            "hard_upper": int(action_hard_upper[task_name]),
            "derived_upper": best,
            "limiting_rows": limiting_rows,
        }

    return {
        "status": "PASS",
        "schema_version": "componentwise_envelope_v1",
        "upper": upper,
        "witnesses": witnesses,
    }


def vector_satisfies_rows(
    budgets: Mapping[str, int], rows: Sequence[Mapping[str, Any]]
) -> bool:
    for row in rows:
        lhs = sum(
            int(value) * int(budgets[name])
            for name, value in row["coefficients"].items()
        )
        if lhs > int(row["rhs"]):
            return False
    return True
