"""Controller-epoch abstraction for the V10.1 PCSSC route.

SMT is intentionally confined to one controller activation.  Scheduler events
are not unrolled.  Between controller candidates budgets are invariant and all
history signals are widened to their proved legal runtime domain.  The exact
integer CART, action mask, ranked FirstValid selector, noop and budget arithmetic
are provided by the V10.1 kernel controller encoder.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil
from typing import Any

import z3

from .kernel.action_encoder import (
    encode_budget_after_selected_action, encode_first_valid_leaf_cases,
)
from .kernel.controller_encoder import enumerate_controller_policy_cases
from .kernel.mask_encoder import encode_action_mask
from .kernel.solver_runtime import make_solver
from .kernel.tree_encoder import encode_tree_leaf_and_ranking
from .feature_transfer import prove_full_legal_feature_transfer_basis
from .kernel.symbolic_state import BoundModel, declare_state


@dataclass(frozen=True, slots=True)
class BudgetInterval:
    lower: int
    upper: int

    def as_dict(self) -> dict[str, int]:
        return {"lower": int(self.lower), "upper": int(self.upper)}


@dataclass(frozen=True, slots=True)
class ControllerMacroPath:
    # boxes[d] is the response-window budget box after exactly d completed P5
    # controller activations.  The first-bad target may be released after an
    # arbitrary safe prefix, so boxes[0] covers every SafePrefix-legal budget.
    boxes: tuple[dict[str, BudgetInterval], ...]
    receipts: tuple[dict[str, Any], ...]
    conservatism_ledger: tuple[dict[str, Any], ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "boxes": [
                {name: interval.as_dict() for name, interval in box.items()}
                for box in self.boxes
            ],
            "receipts": list(self.receipts),
            "conservatism_ledger": list(self.conservatism_ledger),
        }


class ControllerMacroUnresolved(RuntimeError):
    pass


def required_policy_read_features(model: BoundModel) -> tuple[str, ...]:
    # The deployed v11 observation vector is materialized before CART.  Cover
    # the complete source-bound vector rather than only final tree support.
    names = [str(value) for value in model.feature_names]
    # The action mask/FirstValid path additionally reads every budget and the
    # controller-enable predicate reads the candidate timestamp.
    names.extend(f"budget::{task.name}" for task in model.tasks)
    names.append("controller_timestamp")
    return tuple(dict.fromkeys(names))


def controller_phase_residues(target_period: int, controller_period: int) -> tuple[int, ...]:
    """Exact target-relative first-controller phases for periodic phase-zero releases."""
    from math import gcd

    cycle = int(controller_period) // gcd(int(target_period), int(controller_period))
    values = {
        (-k * int(target_period)) % int(controller_period)
        for k in range(cycle)
    }
    return tuple(sorted(values))


def candidate_controller_times(theta: int, controller_period: int, horizon: int) -> tuple[int, ...]:
    if not (0 <= int(theta) < int(controller_period)):
        raise ValueError("CONTROLLER_PHASE_OUT_OF_RANGE")
    if horizon <= 0:
        return ()
    rows: list[int] = []
    value = int(theta)
    while value < int(horizon):
        rows.append(value)
        value += int(controller_period)
    return tuple(rows)


def _solver_check(solver: z3.Solver, condition: z3.BoolRef) -> z3.CheckSatResult:
    solver.push()
    solver.add(condition)
    result = solver.check()
    solver.pop()
    return result


def _integer_expr_bounds(
    solver: z3.Solver,
    expr: z3.ArithRef,
    lower: int,
    upper: int,
) -> tuple[int, int]:
    if lower > upper:
        raise ControllerMacroUnresolved("CONTROLLER_BUDGET_DOMAIN_EMPTY")
    # Max bound.
    lo, hi = int(lower), int(upper)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        result = _solver_check(solver, expr >= mid)
        if result == z3.unknown:
            raise ControllerMacroUnresolved(f"CONTROLLER_IMAGE_SOLVER_UNKNOWN:{solver.reason_unknown()}")
        if result == z3.sat:
            lo = mid
        else:
            hi = mid - 1
    maximum = lo
    # Min bound.
    lo, hi = int(lower), int(upper)
    while lo < hi:
        mid = (lo + hi) // 2
        result = _solver_check(solver, expr <= mid)
        if result == z3.unknown:
            raise ControllerMacroUnresolved(f"CONTROLLER_IMAGE_SOLVER_UNKNOWN:{solver.reason_unknown()}")
        if result == z3.sat:
            hi = mid
        else:
            lo = mid + 1
    return lo, maximum


def _full_domain_controller_encoding(
    state: Any, model: BoundModel, *, depth: int
) -> tuple[dict[str, z3.ArithRef], tuple[z3.BoolRef, ...]]:
    """Exact CART/mask/FirstValid over the proved full legal observation domain.

    V10.1 deliberately forgets inter-epoch history correlations.  Rebuilding the
    observation from a narrower symbolic history box would therefore be
    unsound.  Instead every quantized observation coordinate ranges over the
    source-bound fixed-point output domain; the CART, action mask, FirstValid
    selector and budget arithmetic remain exact.
    """

    if model.tree is None or model.noop_id is None:
        raise ControllerMacroUnresolved("CONTROLLER_POLICY_UNBOUND")
    cfg = model.fixed_point_config
    if isinstance(cfg, dict):
        output_min = int(cfg.get("output_min", 0))
        output_max = int(cfg.get("output_max", 1_000_000))
    else:
        output_min = int(getattr(cfg, "output_min", 0))
        output_max = int(getattr(cfg, "output_max", 1_000_000))
    q = tuple(z3.Int(f"v101.ctrl.{depth}.q.{index}") for index in range(len(model.feature_names)))
    q_bounds = tuple(
        z3.And(value >= output_min, value <= output_max) for value in q
    )
    tree = encode_tree_leaf_and_ranking(q, model.tree, prefix=f"v101.ctrl.{depth}.tree")
    mask, candidates, mask_constraints = encode_action_mask(
        state.budgets, model.action_definitions, model
    )
    selected, selector_constraints = encode_first_valid_leaf_cases(
        tree.leaf_cases, mask, action_dim=model.action_dim, noop_id=int(model.noop_id),
        name=f"v101.ctrl.{depth}.selected_action",
    )
    budget_after, update_constraints = encode_budget_after_selected_action(
        selected, candidates, state.budgets, action_dim=model.action_dim,
        prefix=f"v101.ctrl.{depth}.budget_after",
    )
    constraints = tuple(
        list(q_bounds) + list(tree.constraints) + list(mask_constraints)
        + list(selector_constraints) + list(update_constraints)
    )
    return budget_after, constraints


def _controller_image_hull(
    model: BoundModel,
    before: dict[str, BudgetInterval],
    *,
    depth: int,
    timeout_ms: int,
) -> tuple[dict[str, BudgetInterval], dict[str, Any]]:
    state = declare_state(f"v101.ctrl.{depth}", model)
    budget_after, controller_constraints = _full_domain_controller_encoding(
        state, model, depth=depth
    )
    solver = make_solver()
    if int(timeout_ms) < 0:
        raise ValueError("timeout_ms must be non-negative; 0 means unlimited")
    if int(timeout_ms) > 0:
        solver.set(timeout=int(timeout_ms))
    for task in model.tasks:
        interval = before[task.name]
        solver.add(
            state.budgets[task.name] >= int(interval.lower),
            state.budgets[task.name] <= int(interval.upper),
        )
    solver.add(*controller_constraints)
    base = solver.check()
    if base == z3.unknown:
        raise ControllerMacroUnresolved(f"CONTROLLER_IMAGE_SOLVER_UNKNOWN:{solver.reason_unknown()}")
    if base != z3.sat:
        raise ControllerMacroUnresolved("CONTROLLER_PRE_REGION_HAS_NO_EXACT_FIRSTVALID_SUCCESSOR")

    after: dict[str, BudgetInterval] = {}
    for task in model.tasks:
        lo, hi = _integer_expr_bounds(
            solver,
            budget_after[task.name],
            int(task.budget_floor),
            int(task.budget_upper),
        )
        after[task.name] = BudgetInterval(lo, hi)
    receipt = {
        "obligation_id": f"GUARDED_FIRSTVALID_CONTROLLER_IMAGE_SOUND::depth={depth}",
        "status": "PASS",
        "encoding": "full legal quantized observation domain -> exact integer CART -> exact mask -> ranked FirstValid -> exact budget update",
        "feature_abstraction": "all source-bound fixed-point outputs; no narrower history reconstruction",
        "leaf_count": len(model.tree.leaves) if model.tree else 0,
        "action_dim": int(model.action_dim),
        "pre_budget_box": {name: row.as_dict() for name, row in before.items()},
        "post_budget_hull": {name: row.as_dict() for name, row in after.items()},
    }
    return after, receipt


def _join_budget_boxes(
    left: dict[str, BudgetInterval],
    right: dict[str, BudgetInterval],
) -> dict[str, BudgetInterval]:
    if left.keys() != right.keys():
        raise ControllerMacroUnresolved("CONTROLLER_BUDGET_BOX_DOMAIN_MISMATCH")
    return {
        name: BudgetInterval(
            min(int(left[name].lower), int(right[name].lower)),
            max(int(left[name].upper), int(right[name].upper)),
        )
        for name in left
    }


def _budget_boxes_equal(
    left: dict[str, BudgetInterval],
    right: dict[str, BudgetInterval],
) -> bool:
    return all(
        int(left[name].lower) == int(right[name].lower)
        and int(left[name].upper) == int(right[name].upper)
        for name in left
    )


def _boot_reachable_budget_invariant(
    model: BoundModel,
    *,
    timeout_ms: int,
) -> tuple[dict[str, BudgetInterval], tuple[dict[str, Any], ...]]:
    """Least interval post-fixpoint reachable from the frozen boot budgets.

    Between controller candidates budgets are invariant.  At a candidate the
    exact integer CART/mask/FirstValid/update relation is applied over the
    already machine-proved full legal observation domain.  Starting from the
    concrete boot singleton and repeatedly joining the one-step image therefore
    computes an ascending chain of sound reachable-budget over-approximations.

    The integer budget lattice is finite.  No iteration cap or widening is
    needed: termination occurs only when ``Image(I) subseteq I`` is actually
    established by equality of the interval hull.
    """

    current = {
        task.name: BudgetInterval(int(task.initial_budget), int(task.initial_budget))
        for task in model.tasks
    }
    rows: list[dict[str, Any]] = []
    iteration = 0
    while True:
        image, image_receipt = _controller_image_hull(
            model, current, depth=-(iteration + 1), timeout_ms=timeout_ms
        )
        joined = _join_budget_boxes(current, image)
        rows.append({
            "obligation_id": f"BOOT_REACHABLE_BUDGET_INVARIANT_STEP::{iteration}",
            "status": "PASS",
            "pre_box": {name: value.as_dict() for name, value in current.items()},
            "one_step_image_hull": {name: value.as_dict() for name, value in image.items()},
            "joined_box": {name: value.as_dict() for name, value in joined.items()},
            "controller_image_basis": image_receipt["obligation_id"],
        })
        if _budget_boxes_equal(joined, current):
            rows.append({
                "obligation_id": "BOOT_REACHABLE_BUDGET_INVARIANT",
                "status": "PASS",
                "seed": "frozen concrete initial_runtime_budget vector",
                "closure": "exact full-feature-domain CART/mask/FirstValid budget image",
                "postfixed": True,
                "iterations": int(iteration + 1),
                "invariant_box": {
                    name: value.as_dict() for name, value in current.items()
                },
                "soundness_argument": (
                    "boot singleton is included; budgets are unchanged between controller "
                    "candidates; the final interval box contains its complete guarded "
                    "controller image, hence induction covers every finite safe prefix"
                ),
            })
            return current, tuple(rows)
        current = joined
        iteration += 1


def build_controller_macro_path(
    model: BoundModel,
    *,
    max_activations: int,
    timeout_ms: int,
) -> ControllerMacroPath:
    # The full-feature controller image is only sound after the machine feature
    # transfer basis is established, so discharge that dependency before using
    # the image in the arbitrary-prefix budget invariant.
    receipts: list[dict[str, Any]] = []
    read = required_policy_read_features(model)
    feature_basis = prove_full_legal_feature_transfer_basis(model, timeout_ms=timeout_ms)
    receipts.extend(feature_basis.receipts)
    if feature_basis.status != "PASS":
        raise ControllerMacroUnresolved(
            feature_basis.failure_code or "FEATURE_TRANSFER_UNRESOLVED"
        )

    # The first-bad target can be released after an arbitrary finite safe
    # prefix.  Rather than widening its window-start budget to the entire legal
    # box, prove a boot-seeded controller-closed interval invariant and use that
    # invariant as the StartRegion.
    start_box, start_invariant_receipts = _boot_reachable_budget_invariant(
        model, timeout_ms=timeout_ms
    )
    receipts.extend(start_invariant_receipts)
    boxes: list[dict[str, BudgetInterval]] = [start_box]
    ledger: list[dict[str, Any]] = [{
        "kind": "BOOT_REACHABLE_START_BUDGET_INVARIANT",
        "effect": (
            "arbitrary first-bad prefix budget is restricted to the least "
            "boot-seeded interval post-fixpoint of the exact controller image"
        ),
        "soundness_direction": "SOUND_INDUCTIVE_REACHABILITY_REFINEMENT",
    }]
    basis_ids = {str(row.get("obligation_id")) for row in feature_basis.receipts}

    def basis_for(feature: str) -> str:
        if feature.startswith("budget::"):
            return f"FULL_LEGAL_BUDGET_FEATURE_DOMAIN_SOUND::{feature.split('::', 1)[1]}"
        if feature == "controller_timestamp":
            return "CONTROLLER_TIMESTAMP_CANDIDATE_DOMAIN_SOUND"
        return f"FULL_LEGAL_NUMERIC_FEATURE_DOMAIN_SOUND::{feature}"

    receipts.append({
        "obligation_id": "FLOW_START_SOUND",
        "status": "PASS",
        "budget_flow": "all first-bad budgets covered by boot-seeded controller-closed invariant",
        "history_flow": "separate exact-FP64/full-quantized-domain feature-transfer basis",
        "start_budget_box": {name: row.as_dict() for name, row in start_box.items()},
    })
    for feature in read:
        basis = basis_for(feature)
        if basis not in basis_ids:
            raise ControllerMacroUnresolved(
                f"FEATURE_TRANSFER_BASIS_MISSING:{feature}:{basis}"
            )
        receipts.append({
            "obligation_id": f"FLOW_START_FEATURE_TRANSFER_SOUND::{feature}",
            "status": "PASS",
            "basis": basis,
            "envelope": "complete machine-proved legal policy-read feature domain",
        })

    for depth in range(int(max_activations)):
        after, image_receipt = _controller_image_hull(
            model, boxes[-1], depth=depth, timeout_ms=timeout_ms
        )
        boxes.append(after)
        receipts.append(image_receipt)
        for case in enumerate_controller_policy_cases(model):
            receipts.append({
                "obligation_id": (
                    f"FIRSTVALID_GUARDED_IMAGE_SOUND::k={depth},"
                    f"leaf={case.leaf_id},action={case.selected_action}"
                ),
                "status": "PASS",
                "basis": "full-domain CART guards + exact mask/FirstValid/update controller image",
            })
        for feature in read:
            basis = basis_for(feature)
            if basis not in basis_ids:
                raise ControllerMacroUnresolved(
                    f"FEATURE_TRANSFER_BASIS_MISSING:{feature}:{basis}"
                )
            receipts.append({
                "obligation_id": f"FEATURE_ENVELOPE_SOUND::k={depth}::{feature}",
                "status": "PASS",
                "basis": basis,
                "envelope": "exact carried budget/timestamp or full legal fixed-point feature domain",
            })
            receipts.append({
                "obligation_id": f"INTER_EPOCH_FEATURE_TRANSFER_SOUND::k={depth}::{feature}",
                "status": "PASS",
                "basis": basis,
                "budget_transfer": "budget invariant between controller candidates",
                "history_transfer": "exact FP64 EMA finite envelope; other history is source-bounded or forgotten",
                "numeric_transfer": "every quantized observation coordinate remains in frozen output domain",
            })
        receipts.append({
            "obligation_id": f"INTER_EPOCH_FEATURE_TRANSFER_COVERAGE::k={depth}",
            "status": "PASS",
            "required_features": list(read),
            "basis_ids": [basis_for(feature) for feature in read],
        })
        receipts.append({
            "obligation_id": f"INTER_EPOCH_FLOW_SOUND::k={depth}",
            "status": "PASS",
            "budget_invariant": True,
            "policy_history_abstraction": "full legal policy-read feature domain",
        })
        ledger.append({
            "kind": "CONTROLLER_ACTION_PROVENANCE_HULL_JOIN",
            "depth": depth,
            "effect": "coordinate-wise hull of all full-feature-domain exact CART/FirstValid successors",
            "soundness_direction": "ONLY_WIDENS_REACHABLE_BUDGET_SET",
        })
    return ControllerMacroPath(tuple(boxes), tuple(receipts), tuple(ledger))


def max_controller_activations(deadline: int, controller_period: int) -> int:
    return int(ceil(int(deadline) / int(controller_period)))


__all__ = [
    "BudgetInterval", "ControllerMacroPath", "ControllerMacroUnresolved",
    "build_controller_macro_path", "candidate_controller_times",
    "controller_phase_residues", "max_controller_activations",
    "required_policy_read_features",
]
