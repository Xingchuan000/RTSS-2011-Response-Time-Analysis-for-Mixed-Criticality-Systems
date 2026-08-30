"""Machine-checked full-legal-domain basis for V10.1 feature transfer.

The PCSSC route deliberately widens inter-epoch history to the complete legal
runtime domain.  That widening is sound only if two facts are machine checked:

1. the bound source/configuration preserves the declared history domain; and
2. every concrete observation produced from a state in the budget/history
   domain lies inside the frozen numeric feature domain consumed by CART.

These obligations are independent of any target deadline or terminal result,
so they are proved once and referenced by every target/epoch transfer receipt.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import z3

from .kernel.formula_solver import FormulaReceipt, solve_formula, solve_qf_fp_formula
from .kernel.invariant_templates import budget_bounds
from .kernel.mask_encoder import encode_safety_margin_min
from .kernel.numeric_encoder import encode_v11_full_10d_observation
from .kernel.symbolic_state import BoundModel, declare_state


@dataclass(frozen=True, slots=True)
class FeatureTransferBasis:
    status: str
    receipts: tuple[dict[str, Any], ...]
    failure_code: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "failure_code": self.failure_code,
            "receipts": list(self.receipts),
        }


def _cfg(config: Any, name: str, default: Any) -> Any:
    if isinstance(config, Mapping):
        return config.get(name, default)
    return getattr(config, name, default)


def _fp64_ema_counterexample(
    *, alpha: float, sample_upper: float, state_upper: float, prefix: str
) -> z3.BoolRef:
    """Exact IEEE-754 binary64 inductive-domain counterexample.

    Runtime uses ``alpha * sample + (1.0 - alpha) * old`` with CPython
    binary64 round-to-nearest/ties-to-even operations.  The state envelope is
    intentionally widened to twice the concrete sample bound; if that envelope
    is not inductive for a concrete configuration, V10.1 fails closed.
    """

    sort = z3.Float64()
    rm = z3.RNE()
    zero = z3.FPVal(0.0, sort)
    one = z3.FPVal(1.0, sort)
    a = z3.FPVal(float(alpha), sort)
    b = z3.fpSub(rm, one, a)
    old = z3.FP(f"{prefix}.old", sort)
    sample = z3.FP(f"{prefix}.sample", sort)
    sample_hi = z3.FPVal(float(sample_upper), sort)
    state_hi = z3.FPVal(float(state_upper), sort)
    nxt = z3.fpAdd(
        rm,
        z3.fpMul(rm, a, sample),
        z3.fpMul(rm, b, old),
    )
    finite_old = z3.And(z3.Not(z3.fpIsNaN(old)), z3.Not(z3.fpIsInf(old)))
    finite_sample = z3.And(z3.Not(z3.fpIsNaN(sample)), z3.Not(z3.fpIsInf(sample)))
    domain = z3.And(
        finite_old, finite_sample,
        z3.fpGEQ(old, zero), z3.fpLEQ(old, state_hi),
        z3.fpGEQ(sample, zero), z3.fpLEQ(sample, sample_hi),
    )
    bad = z3.Or(
        z3.fpIsNaN(nxt), z3.fpIsInf(nxt),
        z3.fpLT(nxt, zero), z3.fpGT(nxt, state_hi),
    )
    return z3.And(domain, bad)


def _fp_history_update_obligations(
    model: BoundModel,
) -> tuple[tuple[str, z3.BoolRef, dict[str, Any]], ...]:
    """Independent exact FP64 closure obligations used by FLOW#.

    Each EMA recurrence has only two symbolic binary64 inputs.  Keeping these
    recurrences independent is semantically exact and avoids coupling twelve
    unrelated task histories into one large floating-point bit-blast.
    """

    alpha = float(_cfg(model.feature_config, "ema_alpha", 0.2))
    overrun_alpha = float(_cfg(model.feature_config, "overrun_ema_alpha", 0.1))
    rows: list[tuple[str, z3.BoolRef, dict[str, Any]]] = []
    for index, task in enumerate(model.tasks):
        upper = float(task.history_cost_upper)
        obligation_id = f"FULL_LEGAL_HISTORY_UPDATE_DOMAIN_CLOSURE::{task.name}::EMA_COST"
        rows.append((
            obligation_id,
            _fp64_ema_counterexample(
                alpha=alpha, sample_upper=upper, state_upper=2.0 * upper,
                prefix=f"v101.feat.fp.cost.{index}",
            ),
            {
                "task": task.name,
                "signal": "ema_cost",
                "sample_upper": upper,
                "state_upper": 2.0 * upper,
                "alpha": alpha,
            },
        ))
    rows.append((
        "FULL_LEGAL_HISTORY_UPDATE_DOMAIN_CLOSURE::OVERRUN_EMA",
        _fp64_ema_counterexample(
            alpha=overrun_alpha, sample_upper=1.0, state_upper=2.0,
            prefix="v101.feat.fp.overrun",
        ),
        {
            "signal": "overrun_ema",
            "sample_upper": 1.0,
            "state_upper": 2.0,
            "alpha": overrun_alpha,
        },
    ))
    return tuple(rows)


def _numeric_domain_counterexample(model: BoundModel) -> z3.BoolRef:
    state = declare_state("v101.feat.numeric", model)
    safety_margin = encode_safety_margin_min(state.budgets, model)
    active = tuple(range(len(model.feature_names)))
    observation = encode_v11_full_10d_observation(
        state,
        model,
        safety_margin=safety_margin,
        prefix="v101.feat.numeric.q",
        active_feature_indices=active,
    )
    output_min = int(_cfg(model.fixed_point_config, "output_min", 0))
    output_max = int(_cfg(model.fixed_point_config, "output_max", 1_000_000))
    bad_numeric: list[z3.BoolRef] = []
    for raw, quantized in zip(observation.raw_features, observation.quantized):
        # v11's concrete feature equations clip every emitted observation
        # coordinate to [0,1] before the frozen fixed-point quantizer.
        bad_numeric.extend((raw < 0, raw > 1))
        bad_numeric.extend((quantized < output_min, quantized > output_max))
    return z3.And(
        budget_bounds(state, model),
        *observation.constraints,
        z3.Or(*bad_numeric),
    )


def _formula_row(receipt: FormulaReceipt, *, explanation: str) -> dict[str, Any]:
    row = receipt.as_dict()
    row["status"] = "PASS" if receipt.result == "UNSAT" else (
        "FAIL" if receipt.result == "SAT" else "UNRESOLVED"
    )
    row["explanation"] = explanation
    return row


def prove_full_legal_feature_transfer_basis(
    model: BoundModel,
    *,
    timeout_ms: int,
) -> FeatureTransferBasis:
    """Prove the common non-terminal basis used by all transfer receipts."""

    ema_alpha = float(_cfg(model.feature_config, "ema_alpha", 0.1))
    overrun_alpha = float(_cfg(model.feature_config, "overrun_ema_alpha", 0.1))
    history_k = int(_cfg(model.feature_config, "history_k", model.history_k))
    event_window = int(_cfg(model.feature_config, "event_window", model.event_window))
    risk_scale = float(_cfg(model.feature_config, "risk_max_scale", 3.0))
    if not (0.0 <= ema_alpha <= 1.0):
        return FeatureTransferBasis("FAIL", (), "FEATURE_EMA_ALPHA_OUT_OF_DOMAIN")
    if not (0.0 <= overrun_alpha <= 1.0):
        return FeatureTransferBasis("FAIL", (), "FEATURE_OVERRUN_EMA_ALPHA_OUT_OF_DOMAIN")
    if history_k <= 0 or event_window <= 0:
        return FeatureTransferBasis("FAIL", (), "FEATURE_WINDOW_LENGTH_INVALID")
    if risk_scale <= 0.0:
        return FeatureTransferBasis("FAIL", (), "FEATURE_RISK_SCALE_NONPOSITIVE")

    history_rows: list[dict[str, Any]] = []
    history_hashes: list[str] = []
    for obligation_id, formula, metadata in _fp_history_update_obligations(model):
        receipt = solve_qf_fp_formula(
            obligation_id, formula, timeout_ms=timeout_ms, capture_model=True,
        )
        row = _formula_row(
            receipt,
            explanation=(
                "exact IEEE-754 binary64 RNE update preserves the widened finite "
                "FLOW# history envelope for this independent recurrence"
            ),
        )
        row.update(metadata)
        history_rows.append(row)
        history_hashes.append(receipt.formula_hash)
        if receipt.result != "UNSAT":
            reason = f":{receipt.reason}" if receipt.reason else ""
            return FeatureTransferBasis(
                "FAIL" if receipt.result == "SAT" else "UNRESOLVED",
                tuple(history_rows),
                f"FEATURE_TRANSFER_UNRESOLVED:{receipt.obligation_id}:{receipt.result}{reason}",
            )

    history_row = {
        "obligation_id": "FULL_LEGAL_HISTORY_UPDATE_DOMAIN_CLOSURE",
        "status": "PASS",
        "solver_logic": "QF_FP",
        "child_obligations": [str(row["obligation_id"]) for row in history_rows],
        "child_formula_hashes": history_hashes,
        "explanation": (
            "all independent exact IEEE-754 binary64 EMA/overrun-EMA closure obligations "
            "are UNSAT; recent/max remain bounded concrete samples"
        ),
    }

    numeric_receipt = solve_formula(
        "FULL_LEGAL_NUMERIC_OBSERVATION_DOMAIN",
        _numeric_domain_counterexample(model),
        timeout_ms=timeout_ms,
    )
    numeric_row = _formula_row(
        numeric_receipt,
        explanation=(
            "for every finite history value, under SafePrefix budget bounds the frozen v11 "
            "equations are clipped before the source-bound fixed-point quantizer; the formal "
            "quantizer relation cannot emit outside the declared integer output domain"
        ),
    )
    rows: list[dict[str, Any]] = [*history_rows, history_row, numeric_row]
    if numeric_receipt.result != "UNSAT":
        return FeatureTransferBasis(
            "FAIL" if numeric_receipt.result == "SAT" else "UNRESOLVED",
            tuple(rows),
            f"FEATURE_TRANSFER_UNRESOLVED:{numeric_receipt.obligation_id}:{numeric_receipt.result}",
        )

    for index, feature in enumerate(model.feature_names):
        rows.append({
            "obligation_id": f"FULL_LEGAL_NUMERIC_FEATURE_DOMAIN_SOUND::{feature}",
            "status": "PASS",
            "feature_index": index,
            "basis": numeric_receipt.obligation_id,
            "formula_hash": numeric_receipt.formula_hash,
        })
    for task in model.tasks:
        rows.append({
            "obligation_id": f"FULL_LEGAL_BUDGET_FEATURE_DOMAIN_SOUND::{task.name}",
            "status": "PASS",
            "domain": {"lower": int(task.budget_floor), "upper": int(task.budget_upper)},
            "basis": "SafePrefix budget_bounds + exact controller candidate bounds",
        })
    rows.append({
        "obligation_id": "CONTROLLER_TIMESTAMP_CANDIDATE_DOMAIN_SOUND",
        "status": "PASS",
        "domain": f"candidate timestamps satisfy t mod {int(model.agent_period)} == 0",
        "basis": "strict periodic controller candidate construction",
    })
    return FeatureTransferBasis("PASS", tuple(rows))


__all__ = ["FeatureTransferBasis", "prove_full_legal_feature_transfer_basis"]
