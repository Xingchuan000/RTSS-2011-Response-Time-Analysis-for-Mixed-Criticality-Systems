"""SMT binding for the deployed v11_full_10d integer observation."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, ROUND_FLOOR
from typing import Any, Mapping, Sequence

import z3

from .symbolic_state import BoundModel, SymbolicKernelState


@dataclass(frozen=True, slots=True)
class NumericEncoding:
    quantized: tuple[z3.ArithRef, ...]
    constraints: tuple[z3.BoolRef, ...]
    raw_features: tuple[z3.ArithRef, ...]


def _config_value(config: Any, name: str, default: Any) -> Any:
    if config is None:
        return default
    if isinstance(config, Mapping):
        return config.get(name, default)
    return getattr(config, name, default)


def _real(value: Any) -> z3.ArithRef:
    return value if isinstance(value, z3.ArithRef) else z3.RealVal(str(value))


def _clip(value: z3.ArithRef, lo: Any = 0, hi: Any = 1) -> z3.ArithRef:
    return z3.If(value < _real(lo), _real(lo), z3.If(value > _real(hi), _real(hi), value))


def _max(*values: z3.ArithRef) -> z3.ArithRef:
    result = values[0]
    for value in values[1:]:
        result = z3.If(result >= value, result, value)
    return result


def _fp_real(value: z3.ExprRef) -> z3.ArithRef:
    if z3.is_fp_value(value) or z3.is_fp(value):
        return z3.fpToReal(value)
    if isinstance(value, z3.ArithRef):
        return value
    raise TypeError("numeric encoder expects a Z3 FP or arithmetic expression")


def encode_quantized_feature(
    fp_value: z3.ExprRef,
    fixed_config: Any,
    *,
    name: str,
) -> tuple[z3.ArithRef, list[z3.BoolRef]]:
    """Encode half-up quantization with a sound one-cell FP64 boundary envelope.

    The exact binary64 value is converted with ``fpToReal``.  At a Decimal
    shortest-roundtrip/half-up boundary, either adjacent integer is admitted;
    elsewhere the relation is singleton.  This is intentionally an
    over-approximation, so an UNSAT proof remains sound.
    """

    scale = int(_config_value(fixed_config, "scale", 1_000_000))
    input_min = Decimal(str(_config_value(fixed_config, "input_min", 0.0)))
    input_max = Decimal(str(_config_value(fixed_config, "input_max", 1.0)))
    output_min = int(_config_value(fixed_config, "output_min", 0))
    output_max = int(_config_value(fixed_config, "output_max", scale))
    if scale <= 0 or output_min > output_max:
        raise ValueError("invalid fixed-point configuration")
    x = _fp_real(fp_value)
    clipped = z3.If(x < z3.RealVal(str(input_min)), z3.RealVal(str(input_min)),
                    z3.If(x > z3.RealVal(str(input_max)), z3.RealVal(str(input_max)), x))
    scaled = clipped * scale
    nearest = z3.ToInt(scaled + z3.RealVal("0.5"))
    q = z3.Int(name)
    constraints = [q >= output_min, q <= output_max,
                   z3.Or(q == nearest, q == nearest - 1, q == nearest + 1)]
    return q, constraints


def _quantize_real(value: z3.ArithRef, config: Any, *, name: str):
    # Reuse the public FP relation for real-valued formulas.  Real formulas are
    # exact inputs to the over-approx relation and are useful in differential
    # tests where the concrete runtime has already produced binary64 values.
    return encode_quantized_feature(value, config, name=name)


def encode_v11_full_10d_observation(
    state: SymbolicKernelState,
    model: BoundModel,
    *,
    safety_margin: z3.ArithRef | int | float = 1,
    prefix: str = "q",
    active_feature_indices: Sequence[int] | None = None,
) -> NumericEncoding:
    """Translate only tree-observed v11 features to SMT.

    The deployed CART can only inspect feature indices that occur in its
    internal nodes.  Unused observation coordinates are semantically dead for
    policy selection, so they are represented by constant zero and receive no
    quantization constraints.  This is exact support projection, not a policy
    abstraction.
    """

    feature_count = 10 * len(model.tasks) + 8
    if model.feature_names and len(model.feature_names) != feature_count:
        raise ValueError("NUMERIC_OBSERVATION_FEATURE_SCHEMA_MISMATCH")
    if active_feature_indices is None:
        active = frozenset(range(feature_count))
    else:
        active = frozenset(int(index) for index in active_feature_indices)
        if any(index < 0 or index >= feature_count for index in active):
            raise ValueError("TREE_FEATURE_INDEX_OUT_OF_RANGE")
    cfg = model.fixed_point_config
    scale = int(_config_value(cfg, "scale", 1_000_000))
    raw: list[z3.ArithRef] = []
    total_util = z3.RealVal(0)
    hi_util = z3.RealVal(0)
    lo_util = z3.RealVal(0)
    for rank, task in enumerate(model.tasks):
        norm_lo = z3.RealVal(task.normalization_min_cost)
        norm_hi = z3.RealVal(task.normalization_upper)
        norm_width = norm_hi - norm_lo
        budget = _real(state.budgets[task.name])
        recent = _real(state.chi.recent_cost[task.name])
        ema = _real(state.chi.ema_cost[task.name])
        maxk = _real(state.chi.max_cost_k[task.name])
        # Production v11 uses the dedicated RuntimeFeatureState.overrun_ema
        # signal directly; it is already an EMA of a 0/1 overrun event.
        overrun = _clip(_real(state.chi.overrun_ema[task.name]))
        criticality = z3.RealVal(1 if task.criticality == "HI" else 0)
        priority = z3.RealVal(str(1.0 - rank / max(1, len(model.tasks) - 1)))
        util = _clip(budget / task.period)
        max_weight = z3.RealVal(str(_config_value(model.feature_config, "max_cost_weight", 0.7)))
        pred = _max(recent, ema, max_weight * maxk)
        risk_scale = z3.RealVal(str(_config_value(model.feature_config, "risk_max_scale", 3.0)))
        risk = _clip((pred / z3.If(budget > 1, budget, z3.RealVal(1)) +
                      z3.RealVal("0.5") * overrun + z3.RealVal("0.2") * criticality +
                      z3.RealVal("0.1") * priority) / risk_scale)
        surplus = _clip(((budget - pred) / z3.If(budget > 1, budget, z3.RealVal(1)) + 1) / 2)
        def normalize(value: z3.ArithRef) -> z3.ArithRef:
            # Production _normalize is clip((value-lo)/(hi-lo)).  Bounds are
            # frozen as exact binary64 integers in build_bindings; the public
            # quantizer still keeps its one-cell over-approximation around
            # binary64/Decimal conversion boundaries.
            return _clip((value - norm_lo) / norm_width)

        values = (
            normalize(budget), normalize(recent), normalize(ema),
            normalize(maxk), overrun, risk, surplus, criticality, _clip(priority), util,
        )
        raw.extend(values)
        total_util += budget / task.period
        if task.criticality == "HI":
            hi_util += budget / task.period
        else:
            lo_util += budget / task.period

    denominator = z3.If(z3.Sum(*state.chi.job_start_window) > 0,
                        z3.Sum(*state.chi.job_start_window), z3.IntVal(1))
    rates = (
        z3.Sum(*state.chi.mode_change_window) / denominator,
        z3.Sum(*state.chi.lo_cancel_window) / denominator,
        z3.Sum(*state.chi.hi_overrun_window) / denominator,
        z3.Sum(*state.chi.lo_overrun_window) / denominator,
    )
    raw.extend((_clip(total_util), _clip(hi_util), _clip(lo_util),
                *(_clip(value) for value in rates), _clip(_real(safety_margin))))
    quantized: list[z3.ArithRef] = []
    constraints: list[z3.BoolRef] = []
    for index, value in enumerate(raw):
        if index not in active:
            quantized.append(z3.IntVal(0))
            continue
        q, relation = _quantize_real(value, cfg, name=f"{prefix}.{index}")
        quantized.append(q)
        constraints.extend(relation)
    if len(quantized) != 10 * len(model.tasks) + 8:
        raise AssertionError("v11_full_10d encoder produced an unexpected dimension")
    return NumericEncoding(tuple(quantized), tuple(constraints), tuple(raw))


__all__ = ["NumericEncoding", "encode_quantized_feature", "encode_v11_full_10d_observation"]
