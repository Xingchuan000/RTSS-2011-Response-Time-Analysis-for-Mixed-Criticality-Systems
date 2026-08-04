from __future__ import annotations

from fractions import Fraction


def parse_ratio(value) -> Fraction:
    if isinstance(value, Fraction):
        return value
    if isinstance(value, str) and "/" in value:
        numerator, denominator = value.split("/", 1)
        return Fraction(int(numerator), int(denominator))
    return Fraction(str(value))


def z3_floor_fraction(value, ratio: Fraction):
    return (value * ratio.numerator) / ratio.denominator


def z3_ceil_fraction(value, ratio: Fraction):
    return (value * ratio.numerator + ratio.denominator - 1) / ratio.denominator


def z3_nearest_half_up(value, ratio: Fraction):
    return (2 * value * ratio.numerator + ratio.denominator) / (2 * ratio.denominator)


def encode_candidate_budget(action, budget_var):
    import z3
    if action.operation == "noop":
        return budget_var
    if action.ratio is None:
        raise ValueError("ratio missing for budget action")
    if action.rounding_mode == "ceil":
        delta = z3_ceil_fraction(budget_var, action.ratio)
    elif action.rounding_mode == "floor":
        delta = z3_floor_fraction(budget_var, action.ratio)
    elif action.rounding_mode == "nearest_half_up":
        delta = z3_nearest_half_up(budget_var, action.ratio)
    else:
        raise ValueError(f"unsupported rounding {action.rounding_mode}")
    delta = z3.If(delta < action.minimum_increment, action.minimum_increment, delta)
    return budget_var + delta if action.operation == "increase" else budget_var - delta
