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


def z3_nearest_even(value, ratio: Fraction):
    """Encode Python ``round`` for a non-negative rational value."""
    import z3
    scaled = value * ratio.numerator
    denominator = ratio.denominator
    quotient = scaled / denominator
    remainder = scaled % denominator
    twice = 2 * remainder
    return z3.If(
        twice < denominator,
        quotient,
        z3.If(
            twice > denominator,
            quotient + 1,
            z3.If(quotient % 2 == 0, quotient, quotient + 1),
        ),
    )


def encode_candidate_budget(action, budget_var):
    """Encode the executable whole-budget rounding used by amc_py.rl.actions."""
    import z3
    if action.operation == "noop":
        return budget_var
    if action.ratio is None:
        raise ValueError("ratio missing for budget action")
    ratio = action.ratio
    if action.operation == "increase":
        numerator = ratio.denominator + ratio.numerator
        direction = "increase"
    elif action.operation == "decrease":
        numerator = ratio.denominator - ratio.numerator
        direction = "decrease"
    else:
        raise ValueError(f"unsupported operation {action.operation}")
    effective_mode = action.rounding_mode
    if effective_mode in {"ceil_floor", "ceil", "floor"}:
        rounded = (
            z3_ceil_fraction(budget_var, Fraction(numerator, ratio.denominator))
            if direction == "increase"
            else z3_floor_fraction(budget_var, Fraction(numerator, ratio.denominator))
        )
    elif effective_mode == "nearest":
        rounded = z3_nearest_even(
            budget_var, Fraction(numerator, ratio.denominator)
        )
    elif effective_mode == "nearest_half_up":
        rounded = (
            2 * budget_var * numerator + ratio.denominator
        ) / (2 * ratio.denominator)
    else:
        raise ValueError(f"unsupported rounding {effective_mode}")
    if direction == "increase":
        return z3.If(rounded < budget_var + action.minimum_increment,
                     budget_var + action.minimum_increment, rounded)
    return z3.If(rounded > budget_var - action.minimum_increment,
                 budget_var - action.minimum_increment, rounded)


def find_rounding_difference(*, floor: int, upper: int, ratio: float, direction: str = "increase") -> dict:
    """Find a concrete integer state where C2 changes executable semantics."""
    import math
    if floor > upper or direction not in {"increase", "decrease"}:
        raise ValueError("invalid rounding search domain")
    for current in range(int(floor), int(upper) + 1):
        product = current * (1.0 + ratio) if direction == "increase" else current * (1.0 - ratio)
        ceil_floor = math.ceil(product) if direction == "increase" else math.floor(product)
        nearest = int(round(product))
        if ceil_floor != nearest:
            return {"status": "ACTIVATED", "current": current, "ceil_floor": ceil_floor, "nearest": nearest}
    return {"status": "NOT_ACTIVATED"}
