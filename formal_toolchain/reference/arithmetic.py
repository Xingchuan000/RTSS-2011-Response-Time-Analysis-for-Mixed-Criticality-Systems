"""Phase J01：仅使用非负整数的精确算术辅助函数。"""

from __future__ import annotations


def _int(value: object, name: str, *, nonnegative: bool = True) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} 必须是 int")
    if nonnegative and value < 0:
        raise ValueError(f"{name} 不能为负数")
    return value


def ceil_div_nonnegative(a: int, b: int) -> int:
    a, b = _int(a, "a"), _int(b, "b")
    if b == 0:
        raise ZeroDivisionError("b 不能为 0")
    return (a + b - 1) // b


def floor_div_nonnegative(a: int, b: int) -> int:
    a, b = _int(a, "a"), _int(b, "b")
    if b == 0:
        raise ZeroDivisionError("b 不能为 0")
    return a // b


def post_count(s: int, t: int, period: int) -> int:
    """返回 release time 严格位于 ``[s,t)`` 后的数量。"""
    s, t, period = _int(s, "s"), _int(t, "t"), _int(period, "period")
    if period == 0:
        raise ValueError("period 必须为正整数")
    if t <= s:
        return 0
    return ceil_div_nonnegative(t - s, period)
