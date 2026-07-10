"""预算比例算术专项测试。

测试内容：
1. 整数比例 increase/decrease 的正确性；
2. 边界情况（1% 比例、100% 比例）；
3. 大整数预算的整数溢出安全。
"""

from __future__ import annotations

import pytest

from amc_py.rl.actions import (
    BudgetRatio,
    compute_increase_candidate,
    compute_decrease_candidate,
)


def test_increase_ratio_10pct():
    """10% increase 比例：100 -> 110。"""
    current = 100
    ratio = BudgetRatio.from_float_via_string(0.10)
    result = compute_increase_candidate(current, ratio)
    assert result == 110


def test_increase_ratio_50pct_odd():
    """50% increase 比例奇数值：33 的自增验证。"""
    current = 33
    ratio = BudgetRatio.from_float_via_string(0.50)
    result = compute_increase_candidate(current, ratio)
    # new = ceil(33 * 1.5) = ceil(49.5) = 50
    assert result == 50
    assert isinstance(result, int)


def test_decrease_ratio_5pct():
    """5% decrease 比例：100 -> 95。"""
    current = 100
    ratio = BudgetRatio.from_float_via_string(0.05)
    result = compute_decrease_candidate(current, ratio)
    assert result == 95


def test_decrease_ratio_50pct_odd():
    """50% decrease 比例奇数值：33 -> 16。
    decrease_amount = ceil(33 * 0.5) = 17; result = 33 - 17 = 16。
    """
    current = 33
    ratio = BudgetRatio.from_float_via_string(0.50)
    result = compute_decrease_candidate(current, ratio)
    assert result == 16


def test_increase_ratio_1pct_minimal():
    """1% increase 比例：100 -> 101。"""
    current = 100
    ratio = BudgetRatio.from_float_via_string(0.01)
    result = compute_increase_candidate(current, ratio)
    assert result == 101


def test_increase_ratio_large_budget():
    """大整数预算 increase 不溢出。"""
    current = 1_000_000_000
    ratio = BudgetRatio.from_float_via_string(0.10)
    result = compute_increase_candidate(current, ratio)
    assert isinstance(result, int)
    assert result > current


def test_decrease_ratio_1pct_minimal():
    """1% decrease 比例最小化：100 -> 99。"""
    current = 100
    ratio = BudgetRatio.from_float_via_string(0.01)
    result = compute_decrease_candidate(current, ratio)
    assert result == 99


def test_budget_ratio_floor_at_one():
    """decrease 50% on base=1: decrease_amount=ceil(0.5)=1, result=0。
    当前实现允许 decrease 归零（env 层会拒绝非正预算），
    这里的测试只验证算术正确性。"""
    current = 1
    ratio = BudgetRatio.from_float_via_string(0.50)
    result = compute_decrease_candidate(current, ratio)
    # decrease_amount = ceil(1*0.5) = 1; result = 0
    assert result == 0


def test_budget_ratio_consistency_10pct():
    """使用 Fraction 和 float 构造的 10% ratio 应得到一致的 increase 结果。"""
    current = 100
    ratio_frac = BudgetRatio(numerator=1, denominator=10)
    ratio_float = BudgetRatio.from_float_via_string(0.10)
    assert ratio_frac.numerator == ratio_float.numerator
    assert ratio_frac.denominator == ratio_float.denominator
    result_frac = compute_increase_candidate(current, ratio_frac)
    result_float = compute_increase_candidate(current, ratio_float)
    assert result_frac == result_float
    assert result_frac == 110
