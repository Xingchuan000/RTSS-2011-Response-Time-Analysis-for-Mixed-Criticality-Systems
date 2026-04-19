"""阶段5：统计聚合模块测试。"""

from __future__ import annotations

import pandas as pd

from amc_py.aggregation import aggregate_by_util, aggregate_weighted_schedulability


def test_aggregate_by_util_matches_hand_calculation() -> None:
    """util 层内聚合应与手算一致。"""

    # 手算样本：
    # util=0.5: (U=0.5,1), (U=0.4,0) => util_sum=0.9, weighted_success=0.5
    # util=0.7: (U=0.7,1), (U=0.8,1) => util_sum=1.5, weighted_success=1.5
    df = pd.DataFrame(
        {
            "util_level": [0.5, 0.5, 0.7, 0.7],
            "actual_total_util_lo": [0.5, 0.4, 0.7, 0.8],
            "schedulable": [True, False, True, True],
        }
    )

    agg = aggregate_by_util(df, util_col="util_level")
    by_util = {row["util_level"]: row for row in agg.to_dict("records")}

    assert abs(by_util[0.5]["util_sum"] - 0.9) < 1e-9
    assert abs(by_util[0.5]["weighted_success_sum"] - 0.5) < 1e-9
    assert abs(by_util[0.5]["weighted_schedulability_at_util"] - (0.5 / 0.9)) < 1e-9
    assert abs(by_util[0.5]["schedulable_ratio"] - 0.5) < 1e-9

    assert abs(by_util[0.7]["util_sum"] - 1.5) < 1e-9
    assert abs(by_util[0.7]["weighted_success_sum"] - 1.5) < 1e-9
    assert abs(by_util[0.7]["weighted_schedulability_at_util"] - 1.0) < 1e-9
    assert abs(by_util[0.7]["schedulable_ratio"] - 1.0) < 1e-9


def test_aggregate_weighted_schedulability_across_utils() -> None:
    """跨 util 聚合应等于两层手算公式。"""

    raw = pd.DataFrame(
        {
            "method": ["m", "m", "m", "m", "m", "m"],
            "sweep_value": [1.0, 1.0, 1.0, 2.0, 2.0, 2.0],
            "util_level": [0.4, 0.4, 0.8, 0.4, 0.8, 0.8],
            "actual_total_util_lo": [0.4, 0.5, 0.8, 0.4, 0.7, 0.9],
            "schedulable": [True, False, True, False, True, False],
        }
    )

    util_agg = aggregate_by_util(
        raw,
        util_col="util_level",
        outer_group_cols=["method", "sweep_value"],
    )
    weighted = aggregate_weighted_schedulability(
        util_agg,
        util_col="util_level",
        outer_group_cols=["method", "sweep_value"],
    )

    rows = {(row["method"], row["sweep_value"]): row for row in weighted.to_dict("records")}

    # sweep=1.0 手算：
    # util_sum = 0.4+0.5+0.8 = 1.7
    # weighted_success = 0.4+0+0.8 = 1.2
    assert abs(rows[("m", 1.0)]["util_sum"] - 1.7) < 1e-9
    assert abs(rows[("m", 1.0)]["weighted_success_sum"] - 1.2) < 1e-9
    assert abs(rows[("m", 1.0)]["weighted_schedulability"] - (1.2 / 1.7)) < 1e-9

    # sweep=2.0 手算：
    # util_sum = 0.4+0.7+0.9 = 2.0
    # weighted_success = 0+0.7+0 = 0.7
    assert abs(rows[("m", 2.0)]["util_sum"] - 2.0) < 1e-9
    assert abs(rows[("m", 2.0)]["weighted_success_sum"] - 0.7) < 1e-9
    assert abs(rows[("m", 2.0)]["weighted_schedulability"] - 0.35) < 1e-9


def test_statistics_empty_input() -> None:
    """空样本应返回空表而不是崩溃。"""

    empty = pd.DataFrame(columns=["util_level", "actual_total_util_lo", "schedulable"])
    util_agg = aggregate_by_util(empty, util_col="util_level")
    assert util_agg.empty

    weighted = aggregate_weighted_schedulability(util_agg, util_col="util_level")
    assert weighted.empty


def test_statistics_all_success_and_all_failure() -> None:
    """边界：全成功与全失败。"""

    all_success = pd.DataFrame(
        {
            "util_level": [0.3, 0.3, 0.6],
            "actual_total_util_lo": [0.3, 0.2, 0.6],
            "schedulable": [True, True, True],
        }
    )
    success_util = aggregate_by_util(all_success, util_col="util_level")
    success_weighted = aggregate_weighted_schedulability(success_util, util_col="util_level")
    assert abs(success_weighted.iloc[0]["weighted_schedulability"] - 1.0) < 1e-9
    assert abs(success_weighted.iloc[0]["schedulable_ratio"] - 1.0) < 1e-9

    all_failure = pd.DataFrame(
        {
            "util_level": [0.3, 0.6],
            "actual_total_util_lo": [0.3, 0.6],
            "schedulable": [False, False],
        }
    )
    fail_util = aggregate_by_util(all_failure, util_col="util_level")
    fail_weighted = aggregate_weighted_schedulability(fail_util, util_col="util_level")
    assert abs(fail_weighted.iloc[0]["weighted_schedulability"] - 0.0) < 1e-9
    assert abs(fail_weighted.iloc[0]["schedulable_ratio"] - 0.0) < 1e-9
