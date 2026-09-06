"""Structure-only five-strata selector tests."""

from __future__ import annotations

import copy

import pytest

from scripts.select_mc_stratified_dynamic_primary10 import (
    DIAGNOSTICS_SCHEMA_VERSION,
    SelectionShortageError,
    SELECTION_FEATURES,
    assert_selection_feature_guard,
    build_percentile_features,
    select_primary10,
    validate_diagnostics_rows,
)


def _diagnostic_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for seed in range(10):
        family = "semi-harmonic" if seed < 5 else "log-uniform"
        rows.append(
            {
                "schema_version": DIAGNOSTICS_SCHEMA_VERSION,
                "input_schema_version": "mc_stratified_dynamic_manifest_v1",
                "candidate_seed": seed,
                "period_family": family,
                "baseline_deadline_misses_sum": 0,
                "baseline_lo_quality_qos": 0.5 + seed * 0.01,
                "valid_lo_increase_count_mean": 2,
                "valid_lo_decrease_count_mean": 2,
                "mask_observation_count": 4,
                "total_util_lo_mode": 0.35 + seed * 0.04,
                "analysis_normalized_slack": 0.8 - seed * 0.03,
                "analysis_schedulable": True,
                "admission_method": "c_amc_sem",
                "admission_priority_policy": "opa",
                "c_amc_sem_xf": 0.5,
                "amc_rtb_normalized_slack": 0.9 - seed * 0.02,
                "lo_cost_lag1_autocorr_mean": 0.1 + seed * 0.03,
                "hi_cost_lag1_autocorr_mean": 0.2 + seed * 0.02,
                "stress_duty_empirical_mean": 0.1 + seed * 0.02,
                "stress_leader_turnover_rate": 0.1 + seed * 0.01,
                "lo_pressure_leader_turnover_rate": 0.2 + seed * 0.01,
                "mask_turnover_rate": 0.05 + seed * 0.01,
                "budget_competition_index": 0.1 + seed * 0.02,
                "mode_change_rate": seed * 0.01,
                "hi_overrun_event_rate": seed * 0.02,
                "fraction_time_hi_mode": seed * 0.01,
            }
        )
    return rows


def _selection_signature(result: tuple[object, object, object]) -> list[tuple[int, str, str]]:
    selected = result[0]
    return sorted((item.row["candidate_seed"], item.stratum, item.period_family) for item in selected)  # type: ignore[union-attr]


def test_schema_guard_rejects_old_diagnostics() -> None:
    with pytest.raises(ValueError, match="schema_version"):
        validate_diagnostics_rows([{"schema_version": "old", "candidate_seed": 1}])


def test_primary10_is_exactly_five_strata_by_two_period_families() -> None:
    selected, audit, _ = select_primary10(_diagnostic_rows())
    assert len(selected) == 10
    assert len(audit) == 10
    assert {item.stratum for item in selected} == {"S1", "S2", "S3", "S4", "S5"}
    assert {(item.stratum, item.period_family) for item in selected} == {
        (stratum, family)
        for stratum in ("S1", "S2", "S3", "S4", "S5")
        for family in ("semi-harmonic", "log-uniform")
    }
    assert len({item.row["candidate_seed"] for item in selected}) == 10


def test_same_input_is_bitwise_selection_deterministic() -> None:
    rows = _diagnostic_rows()
    first = select_primary10(rows)
    second = select_primary10(rows)
    assert _selection_signature(first) == _selection_signature(second)
    assert first[2] == second[2]



def test_legacy_amc_rtb_slack_does_not_affect_selection() -> None:
    rows = _diagnostic_rows()
    baseline = select_primary10(rows)
    changed = copy.deepcopy(rows)
    for row in changed:
        row["amc_rtb_normalized_slack"] = 1000.0 - int(row["candidate_seed"]) * 100.0
    assert _selection_signature(baseline) == _selection_signature(select_primary10(changed))

def test_performance_fields_do_not_change_selection() -> None:
    baseline = select_primary10(_diagnostic_rows())
    changed = copy.deepcopy(_diagnostic_rows())
    for row in changed:
        row["dqn_validation_reward"] = 10_000 - int(row["candidate_seed"])
        row["heuristic_qos"] = -100.0
        row["viper_retention"] = 0.0
        row["pressure_score"] = 999.0
    assert _selection_signature(baseline) == _selection_signature(select_primary10(changed))


def test_selection_feature_guard_has_no_forbidden_intersection() -> None:
    assert_selection_feature_guard(SELECTION_FEATURES)
    assert not any(name.startswith(("dqn_", "viper_", "heuristic_", "pressure_", "reward_", "train_", "hout_")) for name in SELECTION_FEATURES)


def test_percentile_features_are_deterministic_and_bounded() -> None:
    first = build_percentile_features(_diagnostic_rows())
    second = build_percentile_features(_diagnostic_rows())
    assert first == second
    assert all(0.0 <= value <= 1.0 for vector in first.values() for value in vector.values())


def test_shortage_fails_closed_without_cross_stratum_fill() -> None:
    rows = _diagnostic_rows()[:9]
    with pytest.raises(SelectionShortageError) as exc_info:
        select_primary10(rows)
    assert "shortage" in exc_info.value.report
    assert "missing_slot" in exc_info.value.report



def test_degenerate_structural_metrics_fail_closed() -> None:
    rows = _diagnostic_rows()
    for row in rows:
        row["mask_turnover_rate"] = 0.0
        row["budget_competition_index"] = 0.0
    with pytest.raises(SelectionShortageError) as exc_info:
        select_primary10(rows)
    report = exc_info.value.report
    assert report["schema_version"] == "mc_stratified_dynamic_structural_variation_report_v1"
    assert {item["field"] for item in report["problems"]} == {
        "mask_turnover_rate",
        "budget_competition_index",
    }
