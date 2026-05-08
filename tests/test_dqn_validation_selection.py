"""DQN 验证集 best checkpoint 选择规则测试。"""

from __future__ import annotations

from scripts.train_dqn_amc import _is_better_validation_row, _is_pareto_valid_checkpoint


def test_reward_metric_should_prefer_larger_value() -> None:
    """save_best_by=reward 时，应以更大的 reward_mean 作为更优。"""

    best = {"deadline_misses_sum": 0, "reward_mean": 10.0}
    candidate = {"deadline_misses_sum": 0, "reward_mean": 12.0}
    assert _is_better_validation_row(
        candidate_row=candidate,
        best_row=best,
        save_best_by="reward",
    ) is True


def test_lo_cancellations_metric_should_prefer_smaller_value() -> None:
    """save_best_by=lo_cancellations 时，应保持越小越好。"""

    best = {
        "deadline_misses_sum": 0,
        "lo_cancellations_mean": 5.0,
        "mode_changes_mean": 7.0,
        "baseline_mode_changes_mean": 8.0,
    }
    candidate = {
        "deadline_misses_sum": 0,
        "lo_cancellations_mean": 6.0,
        "mode_changes_mean": 7.0,
        "baseline_mode_changes_mean": 8.0,
    }
    assert _is_better_validation_row(
        candidate_row=candidate,
        best_row=best,
        save_best_by="lo_cancellations",
    ) is False


def test_lo_cancellations_rule_requires_mode_changes_not_worse_than_baseline() -> None:
    """save_best_by=lo_cancellations 时，mode_changes 超过 baseline 的候选不可入选。"""

    best = {
        "deadline_misses_sum": 0,
        "lo_cancellations_mean": 4.0,
        "mode_changes_mean": 6.0,
        "baseline_mode_changes_mean": 8.0,
    }
    candidate = {
        "deadline_misses_sum": 0,
        "lo_cancellations_mean": 1.0,
        "mode_changes_mean": 9.0,
        "baseline_mode_changes_mean": 8.0,
    }
    assert _is_better_validation_row(
        candidate_row=candidate,
        best_row=best,
        save_best_by="lo_cancellations",
    ) is False


def test_lo_cancellations_rule_prefers_smaller_lo_after_mode_gate() -> None:
    """save_best_by=lo_cancellations 时，先过 mode gate，再比较 lo_cancellations。"""

    best = {
        "deadline_misses_sum": 0,
        "lo_cancellations_mean": 5.0,
        "mode_changes_mean": 7.0,
        "baseline_mode_changes_mean": 8.0,
    }
    candidate = {
        "deadline_misses_sum": 0,
        "lo_cancellations_mean": 4.0,
        "mode_changes_mean": 8.0,
        "baseline_mode_changes_mean": 8.0,
        "relative_score": 999.0,
    }
    assert _is_better_validation_row(
        candidate_row=candidate,
        best_row=best,
        save_best_by="lo_cancellations",
    ) is True


def test_pareto_relative_score_adds_soft_penalty_when_worse_than_baseline() -> None:
    """温和版 pareto_relative_score 应对劣于 baseline 的维度增加软惩罚。"""

    # 候选虽然 relative_score 更小，但 mode_changes 相对 baseline 变差较多；
    # 在软惩罚后应不优于 best。
    best = {
        "deadline_misses_sum": 0,
        "relative_score": -0.10,
        "relative_delta_mode_changes": -0.05,
        "relative_delta_lo_cancellations": -0.02,
    }
    candidate = {
        "deadline_misses_sum": 0,
        "relative_score": -0.20,
        "relative_delta_mode_changes": 0.20,
        "relative_delta_lo_cancellations": -0.01,
    }
    assert _is_better_validation_row(
        candidate_row=candidate,
        best_row=best,
        save_best_by="pareto_relative_score",
    ) is False


def test_pareto_relative_score_reduces_to_relative_score_when_no_positive_deltas() -> None:
    """当两个 delta 都 <=0 时，pareto_relative_score 应退化为 relative_score 比较。"""

    best = {
        "deadline_misses_sum": 0,
        "relative_score": -0.10,
        "relative_delta_mode_changes": -0.01,
        "relative_delta_lo_cancellations": 0.0,
    }
    candidate = {
        "deadline_misses_sum": 0,
        "relative_score": -0.20,
        "relative_delta_mode_changes": -0.20,
        "relative_delta_lo_cancellations": -0.10,
    }
    assert _is_better_validation_row(
        candidate_row=candidate,
        best_row=best,
        save_best_by="pareto_relative_score",
    ) is True


def test_is_pareto_valid_checkpoint_requires_all_three_conditions() -> None:
    """Pareto-valid 必须同时满足 miss=0、mode 不劣化、lo_cancel 不劣化。"""

    row_ok = {
        "deadline_misses_sum": 0,
        "mode_changes_mean": 20.0,
        "baseline_mode_changes_mean": 20.0,
        "lo_cancellations_mean": 30.0,
        "baseline_lo_cancellations_mean": 31.0,
    }
    row_bad_mode = {
        "deadline_misses_sum": 0,
        "mode_changes_mean": 20.1,
        "baseline_mode_changes_mean": 20.0,
        "lo_cancellations_mean": 30.0,
        "baseline_lo_cancellations_mean": 31.0,
    }
    row_bad_miss = {
        "deadline_misses_sum": 1,
        "mode_changes_mean": 19.0,
        "baseline_mode_changes_mean": 20.0,
        "lo_cancellations_mean": 30.0,
        "baseline_lo_cancellations_mean": 31.0,
    }
    assert _is_pareto_valid_checkpoint(row_ok) is True
    assert _is_pareto_valid_checkpoint(row_bad_mode) is False
    assert _is_pareto_valid_checkpoint(row_bad_miss) is False
