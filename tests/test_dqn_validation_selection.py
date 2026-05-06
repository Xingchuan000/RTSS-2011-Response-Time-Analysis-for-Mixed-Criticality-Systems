"""DQN 验证集 best checkpoint 选择规则测试。"""

from __future__ import annotations

from scripts.train_dqn_amc import _is_better_validation_row


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
