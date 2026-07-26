from __future__ import annotations

import csv
import json
from pathlib import Path

from scripts.check_qamc_pilot_readiness import (
    EXIT_HI_SAFETY_FAILURE,
    EXIT_HOUT_INCOMPLETE,
    EXIT_METRIC_MISMATCH,
    EXIT_OK,
    READY,
    check_readiness,
)


LOSS_VALUES = {
    "qamc_loss_released_lo_jobs": 4,
    "qamc_loss_completed_positive_quality_jobs": 2,
    "qamc_loss_overrun_stopped_zero_quality_jobs": 1,
    "qamc_loss_deadline_lost_zero_quality_jobs": 0,
    "qamc_loss_min_threshold_fallback_zero_quality_jobs": 0,
    "qamc_loss_hi_mode_discard_zero_quality_jobs": 1,
    "qamc_loss_other_zero_quality_jobs": 0,
}


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _validation_row() -> dict[str, object]:
    row: dict[str, object] = {
        "lo_quality_qos_mean": 0.5,
        "qamc_normalized_quality_qos_mean": 0.5,
        "qamc_release_count_mean": 4,
        "qamc_overrun_stop_count_mean": 1,
        "qamc_quality_transition_count_mean": 1,
        "qamc_dqn_budget_update_event_count_mean": 2,
        "hi_deadline_misses_sum": 0,
    }
    row.update({f"{key}_mean": value for key, value in LOSS_VALUES.items()})
    return row


def _hout_row(method: str) -> dict[str, object]:
    row: dict[str, object] = {
        "seed": 200,
        "method": method,
        "dqn_runtime_semantics": "Q_AMC",
        "lo_quality_qos": 0.5,
        "qamc_normalized_quality_qos": 0.5,
        "qamc_overrun_stop_count": 1,
        "qamc_quality_transition_count": 1,
        "qamc_dqn_budget_update_event_count": (
            1 if method == "q_amc_dqn_budget_overlay" else 0
        ),
        "hi_deadline_misses": 0,
    }
    row.update(LOSS_VALUES)
    return row


def _ready_outputs(tmp_path: Path) -> tuple[Path, Path]:
    train = tmp_path / "train" / "r0_s185"
    hout = tmp_path / "hout"
    train.mkdir(parents=True)
    (train / "config.json").write_text(
        json.dumps(
            {
                "fixed_taskset_seed": 185,
                "runtime_config": {"semantics": "Q_AMC"},
            }
        ),
        encoding="utf-8",
    )
    (train / "checkpoints").mkdir()
    (train / "checkpoints" / "model_episode_0001.pt").touch()
    (train / "model_best.pt").touch()
    _write_csv(train / "validation_metrics.csv", [_validation_row()])
    _write_csv(
        hout / "hout.csv",
        [_hout_row("q_amc_native"), _hout_row("q_amc_dqn_budget_overlay")],
    )
    return train, hout


def test_ready_pilot_outputs_pass_all_checks(tmp_path: Path) -> None:
    train, hout = _ready_outputs(tmp_path)

    summary, exit_code = check_readiness(train, hout)

    assert exit_code == EXIT_OK
    assert summary["status"] == READY
    assert summary["taskset_seeds"] == [185]
    assert all(summary["checks"].values())


def test_qos_mismatch_uses_metric_mismatch_exit_code(tmp_path: Path) -> None:
    train, hout = _ready_outputs(tmp_path)
    row = _validation_row()
    row["qamc_normalized_quality_qos_mean"] = 0.75
    _write_csv(train / "validation_metrics.csv", [row])

    summary, exit_code = check_readiness(train, hout)

    assert exit_code == EXIT_METRIC_MISMATCH
    assert summary["failure_code"] == "QAMC_GENERIC_QOS_MISMATCH"


def test_hi_deadline_miss_uses_safety_exit_code(tmp_path: Path) -> None:
    train, hout = _ready_outputs(tmp_path)
    row = _validation_row()
    row["hi_deadline_misses_sum"] = 1
    _write_csv(train / "validation_metrics.csv", [row])

    summary, exit_code = check_readiness(train, hout)

    assert exit_code == EXIT_HI_SAFETY_FAILURE
    assert summary["failure_code"] == "HI_DEADLINE_MISS"


def test_missing_qamc_overlay_uses_hout_exit_code(tmp_path: Path) -> None:
    train, hout = _ready_outputs(tmp_path)
    _write_csv(hout / "hout.csv", [_hout_row("q_amc_native")])

    summary, exit_code = check_readiness(train, hout)

    assert exit_code == EXIT_HOUT_INCOMPLETE
    assert summary["failure_code"] == "HOUT_INCOMPLETE"
