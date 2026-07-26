"""Fail-closed readiness gate for real-seed q-AMC pilot outputs."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Iterable, Mapping


EXIT_OK = 0
EXIT_MISSING_ARTIFACT = 10
EXIT_METRIC_MISMATCH = 20
EXIT_NO_QAMC_EVENT = 21
EXIT_NO_DQN_ACTION = 22
EXIT_HI_SAFETY_FAILURE = 30
EXIT_HOUT_INCOMPLETE = 40

SCHEMA_VERSION = "qamc_pilot_readiness_v1"
READY = "READY_FOR_FULL_TRAINING"
NOT_READY = "NOT_READY"

QAMC_NATIVE_METHOD = "q_amc_native"
QAMC_DQN_METHOD = "q_amc_dqn_budget_overlay"

LOSS_COMPONENT_FIELDS = (
    "qamc_loss_completed_positive_quality_jobs",
    "qamc_loss_overrun_stopped_zero_quality_jobs",
    "qamc_loss_deadline_lost_zero_quality_jobs",
    "qamc_loss_min_threshold_fallback_zero_quality_jobs",
    "qamc_loss_hi_mode_discard_zero_quality_jobs",
    "qamc_loss_other_zero_quality_jobs",
)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def _first_value(
    row: Mapping[str, object],
    *names: str,
) -> object | None:
    for name in names:
        value = row.get(name)
        if value not in (None, ""):
            return value
    return None


def _number(row: Mapping[str, object], *names: str) -> float:
    value = _first_value(row, *names)
    if value is None:
        raise KeyError("|".join(names))
    return float(value)


def _metric(row: Mapping[str, object], name: str) -> float:
    return _number(row, f"{name}_mean", name)


def _discover_train_runs(train_output: Path) -> list[Path]:
    if (train_output / "config.json").is_file():
        return [train_output]
    return sorted(
        path.parent
        for path in train_output.rglob("config.json")
        if (path.parent / "validation_metrics.csv").is_file()
    )


def _discover_hout_csvs(hout_output: Path) -> list[Path]:
    if hout_output.is_file() and hout_output.suffix.lower() == ".csv":
        return [hout_output]
    if not hout_output.is_dir():
        return []
    return sorted(hout_output.rglob("*.csv"))


def _loss_is_conservative(row: Mapping[str, object]) -> bool:
    released = _metric(row, "qamc_loss_released_lo_jobs")
    classified = sum(_metric(row, name) for name in LOSS_COMPONENT_FIELDS)
    return math.isclose(classified, released, rel_tol=0.0, abs_tol=1e-12)


def _qos_matches(row: Mapping[str, object]) -> bool:
    generic = _number(row, "lo_quality_qos_mean", "lo_quality_qos")
    qamc = _number(
        row,
        "qamc_normalized_quality_qos_mean",
        "qamc_normalized_quality_qos",
    )
    return math.isclose(generic, qamc, rel_tol=0.0, abs_tol=1e-12)


def _taskset_seed(config: Mapping[str, object]) -> int | None:
    value = _first_value(config, "fixed_taskset_seed", "effective_taskset_seed")
    return None if value is None else int(value)


def _runtime_semantics(config: Mapping[str, object]) -> str:
    runtime_config = config.get("runtime_config")
    if isinstance(runtime_config, Mapping):
        value = _first_value(runtime_config, "semantics", "runtime_semantics")
        if value is not None:
            return str(value)
    value = _first_value(config, "runtime_semantics", "dqn_runtime_semantics")
    return "" if value is None else str(value)


def _all_have_columns(
    rows: Iterable[Mapping[str, object]],
    aliases: tuple[tuple[str, ...], ...],
) -> bool:
    return all(
        all(
            any(name in row for name in column_aliases)
            for column_aliases in aliases
        )
        for row in rows
    )


def check_readiness(train_output: Path, hout_output: Path) -> tuple[dict[str, object], int]:
    checks = {
        "checkpoint_present": False,
        "best_checkpoint_present": False,
        "validation_completed": False,
        "qamc_release_count_positive": False,
        "generic_qos_matches_qamc_qos": False,
        "loss_conservation": False,
        "qamc_events_nonzero": False,
        "dqn_updates_nonzero": False,
        "hi_deadline_miss_zero": False,
        "hout_complete": False,
    }
    summary: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "status": NOT_READY,
        "runtime_semantics": "Q_AMC",
        "taskset_seeds": [],
        "checks": checks,
    }

    train_runs = _discover_train_runs(train_output) if train_output.exists() else []
    if not train_runs:
        summary["failure_code"] = "MISSING_TRAIN_OUTPUT"
        return summary, EXIT_MISSING_ARTIFACT

    validation_rows: list[dict[str, str]] = []
    taskset_seeds: list[int] = []
    semantics_are_qamc = True
    checkpoints_present = True
    best_checkpoints_present = True
    validation_completed = True
    for run_dir in train_runs:
        try:
            config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError):
            summary["failure_code"] = "MISSING_OR_INVALID_TRAIN_CONFIG"
            return summary, EXIT_MISSING_ARTIFACT
        seed = _taskset_seed(config)
        if seed is None:
            summary["failure_code"] = "MISSING_TASKSET_SEED"
            return summary, EXIT_MISSING_ARTIFACT
        taskset_seeds.append(seed)
        semantics_are_qamc &= _runtime_semantics(config) == "Q_AMC"

        checkpoint_files = list((run_dir / "checkpoints").glob("*.pt"))
        checkpoints_present &= bool(checkpoint_files)
        best_checkpoints_present &= bool(list(run_dir.glob("model_best*.pt")))
        validation_path = run_dir / "validation_metrics.csv"
        rows = _read_csv(validation_path) if validation_path.is_file() else []
        validation_completed &= bool(rows)
        validation_rows.extend(rows)

    summary["taskset_seeds"] = sorted(set(taskset_seeds))
    checks["checkpoint_present"] = checkpoints_present
    checks["best_checkpoint_present"] = best_checkpoints_present
    checks["validation_completed"] = validation_completed
    if not (
        semantics_are_qamc
        and checkpoints_present
        and best_checkpoints_present
        and validation_completed
    ):
        summary["failure_code"] = (
            "RUNTIME_SEMANTICS_NOT_Q_AMC"
            if not semantics_are_qamc
            else "MISSING_TRAIN_ARTIFACT"
        )
        return summary, EXIT_MISSING_ARTIFACT

    try:
        checks["qamc_release_count_positive"] = all(
            _metric(row, "qamc_release_count") > 0.0 for row in validation_rows
        )
        checks["generic_qos_matches_qamc_qos"] = all(
            _qos_matches(row) for row in validation_rows
        )
        checks["loss_conservation"] = all(
            _loss_is_conservative(row) for row in validation_rows
        )
        qamc_overrun_stops = sum(
            _metric(row, "qamc_overrun_stop_count") for row in validation_rows
        )
        qamc_quality_transitions = sum(
            _metric(row, "qamc_quality_transition_count")
            for row in validation_rows
        )
        dqn_budget_updates = sum(
            _metric(row, "qamc_dqn_budget_update_event_count")
            for row in validation_rows
        )
        hi_deadline_misses = sum(
            _number(
                row,
                "hi_deadline_misses_sum",
                "hi_deadline_miss_count",
                "hi_deadline_misses",
            )
            for row in validation_rows
        )
    except (KeyError, TypeError, ValueError):
        summary["failure_code"] = "MISSING_TRAIN_METRIC"
        return summary, EXIT_MISSING_ARTIFACT

    checks["qamc_events_nonzero"] = (
        qamc_overrun_stops > 0.0 and qamc_quality_transitions > 0.0
    )
    checks["dqn_updates_nonzero"] = dqn_budget_updates > 0.0
    checks["hi_deadline_miss_zero"] = math.isclose(
        hi_deadline_misses, 0.0, rel_tol=0.0, abs_tol=0.0
    )
    summary["metrics"] = {
        "qamc_overrun_stop_count": qamc_overrun_stops,
        "qamc_quality_transition_count": qamc_quality_transitions,
        "dqn_budget_update_count": dqn_budget_updates,
        "hi_deadline_miss_count": hi_deadline_misses,
    }

    if not checks["qamc_release_count_positive"]:
        summary["failure_code"] = "QAMC_RELEASE_COUNT_NOT_POSITIVE"
        return summary, EXIT_METRIC_MISMATCH
    if not checks["generic_qos_matches_qamc_qos"]:
        summary["failure_code"] = "QAMC_GENERIC_QOS_MISMATCH"
        return summary, EXIT_METRIC_MISMATCH
    if not checks["loss_conservation"]:
        summary["failure_code"] = "QAMC_LOSS_NOT_CONSERVATIVE"
        return summary, EXIT_METRIC_MISMATCH
    if not checks["qamc_events_nonzero"]:
        summary["failure_code"] = "QAMC_EVENTS_NOT_OBSERVED"
        return summary, EXIT_NO_QAMC_EVENT
    if not checks["dqn_updates_nonzero"]:
        summary["failure_code"] = "DQN_BUDGET_UPDATE_NOT_OBSERVED"
        return summary, EXIT_NO_DQN_ACTION
    if not checks["hi_deadline_miss_zero"]:
        summary["failure_code"] = "HI_DEADLINE_MISS"
        return summary, EXIT_HI_SAFETY_FAILURE

    hout_csvs = _discover_hout_csvs(hout_output)
    hout_rows = [row for path in hout_csvs for row in _read_csv(path)]
    required_aliases = (
        ("seed",),
        ("method",),
        ("runtime_semantics", "dqn_runtime_semantics"),
        ("lo_quality_qos",),
        ("qamc_normalized_quality_qos",),
        ("qamc_overrun_stop_count",),
        ("qamc_quality_transition_count",),
        ("dqn_budget_update_count", "qamc_dqn_budget_update_event_count"),
        ("hi_deadline_miss_count", "hi_deadline_misses"),
    )
    methods = {str(row.get("method", "")) for row in hout_rows}
    hout_complete = (
        bool(hout_rows)
        and QAMC_NATIVE_METHOD in methods
        and QAMC_DQN_METHOD in methods
        and _all_have_columns(hout_rows, required_aliases)
    )
    if hout_complete:
        try:
            qamc_hout_rows = [
                row
                for row in hout_rows
                if str(
                    _first_value(row, "runtime_semantics", "dqn_runtime_semantics")
                )
                == "Q_AMC"
                and str(row.get("method", ""))
                in {QAMC_NATIVE_METHOD, QAMC_DQN_METHOD}
            ]
            hout_complete = (
                bool(qamc_hout_rows)
                and all(_qos_matches(row) for row in qamc_hout_rows)
                and all(_loss_is_conservative(row) for row in qamc_hout_rows)
            )
        except (KeyError, TypeError, ValueError):
            hout_complete = False
    checks["hout_complete"] = hout_complete
    if not hout_complete:
        summary["failure_code"] = "HOUT_INCOMPLETE"
        return summary, EXIT_HOUT_INCOMPLETE

    summary["status"] = READY
    return summary, EXIT_OK


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-output", required=True, type=Path)
    parser.add_argument("--hout-output", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    summary, exit_code = check_readiness(args.train_output, args.hout_output)
    summary_path = args.train_output / "pilot_readiness_summary.json"
    if args.train_output.is_file():
        summary_path = args.train_output.parent / "pilot_readiness_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(summary["status"])
    print(summary_path)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
