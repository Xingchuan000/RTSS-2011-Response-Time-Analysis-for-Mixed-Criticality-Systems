"""Aggregate the formal10 C-AMC-sem four-baseline HOUT CSVs."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from statistics import mean


METHODS = (
    "c_amc_sem_baseline",
    "noop_agent",
    "static_tuned_budget",
    "random_valid_agent",
    "pressure_threshold_valid_agent",
)
METHOD_ORDER = {method: index for index, method in enumerate(METHODS)}
METRICS = (
    "lo_quality_qos",
    "lo_zero_service_ratio",
    "lo_equiv_jne",
    "tid_ratio",
    "deadline_misses",
    "hi_deadline_misses",
    "lo_deadline_misses",
    "accepted_action_rate",
    "noop_action_rate",
    "selected_invalid_mask_actions",
    "masked_deploy_cap_increase_rate",
)
SUM_FIELDS = (
    "deadline_misses",
    "hi_deadline_misses",
    "lo_deadline_misses",
    "selected_invalid_mask_actions",
    "masked_deploy_cap_increase_count",
)
DEFAULT_TASKSET_SEEDS = "2221,397,861,639,1264,1502,358,185,2535,2829"


def _parse_taskset_seeds(raw_value: str) -> list[int]:
    seeds = [int(part.strip()) for part in raw_value.split(",") if part.strip()]
    if not seeds:
        raise ValueError("--taskset-seeds must not be empty")
    return seeds


def _to_float(value: object) -> float | None:
    if value in (None, "", "None", "NA"):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _values(rows: list[dict[str, str]], field: str) -> list[float]:
    return [value for row in rows if (value := _to_float(row.get(field))) is not None]


def _sum(rows: list[dict[str, str]], field: str) -> float:
    return sum(_to_float(row.get(field)) or 0.0 for row in rows)


def _format_number(value: float | int | None) -> str:
    if value is None:
        return ""
    if isinstance(value, int):
        return str(value)
    return f"{value:.12g}"


def _read_hout(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        rows = [dict(row) for row in reader]
    return rows, fieldnames


def _write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _group_rows(
    rows: list[dict[str, str]],
    *,
    by_taskset: bool,
) -> list[tuple[str | None, str, list[dict[str, str]]]]:
    grouped: dict[tuple[str | None, str], list[dict[str, str]]] = {}
    for row in rows:
        method = str(row.get("method", ""))
        if method not in METHOD_ORDER:
            continue
        taskset = str(row.get("taskset_seed", "")) if by_taskset else None
        grouped.setdefault((taskset, method), []).append(row)
    return sorted(
        grouped.items(),
        key=lambda item: (
            int(item[0][0]) if item[0][0] not in (None, "") else -1,
            METHOD_ORDER[item[0][1]],
        ),
    )


def _aggregate_rows(
    rows: list[dict[str, str]],
    *,
    by_taskset: bool,
) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    for (taskset_seed, method), method_rows in _group_rows(
        rows, by_taskset=by_taskset
    ):
        result: dict[str, object] = {
            "method": method,
            "n_cases": len(method_rows),
        }
        if by_taskset:
            result["taskset_seed"] = taskset_seed
        for field in METRICS:
            values = _values(method_rows, field)
            result[f"{field}_mean"] = _format_number(mean(values) if values else None)
        for field in SUM_FIELDS:
            result[f"{field}_sum"] = _format_number(_sum(method_rows, field))
        output.append(result)
    return output


def _aggregate_fieldnames(*, by_taskset: bool) -> list[str]:
    fields = ["method", "n_cases"]
    if by_taskset:
        fields.insert(0, "taskset_seed")
    fields.extend(f"{field}_mean" for field in METRICS)
    fields.extend(f"{field}_sum" for field in SUM_FIELDS)
    return fields


def _load_hout_rows(
    formal_root: Path,
    taskset_seeds: list[int],
    label: str,
) -> tuple[list[dict[str, str]], list[str]]:
    rows: list[dict[str, str]] = []
    fieldnames: list[str] = []
    for taskset_seed in taskset_seeds:
        path = formal_root / f"r0_s{taskset_seed}" / "baseline_suite" / f"hout_{label}.csv"
        if not path.is_file():
            raise FileNotFoundError(f"Missing formal10 HOUT CSV: {path}")
        source_rows, source_fields = _read_hout(path)
        for field in source_fields:
            if field not in fieldnames:
                fieldnames.append(field)
        rows.extend(
            row for row in source_rows if str(row.get("method", "")) in METHOD_ORDER
        )
    if not rows:
        raise ValueError(f"No formal baseline rows found for horizon {label}")
    return rows, fieldnames


def _aggregate_horizon(
    *,
    formal_root: Path,
    taskset_seeds: list[int],
    output_dir: Path,
    label: str,
) -> None:
    rows, fieldnames = _load_hout_rows(formal_root, taskset_seeds, label)
    _write_csv(
        output_dir / f"baseline_combined_{label}.csv",
        rows,
        fieldnames,
    )
    _write_csv(
        output_dir / f"baseline_method_means_{label}.csv",
        _aggregate_rows(rows, by_taskset=False),
        _aggregate_fieldnames(by_taskset=False),
    )
    _write_csv(
        output_dir / f"baseline_by_taskset_{label}.csv",
        _aggregate_rows(rows, by_taskset=True),
        _aggregate_fieldnames(by_taskset=True),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--formal-root", type=Path, required=True)
    parser.add_argument("--taskset-seeds", default=DEFAULT_TASKSET_SEEDS)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    taskset_seeds = _parse_taskset_seeds(args.taskset_seeds)
    output_dir = args.output_dir or (args.formal_root / "baseline_aggregate")
    for label in ("h2", "h5"):
        _aggregate_horizon(
            formal_root=args.formal_root,
            taskset_seeds=taskset_seeds,
            output_dir=output_dir,
            label=label,
        )
        print(f"WROTE baseline aggregate files for {label}: {output_dir}")


if __name__ == "__main__":
    main()
