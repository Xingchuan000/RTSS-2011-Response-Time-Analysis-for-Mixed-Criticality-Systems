"""Audit existing formal-seed tasksets with C-AMC-sem and Audsley OPA.

The script intentionally reads the frozen formal input files rather than
regenerating tasksets from a seed.  This makes the audit exact even when the
workload-generator configuration or defaults have changed since training.

Example (PowerShell / Windows path is accepted by Python):

    python scripts/analyze_c_amc_sem_seed_results.py ^
      --formal-root D:\\AMC\\build\\formal ^
      --seeds 313,558,603,715,814,1012,1555,1775,2408,2942 ^
      --output-dir build\\c_amc_sem_baseline_audit

The report contains three distinct checks:
1. legacy AMC-rtb + DM (for comparison with the historical admission gate),
2. C-AMC-sem on the frozen/current priority order,
3. C-AMC-sem + Audsley OPA on the same task parameters.

A C-AMC-sem+OPA PASS only proves that *some* C-AMC-sem-safe priority order
exists.  If that order differs from the trained/frozen order, existing DQN /
VIPER artifacts must not be silently reused under the new order.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
import json
from pathlib import Path
from typing import Any, Iterable

from amc_py.c_amc_sem import analyze_c_amc_sem_task
from amc_py.experiments import evaluate_taskset, resolve_ordering
from amc_py.models import Criticality, Task


DEFAULT_SEEDS = (313, 558, 603, 715, 814, 1012, 1555, 1775, 2408, 2942)


def _parse_seeds(raw: str) -> tuple[int, ...]:
    seeds: list[int] = []
    for token in raw.replace(";", ",").split(","):
        token = token.strip()
        if not token:
            continue
        seeds.append(int(token))
    result = tuple(dict.fromkeys(seeds))
    if not result:
        raise ValueError("--seeds must contain at least one integer seed")
    return result


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _seed_dir(formal_root: Path, seed: int, folder_template: str) -> Path:
    candidate = formal_root / folder_template.format(seed=seed)
    if candidate.is_dir():
        return candidate
    # Small convenience for renamed result folders: accept a unique s<seed>_* match.
    matches = sorted(path for path in formal_root.glob(f"s{seed}_*") if path.is_dir())
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise FileNotFoundError(f"formal result directory not found for seed {seed}: {candidate}")
    raise FileNotFoundError(
        f"multiple formal result directories match seed {seed}; use --folder-template: {matches}"
    )


def _load_frozen_tasks(seed_dir: Path) -> tuple[list[Task], list[str]]:
    path = seed_dir / "request" / "inputs" / "formal_inputs" / "code_taskset_canonical.json"
    data = _load_json(path)
    rows = data.get("ordered_tasks")
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"missing ordered_tasks in {path}")

    tasks: list[Task] = []
    declared_order: list[str] = []
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError(f"invalid task row in {path}")
        criticality = Criticality(str(row["criticality"]))
        task = Task(
            name=str(row["name"]),
            period=int(row["period"]),
            deadline=int(row["deadline"]),
            c_lo=int(row["code_c_lo"]),
            c_hi=int(row["code_c_hi"]),
            criticality=criticality,
        )
        tasks.append(task)
        declared_order.append(task.name)

    priority_order = data.get("priority_order")
    if isinstance(priority_order, list):
        normalized = [str(name) for name in priority_order]
        if normalized != declared_order:
            raise ValueError(f"ordered_tasks and priority_order disagree in {path}")
    return tasks, declared_order


def _runtime_field(fields: dict[str, Any], name: str) -> Any:
    row = fields.get(name)
    if not isinstance(row, dict) or "value" not in row:
        raise ValueError(f"effective_runtime_config missing field {name}")
    return row["value"]


def _load_runtime_csem_settings(seed_dir: Path) -> tuple[float, bool]:
    path = seed_dir / "request" / "inputs" / "formal_inputs" / "effective_runtime_config.json"
    data = _load_json(path)
    fields = data.get("fields")
    if not isinstance(fields, dict):
        raise ValueError(f"missing fields in {path}")
    xf = float(_runtime_field(fields, "c_amc_sem_lo_degradation_ratio"))
    primary_on_switch = bool(_runtime_field(fields, "c_amc_sem_primary_on_switch_time"))
    return xf, primary_on_switch


def _task_details(ordered_tasks: list[Task], xf: float) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for idx, task in enumerate(ordered_tasks):
        analysis = analyze_c_amc_sem_task(
            task,
            ordered_tasks[:idx],
            lo_degradation_ratio=xf,
        )
        row = asdict(analysis)
        row.update(
            {
                "priority_index": idx,
                "criticality": task.criticality.value,
                "period": task.period,
                "deadline": task.deadline,
                "c_lo": task.c_lo,
                "c_hi_code": task.c_hi,
            }
        )
        rows.append(row)
        if not analysis.schedulable:
            break
    return rows


def _min_slack(tasks: Iterable[Task], response_times: dict[str, int]) -> int | None:
    values = [task.deadline - response_times[task.name] for task in tasks if task.name in response_times]
    return min(values) if values else None


def _audit_seed(
    formal_root: Path,
    seed: int,
    *,
    folder_template: str,
    xf_override: float | None,
) -> dict[str, Any]:
    seed_dir = _seed_dir(formal_root, seed, folder_template)
    frozen_tasks, frozen_order = _load_frozen_tasks(seed_dir)
    runtime_xf, primary_on_switch = _load_runtime_csem_settings(seed_dir)
    xf = runtime_xf if xf_override is None else float(xf_override)

    if not primary_on_switch:
        raise ValueError(
            f"seed {seed}: frozen runtime has c_amc_sem_primary_on_switch_time=False; "
            "the paper Eq.(6)/(16)/(17) binding used by this audit assumes True"
        )

    legacy = evaluate_taskset(frozen_tasks, method="amc_rtb", priority_policy="dm")
    fixed = evaluate_taskset(
        frozen_tasks,
        method="c_amc_sem",
        priority_policy="dm",  # frozen task files are DM-ordered in the current experiment
        c_amc_sem_xf=xf,
    )

    # The fixed file order is authoritative; also calculate directly from that
    # order in case future artifacts were not produced by DM.
    from amc_py.c_amc_sem import c_amc_sem_sched_test

    fixed_exact = c_amc_sem_sched_test(frozen_tasks, lo_degradation_ratio=xf)

    opa = evaluate_taskset(
        frozen_tasks,
        method="c_amc_sem",
        priority_policy="opa",
        c_amc_sem_xf=xf,
    )
    opa_order: list[str] = []
    opa_details: list[dict[str, Any]] = []
    if opa.schedulable:
        ordered_opa = resolve_ordering(
            frozen_tasks,
            priority_policy="opa",
            method="c_amc_sem",
            c_amc_sem_xf=xf,
        )
        opa_order = [task.name for task in ordered_opa]
        opa_details = _task_details(ordered_opa, xf)

    fixed_details = _task_details(frozen_tasks, xf)
    first_failure = next((row for row in fixed_details if not row["schedulable"]), None)

    return {
        "seed": seed,
        "seed_dir": str(seed_dir),
        "xf": xf,
        "runtime_xf": runtime_xf,
        "primary_on_switch_time": primary_on_switch,
        "frozen_priority_order": frozen_order,
        "legacy_amc_rtb_dm": {
            "schedulable": legacy.schedulable,
            "min_slack": _min_slack(frozen_tasks, legacy.response_times),
            "details": legacy.details,
        },
        "c_amc_sem_frozen_order": {
            "schedulable": fixed_exact.schedulable,
            "min_slack": _min_slack(frozen_tasks, fixed_exact.response_times),
            "details": fixed_exact.details,
            "task_details_until_first_failure": fixed_details,
        },
        "c_amc_sem_dm_crosscheck": {
            "schedulable": fixed.schedulable,
            "details": fixed.details,
        },
        "c_amc_sem_opa": {
            "schedulable": opa.schedulable,
            "min_slack": _min_slack(
                [next(task for task in frozen_tasks if task.name == name) for name in opa_order],
                opa.response_times,
            )
            if opa_order
            else None,
            "details": opa.details,
            "priority_order": opa_order,
            "order_changed_from_frozen": bool(opa_order and opa_order != frozen_order),
            "task_details": opa_details,
        },
        "first_frozen_order_failure": first_failure,
    }


def _write_summary_csv(path: Path, reports: list[dict[str, Any]]) -> None:
    fields = [
        "seed",
        "xf",
        "legacy_amc_rtb_dm_schedulable",
        "c_amc_sem_frozen_order_schedulable",
        "c_amc_sem_opa_schedulable",
        "opa_order_changed_from_frozen",
        "first_failure_task",
        "first_failure_case",
        "first_failure_switch_time",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for report in reports:
            failure = report.get("first_frozen_order_failure") or {}
            writer.writerow(
                {
                    "seed": report["seed"],
                    "xf": report["xf"],
                    "legacy_amc_rtb_dm_schedulable": report["legacy_amc_rtb_dm"]["schedulable"],
                    "c_amc_sem_frozen_order_schedulable": report["c_amc_sem_frozen_order"]["schedulable"],
                    "c_amc_sem_opa_schedulable": report["c_amc_sem_opa"]["schedulable"],
                    "opa_order_changed_from_frozen": report["c_amc_sem_opa"]["order_changed_from_frozen"],
                    "first_failure_task": failure.get("task_name", ""),
                    "first_failure_case": failure.get("worst_case", ""),
                    "first_failure_switch_time": failure.get("worst_switch_time", ""),
                }
            )


def _write_markdown(path: Path, reports: list[dict[str, Any]]) -> None:
    lines = [
        "# C-AMC-sem baseline schedulability audit",
        "",
        "The frozen priority column analyzes the exact order stored in each formal artifact. "
        "The OPA column asks whether the same task parameters admit *some* C-AMC-sem-safe "
        "Audsley priority assignment.",
        "",
        "| seed | legacy AMC-rtb+DM | C-AMC-sem frozen order | C-AMC-sem+OPA | OPA order changed | first frozen-order failure |",
        "|---:|:---:|:---:|:---:|:---:|---|",
    ]
    for report in reports:
        failure = report.get("first_frozen_order_failure") or {}
        failure_text = "-"
        if failure:
            failure_text = (
                f"{failure.get('task_name')} / {failure.get('worst_case')} / "
                f"s={failure.get('worst_switch_time')}"
            )
        lines.append(
            "| {seed} | {legacy} | {fixed} | {opa} | {changed} | {failure} |".format(
                seed=report["seed"],
                legacy="PASS" if report["legacy_amc_rtb_dm"]["schedulable"] else "FAIL",
                fixed="PASS" if report["c_amc_sem_frozen_order"]["schedulable"] else "FAIL",
                opa="PASS" if report["c_amc_sem_opa"]["schedulable"] else "FAIL",
                changed="YES" if report["c_amc_sem_opa"]["order_changed_from_frozen"] else "NO",
                failure=failure_text,
            )
        )

    lines.extend(["", "## OPA priority orders", ""])
    for report in reports:
        order = report["c_amc_sem_opa"]["priority_order"]
        lines.append(f"- s{report['seed']}: " + (" > ".join(order) if order else "OPA FAILED"))

    lines.extend(
        [
            "",
            "## Interpretation guard",
            "",
            "An OPA order different from the frozen/trained order is a new baseline configuration. "
            "Do not reuse a DQN/VIPER policy trained with the old priority-indexed observation/action "
            "layout without rebuilding/retraining and re-running formal verification.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--formal-root", type=Path, required=True)
    parser.add_argument(
        "--seeds",
        default=",".join(str(seed) for seed in DEFAULT_SEEDS),
        help="comma-separated taskset seeds",
    )
    parser.add_argument(
        "--folder-template",
        default="s{seed}_best_overall_v9_1_e2e",
        help="relative formal result folder template; must contain {seed}",
    )
    parser.add_argument(
        "--xf",
        type=float,
        default=None,
        help="optional XF override; default reads each frozen effective_runtime_config.json",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if "{seed}" not in args.folder_template:
        raise ValueError("--folder-template must contain {seed}")

    reports = [
        _audit_seed(
            args.formal_root,
            seed,
            folder_template=args.folder_template,
            xf_override=args.xf,
        )
        for seed in _parse_seeds(args.seeds)
    ]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "c_amc_sem_seed_audit.json").write_text(
        json.dumps({"schema_version": "c_amc_sem_seed_audit_v1", "seeds": reports}, indent=2),
        encoding="utf-8",
    )
    _write_summary_csv(args.output_dir / "c_amc_sem_seed_audit.csv", reports)
    _write_markdown(args.output_dir / "c_amc_sem_seed_audit.md", reports)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
