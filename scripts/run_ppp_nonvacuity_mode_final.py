from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import traceback
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from typing import Any, TextIO

# Support direct execution after copying to <repo>/scripts/.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from nonvacuity_lab.canonical import IGNORED_TREE_PARTS, tree_hash
from nonvacuity_lab.config_io import (
    validate_config_kind,
    verify_config_hash,
    write_resolved_campaign,
)
from nonvacuity_lab.doctor.runner import run_doctor
from nonvacuity_lab.preflight import audit_v2_campaign_path
from nonvacuity_lab.v2_runner import run_v2_campaign


MODE_MUTATIONS: dict[str, tuple[str, ...]] = {
    "S185Core": (
        "P0_s185_compact",
        "A1_s185_dangerous_top1_masked",
        "B1_s185_mask_bypass",
    ),
    "PositiveControls": (
        "P0_s185_compact",
        "P1_s1264_compact",
        "P2_s1264_balanced",
        "P3_s397_balanced",
    ),
    "S397Pair": (
        "P3_s397_balanced",
        "A2_s397_dangerous_top1_masked",
        "B5_s397_mask_bypass",
    ),
    "B234": (
        "P1_s1264_compact",
        "P2_s1264_balanced",
        "B2_s1264_no_first_valid",
        "B3_s1264_all_invalid_force_top1",
        "B4_s1264_guard_ablation",
    ),
    "Integrity": (
        "F1_tree_tamper",
        "F2_cross_seed",
        "F3_cross_variant",
    ),
    "IntegrityAll": (
        "F1_tree_tamper",
        "F2_cross_seed",
        "F3_cross_variant",
        "F4_priority_tamper",
        "F5_witness_tamper",
        "F6_delete_obligation_artifact",
        "F7_source_binding_tamper",
    ),
    "ModelE1": ("E1_deadline_cleanup_remove",),
    "ModelEAll": (
        "E1_deadline_cleanup_remove",
        "E2_hi_job_truncate",
        "E3_event_order",
        "E4_controller_overhead",
        "E5_nonquiescent_recovery",
        "E6_unstable_demand_reads",
    ),
    "C123": (
        "C1_action_ratio",
        "C2_rounding",
        "C3_retroactive_release",
    ),
    "D1": ("D1_dynamic_envelope_gradient",),
    "PaperMinimum": (
        "P0_s185_compact",
        "P1_s1264_compact",
        "P2_s1264_balanced",
        "P3_s397_balanced",
        "A1_s185_dangerous_top1_masked",
        "B1_s185_mask_bypass",
        "B2_s1264_no_first_valid",
        "B3_s1264_all_invalid_force_top1",
        "B4_s1264_guard_ablation",
        "F1_tree_tamper",
        "F2_cross_seed",
        "F3_cross_variant",
        "E1_deadline_cleanup_remove",
        "A2_s397_dangerous_top1_masked",
        "B5_s397_mask_bypass",
        "D1_dynamic_envelope_gradient",
    ),
}


class Tee(TextIO):
    def __init__(self, *streams: TextIO) -> None:
        self._streams = streams

    def write(self, text: str) -> int:
        for stream in self._streams:
            stream.write(text)
            stream.flush()
        return len(text)

    def flush(self) -> None:
        for stream in self._streams:
            stream.flush()


class SetupFailure(RuntimeError):
    pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Deterministically prepare, doctor and run any PPP non-vacuity mode "
            "without PowerShell native-command wrapping."
        )
    )
    parser.add_argument("--mode", choices=tuple(MODE_MUTATIONS), required=True)
    parser.add_argument("--project-root", type=Path, default=Path(r"D:\AMC"))
    parser.add_argument(
        "--output-tag",
        default="formalv1_csem_t10_s1550_1599_tr8e6_v2e7_h2_h5",
    )
    parser.add_argument("--experiment-data-root", type=Path)
    parser.add_argument("--timeout-seconds", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Stop after preflight and two stable PASS doctor runs.",
    )
    parser.add_argument("--list-modes", action="store_true")
    return parser.parse_args()


def load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise SetupFailure(f"JSON object required: {path}")
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def fail_checks(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [row for row in rows if row.get("status") == "FAIL"]


def result_exit_code(result: dict[str, Any]) -> int:
    status = str(result.get("status", ""))
    if status == "COMPLETED":
        return 0
    if status == "FAILED_NOT_ACTIVATED":
        return 3
    if status == "COMPLETED_WITH_INVALID_RESULTS":
        return 4
    if status in {"PASS", "DISABLED"}:
        return 0
    return 2


def main() -> int:
    args = parse_args()
    if args.list_modes:
        print(json.dumps(MODE_MUTATIONS, ensure_ascii=False, indent=2))
        return 0

    project_root = args.project_root.resolve()
    os.chdir(project_root)
    os.environ.setdefault("PYTHONHASHSEED", "0")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

    data_root = (
        args.experiment_data_root.resolve()
        if args.experiment_data_root is not None
        else project_root
        / "experiment_data"
        / f"ppp_nonvacuity_{args.output_tag}"
    )
    config_root = data_root / "config"
    mode = args.mode
    campaign_id = f"ppp_nonvacuity_full_v2_{mode.lower()}"
    base_config = config_root / "ppp_full.resolved.json"
    mode_config = config_root / f"ppp_full.{mode}.json"
    preflight_path = config_root / f"preflight_{mode}.json"
    doctor_path = config_root / f"doctor_{mode}.json"
    doctor_stability_path = config_root / f"doctor_{mode}.stability.json"
    summary_path = config_root / f"orchestration_{mode}.json"
    console_path = config_root / f"orchestration_{mode}.log"
    campaign_dir = data_root / "results" / campaign_id

    config_root.mkdir(parents=True, exist_ok=True)
    with console_path.open("w", encoding="utf-8") as log_stream:
        tee_out = Tee(sys.__stdout__, log_stream)
        tee_err = Tee(sys.__stderr__, log_stream)
        with redirect_stdout(tee_out), redirect_stderr(tee_err):
            try:
                print(f"=== PPP mode: {mode} ===")
                print(f"Project root: {project_root}")
                print(f"Experiment data: {data_root}")

                if not base_config.is_file():
                    raise SetupFailure(
                        f"Resolved base config is missing: {base_config}. "
                        "Run phase-01 preparation and target binding first."
                    )
                if "experiment_data" not in IGNORED_TREE_PARTS:
                    raise SetupFailure(
                        "experiment_data is not excluded from source hashing in "
                        "nonvacuity_lab/canonical.py"
                    )

                base = load_object(base_config)
                validate_config_kind(base)
                verify_config_hash(base)

                config = copy.deepcopy(base)
                config["campaign_id"] = campaign_id
                selected = set(MODE_MUTATIONS[mode])
                known = {
                    str(item.get("mutation_id"))
                    for item in config.get("mutations", [])
                    if isinstance(item, dict)
                }
                unknown = sorted(selected - known)
                if unknown:
                    raise SetupFailure(f"Unknown mutation IDs in mode {mode}: {unknown}")

                for mutation in config.get("mutations", []):
                    if isinstance(mutation, dict):
                        mutation["enabled"] = str(mutation.get("mutation_id")) in selected
                config["enabled"] = True
                config["source_binding"] = {
                    "clean_source_root": str(project_root),
                    "clean_source_root_sha256": tree_hash(project_root),
                }

                for stale in (
                    mode_config,
                    preflight_path,
                    doctor_path,
                    doctor_stability_path,
                    summary_path,
                ):
                    stale.unlink(missing_ok=True)

                write_resolved_campaign(mode_config, config)
                sealed = load_object(mode_config)
                validate_config_kind(sealed)
                verify_config_hash(sealed)

                expected_source = str(
                    sealed.get("source_binding", {}).get("clean_source_root_sha256", "")
                )
                actual_source = tree_hash(project_root)
                if expected_source != actual_source:
                    raise SetupFailure(
                        "Source binding changed immediately after mode-config sealing: "
                        f"expected={expected_source}, actual={actual_source}"
                    )
                print(f"Mode config sealed: {mode_config}")
                print(f"Enabled mutations: {', '.join(MODE_MUTATIONS[mode])}")
                print(f"Source binding: {actual_source}")

                preflight = audit_v2_campaign_path(mode_config)
                write_json(preflight_path, preflight)
                if preflight.get("status") != "PASS":
                    raise SetupFailure(
                        "Preflight failed:\n"
                        + json.dumps(preflight, ensure_ascii=False, indent=2)
                    )
                print("Preflight: PASS")

                doctor = run_doctor(mode_config, doctor_path).to_dict()
                doctor_failures = fail_checks(doctor.get("checks", []))
                if doctor.get("overall_status") != "PASS":
                    raise SetupFailure(
                        "Doctor failed:\n"
                        + json.dumps(doctor_failures or doctor, ensure_ascii=False, indent=2)
                    )
                print("Doctor: PASS")

                doctor2 = run_doctor(mode_config, doctor_stability_path).to_dict()
                doctor2_failures = fail_checks(doctor2.get("checks", []))
                if doctor2.get("overall_status") != "PASS":
                    raise SetupFailure(
                        "Doctor stability check failed:\n"
                        + json.dumps(doctor2_failures or doctor2, ensure_ascii=False, indent=2)
                    )
                if doctor.get("config_sha256") != sealed.get("config_sha256"):
                    raise SetupFailure("Doctor receipt does not bind the current mode config")
                if doctor2.get("config_sha256") != sealed.get("config_sha256"):
                    raise SetupFailure(
                        "Doctor stability receipt does not bind the current mode config"
                    )
                print("Doctor stability: PASS")

                ready_summary: dict[str, Any] = {
                    "schema_version": "ppp_mode_orchestration_v1",
                    "status": "READY",
                    "mode": mode,
                    "campaign_id": campaign_id,
                    "mode_config": str(mode_config),
                    "preflight": str(preflight_path),
                    "doctor": str(doctor_path),
                    "doctor_stability": str(doctor_stability_path),
                    "campaign_dir": str(campaign_dir),
                    "enabled_mutations": list(MODE_MUTATIONS[mode]),
                    "source_sha256": actual_source,
                }
                write_json(summary_path, ready_summary)

                if args.prepare_only:
                    print("Preparation complete; --prepare-only requested.")
                    return 0

                print(f"Starting campaign: {campaign_id}")
                result = run_v2_campaign(
                    mode_config,
                    cli_enable=True,
                    doctor_receipt=doctor_path,
                    timeout_seconds=(
                        args.timeout_seconds if args.timeout_seconds > 0 else None
                    ),
                    overwrite_existing=bool(args.overwrite),
                )
                final_summary = {
                    **ready_summary,
                    "status": str(result.get("status")),
                    "campaign_result": str(campaign_dir / "campaign_result.json"),
                    "report": str(campaign_dir / "report.md"),
                    "result_summary": result.get("summary"),
                }
                write_json(summary_path, final_summary)
                print(json.dumps(final_summary, ensure_ascii=False, indent=2))
                return result_exit_code(result)

            except Exception as exc:
                failure = {
                    "schema_version": "ppp_mode_orchestration_v1",
                    "status": "SETUP_OR_EXECUTION_FAILED",
                    "mode": mode,
                    "campaign_id": campaign_id,
                    "exception_type": type(exc).__name__,
                    "message": str(exc),
                    "traceback": traceback.format_exc(),
                    "mode_config": str(mode_config),
                    "preflight": str(preflight_path),
                    "doctor": str(doctor_path),
                    "doctor_stability": str(doctor_stability_path),
                    "campaign_dir": str(campaign_dir),
                    "console_log": str(console_path),
                }
                write_json(summary_path, failure)
                print(json.dumps(failure, ensure_ascii=False, indent=2), file=sys.stderr)
                return 2


if __name__ == "__main__":
    raise SystemExit(main())
