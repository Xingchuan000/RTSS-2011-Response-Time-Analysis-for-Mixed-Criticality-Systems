from __future__ import annotations

import argparse
import csv
import json
import os
import signal
import subprocess
import sys
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable


FULL_MODES: tuple[str, ...] = (
    "S185Core",
    "PositiveControls",
    "S397Pair",
    "B234",
    "C123",
    "IntegrityAll",
    "ModelEAll",
    "D1",
    "PaperMinimum",
)

PAPER_CORE_MODES: tuple[str, ...] = (
    "S185Core",
    "PositiveControls",
    "S397Pair",
    "B234",
    "Integrity",
    "ModelE1",
    "D1",
    "PaperMinimum",
)

SUPPORTED_MODES: frozenset[str] = frozenset(
    {
        "S185Core",
        "PositiveControls",
        "S397Pair",
        "B234",
        "Integrity",
        "IntegrityAll",
        "ModelE1",
        "ModelEAll",
        "C123",
        "D1",
        "PaperMinimum",
    }
)


@dataclass
class ModeResult:
    mode: str
    state: str
    exit_code: int | None
    started_at: str | None
    ended_at: str | None
    duration_seconds: float | None
    orchestration_status: str | None
    result_counts: dict[str, int]
    campaign_result: str | None
    report: str | None
    mode_console_log: str | None
    overnight_console_log: str
    message: str | None = None


_RUNNING: dict[str, subprocess.Popen[str]] = {}
_RUNNING_LOCK = threading.Lock()
_STOP_REQUESTED = threading.Event()


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the PPP non-vacuity experiment modes overnight with at most two "
            "modes in parallel. A mode failure never stops the remaining queue."
        )
    )
    parser.add_argument("--project-root", type=Path, default=Path(r"D:\AMC"))
    parser.add_argument(
        "--output-tag",
        default="formalv1_csem_t10_s1550_1599_tr8e6_v2e7_h2_h5",
    )
    parser.add_argument("--experiment-data-root", type=Path)
    parser.add_argument("--preset", choices=("Full", "PaperCore"), default="Full")
    parser.add_argument(
        "--mode",
        action="append",
        default=[],
        help="Custom mode. Repeat this option to replace the preset mode list.",
    )
    parser.add_argument("--max-parallel", type=int, choices=(1, 2), default=2)
    parser.add_argument("--timeout-seconds-per-mode", type=int, default=0)
    parser.add_argument(
        "--overwrite",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Overwrite each mode's existing campaign directory (default: true).",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip a mode only when its existing orchestration status is COMPLETED.",
    )
    parser.add_argument(
        "--strict-exit",
        action="store_true",
        help="Return non-zero after the queue if any mode is not COMPLETED.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def deduplicate(values: Iterable[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value not in seen:
            result.append(value)
            seen.add(value)
    return result


def resolve_modes(args: argparse.Namespace) -> list[str]:
    if args.mode:
        modes = deduplicate(str(value) for value in args.mode)
    elif args.preset == "PaperCore":
        modes = list(PAPER_CORE_MODES)
    else:
        modes = list(FULL_MODES)

    unknown = sorted(set(modes) - SUPPORTED_MODES)
    if unknown:
        raise ValueError(f"Unsupported modes: {unknown}")

    # PaperMinimum deliberately re-runs the publication subset and must not race
    # with its component campaigns. Always move it to the end and run it alone.
    has_paper_minimum = "PaperMinimum" in modes
    modes = [mode for mode in modes if mode != "PaperMinimum"]
    if has_paper_minimum:
        modes.append("PaperMinimum")
    return modes


def load_json_object(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def counts_from_summary(summary: dict[str, Any] | None) -> dict[str, int]:
    if not isinstance(summary, dict):
        return {}
    counts = summary.get("counts")
    if not isinstance(counts, dict):
        return {}
    result: dict[str, int] = {}
    for key, value in counts.items():
        try:
            result[str(key)] = int(value)
        except (TypeError, ValueError):
            continue
    return result


def orchestration_path(data_root: Path, mode: str) -> Path:
    return data_root / "config" / f"orchestration_{mode}.json"


def existing_completed_result(
    *, mode: str, data_root: Path, overnight_log: Path
) -> ModeResult | None:
    path = orchestration_path(data_root, mode)
    data = load_json_object(path)
    if not data or data.get("status") != "COMPLETED":
        return None
    return ModeResult(
        mode=mode,
        state="SKIPPED_ALREADY_COMPLETED",
        exit_code=0,
        started_at=None,
        ended_at=None,
        duration_seconds=0.0,
        orchestration_status="COMPLETED",
        result_counts=counts_from_summary(data.get("result_summary")),
        campaign_result=data.get("campaign_result"),
        report=data.get("report"),
        mode_console_log=str(data_root / "config" / f"orchestration_{mode}.log"),
        overnight_console_log=str(overnight_log),
        message="Skipped by --resume because an existing COMPLETED result was found.",
    )


def terminate_all_children() -> None:
    _STOP_REQUESTED.set()
    with _RUNNING_LOCK:
        children = list(_RUNNING.items())
    for mode, process in children:
        if process.poll() is not None:
            continue
        try:
            process.terminate()
            print(f"Requested termination of {mode} (PID {process.pid}).", flush=True)
        except OSError:
            pass


def install_signal_handlers() -> None:
    def handler(signum: int, _frame: object) -> None:
        print(f"\nReceived signal {signum}; terminating active modes...", flush=True)
        terminate_all_children()

    for signal_name in ("SIGINT", "SIGTERM"):
        sig = getattr(signal, signal_name, None)
        if sig is not None:
            signal.signal(sig, handler)


def run_mode(
    *,
    mode: str,
    project_root: Path,
    data_root: Path,
    output_tag: str,
    timeout_seconds: int,
    overwrite: bool,
    run_dir: Path,
) -> ModeResult:
    log_path = run_dir / f"{mode}.console.log"
    start_wall = now_iso()
    start_mono = time.monotonic()
    command = [
        sys.executable,
        "-u",
        str(project_root / "scripts" / "run_ppp_nonvacuity_mode_final.py"),
        "--mode",
        mode,
        "--project-root",
        str(project_root),
        "--output-tag",
        output_tag,
        "--experiment-data-root",
        str(data_root),
    ]
    if timeout_seconds > 0:
        command.extend(["--timeout-seconds", str(timeout_seconds)])
    if overwrite:
        command.append("--overwrite")

    env = os.environ.copy()
    env.setdefault("PYTHONHASHSEED", "0")
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("MKL_NUM_THREADS", "1")
    env.setdefault("OPENBLAS_NUM_THREADS", "1")
    env.setdefault("NUMEXPR_NUM_THREADS", "1")
    env["PYTHONUNBUFFERED"] = "1"

    print(f"[{start_wall}] START {mode}", flush=True)
    print(f"  log: {log_path}", flush=True)

    exit_code: int | None = None
    message: str | None = None
    try:
        with log_path.open("w", encoding="utf-8", newline="") as log_stream:
            log_stream.write("COMMAND: " + subprocess.list2cmdline(command) + "\n\n")
            log_stream.flush()
            process = subprocess.Popen(
                command,
                cwd=project_root,
                env=env,
                stdout=log_stream,
                stderr=subprocess.STDOUT,
                text=True,
            )
            with _RUNNING_LOCK:
                _RUNNING[mode] = process
            try:
                exit_code = process.wait()
            finally:
                with _RUNNING_LOCK:
                    _RUNNING.pop(mode, None)
    except Exception as exc:  # scheduler must continue with other modes
        exit_code = None
        message = f"{type(exc).__name__}: {exc}"
        try:
            with log_path.open("a", encoding="utf-8") as log_stream:
                log_stream.write("\nORCHESTRATOR ERROR: " + message + "\n")
        except OSError:
            pass

    end_wall = now_iso()
    duration = round(time.monotonic() - start_mono, 3)
    orchestration = load_json_object(orchestration_path(data_root, mode)) or {}
    orchestration_status = (
        str(orchestration.get("status")) if orchestration.get("status") is not None else None
    )
    counts = counts_from_summary(orchestration.get("result_summary"))

    if _STOP_REQUESTED.is_set() and exit_code not in (0, None):
        state = "INTERRUPTED"
    elif exit_code == 0 and orchestration_status == "COMPLETED":
        state = "COMPLETED"
    elif exit_code is None:
        state = "LAUNCH_FAILED"
    else:
        state = "FINISHED_WITH_FAILURE"

    print(
        f"[{end_wall}] END   {mode}: state={state}, exit={exit_code}, "
        f"status={orchestration_status}, duration={duration:.1f}s",
        flush=True,
    )

    return ModeResult(
        mode=mode,
        state=state,
        exit_code=exit_code,
        started_at=start_wall,
        ended_at=end_wall,
        duration_seconds=duration,
        orchestration_status=orchestration_status,
        result_counts=counts,
        campaign_result=orchestration.get("campaign_result"),
        report=orchestration.get("report"),
        mode_console_log=str(data_root / "config" / f"orchestration_{mode}.log"),
        overnight_console_log=str(log_path),
        message=message or orchestration.get("message"),
    )


def write_outputs(
    *,
    run_dir: Path,
    project_root: Path,
    data_root: Path,
    modes: list[str],
    max_parallel: int,
    started_at: str,
    ended_at: str,
    results: list[ModeResult],
) -> None:
    result_map = {result.mode: result for result in results}
    ordered_results = [result_map[mode] for mode in modes if mode in result_map]

    summary = {
        "schema_version": "ppp_overnight_orchestration_v1",
        "status": (
            "COMPLETED"
            if all(result.state in {"COMPLETED", "SKIPPED_ALREADY_COMPLETED"} for result in ordered_results)
            else "COMPLETED_WITH_MODE_FAILURES"
        ),
        "started_at": started_at,
        "ended_at": ended_at,
        "project_root": str(project_root),
        "experiment_data_root": str(data_root),
        "max_parallel": max_parallel,
        "mode_order": modes,
        "results": [asdict(result) for result in ordered_results],
    }
    (run_dir / "overnight_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    csv_fields = [
        "mode",
        "state",
        "exit_code",
        "orchestration_status",
        "started_at",
        "ended_at",
        "duration_seconds",
        "result_counts",
        "campaign_result",
        "report",
        "overnight_console_log",
        "mode_console_log",
        "message",
    ]
    with (run_dir / "overnight_summary.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=csv_fields)
        writer.writeheader()
        for result in ordered_results:
            row = asdict(result)
            row["result_counts"] = json.dumps(
                result.result_counts, ensure_ascii=False, sort_keys=True
            )
            writer.writerow({field: row.get(field) for field in csv_fields})

    lines = [
        "# PPP 非空洞性夜间实验汇总",
        "",
        f"- 开始时间：`{started_at}`",
        f"- 结束时间：`{ended_at}`",
        f"- 最大并行 mode 数：`{max_parallel}`",
        f"- 总体状态：`{summary['status']}`",
        "",
        "| Mode | 调度状态 | 实验状态 | 退出码 | 结果计数 | 耗时（秒） |",
        "|---|---|---:|---:|---|---:|",
    ]
    for result in ordered_results:
        counts_text = ", ".join(
            f"{key}={value}" for key, value in sorted(result.result_counts.items())
        ) or "-"
        lines.append(
            "| {mode} | {state} | {status} | {exit_code} | {counts} | {duration} |".format(
                mode=result.mode,
                state=result.state,
                status=result.orchestration_status or "-",
                exit_code="-" if result.exit_code is None else result.exit_code,
                counts=counts_text,
                duration=(
                    "-"
                    if result.duration_seconds is None
                    else f"{result.duration_seconds:.1f}"
                ),
            )
        )
    lines.extend(
        [
            "",
            "## 日志与结果",
            "",
        ]
    )
    for result in ordered_results:
        lines.append(f"### {result.mode}")
        lines.append("")
        lines.append(f"- 夜间控制台日志：`{result.overnight_console_log}`")
        lines.append(f"- mode 内部日志：`{result.mode_console_log or '-'}`")
        lines.append(f"- campaign_result：`{result.campaign_result or '-'}`")
        lines.append(f"- report：`{result.report or '-'}`")
        if result.message:
            lines.append(f"- 消息：`{result.message}`")
        lines.append("")
    (run_dir / "overnight_report.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def main() -> int:
    args = parse_args()
    project_root = args.project_root.resolve()
    if not project_root.is_dir():
        print(f"Project root does not exist: {project_root}", file=sys.stderr)
        return 2

    runner = project_root / "scripts" / "run_ppp_nonvacuity_mode_final.py"
    if not runner.is_file():
        print(f"Mode runner is missing: {runner}", file=sys.stderr)
        return 2

    modes = resolve_modes(args)
    data_root = (
        args.experiment_data_root.resolve()
        if args.experiment_data_root is not None
        else project_root / "experiment_data" / f"ppp_nonvacuity_{args.output_tag}"
    )
    timestamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
    run_dir = data_root / "overnight_runs" / timestamp
    run_dir.mkdir(parents=True, exist_ok=False)

    started_at = now_iso()
    print("=== PPP overnight experiment queue ===")
    print(f"Project root       : {project_root}")
    print(f"Experiment data    : {data_root}")
    print(f"Run directory      : {run_dir}")
    print(f"Python             : {sys.executable}")
    print(f"Maximum parallel   : {args.max_parallel}")
    print(f"Overwrite          : {args.overwrite}")
    print(f"Resume             : {args.resume}")
    print("Mode order         : " + " -> ".join(modes))
    print("PaperMinimum, when selected, always runs last and alone.")

    if args.dry_run:
        dry_run = {
            "project_root": str(project_root),
            "experiment_data_root": str(data_root),
            "run_directory": str(run_dir),
            "max_parallel": args.max_parallel,
            "modes": modes,
            "overwrite": args.overwrite,
            "resume": args.resume,
        }
        (run_dir / "dry_run.json").write_text(
            json.dumps(dry_run, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(dry_run, ensure_ascii=False, indent=2))
        return 0

    install_signal_handlers()
    results: list[ModeResult] = []

    paper_minimum_selected = "PaperMinimum" in modes
    parallel_modes = [mode for mode in modes if mode != "PaperMinimum"]

    modes_to_run: list[str] = []
    for mode in parallel_modes:
        overnight_log = run_dir / f"{mode}.console.log"
        if args.resume:
            existing = existing_completed_result(
                mode=mode, data_root=data_root, overnight_log=overnight_log
            )
            if existing is not None:
                results.append(existing)
                print(f"SKIP {mode}: existing COMPLETED result found.")
                continue
        modes_to_run.append(mode)

    futures: dict[Future[ModeResult], str] = {}
    with ThreadPoolExecutor(max_workers=args.max_parallel) as executor:
        for mode in modes_to_run:
            future = executor.submit(
                run_mode,
                mode=mode,
                project_root=project_root,
                data_root=data_root,
                output_tag=args.output_tag,
                timeout_seconds=args.timeout_seconds_per_mode,
                overwrite=args.overwrite,
                run_dir=run_dir,
            )
            futures[future] = mode

        for future in as_completed(futures):
            mode = futures[future]
            try:
                results.append(future.result())
            except Exception as exc:  # defensive: one worker must never abort queue
                results.append(
                    ModeResult(
                        mode=mode,
                        state="ORCHESTRATOR_WORKER_FAILED",
                        exit_code=None,
                        started_at=None,
                        ended_at=now_iso(),
                        duration_seconds=None,
                        orchestration_status=None,
                        result_counts={},
                        campaign_result=None,
                        report=None,
                        mode_console_log=None,
                        overnight_console_log=str(run_dir / f"{mode}.console.log"),
                        message=f"{type(exc).__name__}: {exc}",
                    )
                )

    # PaperMinimum intentionally waits for every component mode and is executed
    # regardless of their success or failure.
    if paper_minimum_selected and not _STOP_REQUESTED.is_set():
        mode = "PaperMinimum"
        overnight_log = run_dir / f"{mode}.console.log"
        existing = (
            existing_completed_result(
                mode=mode, data_root=data_root, overnight_log=overnight_log
            )
            if args.resume
            else None
        )
        if existing is not None:
            results.append(existing)
            print("SKIP PaperMinimum: existing COMPLETED result found.")
        else:
            results.append(
                run_mode(
                    mode=mode,
                    project_root=project_root,
                    data_root=data_root,
                    output_tag=args.output_tag,
                    timeout_seconds=args.timeout_seconds_per_mode,
                    overwrite=args.overwrite,
                    run_dir=run_dir,
                )
            )

    ended_at = now_iso()
    write_outputs(
        run_dir=run_dir,
        project_root=project_root,
        data_root=data_root,
        modes=modes,
        max_parallel=args.max_parallel,
        started_at=started_at,
        ended_at=ended_at,
        results=results,
    )

    latest_marker = data_root / "overnight_runs" / "LATEST.txt"
    latest_marker.write_text(str(run_dir) + "\n", encoding="utf-8")

    print("\n=== Overnight queue finished ===")
    print(f"Summary JSON : {run_dir / 'overnight_summary.json'}")
    print(f"Summary CSV  : {run_dir / 'overnight_summary.csv'}")
    print(f"Report       : {run_dir / 'overnight_report.md'}")

    failed = [
        result
        for result in results
        if result.state not in {"COMPLETED", "SKIPPED_ALREADY_COMPLETED"}
    ]
    if failed:
        print("Modes requiring review: " + ", ".join(result.mode for result in failed))
    else:
        print("All selected modes completed successfully.")

    if _STOP_REQUESTED.is_set():
        return 130
    if args.strict_exit and failed:
        return 5
    # Default night-run behavior: preserve a zero controller exit after the queue
    # so an individual experiment failure never aborts or invalidates the batch.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
