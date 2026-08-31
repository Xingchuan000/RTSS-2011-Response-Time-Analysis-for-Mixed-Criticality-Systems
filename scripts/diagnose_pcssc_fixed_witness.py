#!/usr/bin/env python3
"""Diagnose V10.11 PCSSC pointwise-max conservatism with fixed outer witnesses.

This script is diagnostic only.  It does NOT change any formal PASS/FAIL rule,
does NOT export a completion certificate, and does NOT modify the proof DAG.

For one already-computed PCSSC target, the normal V10.11 terminal evaluates

    W#(R) = max_omega W(R; omega)

and is free to select a different maximizing outer case at every tested horizon.
This diagnostic instead enumerates the outer PCSSC cases at the target deadline
and, for each fixed case separately, iterates

    R_{k+1} = W(R_k; omega)

until it finds a post-fixed horizon W(R;omega) <= R or crosses the deadline.

"Outer witness" here means the tuple

    (controller theta, switch cell/profile, target classification).

All *inner* V10.11 relaxations are intentionally left unchanged: controller
budget boxes, aggregate carry-in, per-task compatible phase maximization outside
protected PRE_HI, and switch-cell max semantics are exactly the same arithmetic
used by ``formal_toolchain.v10_1.pcssc``.  Therefore this is a clean diagnostic
of cross-horizon outer-case switching, not yet a new theorem and not a concrete
execution validator.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import sys
import tempfile
import time
from typing import Any, Callable, Iterable
import zipfile


@dataclass(frozen=True, slots=True)
class FixedWitnessIteration:
    status: str
    response_bound: int | None
    path: tuple[dict[str, int], ...]
    last_details: dict[str, Any] | None
    failure: str | None = None


def _iterate_fixed_witness(
    *,
    initial_horizon: int,
    deadline: int,
    evaluator: Callable[[int], tuple[int, dict[str, Any]]],
    max_iterations: int = 128,
) -> FixedWitnessIteration:
    """Run one fixed-case post-fixpoint iteration without changing the case."""

    R = max(1, int(initial_horizon))
    D = int(deadline)
    if D <= 0:
        raise ValueError("deadline must be positive")
    rows: list[dict[str, int]] = []
    last_details: dict[str, Any] | None = None

    for _ in range(int(max_iterations)):
        if R > D:
            return FixedWitnessIteration(
                status="EXCEEDS_DEADLINE",
                response_bound=None,
                path=tuple(rows),
                last_details=last_details,
                failure=f"candidate horizon {R} exceeds deadline {D}",
            )
        W, details = evaluator(int(R))
        W = int(W)
        if W <= 0:
            return FixedWitnessIteration(
                status="UNRESOLVED",
                response_bound=None,
                path=tuple(rows),
                last_details=details,
                failure=f"non-positive workload {W}",
            )
        rows.append({"R": int(R), "W": int(W)})
        last_details = details
        if W <= R:
            return FixedWitnessIteration(
                status="POSTFIX_FOUND",
                response_bound=int(R),
                path=tuple(rows),
                last_details=last_details,
            )
        if W > D:
            return FixedWitnessIteration(
                status="EXCEEDS_DEADLINE",
                response_bound=None,
                path=tuple(rows),
                last_details=last_details,
                failure=f"W({R})={W} exceeds deadline {D}",
            )
        # W > R here, so the sequence grows strictly and cannot cycle.
        R = int(W)

    return FixedWitnessIteration(
        status="UNRESOLVED",
        response_bound=None,
        path=tuple(rows),
        last_details=last_details,
        failure=f"iteration limit {max_iterations} reached",
    )


def _diagnostic_signal(*, original_status: str, statuses: Iterable[str]) -> str:
    rows = tuple(str(value) for value in statuses)
    if not rows:
        return "NO_WITNESSES"
    if str(original_status) != "PASS" and all(value == "POSTFIX_FOUND" for value in rows):
        return "STRONG_OUTER_WITNESS_SWITCHING_CONSERVATISM_SIGNAL"
    if any(value == "EXCEEDS_DEADLINE" for value in rows):
        return "FIXED_OUTER_WITNESS_STILL_EXCEEDS_DEADLINE"
    if any(value == "UNRESOLVED" for value in rows):
        return "FIXED_OUTER_WITNESS_DIAGNOSTIC_UNRESOLVED"
    return "NO_POINTWISE_FAILURE_TO_EXPLAIN"


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _find_workspace(root: Path) -> Path:
    if (root / "request" / "proof_request.json").is_file() and (
        root / "verified" / "proof_receipts.json"
    ).is_file():
        return root
    candidates = sorted({
        path.parent.parent.resolve()
        for path in root.rglob("request/proof_request.json")
        if (path.parent.parent / "verified" / "proof_receipts.json").is_file()
    })
    if len(candidates) != 1:
        raise ValueError(
            "could not identify exactly one proof workspace; "
            f"found {len(candidates)} candidates under {root}"
        )
    return candidates[0]


class _WorkspaceInput:
    def __init__(self, path: Path) -> None:
        self.input_path = path.resolve()
        self._temp: tempfile.TemporaryDirectory[str] | None = None
        if self.input_path.is_dir():
            self.workspace = _find_workspace(self.input_path)
        elif self.input_path.is_file() and self.input_path.suffix.lower() == ".zip":
            self._temp = tempfile.TemporaryDirectory(prefix="pcssc_fixed_witness_")
            extract_root = Path(self._temp.name)
            with zipfile.ZipFile(self.input_path) as archive:
                archive.extractall(extract_root)
            self.workspace = _find_workspace(extract_root)
        else:
            raise ValueError("--result must be an extracted proof workspace or a .zip bundle")

    def close(self) -> None:
        if self._temp is not None:
            self._temp.cleanup()


def _find_pcssc_target(receipts: dict[str, Any], target_name: str) -> dict[str, Any]:
    rows = receipts.get("pcssc_targets")
    if not isinstance(rows, list):
        raise ValueError("proof_receipts.pcssc_targets missing")
    matches = [
        row for row in rows
        if isinstance(row, dict) and str(row.get("target")) == str(target_name)
    ]
    if len(matches) != 1:
        raise ValueError(
            f"expected exactly one PCSSC target {target_name!r}, found {len(matches)}"
        )
    return dict(matches[0])


def _effective_completion_map(pcssc_target: dict[str, Any], target_name: str) -> dict[str, int]:
    rows = pcssc_target.get("receipts")
    if not isinstance(rows, list):
        raise ValueError("PCSSC target receipts missing")
    obligation = f"REACHABLE_CARRY_IN::{target_name}"
    matches = [
        row for row in rows
        if isinstance(row, dict) and str(row.get("obligation_id")) == obligation
    ]
    if len(matches) != 1:
        raise ValueError(f"expected exactly one {obligation} receipt, found {len(matches)}")
    raw = matches[0].get("effective_completion_envelopes")
    if not isinstance(raw, dict):
        raise ValueError(f"{obligation} has no effective_completion_envelopes")
    return {str(name): int(value) for name, value in raw.items()}


def _rebuild_model(request_path: Path, source_root: Path) -> Any:
    # Lazy imports keep this module/unit tests usable even on machines without
    # z3.  A real diagnostic run uses the same formal bindings as V10.11.
    from formal_toolchain.v10_1.bindings import build_bindings
    from formal_toolchain.v10_1.kernel.symbolic_state import BoundModel

    bindings = build_bindings(request_path, source_root=source_root)
    return BoundModel.from_bindings(bindings, max_jobs_per_task=2)


def _rebuild_controller_path(receipts: dict[str, Any]) -> Any:
    from formal_toolchain.v10_1.controller_macro import BudgetInterval, ControllerMacroPath

    raw = receipts.get("controller_macro")
    if not isinstance(raw, dict):
        raise ValueError("proof_receipts.controller_macro missing")
    raw_boxes = raw.get("boxes")
    if not isinstance(raw_boxes, list) or not raw_boxes:
        raise ValueError("controller_macro.boxes missing")
    boxes = []
    for raw_box in raw_boxes:
        if not isinstance(raw_box, dict):
            raise ValueError("controller_macro box must be a JSON object")
        boxes.append({
            str(name): BudgetInterval(int(interval["lower"]), int(interval["upper"]))
            for name, interval in raw_box.items()
            if isinstance(interval, dict)
        })
    return ControllerMacroPath(
        boxes=tuple(boxes),
        receipts=tuple(raw.get("receipts") or ()),
        conservatism_ledger=tuple(raw.get("conservatism_ledger") or ()),
    )


def _outer_witness_id(theta: int, switch: Any, classification: str) -> str:
    return f"theta={int(theta)}|switch={switch.id}|class={classification}"


def _compact_case(details: dict[str, Any] | None) -> dict[str, Any] | None:
    if details is None:
        return None
    carry = details.get("carry_in_model")
    carry_value = None
    if isinstance(carry, dict):
        for key in ("carry_in", "total_carry", "carry"):
            if key in carry:
                carry_value = int(carry[key])
                break
    result = {
        "theta": details.get("theta"),
        "switch_profile": details.get("switch_profile"),
        "target_classification": details.get("target_classification"),
        "target_demand": details.get("target_demand"),
        "selected_hp_bound": details.get("selected_hp_bound"),
        "aggregate_hp_bound": details.get("aggregate_hp_bound"),
        "completion_phase_coupled_hp_bound": details.get("completion_phase_coupled_hp_bound"),
        "future_interference_total": details.get("future_interference_total"),
        "carry_in": carry_value,
    }
    return {key: value for key, value in result.items() if value is not None}


def _original_pointwise_trace(pcssc_target: dict[str, Any]) -> list[dict[str, Any]]:
    tested = pcssc_target.get("tested_horizons")
    if not isinstance(tested, list):
        return []
    rows: list[dict[str, Any]] = []
    for row in tested:
        if not isinstance(row, dict):
            continue
        case = row.get("maximizing_case")
        if not isinstance(case, dict):
            case = {}
        rows.append({
            "R": int(row.get("R", 0)),
            "W": int(row.get("W", 0)),
            "postfixed": bool(row.get("postfixed", False)),
            "outer_witness": {
                "theta": case.get("theta"),
                "switch_profile": case.get("switch_profile"),
                "target_classification": case.get("target_classification"),
            },
        })
    return rows


def _render_markdown(report: dict[str, Any]) -> str:
    summary = report["summary"]
    lines = [
        f"# PCSSC fixed-outer-witness diagnostic: {report['target']}",
        "",
        "> 诊断用途：不修改 V10.11 正式 PASS/FAIL 条件，不产生 completion certificate。",
        "",
        "## 结果摘要",
        "",
        f"- 原 PCSSC 状态：`{report['original_pcssc_status']}`",
        f"- deadline：`{report['deadline']}`",
        f"- fixed outer witness 数量：`{summary['witness_count']}`",
        f"- 找到 postfix：`{summary['postfixed_count']}`",
        f"- 固定 witness 仍超过 deadline：`{summary['exceeds_deadline_count']}`",
        f"- 诊断 unresolved：`{summary['unresolved_count']}`",
        f"- 最大已闭合 response bound：`{summary.get('max_postfixed_response_bound')}`",
        f"- 诊断信号：`{summary['diagnostic_signal']}`",
        f"- 总 workload evaluations：`{summary['workload_evaluations']}`",
        f"- wall time：`{summary['elapsed_seconds']:.3f}s`",
        "",
        "## 原 V10.11 pointwise-max 轨迹",
        "",
        "| R | W | theta | switch | class |",
        "|---:|---:|---:|---|---|",
    ]
    for row in report["original_pointwise_trace"]:
        witness = row["outer_witness"]
        lines.append(
            f"| {row['R']} | {row['W']} | {witness.get('theta')} | "
            f"{witness.get('switch_profile')} | {witness.get('target_classification')} |"
        )
    failures = report.get("top_non_postfixed_witnesses") or []
    lines.extend(["", "## 最重要的未闭合 fixed witnesses", ""])
    if not failures:
        lines.append("没有。所有枚举的 fixed outer witnesses 都找到了 deadline 内 postfix。")
    else:
        lines.extend([
            "| witness | status | last R | last W | overshoot |",
            "|---|---|---:|---:|---:|",
        ])
        for row in failures:
            path = row.get("path") or []
            last = path[-1] if path else {"R": None, "W": None}
            overshoot = None if last["W"] is None else int(last["W"]) - int(report["deadline"])
            lines.append(
                f"| `{row['witness_id']}` | {row['status']} | {last['R']} | "
                f"{last['W']} | {overshoot} |"
            )
    lines.extend([
        "",
        "## 解释边界",
        "",
        "本工具只固定 outer PCSSC case（theta / switch cell / target classification）。",
        "controller budget box、aggregate carry-in、非 protected PRE_HI 的 per-task phase relaxation、",
        "以及 Sigma switch-cell 内部的 max relaxation 都保持 V10.11 原样。因此：",
        "",
        "- 若所有 fixed outer witnesses 都闭合，这是 case-consistent theorem 的强诊断证据，但不是正式证明；",
        "- 若仍有 fixed outer witness 超时，应优先对这些少数 witness 做 concrete replay/进一步可达性验证；",
        "- 本输出不得用于把正式 verifier 的 UNRESOLVED 改成 PASS。",
        "",
    ])
    return "\n".join(lines)


def run_diagnostic(
    *,
    result_path: Path,
    source_root: Path,
    target_name: str,
    output_dir: Path,
    max_iterations: int,
    top_n: int,
) -> dict[str, Any]:
    started = time.perf_counter()
    workspace_input = _WorkspaceInput(result_path)
    try:
        workspace = workspace_input.workspace
        receipts = _read_json(workspace / "verified" / "proof_receipts.json")
        request = _read_json(workspace / "request" / "proof_request.json")
        pcssc_target = _find_pcssc_target(receipts, target_name)
        model = _rebuild_model(workspace / "request" / "proof_request.json", source_root)
        target = model.task_by_name.get(target_name)
        if target is None:
            raise ValueError(f"target {target_name!r} not found in rebuilt bound model")
        if str(target.criticality) != "HI":
            raise ValueError("fixed-witness PCSSC diagnostic requires a HI target")
        path = _rebuild_controller_path(receipts)
        protected_response_by_task = _effective_completion_map(pcssc_target, target_name)
        protected = set(protected_response_by_task)

        # Import the exact V10.11 arithmetic only after the model/bindings have
        # been reconstructed.  No alternate workload implementation is used.
        from formal_toolchain.v10_1.controller_macro import (
            candidate_controller_times,
            controller_phase_residues,
        )
        from formal_toolchain.v10_1.pcssc import (
            PCSSCUnresolved,
            SwitchCell,
            _switch_cells,
            _target_cap,
            _valid_target_classes,
            _workload_case,
        )

        target_index = next(
            index for index, task in enumerate(model.tasks) if task.name == target_name
        )
        hp_tasks = tuple(model.tasks[:target_index])
        deadline = int(target.deadline)
        witness_specs: list[tuple[int, Any, str]] = []
        for theta in controller_phase_residues(target.period, model.agent_period):
            controller_times = candidate_controller_times(theta, model.agent_period, deadline)
            profiles: list[Any] = [SwitchCell("PRE_HI"), SwitchCell("LO_NO_SWITCH")]
            profiles.extend(_switch_cells(deadline, controller_times, hp_tasks))
            for switch in profiles:
                for classification in _valid_target_classes(target, switch):
                    witness_specs.append((int(theta), switch, str(classification)))

        results: list[dict[str, Any]] = []
        workload_evaluations = 0
        for index, (theta, switch, classification) in enumerate(witness_specs, start=1):
            witness_id = _outer_witness_id(theta, switch, classification)

            def evaluate(horizon: int) -> tuple[int, dict[str, Any]]:
                nonlocal workload_evaluations
                workload_evaluations += 1
                return _workload_case(
                    model,
                    target,
                    hp_tasks,
                    path,
                    protected,
                    protected_response_by_task,
                    horizon=int(horizon),
                    theta=int(theta),
                    switch=switch,
                    classification=classification,
                )

            try:
                iteration = _iterate_fixed_witness(
                    initial_horizon=int(_target_cap(target, classification)),
                    deadline=deadline,
                    evaluator=evaluate,
                    max_iterations=max_iterations,
                )
            except PCSSCUnresolved as exc:
                iteration = FixedWitnessIteration(
                    status="UNRESOLVED",
                    response_bound=None,
                    path=(),
                    last_details=None,
                    failure=str(exc),
                )
            results.append({
                "witness_id": witness_id,
                "theta": int(theta),
                "switch_profile": switch.id,
                "switch_kind": switch.kind,
                "switch_lower": switch.lower,
                "switch_upper": switch.upper,
                "target_classification": classification,
                "status": iteration.status,
                "response_bound": iteration.response_bound,
                "iterations": len(iteration.path),
                "path": list(iteration.path),
                "last_case": _compact_case(iteration.last_details),
                "failure": iteration.failure,
            })
            if index % 100 == 0 or index == len(witness_specs):
                print(
                    f"[{index}/{len(witness_specs)}] fixed outer witnesses evaluated",
                    file=sys.stderr,
                    flush=True,
                )

        statuses = [row["status"] for row in results]
        postfixed = [row for row in results if row["status"] == "POSTFIX_FOUND"]
        non_postfixed = [row for row in results if row["status"] != "POSTFIX_FOUND"]
        non_postfixed.sort(
            key=lambda row: (
                (row["path"][-1]["W"] - deadline) if row["path"] else -10**18,
                row["witness_id"],
            ),
            reverse=True,
        )
        elapsed = time.perf_counter() - started
        summary = {
            "witness_count": len(results),
            "postfixed_count": sum(row["status"] == "POSTFIX_FOUND" for row in results),
            "exceeds_deadline_count": sum(row["status"] == "EXCEEDS_DEADLINE" for row in results),
            "unresolved_count": sum(row["status"] == "UNRESOLVED" for row in results),
            "max_postfixed_response_bound": (
                max(int(row["response_bound"]) for row in postfixed) if postfixed else None
            ),
            "diagnostic_signal": _diagnostic_signal(
                original_status=str(pcssc_target.get("status")), statuses=statuses
            ),
            "workload_evaluations": int(workload_evaluations),
            "elapsed_seconds": float(elapsed),
        }
        report = {
            "schema_version": "v10_11_pcssc_fixed_outer_witness_diagnostic_v1",
            "diagnostic_only": True,
            "changes_formal_verdict": False,
            "result_input": str(Path(result_path).resolve()),
            "source_root": str(Path(source_root).resolve()),
            "taskset_seed": request.get("taskset_seed"),
            "framework_revision": receipts.get("framework_revision"),
            "target": target_name,
            "deadline": deadline,
            "original_pcssc_status": str(pcssc_target.get("status")),
            "original_pcssc_failure_code": pcssc_target.get("failure_code"),
            "original_pointwise_trace": _original_pointwise_trace(pcssc_target),
            "outer_witness_definition": [
                "controller_theta",
                "switch_cell_or_profile",
                "target_classification",
            ],
            "inner_relaxations_retained": [
                "controller_budget_box",
                "R7_aggregate_carry_in",
                "per_task_compatible_phase_max_outside_protected_PRE_HI",
                "ambiguous_sigma_switch_cell_max",
            ],
            "effective_completion_envelopes": protected_response_by_task,
            "summary": summary,
            "top_non_postfixed_witnesses": non_postfixed[: max(0, int(top_n))],
            "witness_results": results,
        }
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "fixed_witness_diagnostic.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        (output_dir / "fixed_witness_diagnostic.md").write_text(
            _render_markdown(report), encoding="utf-8"
        )
        return report
    finally:
        workspace_input.close()


def _default_output_dir(result: Path, target: str) -> Path:
    stem = result.stem if result.suffix.lower() == ".zip" else result.name
    safe_target = "".join(char if char.isalnum() or char in "-_" else "_" for char in target)
    return Path(f"fixed_witness_diagnostic_{stem}_{safe_target}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Sweep fixed outer PCSSC witnesses without changing V10.11 proof semantics."
    )
    parser.add_argument(
        "--result", required=True, type=Path,
        help="V10.11 proof-result ZIP or extracted proof workspace",
    )
    parser.add_argument(
        "--source-root", type=Path, default=Path("."),
        help="source tree matching the proof request (default: current directory)",
    )
    parser.add_argument(
        "--target", default="mc_sd_hi_0",
        help="PCSSC HI target to diagnose (default: mc_sd_hi_0)",
    )
    parser.add_argument(
        "--output", type=Path, default=None,
        help="output directory; default is derived from result/target",
    )
    parser.add_argument(
        "--max-iterations", type=int, default=128,
        help="per-witness recurrence guard (default: 128)",
    )
    parser.add_argument(
        "--top-n", type=int, default=20,
        help="number of non-postfixed witnesses highlighted in summary (default: 20)",
    )
    args = parser.parse_args()
    if args.max_iterations <= 0:
        parser.error("--max-iterations must be positive")
    if args.top_n < 0:
        parser.error("--top-n must be non-negative")

    output = args.output or _default_output_dir(args.result, args.target)
    report = run_diagnostic(
        result_path=args.result,
        source_root=args.source_root.resolve(),
        target_name=str(args.target),
        output_dir=output,
        max_iterations=int(args.max_iterations),
        top_n=int(args.top_n),
    )
    summary = report["summary"]
    print(json.dumps({
        "target": report["target"],
        "deadline": report["deadline"],
        **summary,
        "output": str(output.resolve()),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
