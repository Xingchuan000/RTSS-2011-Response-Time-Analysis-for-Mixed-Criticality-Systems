"""Single-route V10.1 seed proof workflow."""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any

from formal_toolchain.core.errors import FormalWorkflowError
from formal_toolchain.v10_1.constants import (
    PRIMARY_CLAIM, PROOF_ROUTE, RESULT_INVALID, RESULT_PROVED, RESULT_UNRESOLVED, SCOPE,
)
from formal_toolchain.workflow.seed_workspace_v10_1 import freeze_seed_workspace_v10_1
from formal_toolchain.workflow.subprocess_runner import run_cli

EXIT_CODES = {RESULT_PROVED: 0, RESULT_UNRESOLVED: 20, RESULT_INVALID: 30,
              "CONCRETE_HI_COUNTEREXAMPLE_VERIFIED": 13}


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def prove_seed(*, seed_dir: Path, tree_variant: str, code_root: Path, out: Path,
               target_recipe: Path | None = None, overwrite: bool = False,
               solver_timeout_ms: int = 0) -> tuple[int, dict[str, Any]]:
    """Freeze -> preflight -> fresh verify -> report for V10.1.

    This research workflow writes directly to the requested output directory.
    There is no staging/publication compatibility layer: a failed proof keeps
    its real receipts at the normal output path, and Windows directory-rename
    behaviour cannot overwrite the verifier result after verification ends.
    """
    if solver_timeout_ms < 0:
        raise ValueError("solver_timeout_ms must be non-negative; 0 means unlimited")
    code_root = Path(code_root).resolve()
    out = Path(out).resolve()
    lock = out.parent / f".{out.name}.lock"
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        return 30, {"workflow_status": "FAILED", "result_status": RESULT_INVALID,
                    "failure_code": "WORKSPACE_LOCKED", "proof_route": PROOF_ROUTE}
    try:
        if out.exists():
            if not overwrite:
                return 30, {"workflow_status": "FAILED", "result_status": RESULT_INVALID,
                            "failure_code": "OUTPUT_EXISTS", "proof_route": PROOF_ROUTE}
            shutil.rmtree(out)

        imported = freeze_seed_workspace_v10_1(
            seed_dir, tree_variant, out, code_root=code_root,
            target_recipe=target_recipe, overwrite=False,
        )
        request = Path(imported["request"])
        commands: list[dict[str, Any]] = []

        inspect = run_cli(
            "formal_toolchain.cli.inspect_target",
            ["--request", str(request), "--source-root", str(code_root), "--out", str(out / "preflight")],
            cwd=code_root, log_dir=out / "logs",
        )
        commands.append(inspect)
        if inspect["returncode"] != 0:
            summary = {"workflow_status": "FAILED", "result_status": RESULT_UNRESOLVED,
                       "failure_code": "V10_1_PREFLIGHT_FAILED", "proof_route": PROOF_ROUTE,
                       "scope": SCOPE, "primary_claim": PRIMARY_CLAIM}
        else:
            verify_run = run_cli(
                "formal_toolchain.cli.verify_bundle",
                ["--request", str(request),
                 "--out", str(out / "verified"), "--source-root", str(code_root),
                 "--timeout-ms", str(int(solver_timeout_ms))],
                cwd=code_root, log_dir=out / "logs",
            )
            commands.append(verify_run)
            summary_path = out / "verified/proof_summary.json"
            if not summary_path.is_file():
                stderr_path = out / "logs" / "verify_bundle.stderr.log"
                stderr_text = stderr_path.read_text(encoding="utf-8", errors="replace") if stderr_path.is_file() else ""
                tail = "\n".join(stderr_text.rstrip().splitlines()[-12:]) or None
                summary = {"workflow_status": "FAILED", "result_status": RESULT_UNRESOLVED,
                           "failure_code": "V10_1_VERIFIER_PROCESS_FAILED", "proof_route": PROOF_ROUTE,
                           "scope": SCOPE, "primary_claim": PRIMARY_CLAIM,
                           "failure_message": tail}
            else:
                summary = json.loads(summary_path.read_text(encoding="utf-8"))
                report_run = run_cli(
                    "formal_toolchain.cli.render_report",
                    ["--verified", str(out / "verified"), "--out", str(out / "human_readable_report.md")],
                    cwd=code_root, log_dir=out / "logs",
                )
                commands.append(report_run)

        _write(out / "workflow_manifest.json", {
            "schema_version": "v10_1_workflow_manifest_v1", "proof_route": PROOF_ROUTE,
            "solver_timeout_ms": int(solver_timeout_ms),
            "solver_timeout_policy": "UNLIMITED" if int(solver_timeout_ms) == 0 else "FINITE",
            "workspace_mode": "DIRECT_OUTPUT",
            "commands": commands,
        })
        result_status = str(summary.get("result_status", RESULT_INVALID))
        proof_result = {
            "workflow_schema_version": "prove_seed_v10_1_workflow_v1",
            "proof_route": PROOF_ROUTE,
            "scope": SCOPE,
            "primary_claim": PRIMARY_CLAIM,
            "target_id": imported.get("target_id"),
            "target_kind": imported.get("target_kind"),
            "taskset_seed": json.loads(request.read_text(encoding="utf-8"))["taskset_seed"],
            "tree_variant": tree_variant,
            "workflow_status": summary.get("workflow_status", "FAILED"),
            "result_status": result_status,
            "failure_code": summary.get("failure_code"),
            "failure_message": summary.get("failure_message"),
            "verified_summary": "verified/proof_summary.json" if (out / "verified/proof_summary.json").is_file() else None,
            "exit_code": EXIT_CODES.get(result_status, 30),
            "workspace_mode": "DIRECT_OUTPUT",
        }
        _write(out / "proof_result.json", proof_result)
        if not (out / "human_readable_report.md").is_file():
            (out / "human_readable_report.md").write_text(
                f"# V10.1 Formal Proof Report\n\n- result_status: `{result_status}`\n- failure_code: `{summary.get('failure_code')}`\n",
                encoding="utf-8",
            )
        return int(proof_result["exit_code"]), proof_result
    except (FormalWorkflowError, OSError, ValueError, KeyError) as exc:
        route = exc.route if isinstance(exc, FormalWorkflowError) else RESULT_UNRESOLVED
        code = exc.code if isinstance(exc, FormalWorkflowError) else "V10_1_WORKFLOW_INPUT_ERROR"
        failure = {
            "workflow_schema_version": "prove_seed_v10_1_workflow_v1",
            "proof_route": PROOF_ROUTE, "scope": SCOPE, "primary_claim": PRIMARY_CLAIM,
            "workflow_status": "FAILED", "result_status": route,
            "failure_code": code, "failure_message": str(exc),
            "exit_code": EXIT_CODES.get(route, 20),
            "workspace_mode": "DIRECT_OUTPUT",
        }
        out.mkdir(parents=True, exist_ok=True)
        _write(out / "proof_result.json", failure)
        return int(failure["exit_code"]), failure
    finally:
        os.close(fd)
        try:
            lock.unlink()
        except FileNotFoundError:
            pass
