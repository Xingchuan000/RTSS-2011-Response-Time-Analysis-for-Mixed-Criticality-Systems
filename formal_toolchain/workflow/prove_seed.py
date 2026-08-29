"""Single-route V9.2 seed proof workflow."""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any

from formal_toolchain.core.errors import FormalWorkflowError
from formal_toolchain.v9_2.constants import (
    PRIMARY_CLAIM, PROOF_ROUTE, RESULT_INVALID, RESULT_PROVED, RESULT_UNRESOLVED, SCOPE,
)
from formal_toolchain.workflow.seed_workspace_v9_2 import freeze_seed_workspace_v9_2
from formal_toolchain.workflow.subprocess_runner import run_cli

EXIT_CODES = {RESULT_PROVED: 0, RESULT_UNRESOLVED: 20, RESULT_INVALID: 30,
              "CONCRETE_HI_COUNTEREXAMPLE_VERIFIED": 13}


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _publish_staging(staging: Path, out: Path) -> str:
    """Publish a completed staging directory without losing long proof runs.

    Windows can reject a directory rename when Explorer, antivirus, or another
    reader temporarily holds a handle.  The proof artifacts are already fully
    materialized at this point, so a same-tree copy is a safe publication
    fallback.  The staging directory is intentionally retained after a copy so
    the original proof result remains recoverable.
    """

    try:
        staging.rename(out)
        return "RENAMED"
    except PermissionError as rename_exc:
        if out.exists():
            raise
        try:
            shutil.copytree(staging, out)
        except OSError as copy_exc:
            raise OSError(
                f"failed to publish proof staging directory; "
                f"rename_error={rename_exc}; copy_error={copy_exc}"
            ) from copy_exc
        return "COPIED_AFTER_RENAME_DENIED"


def prove_seed(*, seed_dir: Path, tree_variant: str, code_root: Path, out: Path,
               target_recipe: Path | None = None, overwrite: bool = False,
               solver_timeout_ms: int = 120_000,
               max_boot_replay_ticks: int = 2_000) -> tuple[int, dict[str, Any]]:
    """Freeze -> preflight -> compile -> fresh verify -> report for V9.2 only."""

    if solver_timeout_ms <= 0:
        raise ValueError("solver_timeout_ms must be positive")
    if max_boot_replay_ticks < 0:
        raise ValueError("max_boot_replay_ticks must be non-negative")
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
        staging = out.parent / f".{out.name}.staging"
        if staging.exists():
            shutil.rmtree(staging)

        imported = freeze_seed_workspace_v9_2(
            seed_dir, tree_variant, staging, code_root=code_root,
            target_recipe=target_recipe, overwrite=False,
        )
        request = Path(imported["request"])
        commands: list[dict[str, Any]] = []

        inspect = run_cli(
            "formal_toolchain.cli.inspect_target",
            ["--request", str(request), "--source-root", str(code_root), "--out", str(staging / "preflight")],
            cwd=code_root, log_dir=staging / "logs",
        )
        commands.append(inspect)
        if inspect["returncode"] != 0:
            summary = {"workflow_status": "FAILED", "result_status": RESULT_UNRESOLVED,
                       "failure_code": "V9_2_PREFLIGHT_FAILED", "proof_route": PROOF_ROUTE,
                       "scope": SCOPE, "primary_claim": PRIMARY_CLAIM}
        else:
            compile_run = run_cli(
                "formal_toolchain.cli.compile_seed",
                ["--request", str(request), "--out", str(staging / "candidate"),
                 "--source-root", str(code_root)], cwd=code_root, log_dir=staging / "logs",
            )
            commands.append(compile_run)
            if compile_run["returncode"] != 0:
                summary = {"workflow_status": "FAILED", "result_status": RESULT_INVALID,
                           "failure_code": "V9_2_COMPILER_FAILED", "proof_route": PROOF_ROUTE,
                           "scope": SCOPE, "primary_claim": PRIMARY_CLAIM}
            else:
                verify_run = run_cli(
                    "formal_toolchain.cli.verify_bundle",
                    ["--request", str(request), "--bundle", str(staging / "candidate"),
                     "--out", str(staging / "verified"), "--source-root", str(code_root),
                     "--timeout-ms", str(int(solver_timeout_ms)),
                     "--max-boot-replay-ticks", str(int(max_boot_replay_ticks))],
                    cwd=code_root, log_dir=staging / "logs",
                )
                commands.append(verify_run)
                summary_path = staging / "verified/proof_summary.json"
                if not summary_path.is_file():
                    stderr_path = staging / "logs" / "verify_bundle.stderr.log"
                    stderr_text = stderr_path.read_text(encoding="utf-8", errors="replace") if stderr_path.is_file() else ""
                    tail = "\n".join(stderr_text.rstrip().splitlines()[-12:]) or None
                    summary = {"workflow_status": "FAILED", "result_status": RESULT_UNRESOLVED,
                               "failure_code": "V9_2_VERIFIER_PROCESS_FAILED", "proof_route": PROOF_ROUTE,
                               "scope": SCOPE, "primary_claim": PRIMARY_CLAIM,
                               "failure_message": tail}
                else:
                    summary = json.loads(summary_path.read_text(encoding="utf-8"))
                    report_run = run_cli(
                        "formal_toolchain.cli.render_report",
                        ["--verified", str(staging / "verified"), "--out", str(staging / "human_readable_report.md")],
                        cwd=code_root, log_dir=staging / "logs",
                    )
                    commands.append(report_run)

        _write(staging / "workflow_manifest.json", {
            "schema_version": "v9_2_workflow_manifest_v1", "proof_route": PROOF_ROUTE,
            "solver_timeout_ms": int(solver_timeout_ms),
            "max_boot_replay_ticks": int(max_boot_replay_ticks),
            "commands": commands,
        })
        result_status = str(summary.get("result_status", RESULT_INVALID))
        proof_result = {
            "workflow_schema_version": "prove_seed_v9_2_workflow_v1",
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
            "verified_summary": "verified/proof_summary.json" if (staging / "verified/proof_summary.json").is_file() else None,
            "exit_code": EXIT_CODES.get(result_status, 30),
        }
        _write(staging / "proof_result.json", proof_result)
        if not (staging / "human_readable_report.md").is_file():
            (staging / "human_readable_report.md").write_text(
                f"# V9.2 Formal Proof Report\n\n- result_status: `{result_status}`\n- failure_code: `{summary.get('failure_code')}`\n",
                encoding="utf-8",
            )
        publish_mode = _publish_staging(staging, out)
        proof_result["publish_mode"] = publish_mode
        _write(out / "proof_result.json", proof_result)
        return int(proof_result["exit_code"]), proof_result
    except (FormalWorkflowError, OSError, ValueError, KeyError) as exc:
        route = exc.route if isinstance(exc, FormalWorkflowError) else RESULT_UNRESOLVED
        code = exc.code if isinstance(exc, FormalWorkflowError) else "V9_2_WORKFLOW_INPUT_ERROR"
        failure = {
            "workflow_schema_version": "prove_seed_v9_2_workflow_v1",
            "proof_route": PROOF_ROUTE, "scope": SCOPE, "primary_claim": PRIMARY_CLAIM,
            "workflow_status": "FAILED", "result_status": route,
            "failure_code": code, "failure_message": str(exc),
            "exit_code": EXIT_CODES.get(route, 20),
        }
        if "staging" in locals():
            staging.mkdir(parents=True, exist_ok=True)
            _write(staging / "proof_result.json", failure)
            if not out.exists():
                try:
                    _publish_staging(staging, out)
                except OSError:
                    # Keep staging intact as the recoverable source of truth.
                    pass
        return int(failure["exit_code"]), failure
    finally:
        os.close(fd)
        try:
            lock.unlink()
        except FileNotFoundError:
            pass
