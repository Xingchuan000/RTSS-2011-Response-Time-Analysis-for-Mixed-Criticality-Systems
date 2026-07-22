"""Phase L/M 顶层单命令工作流。"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any

from formal_toolchain.workflow.seed_workspace import freeze_seed_workspace
from formal_toolchain.workflow.subprocess_runner import run_cli
from formal_toolchain.core.errors import FormalWorkflowError


EXIT_CODES = {"DEPLOYED_TREE_PROVED": 0, "MODEL_CONFORMANCE_FAILED": 10,
              "POLICY_CONTRACT_VIOLATION": 11, "REFERENCE_CERTIFICATE_FAILED": 12,
              "CONCRETE_TIMING_COUNTEREXAMPLE": 13, "REFERENCE_COUNTEREXAMPLE": 14,
              "UNRESOLVED": 20, "PROOF_BUNDLE_INVALID": 30}


def _write(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _dependency_preflight(source_root: Path) -> dict[str, Any] | None:
    """前置依赖检查：lock 文件中的精确版本缺失时直接拒绝。"""
    lock_path = source_root / "formal_toolchain" / "specs" / "proof_dependency_lock.json"
    if not lock_path.is_file():
        return {"code": "DEPENDENCY_LOCK_FILE_MISSING", "message": f"lock file not found: {lock_path}"}
    import json
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    from formal_toolchain.adapters.runtime_manifest import build_dependency_manifest, check_dependency_policy
    manifest = build_dependency_manifest()
    result = check_dependency_policy(manifest, lock=lock)
    if result.get("status") != "PASS":
        return result
    return None


def prove_seed(*, seed_dir: Path, tree_variant: str, code_root: Path, out: Path,
               target_recipe: Path | None = None, overwrite: bool = False,
               nonvacuity_profile: str = "off",
               nonvacuity_params: dict[str, Any] | None = None,
               refresh_phase_k_map: bool = False,
               dependency_manifest_override: dict[str, Any] | None = None) -> tuple[int, dict[str, Any]]:
    """执行 discovery→preflight→compile→fresh verify→report。

    dependency_manifest_override: 测试用注入 dependency manifest，绕过真实环境检查。
    """
    if dependency_manifest_override is None:
        preflight = _dependency_preflight(code_root)
        if preflight is not None:
            return 30, {"workflow_status": "FAILED", "result_status": "PROOF_BUNDLE_INVALID",
                         "failure_route": "PROOF_BUNDLE_INVALID",
                         "failure_code": preflight.get("code", "DEPENDENCY_LOCK_INCOMPLETE"),
                         "failure_message": str(preflight.get("message", preflight))}
    else:
        from formal_toolchain.adapters.runtime_manifest import check_dependency_policy
        lock_path = code_root / "formal_toolchain" / "specs" / "proof_dependency_lock.json"
        import json
        lock = json.loads(lock_path.read_text(encoding="utf-8")) if lock_path.is_file() else None
        result = check_dependency_policy(dependency_manifest_override, lock=lock)
        if result.get("status") != "PASS":
            return 30, {"workflow_status": "FAILED", "result_status": "PROOF_BUNDLE_INVALID",
                         "failure_route": "PROOF_BUNDLE_INVALID",
                         "failure_code": result.get("code", "DEPENDENCY_LOCK_INCOMPLETE"),
                         "failure_message": str(result)}

    out = Path(out).resolve()
    lock = out.parent / f".{out.name}.lock"
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        return 2, {"workflow_status": "FAILED", "result_status": "PROOF_BUNDLE_INVALID",
                   "failure_code": "WORKSPACE_LOCKED"}
    try:
        if out.exists():
            if not overwrite:
                return 2, {"workflow_status": "FAILED", "result_status": "PROOF_BUNDLE_INVALID",
                           "failure_code": "OUTPUT_EXISTS"}
            shutil.rmtree(out)
        staging = out.parent / f".{out.name}.staging"
        if staging.exists():
            shutil.rmtree(staging)
        imported = freeze_seed_workspace(seed_dir, tree_variant, staging,
                                         code_root=code_root, target_recipe=target_recipe,
                                         overwrite=False,
                                         nonvacuity_profile=nonvacuity_profile,
                                         nonvacuity_params=nonvacuity_params,
                                         refresh_phase_k_map=refresh_phase_k_map)
        request = Path(imported["request"])
        manifest: dict[str, Any] = {"schema_version": "workflow_manifest_v1", "commands": []}
        inspect = run_cli("formal_toolchain.cli.inspect_target", ["--request", str(request), "--out", str(staging / "preflight")], cwd=Path(code_root), log_dir=staging / "logs")
        manifest["commands"].append(inspect)
        if inspect["returncode"] != 0:
            summary = {"workflow_status": "FAILED", "result_status": "MODEL_CONFORMANCE_FAILED",
                       "failure_route": "MODEL_CONFORMANCE_FAILED", "failure_code": "PREFLIGHT_FAILED",
                       "exit_code": 10}
        else:
            compile_out = staging / "candidate"
            compile_run = run_cli("formal_toolchain.cli.compile_seed", ["--request", str(request), "--out", str(compile_out), "--source-root", str(code_root.resolve())], cwd=Path(code_root), log_dir=staging / "logs")
            manifest["commands"].append(compile_run)
            if compile_run["returncode"] != 0:
                summary = {"workflow_status": "FAILED", "result_status": "PROOF_BUNDLE_INVALID",
                           "failure_route": "PROOF_BUNDLE_INVALID", "failure_code": "COMPILER_FAILED",
                           "exit_code": 30}
            else:
                verify_out = staging / "verified"
                verify_run = run_cli("formal_toolchain.cli.verify_bundle", ["--request", str(request), "--bundle", str(compile_out), "--out", str(verify_out), "--source-root", str(code_root.resolve())], cwd=Path(code_root), log_dir=staging / "logs")
                manifest["commands"].append(verify_run)
                if not (verify_out / "proof_summary.json").is_file():
                    summary = {"workflow_status": "FAILED", "result_status": "PROOF_BUNDLE_INVALID",
                               "failure_route": "PROOF_BUNDLE_INVALID", "failure_code": "VERIFIER_SUMMARY_MISSING",
                               "exit_code": 30}
                else:
                    summary = json.loads((verify_out / "proof_summary.json").read_text(encoding="utf-8"))
                    report_run = run_cli("formal_toolchain.cli.render_report", ["--verified", str(verify_out), "--out", str(staging / "human_readable_report.md")], cwd=Path(code_root), log_dir=staging / "logs")
                    manifest["commands"].append(report_run)
                    if report_run["returncode"] != 0:
                        (staging / "logs" / "internal_error.log").write_text("report renderer failed\n", encoding="utf-8")
                        summary = {**summary, "workflow_status": "FAILED", "internal_error": "REPORT_RENDER_FAILED"}
        _write(staging / "workflow_manifest.json", manifest)
        final_status = str(summary.get("result_status", "PROOF_BUNDLE_INVALID"))
        request_data = json.loads(request.read_text(encoding="utf-8"))
        target_kind = request_data.get("target_kind")
        proof_result = {"workflow_schema_version": "prove_seed_workflow_v1",
                        "taskset_seed": request_data.get("taskset_seed"),
                        "target_id": request_data.get("target_id"),
                        "target_kind": target_kind,
                        "tree_variant": tree_variant, "profile": "P0",
                        "nonvacuity_profile": nonvacuity_profile,
                        "nonvacuity_params": dict(nonvacuity_params or {}),
                        "phase_k_map_refreshed": bool(refresh_phase_k_map),
                        "primary_claim": "DEPLOYED_HI_SAFETY",
                        "workflow_status": summary.get("workflow_status", "FAILED"),
                        "result_status": final_status,
                        "failure_route": summary.get("failure_route"),
                        "failure_code": summary.get("failure_code"),
                        "violated_obligation_id": summary.get("violated_obligation_id"),
                        "failure_message": summary.get("failure_message"),
                        "verified_summary": "verified/proof_summary.json" if (staging / "verified/proof_summary.json").is_file() else None,
                        "outer_bundle_root": summary.get("outer_bundle_root"),
                        "fixture_claim_result": summary.get("fixture_claim_result", final_status),
                        "fixture_id": summary.get("fixture_id", summary.get("target_id")),
                        "fixture_kind": summary.get("fixture_kind", summary.get("target_kind")),
                        "real_seed_evaluation": "DEFERRED" if target_kind == "SYNTHETIC_P0"
                        else "COMPLETED" if target_kind is not None else "UNRESOLVED",
                        "exit_code": 70 if summary.get("internal_error") else EXIT_CODES.get(final_status, 70)}
        _write(staging / "proof_result.json", proof_result)
        if not (staging / "human_readable_report.md").is_file():
            (staging / "human_readable_report.md").write_text(
                f"# Formal proof report\n\n- result_status: `{final_status}`\n", encoding="utf-8")
        staging.rename(out)
        return int(proof_result["exit_code"]), proof_result
    except Exception as exc:
        # 输入合同错误不是内部异常：保留一个最小 staging workspace，写出
        # 可诊断的 fail-closed result；真正未映射异常仍使用 70 并保留 traceback
        # 位置，不能被伪装成 UNRESOLVED。
        if isinstance(exc, FileExistsError):
            code, result_status, failure_code = 2, "PROOF_BUNDLE_INVALID", "OUTPUT_EXISTS"
        elif isinstance(exc, FormalWorkflowError):
            code, result_status, failure_code = exc.exit_code, exc.route, exc.code
        elif isinstance(exc, ValueError):
            code, result_status, failure_code = 20, "UNRESOLVED", "SEED_IMPORT_OR_PREFLIGHT_FAILED"
        else:
            code, result_status, failure_code = 70, "PROOF_BUNDLE_INVALID", "INTERNAL_WORKFLOW_ERROR"
        if "staging" not in locals():
            staging = out.parent / f".{out.name}.staging"
        if not staging.exists():
            staging.mkdir(parents=True, exist_ok=True)
        (staging / "logs").mkdir(exist_ok=True)
        (staging / "logs" / ("internal_error.log" if code == 70 else "seed_import_error.log")).write_text(str(exc), encoding="utf-8")
        failure = {"workflow_schema_version": "prove_seed_workflow_v1",
                   "profile": "P0", "primary_claim": "DEPLOYED_HI_SAFETY",
                   "nonvacuity_profile": nonvacuity_profile,
                   "nonvacuity_params": dict(nonvacuity_params or {}),
                   "workflow_status": "FAILED", "result_status": result_status,
                   "failure_route": result_status, "failure_code": failure_code,
                   "failure_message": str(exc), "verified_summary": None,
                   "outer_bundle_root": None, "exit_code": code}
        _write(staging / "proof_result.json", failure)
        (staging / "human_readable_report.md").write_text(
            f"# Formal proof report\n\n- result_status: `{result_status}`\n- failure_code: `{failure_code}`\n",
            encoding="utf-8")
        if not out.exists():
            staging.rename(out)
        return code, failure
    finally:
        os.close(fd)
        try:
            lock.unlink()
        except FileNotFoundError:
            pass
