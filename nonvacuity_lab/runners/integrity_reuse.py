from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any

from ..mutators.bundle_tamper import BundleTamperMutation
from ..mutators.base import MutationContext
from ..schema import MutationManifest
from ..subprocess_runner import ordinary_verify_command, run_command
from ..workspace import ExperimentWorkspace


def run_integrity_reuse(
    *,
    manifest: MutationManifest,
    workspace: ExperimentWorkspace,
    source_root: Path,
    timeout_seconds: int | None = None,
) -> dict[str, Any]:
    if manifest.reuse_source_bundle is None:
        raise ValueError("integrity mutation 缺少 reuse_source_bundle")
    proof_root = manifest.reuse_source_bundle
    request_source, bundle_source = _resolve_proof_inputs(proof_root)
    tampered = workspace.root / "integrity_reuse" / "tampered_inputs"
    tampered.mkdir(parents=True)
    shutil.copytree(request_source.parent, tampered / "request", symlinks=False)
    shutil.copytree(bundle_source, tampered / "candidate", symlinks=False)
    source_overlay = None
    parameters = dict(
        manifest.mutator.get(
            "integrity_parameters",
            manifest.mutator.get("parameters", {}),
        )
    )
    if isinstance(parameters.get("source_file"), str):
        parameters["source_file"] = parameters["source_file"].format(
            mutated_seed=str(workspace.mutated_seed),
            source_root=str(Path(source_root).resolve()),
        )
    if parameters.get("tamper_kind") == "source_file":
        source_overlay = workspace.create_source_overlay(
            source_root,
            destination=workspace.root / "integrity_reuse" / "source_overlay",
        )
        parameters["workspace_root"] = str(source_overlay)
    else:
        parameters["workspace_root"] = str(tampered)
    mutator = BundleTamperMutation()
    context = MutationContext(
        mutation_id=manifest.mutation_id,
        source_root=source_root,
        mutated_seed=workspace.mutated_seed,
        source_overlay=source_overlay,
        parameters=parameters,
    )
    mutation_result = mutator.apply(context)
    single_change = mutator.verify_single_change(mutation_result)
    (workspace.root / "integrity_reuse" / "mutation_diff.json").write_text(
        json.dumps(mutation_result.to_dict(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    verify_source = source_overlay or source_root
    environment = {}
    if source_overlay is not None:
        # Integrity verification must execute one coherent source tree; a
        # clean-tree fallback could silently mix patched and unpatched code.
        environment["PYTHONPATH"] = str(source_overlay)
    receipt = run_command(
        ordinary_verify_command(
            request=tampered / "request" / request_source.name,
            bundle=tampered / "candidate",
            output_dir=workspace.integrity_output,
            source_root=verify_source,
        ),
        cwd=verify_source,
        log_dir=workspace.root / "integrity_reuse" / "logs",
        env=environment,
        timeout_seconds=timeout_seconds,
    )
    proof_result = _read_integrity_result(workspace.integrity_output, receipt)
    (workspace.root / "integrity_reuse" / "original_bundle_ref.json").write_text(
        json.dumps(
            {
                "request": str(request_source),
                "bundle": str(bundle_source),
                "recompiled": False,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return {
        "schema_version": "integrity_reuse_run_v1",
        "setup_valid": single_change.status == "PASS",
        "recompiled": False,
        "mutation_result": mutation_result.to_dict(),
        "receipt": receipt,
        "proof_result": proof_result,
    }


def _resolve_proof_inputs(root: Path) -> tuple[Path, Path]:
    root = Path(root).resolve()
    request_candidates = (
        root / "request" / "proof_request.json",
        root / "proof_request.json",
    )
    request = next((path for path in request_candidates if path.is_file()), None)
    bundle_candidates = (root / "candidate", root)
    bundle = next(
        (
            path
            for path in bundle_candidates
            if path.is_dir()
            and any(path.glob("*.json"))
        ),
        None,
    )
    if request is None or bundle is None:
        raise ValueError(f"reuse_source_bundle 不含 request/candidate: {root}")
    return request, bundle


def _read_integrity_result(output: Path, receipt: dict[str, object]) -> dict[str, Any]:
    summary = output / "proof_summary.json"
    if summary.is_file():
        raw = json.loads(summary.read_text(encoding="utf-8"))
        if isinstance(raw, dict) and isinstance(raw.get("result_status"), str):
            return dict(raw)
        return _execution_failure(
            "VERIFIER_OUTPUT_MISSING",
            "proof_summary.json 缺少 canonical result_status",
        )
    stdout = Path(str(receipt["stdout"]))
    for line in reversed(stdout.read_text(encoding="utf-8").splitlines()):
        try:
            raw = json.loads(line)
        except ValueError:
            continue
        if isinstance(raw, dict) and isinstance(raw.get("result_status"), str):
            return raw
    if receipt.get("timed_out"):
        return _execution_failure("VERIFIER_TIMEOUT", "verifier process timed out")
    if receipt.get("traceback_marker"):
        return _execution_failure("TOOL_EXECUTION_FAILED", "verifier emitted Python traceback")
    return _execution_failure(
        "VERIFIER_OUTPUT_MISSING",
        "verifier 未生成 canonical summary/claim",
    )


def _execution_failure(status: str, reason: str) -> dict[str, Any]:
    return {
        "workflow_status": "FAILED",
        "result_status": status,
        "failure_code": status,
        "reason": reason,
    }
