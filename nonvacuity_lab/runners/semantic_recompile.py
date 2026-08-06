from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from ..mutators import (
    ActionConfigMutation,
    ActionStepMutation,
    DangerousTop1Mutation,
    EnvelopeMutation,
    JsonPatchMutation,
    MultiPythonSymbolMutation,
    CoherentSourcePatchMutation,
    PythonSymbolMutation,
)
from ..mutators.retroactive_release_budget import RetroactiveReleaseBudgetMutation
from ..mutators.base import MutationContext
from ..schema import MutationManifest
from ..subprocess_runner import ordinary_prove_command, run_command
from ..workspace import ExperimentWorkspace, write_command_receipt, write_input_snapshot
from .baseline import _read_proof_result, _verify_request_is_blind


def run_semantic_recompile(
    *,
    manifest: MutationManifest,
    workspace: ExperimentWorkspace,
    source_root: Path,
    timeout_seconds: int | None = None,
) -> dict[str, Any]:
    if manifest.seed_dir is None:
        raise ValueError("semantic mutation 缺少 seed_dir")
    kind = str(manifest.mutator.get("kind", ""))
    parameters = dict(manifest.mutator.get("parameters", {}))
    parameters.setdefault("tree_variant", manifest.tree_variant)
    source_overlay: Path | None = None
    if kind in {"python_symbol", "source_overlay", "coherent_source_patch", "retroactive_release_budget"}:
        source_overlay = workspace.create_source_overlay(source_root)
        if isinstance(parameters.get("patches"), list):
            parameters.setdefault(
                "diff_dir",
                str(workspace.diff_output.parent / "source_patches"),
            )
        else:
            parameters.setdefault(
                "diff_file",
                str(workspace.diff_output.with_suffix(".patch")),
            )
    mutator = _mutator_for_kind(kind, parameters)
    context = MutationContext(
        mutation_id=manifest.mutation_id,
        source_root=source_root,
        mutated_seed=workspace.mutated_seed,
        source_overlay=source_overlay,
        parameters=parameters,
    )
    mutation_result = mutator.apply(context)
    single_change = mutator.verify_single_change(mutation_result)
    workspace.diff_output.parent.mkdir(parents=True, exist_ok=True)
    workspace.diff_output.write_text(
        json.dumps(mutation_result.to_dict(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    if mutation_result.details:
        write_input_snapshot(workspace.coherence_output, {
            "schema_version": "nonvacuity_coherence_receipt_v1",
            **dict(mutation_result.details),
        })
    if single_change.status != "PASS":
        return {
            "schema_version": "semantic_recompile_run_v1",
            "setup_valid": False,
            "mutation_result": mutation_result.to_dict(),
            "single_change": dict(single_change.details),
            "proof_result": None,
        }
    proof_source = source_overlay or source_root
    argv = ordinary_prove_command(
        seed_dir=workspace.mutated_seed,
        tree_variant=manifest.tree_variant,
        source_root=proof_source,
        output_dir=workspace.semantic_output,
    )
    forbidden_args = [
        item for item in argv
        if item.startswith("--nonvacuity")
        or item.startswith("--mutation")
        or item.startswith("--expected")
    ]
    if forbidden_args:
        raise ValueError(f"PROOF_COMMAND_NOT_MUTATION_BLIND:{forbidden_args}")
    environment = {}
    if source_overlay is not None:
        # Do not retain the original checkout as an import fallback.  The
        # proof process must execute the copied overlay exclusively; otherwise
        # an omitted mirrored file could silently resolve from the clean tree.
        environment["PYTHONPATH"] = str(source_overlay)
    receipt = run_command(
        argv,
        cwd=proof_source,
        log_dir=workspace.semantic_output.parent / "logs",
        env=environment,
        timeout_seconds=timeout_seconds,
    )
    proof_result = _read_proof_result(workspace.semantic_output)
    diagnostic = {
        "schema_version": "semantic_recompile_diagnostic_v1",
        "mutation_id": manifest.mutation_id,
        "mutator_kind": kind,
        "seed_dir": str(workspace.mutated_seed),
        "source_root": str(proof_source),
        "proof_output": str(workspace.semantic_output),
        "returncode": int(receipt.get("returncode", -1)),
        "proof_result_status": proof_result.get("result_status"),
        "failure_route": proof_result.get("failure_route"),
        "failure_code": proof_result.get("failure_code"),
        "violated_obligation_id": proof_result.get("violated_obligation_id"),
        "failure_message": proof_result.get("failure_message"),
        "seed_import_error_log": str(workspace.semantic_output / "logs" / "seed_import_error.log"),
        "candidate_failure": str(workspace.semantic_output / "candidate" / "candidate_failure.json"),
    }
    write_input_snapshot(
        workspace.semantic_output.parent / "semantic_recompile_diagnostic.json",
        diagnostic,
    )
    write_command_receipt(
        workspace.command_output / "semantic_recompile.json",
        argv=argv,
        cwd=proof_source,
        env={str(key): str(value) for key, value in environment.items()},
        returncode=int(receipt.get("returncode", -1)),
    )
    request_guard = _verify_request_is_blind(
        workspace.semantic_output / "request" / "proof_request.json"
    )
    return {
        "schema_version": "semantic_recompile_run_v1",
        "setup_valid": request_guard["status"] == "PASS",
        "mutation_result": mutation_result.to_dict(),
        "single_change": dict(single_change.details),
        "receipt": receipt,
        "proof_result": proof_result,
        "request_guard": request_guard,
    }


def _mutator_for_kind(kind: str, parameters: dict[str, Any]):
    if kind in {"dangerous_top1", "tree_ranking"}:
        return DangerousTop1Mutation()
    if kind == "json_patch":
        return JsonPatchMutation()
    if kind == "action_config":
        return ActionConfigMutation()
    if kind == "action_step":
        return ActionStepMutation()
    if kind == "envelope":
        return EnvelopeMutation()
    if kind in {"python_symbol", "source_overlay"}:
        return (
            MultiPythonSymbolMutation()
            if isinstance(parameters.get("patches"), list)
            else PythonSymbolMutation()
        )
    if kind == "coherent_source_patch":
        return CoherentSourcePatchMutation()
    if kind == "retroactive_release_budget":
        return RetroactiveReleaseBudgetMutation()
    raise ValueError(f"不支持的 semantic mutator kind: {kind}")
