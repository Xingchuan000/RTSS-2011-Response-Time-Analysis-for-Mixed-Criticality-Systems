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
    PythonSymbolMutation,
)
from ..mutators.base import MutationContext
from ..schema import MutationManifest
from ..subprocess_runner import ordinary_prove_command, run_command
from ..workspace import ExperimentWorkspace
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
    if kind in {"python_symbol", "source_overlay"}:
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
    if single_change.status != "PASS":
        return {
            "schema_version": "semantic_recompile_run_v1",
            "setup_valid": False,
            "mutation_result": mutation_result.to_dict(),
            "single_change": dict(single_change.details),
            "proof_result": None,
        }
    proof_source = source_overlay or source_root
    environment = {}
    if source_overlay is not None:
        environment["PYTHONPATH"] = os.pathsep.join(
            [str(source_overlay), str(Path(source_root).resolve())]
        )
    receipt = run_command(
        ordinary_prove_command(
            seed_dir=workspace.mutated_seed,
            tree_variant=manifest.tree_variant,
            source_root=proof_source,
            output_dir=workspace.semantic_output,
        ),
        cwd=proof_source,
        log_dir=workspace.semantic_output.parent / "logs",
        env=environment,
        timeout_seconds=timeout_seconds,
    )
    proof_result = _read_proof_result(workspace.semantic_output)
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
    raise ValueError(f"不支持的 semantic mutator kind: {kind}")
