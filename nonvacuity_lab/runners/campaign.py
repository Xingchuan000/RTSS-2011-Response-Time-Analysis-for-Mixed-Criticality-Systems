from __future__ import annotations

import json
import platform
from pathlib import Path
import sys
from typing import Any

from ..activation import evaluate_hout_activation, solve_symbolic_activation
from ..activation.hout_activation import load_events
from ..analysis.comparison import compare_proofs
from ..analysis.expectations import classify_experiment
from ..analysis.rta_slack import scan_rta_slack, select_minimum_slack
from ..canonical import canonical_json_hash
from ..manifest import write_resolved_manifest
from ..preflight import audit_campaign, audit_mutation
from ..reporting.json_report import write_json
from ..reporting.markdown_report import write_campaign_report, write_experiment_report
from ..schema import (
    CampaignConfig,
    ExperimentStatus,
    MutationClass,
    MutationManifest,
    experiment_envelope,
)
from ..workspace import (
    ExperimentWorkspace,
    verify_original_inputs_unchanged,
    write_input_snapshot,
)
from .baseline import run_baseline
from .integrity_reuse import run_integrity_reuse
from .paired_hout import run_paired_hout
from .semantic_recompile import run_semantic_recompile


def run_campaign(
    config: CampaignConfig,
    *,
    enabled_by_cli: bool,
    timeout_seconds: int | None = None,
) -> dict[str, Any]:
    if not config.enabled or not enabled_by_cli:
        return experiment_envelope(
            campaign_id=config.campaign_id,
            status=ExperimentStatus.EXPERIMENT_DISABLED.value,
            mutation_results=[],
        )
    preflight = audit_campaign(config)
    if preflight["status"] != "PASS":
        return experiment_envelope(
            campaign_id=config.campaign_id,
            status=ExperimentStatus.CAMPAIGN_PREFLIGHT_FAILED.value,
            preflight=preflight,
            mutation_results=[],
        )
    campaign_dir = config.output_root / config.campaign_id
    campaign_dir.mkdir(parents=True, exist_ok=True)
    results = []
    paired_hashes: dict[str, str] = {}
    paired_seed_dirs: dict[str, Path] = {}
    for manifest in config.mutations:
        if not manifest.enabled:
            results.append(
                experiment_envelope(
                    campaign_id=config.campaign_id,
                    mutation_id=manifest.mutation_id,
                    status=ExperimentStatus.EXPERIMENT_DISABLED.value,
                )
            )
            continue
        effective_manifest = manifest
        if (
            manifest.mutation_class is MutationClass.ENVELOPE
            and manifest.metadata.get("dynamic_minimum_slack_selection")
        ):
            try:
                effective_manifest = _resolve_dynamic_envelope_manifest(
                    manifest,
                    source_root=config.source_root,
                )
            except (OSError, ValueError, KeyError, TypeError) as exc:
                results.append(
                    experiment_envelope(
                        campaign_id=config.campaign_id,
                        mutation_id=manifest.mutation_id,
                        status=ExperimentStatus.SETUP_INVALID.value,
                        reason=str(exc),
                    )
                )
                continue
        if manifest.paired_with:
            paired_seed = paired_seed_dirs.get(manifest.paired_with)
            if paired_seed is None:
                result = experiment_envelope(
                    campaign_id=config.campaign_id,
                    mutation_id=manifest.mutation_id,
                    status=ExperimentStatus.SETUP_INVALID.value,
                    reason=f"paired mutation 尚无可用工作区: {manifest.paired_with}",
                )
                results.append(result)
                continue
            from dataclasses import replace

            effective_manifest = replace(manifest, seed_dir=paired_seed)
        result = run_one(
            effective_manifest,
            campaign_id=config.campaign_id,
            output_root=config.output_root,
            source_root=config.source_root,
            enabled_by_cli=enabled_by_cli,
            run_baseline_first=config.run_baselines,
            run_semantic=config.run_semantic_recompile,
            run_integrity=config.run_integrity_reuse,
            run_hout=config.run_hout,
            timeout_seconds=timeout_seconds,
            paired_hashes=paired_hashes,
        )
        results.append(result)
        mutation = result.get("semantic_recompile", {}).get("mutation_result", {})
        after_hash = mutation.get("after_hash")
        if after_hash:
            paired_hashes[manifest.mutation_id] = str(after_hash)
            paired_seed_dirs[manifest.mutation_id] = (
                config.output_root
                / config.campaign_id
                / manifest.mutation_id
                / "semantic_recompile"
                / "mutated_seed"
            )
    summary = _campaign_summary(
        results,
        fail_on_not_activated=config.fail_on_not_activated,
    )
    envelope = experiment_envelope(
        campaign_id=config.campaign_id,
        status=summary["status"],
        mutation_results=results,
        summary=summary,
        preflight=preflight,
    )
    (campaign_dir / "campaign_result.json").write_text(
        json.dumps(envelope, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    write_campaign_report(campaign_dir / "report.md", envelope)
    return envelope


def run_one(
    manifest: MutationManifest,
    *,
    campaign_id: str,
    output_root: Path,
    source_root: Path,
    enabled_by_cli: bool,
    run_baseline_first: bool = True,
    run_semantic: bool = True,
    run_integrity: bool = True,
    run_hout: bool = True,
    timeout_seconds: int | None = None,
    paired_hashes: dict[str, str] | None = None,
) -> dict[str, Any]:
    if not manifest.enabled or not enabled_by_cli:
        return experiment_envelope(
            campaign_id=campaign_id,
            mutation_id=manifest.mutation_id,
            status=ExperimentStatus.EXPERIMENT_DISABLED.value,
        )
    preflight = audit_mutation(manifest, source_root=source_root)
    if preflight["status"] != "PASS":
        return experiment_envelope(
            campaign_id=campaign_id,
            mutation_id=manifest.mutation_id,
            status=ExperimentStatus.CAMPAIGN_PREFLIGHT_FAILED.value,
            preflight=preflight,
        )
    if (
        manifest.mutation_class is not MutationClass.BUNDLE_INTEGRITY
        and (manifest.seed_dir is None or not manifest.seed_dir.is_dir())
    ):
        return experiment_envelope(
            campaign_id=campaign_id,
            mutation_id=manifest.mutation_id,
            status=ExperimentStatus.SETUP_INVALID.value,
            reason=f"seed_dir 不存在: {manifest.seed_dir}",
        )
    workspace, hashes_before = ExperimentWorkspace.create(
        output_root=output_root,
        campaign_id=campaign_id,
        mutation_id=manifest.mutation_id,
        seed_dir=manifest.seed_dir,
        source_root=source_root,
    )
    pair_error = None
    if manifest.paired_with:
        expected_tree_hash = (paired_hashes or {}).get(manifest.paired_with)
        actual_tree_hash = _tree_semantic_hash(
            workspace.mutated_seed / manifest.tree_variant / "integer_tree.json"
        )
        if expected_tree_hash is None or actual_tree_hash != expected_tree_hash:
            pair_error = "paired mutation 未使用 A1 的同一 after-tree hash"
    write_resolved_manifest(workspace.root / "manifest_resolved.json", manifest)
    environment = {
        "python": sys.version,
        "platform": platform.platform(),
        "source_root": str(source_root),
        "artifact_class": "NONVACUITY_EXPERIMENT_ONLY",
        "deployment_certificate_eligible": False,
    }
    write_input_snapshot(workspace.root / "environment.json", environment)
    write_input_snapshot(
        workspace.base_snapshot / "input_snapshot.json",
        {
            "schema_version": "nonvacuity_input_snapshot_v1",
            "original_seed": str(manifest.seed_dir) if manifest.seed_dir else None,
            "copied_seed": (
                str(workspace.base_snapshot / "seed")
                if manifest.seed_dir is not None
                else None
            ),
            "tree_variant": manifest.tree_variant,
            "source_root": str(source_root),
            "hashes_before": hashes_before,
        },
    )
    baseline = None
    semantic = None
    integrity = None
    hout = None
    activation = None
    setup_valid = True
    try:
        if pair_error is not None:
            raise ValueError(pair_error)
        if run_baseline_first and manifest.seed_dir is not None:
            baseline = run_baseline(
                seed_dir=workspace.base_snapshot / "seed",
                tree_variant=manifest.tree_variant,
                source_root=source_root,
                output_dir=workspace.base_snapshot,
                timeout_seconds=timeout_seconds,
            )
        if (
            manifest.mutation_class
            not in {MutationClass.BUNDLE_INTEGRITY, MutationClass.BASELINE}
            and run_semantic
        ):
            semantic = run_semantic_recompile(
                manifest=manifest,
                workspace=workspace,
                source_root=source_root,
                timeout_seconds=timeout_seconds,
            )
            setup_valid = setup_valid and bool(semantic.get("setup_valid"))
        if run_hout and manifest.metadata.get("hout"):
            hout = run_paired_hout(
                config=manifest.metadata["hout"],
                workspace_root=workspace.hout_output,
                command_context={
                    "source_root": str(source_root),
                    "base_source_root": str(source_root),
                    "mutated_source_root": str(
                        workspace.source_overlay
                        if workspace.source_overlay.is_dir()
                        else source_root
                    ),
                    "base_seed": str(workspace.base_snapshot / "seed"),
                    "mutated_seed": str(workspace.mutated_seed),
                    "tree_variant": manifest.tree_variant,
                },
                timeout_seconds=timeout_seconds,
            )
        activation = _run_activation(manifest, workspace, hout)
        if (
            run_integrity
            and (
                manifest.mutation_class is MutationClass.BUNDLE_INTEGRITY
                or manifest.reuse_source_bundle is not None
            )
        ):
            integrity = run_integrity_reuse(
                manifest=manifest,
                workspace=workspace,
                source_root=source_root,
                timeout_seconds=timeout_seconds,
            )
        proof_result = (
            integrity.get("proof_result")
            if manifest.mutation_class is MutationClass.BUNDLE_INTEGRITY and integrity
            else (baseline or {}).get("proof_result")
            if manifest.mutation_class is MutationClass.BASELINE
            else (semantic or {}).get("proof_result")
        )
        expectation = classify_experiment(
            expected=manifest.expected,
            proof_result=proof_result,
            activation_result=activation,
            setup_valid=setup_valid,
            baseline_passed=(
                baseline is None
                or bool(baseline.get("passed"))
            ),
            integrity=manifest.mutation_class is MutationClass.BUNDLE_INTEGRITY,
        )
    except (OSError, ValueError, KeyError, TypeError) as exc:
        setup_valid = False
        expectation = {
            "schema_version": "expectation_check_v1",
            "status": ExperimentStatus.SETUP_INVALID.value,
            "reason": str(exc),
        }
    input_integrity = verify_original_inputs_unchanged(
        hashes_before=hashes_before,
        source_root=source_root,
        seed_dir=manifest.seed_dir,
    )
    if input_integrity["status"] != "PASS":
        expectation = {
            "schema_version": "expectation_check_v1",
            "status": ExperimentStatus.SETUP_INVALID.value,
            "reason": "original seed/source changed during experiment",
        }
    comparison = compare_proofs(
        (baseline or {}).get("proof_result"),
        (semantic or {}).get("proof_result"),
    )
    write_json(workspace.comparison_output / "proof_comparison.json", comparison)
    write_json(workspace.comparison_output / "expectation_check.json", expectation)
    if hout and isinstance(hout.get("comparison"), dict):
        write_json(
            workspace.comparison_output / "hout_comparison.json",
            hout["comparison"],
        )
    result = experiment_envelope(
        campaign_id=campaign_id,
        mutation_id=manifest.mutation_id,
        mutation_class=manifest.mutation_class.value,
        status=expectation["status"],
        input_integrity=input_integrity,
        baseline=baseline,
        activation=activation,
        semantic_recompile=semantic,
        integrity_reuse=integrity,
        hout=hout,
        comparison=comparison,
        expectation_check=expectation,
        preflight=preflight,
    )
    (workspace.root / "experiment_result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    write_experiment_report(workspace.report_output / "report.md", result)
    return result


def _run_activation(
    manifest: MutationManifest,
    workspace: ExperimentWorkspace,
    hout: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if manifest.mutation_class in {MutationClass.BASELINE, MutationClass.BUNDLE_INTEGRITY}:
        return None
    mode = str(manifest.activation.get("mode", "symbolic")).lower()
    results = []
    if "symbolic" in mode:
        result = solve_symbolic_activation(
            mutation_id=manifest.mutation_id,
            rule=manifest.activation,
            output_path=workspace.activation_output / "symbolic_witness.json",
        )
        results.append(result)
    if "hout" in mode and hout and hout.get("setup_valid"):
        result = evaluate_hout_activation(
            mutation_id=manifest.mutation_id,
            base_events=load_events(Path(hout["base_events"])),
            mutated_events=load_events(Path(hout["mutated_events"])),
            rule=manifest.activation,
        )
        results.append(result)
    if not results:
        from ..activation.schema import ActivationResult
        from ..schema import ActivationStatus

        results.append(
            ActivationResult(
                mutation_id=manifest.mutation_id,
                status=ActivationStatus.ACTIVATION_INCONCLUSIVE,
                details={"reason": "requested activation evidence unavailable"},
            )
        )
    activated = next((item for item in results if item.status.value == "ACTIVATED"), None)
    selected = activated or results[0]
    payload = selected.to_dict()
    payload["all_evidence"] = [item.to_dict() for item in results]
    (workspace.activation_output / "activation_result.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return payload


def _tree_semantic_hash(path: Path) -> str:
    raw = json.loads(path.read_text(encoding="utf-8"))
    return canonical_json_hash(raw)


def _resolve_dynamic_envelope_manifest(
    manifest: MutationManifest,
    *,
    source_root: Path,
) -> MutationManifest:
    from dataclasses import replace

    roots_raw = manifest.metadata.get("bundle_roots", ("outputs/formal_baselines",))
    if not isinstance(roots_raw, (list, tuple)) or not roots_raw:
        raise ValueError("D1 bundle_roots 必须为非空数组")
    roots = [
        (Path(str(item)) if Path(str(item)).is_absolute() else source_root / str(item)).resolve()
        for item in roots_raw
    ]
    selected = select_minimum_slack(scan_rta_slack(roots))
    seed = int(selected["seed"])
    if seed < 0:
        raise ValueError("D1 minimum-slack artifact 缺少 seed identity")
    seed_dir = (source_root / f"s{seed}").resolve()
    variants = {
        "compact": "best_overall",
        "balanced": "best_balanced",
        "best_overall": "best_overall",
        "best_balanced": "best_balanced",
        "best_performance": "best_performance",
    }
    variant = variants.get(str(selected.get("variant")))
    if variant is None:
        raise ValueError(f"D1 artifact variant 无法映射: {selected.get('variant')}")
    parameters = dict(manifest.mutator.get("parameters", {}))
    target_file = selected.get("envelope_target_file") or parameters.get("target_file")
    pointer = selected.get("envelope_json_pointer")
    if not target_file or not pointer:
        raise ValueError("D1 limiting artifact 缺少 envelope target/pointer adapter 字段")
    parameters["target_file"] = str(target_file)
    parameters["json_pointer"] = str(pointer)
    metadata = {
        **dict(manifest.metadata),
        "dynamic_selection": selected,
    }
    return replace(
        manifest,
        seed_dir=seed_dir,
        base_seed=seed,
        tree_variant=variant,
        mutator={**dict(manifest.mutator), "parameters": parameters},
        metadata=metadata,
    )


def _campaign_summary(
    results: list[dict[str, Any]],
    *,
    fail_on_not_activated: bool,
) -> dict[str, Any]:
    statuses: dict[str, int] = {}
    for result in results:
        status = str(result.get("status"))
        statuses[status] = statuses.get(status, 0) + 1
    excluded = {
        ExperimentStatus.NOT_ACTIVATED.value,
        ExperimentStatus.SETUP_INVALID.value,
        ExperimentStatus.TOOL_EXECUTION_FAILED.value,
        ExperimentStatus.VERIFIER_TIMEOUT.value,
        ExperimentStatus.VERIFIER_OUTPUT_MISSING.value,
        ExperimentStatus.BASELINE_REGRESSION.value,
        ExperimentStatus.EXPERIMENT_DISABLED.value,
    }
    eligible = [item for item in results if item.get("status") not in excluded]
    killed = [
        item
        for item in eligible
        if item.get("status")
        in {
            ExperimentStatus.FAIL_EXPECTED.value,
            ExperimentStatus.INTEGRITY_REJECTION_EXPECTED.value,
        }
    ]
    invalid_statuses = {
        ExperimentStatus.SETUP_INVALID.value,
        ExperimentStatus.TOOL_EXECUTION_FAILED.value,
        ExperimentStatus.VERIFIER_TIMEOUT.value,
        ExperimentStatus.VERIFIER_OUTPUT_MISSING.value,
        ExperimentStatus.BASELINE_REGRESSION.value,
        ExperimentStatus.UNEXPECTED_FAIL.value,
        ExperimentStatus.UNEXPECTED_PASS.value,
        ExperimentStatus.WRONG_FAILURE_LAYER.value,
        ExperimentStatus.INTEGRITY_REJECTION_MISSING.value,
    }
    if any(item.get("status") in invalid_statuses for item in results):
        status = "COMPLETED_WITH_INVALID_RESULTS"
    elif fail_on_not_activated and any(
        item.get("status") == ExperimentStatus.NOT_ACTIVATED.value for item in results
    ):
        status = "FAILED_NOT_ACTIVATED"
    else:
        status = "COMPLETED"
    return {
        "status": status,
        "counts": statuses,
        "kill_rate_numerator": len(killed),
        "kill_rate_denominator": len(eligible),
        "kill_rate": len(killed) / len(eligible) if eligible else None,
    }
