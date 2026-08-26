from __future__ import annotations

import json
import shutil
import platform
from dataclasses import replace
from pathlib import Path
import sys
from typing import Any

from ..activation import evaluate_hout_activation, solve_symbolic_activation
from ..activation.auto_symbolic import run_auto_symbolic_activation
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
    ExpectedResult,
    experiment_envelope,
)
from ..workspace import (
    ExperimentWorkspace,
    verify_original_inputs_unchanged,
    write_input_snapshot,
)
from ..canonical import file_hash, tree_hash
from ..receipts.pair_receipt import PairReceipt, consume_pair_receipt, write_pair_receipt
from ..receipts.mutation_receipt import build_mutation_receipt, write_mutation_receipt
from .baseline import run_baseline
from .integrity_reuse import run_integrity_reuse
from .paired_hout import run_paired_hout
from .semantic_recompile import run_semantic_recompile
from .run_plan import RunKind, build_run_plan
from .envelope_gradient import run_envelope_gradient_experiment


def can_execute_mutation(config: dict, mutation: dict, *, cli_enable: bool) -> tuple[bool, str]:
    """The v2 three-gate check; it has no filesystem side effects."""
    if config.get("config_kind") != "RESOLVED":
        return False, "CONFIG_NOT_RESOLVED"
    if not bool(config.get("enabled", False)):
        return False, "CAMPAIGN_DISABLED"
    if not cli_enable:
        return False, "CLI_ENABLE_MISSING"
    if not bool(mutation.get("enabled", False)):
        return False, "MUTATION_DISABLED"
    return True, "ENABLED"


def run_campaign(
    config: CampaignConfig,
    *,
    enabled_by_cli: bool,
    timeout_seconds: int | None = None,
    overwrite_existing: bool = False,
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
    if campaign_dir.exists() and overwrite_existing:
        shutil.rmtree(campaign_dir)
    campaign_dir.mkdir(parents=True, exist_ok=True)
    results = []
    paired_hashes: dict[str, str] = {}
    paired_receipts: dict[str, Path] = {}
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
            manifest.mutation_class in {MutationClass.ENVELOPE, MutationClass.ENVELOPE_GRADIENT}
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
        plan = build_run_plan({
            "mutation_id": effective_manifest.mutation_id,
            "mutation_class": effective_manifest.mutation_class.value,
            "also_run_old_bundle": effective_manifest.reuse_source_bundle is not None,
            "hout_profile_id": effective_manifest.metadata.get("hout_profile_id"),
            "pair_with": effective_manifest.paired_with,
        })
        if plan.run_kind is RunKind.ENVELOPE_GRADIENT:
            try:
                result = _run_envelope_gradient_campaign(
                    effective_manifest,
                    campaign_id=config.campaign_id,
                    output_root=config.output_root,
                    source_root=config.source_root,
                    enabled_by_cli=enabled_by_cli,
                    timeout_seconds=timeout_seconds,
                )
            except (OSError, ValueError, KeyError, TypeError) as exc:
                result = experiment_envelope(
                    campaign_id=config.campaign_id,
                    mutation_id=effective_manifest.mutation_id,
                    status=ExperimentStatus.SETUP_INVALID.value,
                    reason=str(exc),
                )
            results.append(result)
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
            effective_manifest = replace(effective_manifest, seed_dir=paired_seed)
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
            paired_receipts=paired_receipts,
            overwrite_existing=False,
        )
        results.append(result)
        semantic = result.get("semantic_recompile")
        mutation = (
            semantic.get("mutation_result", {})
            if isinstance(semantic, dict)
            else {}
        )
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
            _write_pair_receipt_if_possible(
                result=result,
                manifest=manifest,
                campaign_dir=campaign_dir,
                paired_receipts=paired_receipts,
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
    paired_receipts: dict[str, Path] | None = None,
    overwrite_existing: bool = False,
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
        overwrite_existing=overwrite_existing,
    )
    pair_error = None
    if manifest.paired_with:
        expected_tree_hash = (paired_hashes or {}).get(manifest.paired_with)
        actual_tree_hash = _tree_semantic_hash(
            workspace.mutated_seed / manifest.tree_variant / "integer_tree.json"
        )
        if expected_tree_hash is None or actual_tree_hash != expected_tree_hash:
            pair_error = "paired mutation 未使用 A1 的同一 after-tree hash"
        receipt_path = (paired_receipts or {}).get(manifest.paired_with)
        if receipt_path is not None:
            try:
                consume_pair_receipt(
                    receipt_path,
                    expected_producer=manifest.paired_with,
                    seed=int(manifest.base_seed or 0),
                    variant=manifest.tree_variant,
                    copied_tree_path=workspace.mutated_seed / manifest.tree_variant / "integer_tree.json",
                    copied_seed_dir=workspace.mutated_seed,
                )
            except ValueError as exc:
                pair_error = str(exc)
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
        activation = _run_activation(manifest, workspace, hout, source_root=source_root)
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
            "status": (ExperimentStatus.PAIR_CONTRACT_FAILED.value if pair_error else ExperimentStatus.SETUP_INVALID.value),
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
    mutation_result_payload = (semantic or integrity or {}).get("mutation_result", {})
    mutation_receipt = build_mutation_receipt(
        mutation_id=manifest.mutation_id,
        mutation_class=manifest.mutation_class.value,
        mutation_result=mutation_result_payload,
        diff_path=workspace.diff_output if workspace.diff_output.is_file() else None,
        coherence_path=workspace.coherence_output if workspace.coherence_output.is_file() else None,
        activation_path=workspace.activation_output / "activation_result.json" if (workspace.activation_output / "activation_result.json").is_file() else None,
        formal_result_path=(workspace.semantic_output / "proof_result.json") if (workspace.semantic_output / "proof_result.json").is_file() else (workspace.integrity_output / "proof_summary.json" if (workspace.integrity_output / "proof_summary.json").is_file() else None),
        hout_result_path=workspace.hout_output / "paired_hout_result.json" if (workspace.hout_output / "paired_hout_result.json").is_file() else None,
        metadata={"experiment_status": expectation["status"]},
    )
    write_mutation_receipt(workspace.root / "mutation_receipt.json", mutation_receipt)
    write_experiment_report(workspace.report_output / "report.md", result)
    return result


def _run_activation(
    manifest: MutationManifest,
    workspace: ExperimentWorkspace,
    hout: dict[str, Any] | None,
    *,
    source_root: Path,
) -> dict[str, Any] | None:
    if manifest.mutation_class in {MutationClass.BASELINE, MutationClass.BUNDLE_INTEGRITY}:
        return None

    from ..activation.schema import ActivationResult
    from ..schema import ActivationStatus

    mode = str(manifest.activation.get("mode", "symbolic")).lower()
    results: list[ActivationResult] = []

    if "symbolic_auto" in mode:
        try:
            results.append(
                run_auto_symbolic_activation(
                    mutation_id=manifest.mutation_id,
                    activation=dict(manifest.activation),
                    resolved_target=dict(manifest.metadata["resolved_target"]),
                    clean_source_root=source_root,
                    overlay_source_root=workspace.source_overlay,
                    output_dir=workspace.activation_output,
                )
            )
        except Exception as exc:  # Evidence-source failure must not suppress paired HOUT.
            results.append(
                ActivationResult(
                    mutation_id=manifest.mutation_id,
                    status=ActivationStatus.ACTIVATION_SETUP_INVALID,
                    evidence_modes=("SYMBOLIC", "CONCRETE_REPLAY"),
                    details={
                        "reason": "automatic symbolic activation unavailable",
                        "exception_type": type(exc).__name__,
                        "exception": str(exc),
                    },
                )
            )
    elif "symbolic" in mode:
        results.append(
            solve_symbolic_activation(
                mutation_id=manifest.mutation_id,
                rule=manifest.activation,
                output_path=workspace.activation_output / "symbolic_witness.json",
            )
        )

    if "hout" in mode:
        if hout and hout.get("setup_valid"):
            results.append(
                evaluate_hout_activation(
                    mutation_id=manifest.mutation_id,
                    base_events=load_events(Path(hout["base_events"])),
                    mutated_events=load_events(Path(hout["mutated_events"])),
                    rule=manifest.activation,
                )
            )
        else:
            results.append(
                ActivationResult(
                    mutation_id=manifest.mutation_id,
                    status=ActivationStatus.ACTIVATION_SETUP_INVALID,
                    evidence_modes=("HOUT",),
                    details={
                        "reason": "paired HOUT activation evidence unavailable",
                        "hout_setup_valid": None if hout is None else bool(hout.get("setup_valid")),
                        "hout_reason": None if hout is None else hout.get("reason", hout.get("hout_schema_error")),
                    },
                )
            )

    if not results:
        results.append(
            ActivationResult(
                mutation_id=manifest.mutation_id,
                status=ActivationStatus.ACTIVATION_INCONCLUSIVE,
                details={"reason": "requested activation evidence unavailable"},
            )
        )

    priority = {
        ActivationStatus.ACTIVATED: 0,
        ActivationStatus.NOT_ACTIVATED: 1,
        ActivationStatus.ACTIVATION_SETUP_INVALID: 2,
        ActivationStatus.ACTIVATION_INCONCLUSIVE: 3,
    }
    selected = min(results, key=lambda item: priority[item.status])
    payload = selected.to_dict()
    payload["all_evidence"] = [item.to_dict() for item in results]
    (workspace.activation_output / "activation_result.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return payload



def _run_envelope_gradient_campaign(
    manifest: MutationManifest,
    *,
    campaign_id: str,
    output_root: Path,
    source_root: Path,
    enabled_by_cli: bool,
    timeout_seconds: int | None,
) -> dict[str, Any]:
    """Run D1 as a sequence of independent ordinary proof invocations.

    Delta zero is an unmodified baseline.  Every positive delta gets its own
    isolated workspace and ordinary protected-prefix proof request.  No proof
    bundle is reused across coordinates.
    """
    if manifest.seed_dir is None or not manifest.seed_dir.is_dir():
        raise ValueError(f"D1 seed_dir 不存在: {manifest.seed_dir}")
    parameters = dict(manifest.mutator.get("parameters", {}))
    if not parameters.get("target_file") or not parameters.get("json_pointer"):
        raise ValueError("D1 需要解析后的 target_file 和 json_pointer")
    initial_step = int(manifest.metadata.get("initial_step", 1))
    maximum_delta = int(manifest.metadata.get("maximum_delta", 1024))
    run_records: dict[int, dict[str, Any]] = {}

    def evaluate_delta(delta: int) -> dict[str, Any]:
        mutation_id = f"{manifest.mutation_id}__delta_{delta:04d}"
        if delta == 0:
            delta_manifest = replace(
                manifest,
                mutation_id=mutation_id,
                mutation_class=MutationClass.BASELINE,
                mutator={},
                activation={},
                expected=ExpectedResult(
                    allowed_result_statuses=("DEPLOYED_TREE_PROVED",),
                    require_proved=True,
                    require_activation=False,
                ),
                reuse_source_bundle=None,
                metadata={**dict(manifest.metadata), "gradient_parent": manifest.mutation_id, "delta": 0},
            )
            result = run_one(
                delta_manifest,
                campaign_id=campaign_id,
                output_root=output_root,
                source_root=source_root,
                enabled_by_cli=enabled_by_cli,
                run_baseline_first=True,
                run_semantic=False,
                run_integrity=False,
                run_hout=False,
                timeout_seconds=timeout_seconds,
            )
            proof = dict((result.get("baseline") or {}).get("proof_result") or {})
        else:
            delta_parameters = {**parameters, "delta": delta}
            delta_manifest = replace(
                manifest,
                mutation_id=mutation_id,
                mutation_class=MutationClass.ENVELOPE,
                mutator={"kind": "envelope", "parameters": delta_parameters},
                activation={"mode": "none"},
                expected=ExpectedResult(
                    allowed_result_statuses=(
                        "DEPLOYED_TREE_PROVED",
                        *tuple(manifest.expected.canonical_statuses),
                    ),
                    allowed_first_failing_obligations=manifest.expected.allowed_first_failing_obligations,
                    allowed_failure_routes=manifest.expected.allowed_failure_routes,
                    allowed_upstream_obligations=manifest.expected.allowed_upstream_obligations,
                    allow_strict_upstream_failure=manifest.expected.allow_strict_upstream_failure,
                    require_activation=False,
                ),
                reuse_source_bundle=None,
                metadata={**dict(manifest.metadata), "gradient_parent": manifest.mutation_id, "delta": delta},
            )
            result = run_one(
                delta_manifest,
                campaign_id=campaign_id,
                output_root=output_root,
                source_root=source_root,
                enabled_by_cli=enabled_by_cli,
                run_baseline_first=False,
                run_semantic=True,
                run_integrity=False,
                run_hout=False,
                timeout_seconds=timeout_seconds,
            )
            proof = dict((result.get("semantic_recompile") or {}).get("proof_result") or {})
        row = {
            "result_status": proof.get("result_status"),
            "violated_obligation_id": proof.get("violated_obligation_id"),
            "failure_route": proof.get("failure_route"),
            "witness": _find_nested_value(proof, ("witness", "counterexample", "first_failing_witness")),
            "slack": _find_numeric_slack(proof),
            "experiment_status": result.get("status"),
            "result_file": str(output_root / campaign_id / mutation_id / "experiment_result.json"),
        }
        run_records[delta] = {"experiment": result, "proof": proof, "row": row}
        return row

    gradient = run_envelope_gradient_experiment(
        None,
        {
            "evaluate_delta": evaluate_delta,
            "initial_step": initial_step,
            "maximum_delta": maximum_delta,
        },
    )
    status = str(gradient.get("experiment_status"))
    if status == ExperimentStatus.GRADIENT_EXPECTED_FAILURE_FOUND.value:
        delta_star = int(gradient["delta_star"])
        failing = run_records[delta_star]["proof"]
        allowed_statuses = set(manifest.expected.canonical_statuses)
        allowed_obligations = set(manifest.expected.allowed_first_failing_obligations)
        actual_status = failing.get("result_status")
        actual_obligation = failing.get("violated_obligation_id")
        if allowed_statuses and actual_status not in allowed_statuses:
            status = ExperimentStatus.WRONG_FAILURE_LAYER.value
            gradient["classification_reason"] = (
                f"delta* status {actual_status!r} not in {sorted(allowed_statuses)}"
            )
        elif allowed_obligations and actual_obligation not in allowed_obligations:
            status = ExperimentStatus.WRONG_FAILURE_LAYER.value
            gradient["classification_reason"] = (
                f"delta* obligation {actual_obligation!r} not in {sorted(allowed_obligations)}"
            )
    campaign_dir = output_root / campaign_id / manifest.mutation_id
    campaign_dir.mkdir(parents=True, exist_ok=True)
    gradient_path = campaign_dir / "gradient_result.json"
    gradient_path.write_text(json.dumps(gradient, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    result = experiment_envelope(
        campaign_id=campaign_id,
        mutation_id=manifest.mutation_id,
        mutation_class=manifest.mutation_class.value,
        status=status,
        gradient=gradient,
        dynamic_selection=manifest.metadata.get("dynamic_selection"),
    )
    (campaign_dir / "experiment_result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    write_experiment_report(campaign_dir / "report.md", result)
    return result


def _find_nested_value(value: Any, keys: tuple[str, ...]) -> Any:
    if isinstance(value, dict):
        for key in keys:
            if key in value:
                return value[key]
        for child in value.values():
            found = _find_nested_value(child, keys)
            if found is not None:
                return found
    elif isinstance(value, list):
        for child in value:
            found = _find_nested_value(child, keys)
            if found is not None:
                return found
    return None


def _find_numeric_slack(value: Any) -> int | float | None:
    candidates: list[int | float] = []

    def visit(node: Any) -> None:
        if isinstance(node, dict):
            for key, child in node.items():
                lowered = str(key).lower()
                if "slack" in lowered and isinstance(child, (int, float)) and not isinstance(child, bool):
                    candidates.append(child)
                else:
                    visit(child)
        elif isinstance(node, list):
            for child in node:
                visit(child)

    visit(value)
    return min(candidates) if candidates else None

def _tree_semantic_hash(path: Path) -> str:
    raw = json.loads(path.read_text(encoding="utf-8"))
    return canonical_json_hash(raw)


def _write_pair_receipt_if_possible(*, result, manifest, campaign_dir, paired_receipts):
    semantic = result.get("semantic_recompile", {})
    mutation = semantic.get("mutation_result", {})
    details = mutation.get("details", {})
    if details.get("leaf_id") is None or details.get("action_id") is None:
        return
    mutation_root = campaign_dir / manifest.mutation_id / "semantic_recompile"
    tree_path = mutation_root / "mutated_seed" / manifest.tree_variant / "integer_tree.json"
    seed_dir = mutation_root / "mutated_seed"
    witness = mutation_root.parent / "activation" / "activation_result.json"
    if not tree_path.is_file() or not seed_dir.is_dir() or not witness.is_file():
        return
    receipt = PairReceipt(
        schema_version="nonvacuity_pair_receipt_v1",
        producer_mutation_id=manifest.mutation_id,
        seed=int(manifest.base_seed or 0),
        tree_variant=manifest.tree_variant,
        leaf_id=int(details["leaf_id"]),
        action_id=int(details["action_id"]),
        base_tree_sha256=str(mutation.get("before_hash", "")),
        mutated_tree_sha256=str(mutation.get("after_hash", "")),
        mutated_tree_file_sha256=file_hash(tree_path),
        activation_witness_sha256=file_hash(witness),
        mutated_seed_snapshot_sha256=tree_hash(seed_dir),
    )
    path = campaign_dir / manifest.mutation_id / "pair_receipt.json"
    write_pair_receipt(path, receipt)
    paired_receipts[manifest.mutation_id] = path


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

    seed_dirs = manifest.metadata.get("seed_dirs_by_seed", {})
    seed_dir_raw = seed_dirs.get(str(seed)) if isinstance(seed_dirs, dict) else None
    seed_dir = Path(str(seed_dir_raw)).resolve() if seed_dir_raw else (source_root / f"s{seed}").resolve()
    if not seed_dir.is_dir():
        raise ValueError(f"D1 resolved seed_dir 不存在: {seed_dir}")

    variant_aliases = {
        "compact": "best_overall",
        "balanced": "best_balanced",
        "best_overall": "best_overall",
        "best_balanced": "best_balanced",
        "best_performance": "best_performance",
    }
    variant = variant_aliases.get(str(selected.get("variant")))
    if variant is None or not (seed_dir / variant).is_dir():
        configured = manifest.metadata.get("tree_variants_by_seed", {})
        candidates = configured.get(str(seed), ()) if isinstance(configured, dict) else ()
        variant = next(
            (variant_aliases.get(str(item), str(item)) for item in candidates
             if (seed_dir / variant_aliases.get(str(item), str(item))).is_dir()),
            variant,
        )
    if variant is None or not (seed_dir / variant).is_dir():
        raise ValueError(f"D1 artifact variant 无法映射到 seed 输入: {selected.get('variant')}")

    parameters = dict(manifest.mutator.get("parameters", {}))
    target_file, pointer = _resolve_d1_envelope_coordinate(
        selected=selected,
        parameters=parameters,
        seed_dir=seed_dir,
    )
    parameters["target_file"] = target_file
    parameters["json_pointer"] = pointer
    dynamic_selection = {
        **selected,
        "resolved_seed_dir": str(seed_dir),
        "resolved_tree_variant": variant,
        "resolved_envelope_target_file": target_file,
        "resolved_envelope_json_pointer": pointer,
    }
    metadata = {
        **dict(manifest.metadata),
        "dynamic_selection": dynamic_selection,
    }
    return replace(
        manifest,
        seed_dir=seed_dir,
        base_seed=seed,
        tree_variant=variant,
        mutator={**dict(manifest.mutator), "parameters": parameters},
        metadata=metadata,
    )


def _resolve_d1_envelope_coordinate(
    *,
    selected: dict[str, Any],
    parameters: dict[str, Any],
    seed_dir: Path,
) -> tuple[str, str]:
    """Resolve only an authoritative reference-envelope coordinate for D1.

    D1 is a reference-certificate sensitivity experiment.  Guessing an
    arbitrary ``*upper`` field from a copied seed can mutate a derived model
    snapshot instead of the certified reference envelope and therefore fail in
    an unrelated model/preflight layer.  The selected RTA artifact (or an
    explicit campaign parameter) must publish the exact seed-relative file and
    JSON pointer.
    """

    raw_file = selected.get("envelope_target_file") or parameters.get("target_file")
    raw_pointer = selected.get("envelope_json_pointer") or parameters.get("json_pointer")
    if not raw_file or not raw_pointer:
        raise ValueError(
            "D1_AUTHORITATIVE_ENVELOPE_BINDING_MISSING:"
            "RTA artifact must provide envelope_target_file and envelope_json_pointer"
        )
    relative = _seed_relative_json_file(seed_dir, str(raw_file), str(raw_pointer))
    return relative, str(raw_pointer)


def _seed_relative_json_file(seed_dir: Path, raw_file: str, pointer: str) -> str:
    candidate = Path(raw_file)
    direct = candidate if candidate.is_absolute() else seed_dir / candidate
    if direct.is_file() and seed_dir in direct.resolve().parents:
        _require_integer_pointer(direct, pointer)
        return direct.resolve().relative_to(seed_dir).as_posix()

    # Proof bundles sometimes publish a candidate-relative or absolute source
    # path.  Match it back to the copied seed input by suffix/basename.
    suffix_parts = tuple(part for part in candidate.parts if part not in {"candidate", "request", "verified"})
    matches = []
    for path in seed_dir.rglob(candidate.name):
        relative_parts = path.relative_to(seed_dir).parts
        suffix_score = 0
        for length in range(1, min(len(relative_parts), len(suffix_parts)) + 1):
            if relative_parts[-length:] == suffix_parts[-length:]:
                suffix_score = length
        try:
            _require_integer_pointer(path, pointer)
        except (OSError, ValueError, KeyError, TypeError, IndexError):
            continue
        matches.append((suffix_score, path))
    if not matches:
        raise ValueError(f"D1 envelope target 无法映射到 seed 输入: {raw_file}:{pointer}")
    _, path = max(matches, key=lambda item: (item[0], -len(item[1].parts)))
    return path.resolve().relative_to(seed_dir).as_posix()


def _require_integer_pointer(path: Path, pointer: str) -> int:
    value: Any = json.loads(path.read_text(encoding="utf-8"))
    if pointer in {"", "/"}:
        current = value
    else:
        current = value
        for raw_token in str(pointer).lstrip("/").split("/"):
            token = raw_token.replace("~1", "/").replace("~0", "~")
            current = current[int(token)] if isinstance(current, list) else current[token]
    if not isinstance(current, int) or isinstance(current, bool):
        raise ValueError(f"D1 envelope coordinate 不是整数: {path}:{pointer}")
    return current



def _iter_envelope_coordinates(
    value: Any,
    *,
    limiting_task: str | None,
    key_priority: tuple[str, ...],
    pointer: str = "",
):
    if isinstance(value, dict):
        identity = value.get("name", value.get("task_id", value.get("task")))
        criticality = str(value.get("criticality", value.get("level", ""))).upper()
        task_match = limiting_task is not None and str(identity) == limiting_task
        lo_fallback = limiting_task is None and criticality in {"LO", "LOW", "0", "FALSE"}
        if task_match or lo_fallback:
            for index, key in enumerate(key_priority):
                if key in value:
                    token = str(key).replace("~", "~0").replace("/", "~1")
                    score = (1000 if task_match else 100) + len(key_priority) - index
                    yield f"{pointer}/{token}", value[key], score
        for key, child in value.items():
            token = str(key).replace("~", "~0").replace("/", "~1")
            yield from _iter_envelope_coordinates(
                child,
                limiting_task=limiting_task,
                key_priority=key_priority,
                pointer=f"{pointer}/{token}",
            )
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from _iter_envelope_coordinates(
                child,
                limiting_task=limiting_task,
                key_priority=key_priority,
                pointer=f"{pointer}/{index}",
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
            ExperimentStatus.GRADIENT_EXPECTED_FAILURE_FOUND.value,
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
        ExperimentStatus.GRADIENT_BASELINE_FAILED.value,
        ExperimentStatus.GRADIENT_BOUND_NOT_FOUND.value,
        ExperimentStatus.GRADIENT_NON_MONOTONIC.value,
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
