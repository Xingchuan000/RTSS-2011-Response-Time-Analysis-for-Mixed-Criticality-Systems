"""Outer-only schemas for non-vacuity experiments.

Nothing in this module is written to an ordinary proof request.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any, Mapping


class ArtifactClass(StrEnum):
    NONVACUITY_EXPERIMENT_ONLY = "NONVACUITY_EXPERIMENT_ONLY"


class ConfigKind(StrEnum):
    TEMPLATE = "TEMPLATE"
    RESOLVED = "RESOLVED"


class ExperimentStatus(StrEnum):
    EXPERIMENT_DISABLED = "EXPERIMENT_DISABLED"
    CAMPAIGN_PREFLIGHT_FAILED = "CAMPAIGN_PREFLIGHT_FAILED"
    SETUP_INVALID = "SETUP_INVALID"
    TOOL_EXECUTION_FAILED = "TOOL_EXECUTION_FAILED"
    VERIFIER_TIMEOUT = "VERIFIER_TIMEOUT"
    VERIFIER_OUTPUT_MISSING = "VERIFIER_OUTPUT_MISSING"
    BASELINE_REGRESSION = "BASELINE_REGRESSION"
    NOT_ACTIVATED = "NOT_ACTIVATED"
    PASS_EXPECTED = "PASS_EXPECTED"
    FAIL_EXPECTED = "FAIL_EXPECTED"
    UNEXPECTED_PASS = "UNEXPECTED_PASS"
    UNEXPECTED_FAIL = "UNEXPECTED_FAIL"
    WRONG_FAILURE_LAYER = "WRONG_FAILURE_LAYER"
    INTEGRITY_REJECTION_EXPECTED = "INTEGRITY_REJECTION_EXPECTED"
    INTEGRITY_REJECTION_MISSING = "INTEGRITY_REJECTION_MISSING"
    PASS = "PASS"
    EXPECTED_REJECTION = "EXPECTED_REJECTION"
    ACCEPTED_SAFE_MUTATION = "ACCEPTED_SAFE_MUTATION"
    MUTATION_COHERENCE_FAILED = "MUTATION_COHERENCE_FAILED"
    PAIR_CONTRACT_FAILED = "PAIR_CONTRACT_FAILED"
    GRADIENT_BASELINE_FAILED = "GRADIENT_BASELINE_FAILED"
    GRADIENT_BOUND_NOT_FOUND = "GRADIENT_BOUND_NOT_FOUND"
    GRADIENT_NON_MONOTONIC = "GRADIENT_NON_MONOTONIC"
    GRADIENT_EXPECTED_FAILURE_FOUND = "GRADIENT_EXPECTED_FAILURE_FOUND"


class ActivationStatus(StrEnum):
    ACTIVATED = "ACTIVATED"
    NOT_ACTIVATED = "NOT_ACTIVATED"
    ACTIVATION_INCONCLUSIVE = "ACTIVATION_INCONCLUSIVE"
    ACTIVATION_SETUP_INVALID = "ACTIVATION_SETUP_INVALID"


class MutationClass(StrEnum):
    BASELINE = "BASELINE"
    DANGEROUS_TOP1 = "DANGEROUS_TOP1"
    MASK_BYPASS = "MASK_BYPASS"
    NO_FIRST_VALID = "NO_FIRST_VALID"
    ALL_INVALID_FORCE_TOP1 = "ALL_INVALID_FORCE_TOP1"
    GUARD_ABLATION = "GUARD_ABLATION"
    ACTION_SEMANTICS = "ACTION_SEMANTICS"
    RUNTIME_SOURCE = "RUNTIME_SOURCE"
    ENVELOPE = "ENVELOPE"
    BUNDLE_INTEGRITY = "BUNDLE_INTEGRITY"
    ENVELOPE_GRADIENT = "ENVELOPE_GRADIENT"
    ROUNDING_TO_NEAREST = "ROUNDING_TO_NEAREST"
    RETROACTIVE_RELEASE_BUDGET = "RETROACTIVE_RELEASE_BUDGET"
    MODEL_SEMANTICS_MUTATION = "MODEL_SEMANTICS_MUTATION"
    SOURCE_BINDING_TAMPER = "SOURCE_BINDING_TAMPER"


class PatchRole(StrEnum):
    DEPLOYED_IMPLEMENTATION = "DEPLOYED_IMPLEMENTATION"
    ADAPTER_BINDING = "ADAPTER_BINDING"
    CONFIG_ARTIFACT = "CONFIG_ARTIFACT"
    DEPLOYED_SELECTION = "DEPLOYED_SELECTION"
    DEPLOYED_APPLY = "DEPLOYED_APPLY"
    FROZEN_SELECTION = "FROZEN_SELECTION"
    FROZEN_APPLY = "FROZEN_APPLY"
    DEPLOYED_GUARD = "DEPLOYED_GUARD"
    FROZEN_GUARD = "FROZEN_GUARD"
    FORMAL_SEMANTIC_MIRROR = "FORMAL_SEMANTIC_MIRROR"


FORBIDDEN_PATCH_ROLES = frozenset({
    "VERIFIER_CHECKER", "AGGREGATOR", "EXPECTED_RESULT_CLASSIFIER",
})


@dataclass(frozen=True)
class SourcePatchSpec:
    role: PatchRole
    target_file: str
    target_symbol: str
    before_ast_hash: str
    before_snippet: str
    after_snippet: str
    occurrence: int = 1

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "SourcePatchSpec":
        raw = dict(value)
        role_raw = str(raw.get("role", ""))
        if role_raw in FORBIDDEN_PATCH_ROLES:
            raise ValueError(f"禁止 patch role: {role_raw}")
        try:
            role = PatchRole(role_raw)
        except ValueError as exc:
            raise ValueError(f"未知 patch role: {role_raw}") from exc
        return cls(
            role=role, target_file=str(raw["target_file"]),
            target_symbol=str(raw["target_symbol"]),
            before_ast_hash=str(raw["before_ast_hash"]),
            before_snippet=str(raw["before_snippet"]),
            after_snippet=str(raw["after_snippet"]),
            occurrence=int(raw.get("occurrence", 1)),
        )


@dataclass(frozen=True)
class ExpectedResult:
    allowed_result_statuses: tuple[str, ...] = ()
    allowed_first_failing_obligations: tuple[str, ...] = ()
    allowed_failure_routes: tuple[str, ...] = ()
    require_failure: bool = False
    require_proved: bool = False
    require_activation: bool = True
    explanation: str = ""
    result_status: str | None = None
    first_failing_obligations: tuple[str, ...] = ()
    allowed_upstream_obligations: tuple[str, ...] = ()
    allow_strict_upstream_failure: bool = False
    integrity_result_status: str = "PROOF_BUNDLE_INVALID"
    performance_degradation_metrics: tuple[str, ...] = ()

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any] | None) -> "ExpectedResult":
        raw = dict(value or {})
        allowed_statuses = _strings(raw.get("allowed_result_statuses", ()))
        legacy_status = _optional_str(raw.get("result_status"))
        if not allowed_statuses and legacy_status:
            allowed_statuses = (legacy_status,)
        allowed_obligations = _strings(raw.get("allowed_first_failing_obligations", ()))
        if not allowed_obligations:
            allowed_obligations = _strings(raw.get("first_failing_obligations", ()))
        return cls(
            allowed_result_statuses=allowed_statuses,
            allowed_first_failing_obligations=allowed_obligations,
            allowed_failure_routes=_strings(raw.get("allowed_failure_routes", ())),
            require_failure=bool(raw.get("require_failure", False)),
            require_proved=bool(raw.get("require_proved", False)),
            require_activation=bool(raw.get("require_activation", True)),
            explanation=str(raw.get("explanation", "")),
            result_status=legacy_status,
            first_failing_obligations=allowed_obligations,
            allowed_upstream_obligations=_strings(raw.get("allowed_upstream_obligations", ())),
            allow_strict_upstream_failure=bool(raw.get("allow_strict_upstream_failure", False)),
            integrity_result_status=str(raw.get("integrity_result_status", "PROOF_BUNDLE_INVALID")),
            performance_degradation_metrics=_strings(raw.get("performance_degradation_metrics", ())),
        )

    def validate(self) -> None:
        if not self.allowed_result_statuses and not self.result_status:
            raise ValueError("allowed_result_statuses must not be empty")
        if self.require_failure and self.require_proved:
            raise ValueError("require_failure and require_proved are mutually exclusive")

    @property
    def canonical_statuses(self) -> tuple[str, ...]:
        return self.allowed_result_statuses or ((self.result_status,) if self.result_status else ())


@dataclass(frozen=True)
class MutationManifest:
    schema_version: str
    enabled: bool
    mutation_id: str
    mutation_class: MutationClass
    seed_dir: Path | None
    base_seed: int | None
    tree_variant: str
    paired_with: str | None
    single_semantic_change: bool
    mutator: Mapping[str, Any]
    activation: Mapping[str, Any]
    expected: ExpectedResult
    reuse_source_bundle: Path | None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, Any],
        *,
        base_dir: Path,
    ) -> "MutationManifest":
        raw = dict(value)
        schema_version = str(raw.get("schema_version", ""))
        if schema_version != "nonvacuity_mutation_v1":
            raise ValueError("mutation schema_version 必须为 nonvacuity_mutation_v1")
        mutation_id = str(raw.get("mutation_id", "")).strip()
        if not mutation_id or any(part in mutation_id for part in ("/", "\\", "..")):
            raise ValueError("mutation_id 必须是非空且不可包含路径片段")
        try:
            mutation_class = MutationClass(str(raw.get("mutation_class", "")))
        except ValueError as exc:
            raise ValueError(f"不支持的 mutation_class: {raw.get('mutation_class')!r}") from exc
        seed_dir = _resolved_optional_path(raw.get("seed_dir"), base_dir)
        reuse = _resolved_optional_path(raw.get("reuse_source_bundle"), base_dir)
        tree_variant = str(raw.get("tree_variant", "best_overall"))
        if tree_variant not in {"best_overall", "best_balanced", "best_performance"}:
            raise ValueError(f"不支持的 tree_variant: {tree_variant}")
        mutator = raw.get("mutator", {})
        activation = raw.get("activation", {})
        if not isinstance(mutator, Mapping) or not isinstance(activation, Mapping):
            raise ValueError("mutator 和 activation 必须为 object")
        if mutation_class not in {MutationClass.BASELINE, MutationClass.BUNDLE_INTEGRITY} and not activation:
            raise ValueError("每个语义 mutation 必须声明 activation rule")
        if mutation_class is MutationClass.BUNDLE_INTEGRITY and reuse is None:
            raise ValueError("integrity mutation 必须声明 reuse_source_bundle")
        base_seed_raw = raw.get("base_seed")
        return cls(
            schema_version=schema_version,
            enabled=bool(raw.get("enabled", False)),
            mutation_id=mutation_id,
            mutation_class=mutation_class,
            seed_dir=seed_dir,
            base_seed=int(base_seed_raw) if base_seed_raw is not None else None,
            tree_variant=tree_variant,
            paired_with=_optional_str(raw.get("paired_with")),
            single_semantic_change=bool(raw.get("single_semantic_change", True)),
            mutator=dict(mutator),
            activation=dict(activation),
            expected=ExpectedResult.from_mapping(raw.get("expected")),
            reuse_source_bundle=reuse,
            metadata=dict(raw.get("metadata", {})),
        )


@dataclass(frozen=True)
class CampaignConfig:
    schema_version: str
    enabled: bool
    campaign_id: str
    proof_route: str
    output_root: Path
    source_root: Path
    preserve_workspaces: bool
    run_baselines: bool
    run_semantic_recompile: bool
    run_integrity_reuse: bool
    run_hout: bool
    fail_on_not_activated: bool
    mutations: tuple[MutationManifest, ...]

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, Any],
        *,
        base_dir: Path,
    ) -> "CampaignConfig":
        raw = dict(value)
        if raw.get("schema_version") != "ppp_nonvacuity_campaign_v1":
            raise ValueError("campaign schema_version 必须为 ppp_nonvacuity_campaign_v1")
        if raw.get("proof_route", "protected_prefix") != "protected_prefix":
            raise ValueError("nonvacuity lab 只允许 protected_prefix")
        campaign_id = str(raw.get("campaign_id", "")).strip()
        if not campaign_id or any(part in campaign_id for part in ("/", "\\", "..")):
            raise ValueError("campaign_id 非法")
        source_root = _resolved_path(raw.get("source_root", "."), base_dir)
        output_root = _resolved_path(raw.get("output_root", "outputs/nonvacuity"), base_dir)
        mutations_raw = raw.get("mutations", ())
        if not isinstance(mutations_raw, list):
            raise ValueError("mutations 必须为 array")
        mutations = tuple(
            MutationManifest.from_mapping(item, base_dir=base_dir)
            for item in mutations_raw
        )
        ids = [item.mutation_id for item in mutations]
        if len(ids) != len(set(ids)):
            raise ValueError("mutation ID 必须唯一")
        _validate_output_outside_inputs(output_root, source_root, mutations)
        return cls(
            schema_version="ppp_nonvacuity_campaign_v1",
            enabled=bool(raw.get("enabled", False)),
            campaign_id=campaign_id,
            proof_route="protected_prefix",
            output_root=output_root,
            source_root=source_root,
            preserve_workspaces=bool(raw.get("preserve_workspaces", True)),
            run_baselines=bool(raw.get("run_baselines", True)),
            run_semantic_recompile=bool(raw.get("run_semantic_recompile", True)),
            run_integrity_reuse=bool(raw.get("run_integrity_reuse", True)),
            run_hout=bool(raw.get("run_hout", True)),
            fail_on_not_activated=bool(raw.get("fail_on_not_activated", True)),
            mutations=mutations,
        )


def experiment_envelope(**payload: Any) -> dict[str, Any]:
    return {
        "schema_version": "nonvacuity_experiment_result_v1",
        "artifact_class": ArtifactClass.NONVACUITY_EXPERIMENT_ONLY.value,
        "deployment_certificate_eligible": False,
        **payload,
    }


def _strings(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, (list, tuple)):
        raise ValueError("期望字符串数组")
    return tuple(str(item) for item in value)


def _optional_str(value: Any) -> str | None:
    return None if value is None else str(value)


def _resolved_path(value: Any, base_dir: Path) -> Path:
    path = Path(str(value))
    return (path if path.is_absolute() else base_dir / path).resolve()


def _resolved_optional_path(value: Any, base_dir: Path) -> Path | None:
    return None if value in (None, "") else _resolved_path(value, base_dir)


def _is_within(path: Path, parent: Path) -> bool:
    return path == parent or parent in path.parents


def _validate_output_outside_inputs(
    output_root: Path,
    source_root: Path,
    mutations: tuple[MutationManifest, ...],
) -> None:
    if _is_within(output_root, source_root):
        # Repository-level outputs/nonvacuity is deliberately allowed; only
        # source package directories and seed inputs are protected below.
        protected = (source_root / "formal_toolchain", source_root / "amc_py")
        if any(_is_within(output_root, item) for item in protected):
            raise ValueError("output_root 不得位于正式源码包内")
    for mutation in mutations:
        if mutation.seed_dir and _is_within(output_root, mutation.seed_dir):
            raise ValueError("output_root 不得位于输入 seed 目录内")
