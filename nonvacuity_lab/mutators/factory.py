from __future__ import annotations

from .action_config import ActionConfigMutation
from .bundle_tamper import BundleTamperMutation
from .coherent_source_patch import CoherentSourcePatchMutation
from .envelope import EnvelopeMutation
from .tree_ranking import DangerousTop1Mutation


MUTATOR_FACTORY = {
    "TREE_DANGEROUS_TOP1_MASKED": DangerousTop1Mutation,
    "DANGEROUS_TOP1": DangerousTop1Mutation,
    "MASK_BYPASS": CoherentSourcePatchMutation,
    "NO_FIRST_VALID": CoherentSourcePatchMutation,
    "ALL_INVALID_FORCE_TOP1": CoherentSourcePatchMutation,
    "GUARD_REMOVAL": CoherentSourcePatchMutation,
    "GUARD_ABLATION": CoherentSourcePatchMutation,
    "ACTION_RATIO_2_TO_5": ActionConfigMutation,
    "ACTION_SEMANTICS": ActionConfigMutation,
    "RUNTIME_SOURCE": CoherentSourcePatchMutation,
    "ENVELOPE": EnvelopeMutation,
    "ROUNDING_TO_NEAREST": CoherentSourcePatchMutation,
    "RETROACTIVE_RELEASE_BUDGET": CoherentSourcePatchMutation,
    "MODEL_SEMANTICS_MUTATION": CoherentSourcePatchMutation,
    "SOURCE_BINDING_TAMPER": BundleTamperMutation,
    "E1_DEADLINE_CLEANUP_REMOVE": CoherentSourcePatchMutation,
    "E2_HI_JOB_TRUNCATE": CoherentSourcePatchMutation,
    "E3_EVENT_ORDER": CoherentSourcePatchMutation,
    "E4_CONTROLLER_OVERHEAD": CoherentSourcePatchMutation,
    "E5_NONQUIESCENT_RECOVERY": CoherentSourcePatchMutation,
    "E6_UNSTABLE_DEMAND_READS": CoherentSourcePatchMutation,
    "BUNDLE_CROSS_SEED": BundleTamperMutation,
    "BUNDLE_CROSS_VARIANT": BundleTamperMutation,
    "BUNDLE_PRIORITY_TAMPER": BundleTamperMutation,
    "BUNDLE_WITNESS_TAMPER": BundleTamperMutation,
    "BUNDLE_ARTIFACT_DELETE": BundleTamperMutation,
    "BUNDLE_TREE_TAMPER": BundleTamperMutation,
    "BUNDLE_INTEGRITY": BundleTamperMutation,
    "ENVELOPE": EnvelopeMutation,
}


def build_mutator(mutation_class: str):
    try:
        return MUTATOR_FACTORY[str(mutation_class)]()
    except KeyError as exc:
        raise ValueError(f"unsupported mutation class: {mutation_class}") from exc
