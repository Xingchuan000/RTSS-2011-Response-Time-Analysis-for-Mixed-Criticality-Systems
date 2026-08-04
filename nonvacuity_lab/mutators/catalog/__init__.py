"""Declarative catalog for the isolated PPP mutation families."""

from dataclasses import dataclass


@dataclass(frozen=True)
class MutationCatalogEntry:
    mutation_id: str
    mutation_class: str
    single_semantic_change: bool = True
    requires_activation: bool = True


PPP_MUTATION_CATALOG = {
    key: MutationCatalogEntry(key, cls)
    for key, cls in {
        "A1": "TREE_DANGEROUS_TOP1_MASKED", "A2": "TREE_DANGEROUS_TOP1_MASKED",
        "B1": "MASK_BYPASS", "B2": "NO_FIRST_VALID", "B3": "ALL_INVALID_FORCE_TOP1", "B4": "GUARD_REMOVAL",
        "B5": "MASK_BYPASS", "C1": "ACTION_RATIO_2_TO_5", "C2": "ROUNDING_TO_NEAREST", "C3": "RETROACTIVE_RELEASE_BUDGET",
        "D1": "ENVELOPE_GRADIENT", "E1": "E1_DEADLINE_CLEANUP_REMOVE", "E2": "E2_HI_JOB_TRUNCATE",
        "E3": "E3_EVENT_ORDER", "E4": "E4_CONTROLLER_OVERHEAD", "E5": "E5_NONQUIESCENT_RECOVERY", "E6": "E6_UNSTABLE_DEMAND_READS",
        "F1": "BUNDLE_TREE_TAMPER", "F2": "BUNDLE_CROSS_SEED", "F3": "BUNDLE_CROSS_VARIANT", "F4": "BUNDLE_PRIORITY_TAMPER",
        "F5": "BUNDLE_WITNESS_TAMPER", "F6": "BUNDLE_ARTIFACT_DELETE", "F7": "SOURCE_BINDING_TAMPER",
    }.items()
}
