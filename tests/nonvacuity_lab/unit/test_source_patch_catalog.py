from __future__ import annotations

from pathlib import Path

from nonvacuity_lab.manifest import load_campaign
from nonvacuity_lab.mutators.base import MutationContext
from nonvacuity_lab.mutators.runtime_source import (
    MultiPythonSymbolMutation,
    PythonSymbolMutation,
)


def test_all_declared_mutation_blind_source_patches_bind_current_symbols():
    root = Path(__file__).resolve().parents[3]
    config = load_campaign(root / "configs/nonvacuity/ppp_full_campaign.json")
    ids = {
        "B1_s185_same_tree_mask_bypass",
        "B5_s397_tail_same_tree_mask_bypass",
        "B2_no_first_valid",
        "B3_all_invalid_force_top1",
        "C2_round_nearest",
        "C3_active_release_budget",
        "E1_deadline_cleanup_remove",
        "E2_hi_budget_truncation",
        "E3_same_timestamp_event_order",
        "E4_nonzero_controller_overhead",
        "E5_nonquiescent_recovery",
        "E6_unstable_demand_reads",
    }
    manifests = {item.mutation_id: item for item in config.mutations}
    assert ids <= set(manifests)
    failures = {}
    for mutation_id in sorted(ids):
        parameters = dict(manifests[mutation_id].mutator["parameters"])
        context = MutationContext(
            mutation_id=mutation_id,
            source_root=root,
            mutated_seed=None,
            source_overlay=root,
            parameters=parameters,
        )
        mutator = (
            MultiPythonSymbolMutation()
            if isinstance(parameters.get("patches"), list)
            else PythonSymbolMutation()
        )
        result = mutator.preflight(context)
        if result.status != "PASS":
            failures[mutation_id] = dict(result.details)
    assert failures == {}
