from pathlib import Path

from formal_toolchain.bridge.handler_decomposition import prove_reschedule_partition
from formal_toolchain.bridge.runtime_branch_map import (
    RESCHEDULE_CASE_IDS,
    bind_reschedule_branch_families,
)


ROOT = Path(__file__).parents[3]


def test_reschedule_partition_is_exhaustive_and_pairwise_exclusive():
    result = prove_reschedule_partition()
    assert result["status"] == "PASS"
    assert result["cases"] == list(RESCHEDULE_CASE_IDS)
    assert result["exhaustive"] is True
    assert result["pairwise_exclusive"] is True


def test_reschedule_families_are_bound_to_real_source_and_not_one_cfg_path():
    bindings = bind_reschedule_branch_families(ROOT)
    assert tuple(bindings) == RESCHEDULE_CASE_IDS
    assert all(binding.entry_function == "_reschedule" for binding in bindings.values())
    assert len(bindings["RESCHEDULE_KEEP_SAME"].effect_ir) == 1
    assert all(binding.branch_family_hash for binding in bindings.values())
    assert bindings["RESCHEDULE_TO_IDLE"].branch_family_hash != bindings["PREEMPTION_DISPATCH"].branch_family_hash

