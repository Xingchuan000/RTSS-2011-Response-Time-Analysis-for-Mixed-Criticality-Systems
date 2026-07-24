from formal_toolchain.reference.protected_priority_prefix.construction import build_saturated_protected_prefix
from formal_toolchain.reference.protected_priority_prefix.macro_step import prove_protected_macro_step_preservation
from formal_toolchain.reference.task_mapping import ReferenceTask, ReferenceTaskset


def test_macro_step_receipt_contains_tail_stutter_and_canonical_closure():
    full = ReferenceTaskset((
        ReferenceTask("lo", 20, 20, 4, 2, "LO", 0, 3, 3, 2, 0),
        ReferenceTask("hi", 25, 25, 2, 5, "HI", 1, 2, 5, None, 0),
        ReferenceTask("tail", 40, 40, 3, 1, "LO", 2, 2, 2, 1, 0),
    ), "a" * 64)
    construction = build_saturated_protected_prefix(full, source_context_hash="a" * 64)
    receipt = prove_protected_macro_step_preservation(construction=construction, full_taskset=full, prefix_taskset=construction.prefix_taskset)
    assert receipt["status"] == "UNRESOLVED"
    assert receipt["canonical_phase_sequence"][:3] == ["SvcEnd", "REM", "REC?"]
    assert receipt["tail_exclusion"]["protected_ready_implies_protected_dispatch"]
