from formal_toolchain.reference.executable_semantics import initial_reference_state
from formal_toolchain.reference.protected_priority_prefix.construction import build_saturated_protected_prefix
from formal_toolchain.reference.protected_priority_prefix.input_projection import (
    build_prefix_initial_state_from_full_inputs, check_projected_demands_legal,
    project_protected_release_stream,
)
from formal_toolchain.reference.task_mapping import ReferenceTask, ReferenceTaskset


def test_projection_preserves_keys_and_demands_without_tail():
    full = ReferenceTaskset((
        ReferenceTask("lo", 20, 20, 4, 2, "LO", 0, 3, 3, 2, 0),
        ReferenceTask("hi", 25, 25, 2, 5, "HI", 1, 2, 5, None, 0),
        ReferenceTask("tail", 40, 40, 3, 1, "LO", 2, 2, 2, 1, 0),
    ), "a" * 64)
    construction = build_saturated_protected_prefix(full, source_context_hash="a" * 64)
    initial = initial_reference_state(full, release_demand_overrides={
        ("lo", 0): 3, ("hi", 0): 2, ("tail", 0): 2,
    })
    projected = project_protected_release_stream(initial, protected_task_names=frozenset(construction.protected_task_names))
    assert {item.job_key for item in projected} == {("lo", 0), ("hi", 0)}
    assert {item.actual_demand for item in projected} == {3, 2}
    assert check_projected_demands_legal(projected, construction.prefix_taskset)["status"] == "PASS"
    prefix = build_prefix_initial_state_from_full_inputs(initial, construction.prefix_taskset, construction)
    assert {key[0] for key in prefix.frontier[0].batch_jobs} <= set(construction.protected_task_names)
