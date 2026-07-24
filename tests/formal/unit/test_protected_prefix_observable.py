from dataclasses import replace

from formal_toolchain.reference.executable_semantics import close_timestamp, initial_reference_state
from formal_toolchain.reference.protected_priority_prefix.construction import build_saturated_protected_prefix
from formal_toolchain.reference.protected_priority_prefix.observable import project_protected_state
from formal_toolchain.reference.protected_priority_prefix.state_relation import rel_pp_close
from formal_toolchain.reference.task_mapping import ReferenceTask, ReferenceTaskset


def _taskset():
    return ReferenceTaskset((
        ReferenceTask("lo", 20, 20, 4, 2, "LO", 0, 3, 3, 2, 0),
        ReferenceTask("hi", 25, 25, 2, 5, "HI", 1, 2, 5, None, 0),
        ReferenceTask("tail", 40, 40, 3, 1, "LO", 2, 2, 2, 1, 0),
    ), "a" * 64)


def test_global_mode_is_not_part_of_protected_observable_relation():
    full = _taskset()
    construction = build_saturated_protected_prefix(full, source_context_hash="a" * 64)
    prefix = construction.prefix_taskset
    full_state = close_timestamp(initial_reference_state(full), full)
    prefix_state = close_timestamp(initial_reference_state(prefix), prefix)
    assert rel_pp_close(full_state, prefix_state, construction=construction, full_taskset=full, prefix_taskset=prefix)
    assert rel_pp_close(replace(full_state, mode="HI"), prefix_state, construction=construction, full_taskset=full, prefix_taskset=prefix)
    assert project_protected_state(full_state, protected_task_names=frozenset(construction.protected_task_names), taskset=full).jobs
