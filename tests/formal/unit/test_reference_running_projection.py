from formal_toolchain.bridge.model_bounds import _legacy_test_bounds
from formal_toolchain.reference.p0_projection import project_executable_reference_state
from formal_toolchain.reference.reference_state import ReferenceState


def test_running_scalar_is_boolean_and_identity_uses_running_job_key():
    state = ReferenceState(
        time=0,
        mode="LO",
        jobs={},
        released={},
        terminal={},
        misses=(),
        ready_order=(),
        running=("TaskX", 7),
        frontier=(),
    )
    projected = project_executable_reference_state(
        state=state,
        taskset={"tasks": []},
        bounds=_legacy_test_bounds(),
        job_slot_by_key={},
        task_slot_by_name={},
    )
    assert projected["running"] == 1
    assert projected["running_job_key"] != projected["running"]


def test_idle_running_scalar_and_key_are_zero():
    state = ReferenceState(
        time=0,
        mode="LO",
        jobs={},
        released={},
        terminal={},
        misses=(),
        ready_order=(),
        running=None,
        frontier=(),
    )
    projected = project_executable_reference_state(
        state=state,
        taskset={"tasks": []},
        bounds=_legacy_test_bounds(),
        job_slot_by_key={},
        task_slot_by_name={},
    )
    assert projected["running"] == 0
    assert projected["running_job_key"] == 0
