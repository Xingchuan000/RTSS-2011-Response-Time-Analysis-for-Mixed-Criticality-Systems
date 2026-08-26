from __future__ import annotations

from amc_py.budget_runtime import BudgetState
from amc_py.event_runtime import EventRuntimeEngine
from amc_py.models import Criticality, Task
from amc_py.runtime_models import RuntimeConfig, RuntimeSemantics
from amc_py.runtime_scenarios import make_nominal_scenario
from formal_toolchain.adapters.formal_runtime_snapshot import build_formal_runtime_snapshot
from formal_toolchain.binding.controller_binding import bind_controller_runtime
from formal_toolchain.bridge.controller_reschedule import compare_controller_state_frame


def test_selected_action_binding_contains_complete_n3_witness() -> None:
    binding = bind_controller_runtime(".")
    selected = binding["selected_action_runtime_binding"]

    assert selected["status"] == "PASS"
    assert selected["source"] == "amc_py/rl/env.py:AmcBudgetEnv.step"
    assert selected["source_kind"] == "CONTROLLER_SYNCHRONOUS"
    assert selected["source_binding"] == "self._engine.apply_budget_updates"
    for field in (
        "payload_prevalidated", "no_partial_mutation", "zero_time", "time_unchanged",
        "mode_unchanged", "active_jobs_unchanged", "ready_jobs_unchanged",
        "running_job_unchanged", "released_job_fields_unchanged",
        "released_job_snapshot_unchanged", "released_job_service_unchanged",
        "released_job_demand_unchanged", "released_job_classification_unchanged",
        "completion_miss_unchanged", "service_unchanged",
        "effective_event_frontier_unchanged", "plant_progression_separated",
    ):
        assert selected[field] is True
    assert selected["timing_projection"] == "STUTTER"


def test_production_engine_preserves_concrete_preclosed_frame_on_force_reschedule() -> None:
    task = Task("task", 10, 10, 2, 2, Criticality.LO)
    engine = EventRuntimeEngine.build(
        ordered_tasks=[task],
        scenario=make_nominal_scenario(),
        config=RuntimeConfig(end_time=20, semantics=RuntimeSemantics.C_AMC_SEM),
        budget_state=BudgetState.from_tasks([task]),
    )
    engine.run_until(0, include_boundary=True)
    before_snapshot = build_formal_runtime_snapshot(engine, {"task": 0})
    before_running = engine.state.running_job
    assert before_running is not None

    engine.apply_budget_updates({"task": 3})

    after_snapshot = build_formal_runtime_snapshot(engine, {"task": 0})
    after_running = engine.state.running_job
    assert after_running is not None

    def frame(snapshot, running):
        frontier = tuple(
            (event.time, event.phase_rank, event.kind.value, event.job_key, event.batch_jobs)
            for event in snapshot.effective_event_frontier
        )
        return {
            "current_time": snapshot.time,
            "mode": snapshot.mode,
            "active_keys": snapshot.active_job_keys,
            "ready_keys": snapshot.active_job_keys,
            "running_key": (running.task.name, running.release_index),
            "released_job_snapshot": snapshot.released_ledger,
            "released_job_service": running.executed_time,
            "released_job_demand": snapshot.released_ledger[0].removal_demand,
            "released_job_classification": snapshot.released_ledger[0].release_class,
            "completion_status": snapshot.terminal_ledger,
            "miss_status": snapshot.miss_ledger,
            "service": running.executed_time,
            "effective_event_frontier": frontier,
        }

    relation = compare_controller_state_frame(
        before=frame(before_snapshot, before_running),
        after=frame(after_snapshot, after_running),
    )
    assert relation["status"] == "PASS"
    assert relation["running_key_unchanged"] is True
    assert relation["completion_miss_unchanged"] is True
    assert relation["effective_event_frontier_unchanged"] is True
