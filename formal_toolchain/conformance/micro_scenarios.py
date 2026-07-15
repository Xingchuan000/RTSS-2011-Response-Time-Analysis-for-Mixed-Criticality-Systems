"""通过真实 amc_py event runtime 执行的六个 P0 合成微场景。"""

from __future__ import annotations

from amc_py.event_runtime import simulate_ordered_taskset_event_driven
from amc_py.models import Criticality, Task
from amc_py.runtime_models import RuntimeConfig, RuntimeSemantics
from amc_py.runtime_scenarios import make_table_scenario

SCENARIOS = ("completion_at_deadline", "deadline_observe_only", "primary_lo_b_plus_one",
             "hi_nontruncation", "idle_recovery", "c_amc_single_switch")


def _run(name: str) -> dict[str, object]:
    if name in {"completion_at_deadline", "deadline_observe_only"}:
        tasks = (Task("SYN_HI", 10, 2, 2, 3, Criticality.HI), Task("SYN_LO", 12, 6, 1, 1, Criticality.LO))
        actual = {("SYN_HI", 0): 2 if name == "completion_at_deadline" else 3, ("SYN_LO", 0): 1}
        end_time = 3
    elif name == "c_amc_single_switch":
        tasks = (Task("SYN_HI_A", 10, 5, 1, 2, Criticality.HI), Task("SYN_HI_B", 10, 5, 1, 2, Criticality.HI), Task("SYN_LO", 12, 6, 1, 1, Criticality.LO))
        actual = {("SYN_HI_A", 0): 2, ("SYN_HI_B", 0): 2, ("SYN_LO", 0): 1}
        end_time = 6
    else:
        tasks = (Task("SYN_HI", 10, 5, 1, 3, Criticality.HI), Task("SYN_LO", 12, 6, 1, 1, Criticality.LO))
        actual = {("SYN_HI", 0): 3 if name in {"hi_nontruncation", "idle_recovery"} else 1,
                  ("SYN_LO", 0): 3 if name == "primary_lo_b_plus_one" else 1}
        end_time = 6
    config = RuntimeConfig(end_time=end_time, capture_trace=True, capture_debug_events=True,
                           semantics=RuntimeSemantics.C_AMC_SEM if name != "primary_lo_b_plus_one" else RuntimeSemantics.AMC_PLUS,
                           c_amc_sem_primary_on_switch_time=True, drop_lo_jobs_on_hi_switch=False)
    scenario = make_table_scenario(actual, default_hi="c_lo", default_lo="c_lo")
    result = simulate_ordered_taskset_event_driven(tasks, scenario, config=config)
    snapshots = [{"time": tick.time, "running_task": tick.executing_task,
                  "mode": tick.mode.value if hasattr(tick.mode, "value") else str(tick.mode)}
                 for tick in result.trace]
    events = [str(item.get("event")) for item in result.debug_events]
    jobs = {job.task.name: job for job in result.jobs}
    assertions = {"runtime_executed": bool(result.trace), "jobs_created": bool(result.jobs),
                  "final_mode_recorded": result.final_mode is not None}
    if name == "completion_at_deadline":
        assertions.update({"completion_equals_deadline": jobs["SYN_HI"].completion_time == jobs["SYN_HI"].absolute_deadline,
                           "no_deadline_miss": not result.deadline_misses})
    elif name == "deadline_observe_only":
        assertions.update({"deadline_observed": bool(result.deadline_misses), "job_not_removed": not jobs["SYN_HI"].dropped})
    elif name == "hi_nontruncation":
        assertions.update({"hi_completed_full_demand": jobs["SYN_HI"].completion_time == 3,
                           "hi_not_dropped": not jobs["SYN_HI"].dropped, "mode_switched": len(result.mode_switches) == 1})
    elif name == "primary_lo_b_plus_one":
        assertions.update({"service_ticks_recorded": jobs["SYN_LO"].executed_time == 2,
                           "completion_or_explicit_removal": jobs["SYN_LO"].completion_time is not None or bool(result.job_cancellations)})
    elif name == "idle_recovery":
        assertions.update({"mode_switched": len(result.mode_switches) == 1, "recovered_to_lo": len(result.mode_recoveries) == 1})
    elif name == "c_amc_single_switch":
        assertions.update({"single_switch": len(result.mode_switches) == 1,
                           "same_batch_jobs_preserved": all(not job.dropped for job in result.jobs)})
    if not all(assertions.values()):
        raise AssertionError(f"synthetic scenario {name} runtime evidence incomplete")
    return {"initial_state": {"mode": "LO", "tasks": [task.name for task in tasks]},
            "event_sequence": events, "state_snapshots": snapshots,
            "service_by_job": {key: {"executed_time": job.executed_time, "completion_time": job.completion_time,
                                     "dropped": job.dropped} for key, job in jobs.items()},
            "removal_reason": [event.reason for event in result.job_cancellations],
            "deadline_flags": {"deadline_miss_count": len(result.deadline_misses)},
            "mode_changes": {"switches": len(result.mode_switches), "recoveries": len(result.mode_recoveries)},
            "assertions": assertions,
            "final_state": {"mode": result.final_mode.value, "job_count": len(result.jobs),
                            "mode_switch_count": len(result.mode_switches)},
            "status": "PASS"}


def run_p0_micro_scenarios(*, target_available: bool) -> dict[str, object]:
    if not target_available:
        return {"status": "UNRESOLVED", "failure": {"code": "AUTHORITATIVE_TARGET_MISSING",
                "route": "MODEL_CONFORMANCE_FAILED"}, "scenarios": list(SCENARIOS)}
    scenarios: dict[str, object] = {}
    try:
        for name in SCENARIOS:
            scenarios[name] = _run(name)
    except Exception as exc:
        return {"status": "UNRESOLVED", "failure": {"code": "MICRO_SCENARIO_RUNTIME_FAILED",
                "route": "MODEL_CONFORMANCE_FAILED", "detail": str(exc)}, "scenarios": scenarios}
    return {"status": "PASS", "failure": None, "scenarios": scenarios,
            "fixture": "synthetic_p0_micro_scenarios_v2"}
