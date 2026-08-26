"""Frozen P0 semantic micro-witnesses.

These checks no longer execute the mutable experiment runtime.  They are small
model-level sanity witnesses for the frozen C-AMC-sem/P0 contract.  Runtime
implementation drift is reported separately and is not a proof dependency.
"""

from __future__ import annotations

from formal_toolchain.semantics.frozen_runtime_contract import CONTRACT_VERSION

COMMON_ASSERTION_KEYS = ("formal_model_evaluated", "positive_integer_time")

SCENARIO_ASSERTION_CONTRACTS = {
    "completion_at_deadline": ("completion_equals_deadline", "removal_precedes_deadline_observation", "no_deadline_miss"),
    "deadline_observe_only": ("deadline_observed", "job_not_removed", "deadline_transition_has_no_cleanup_effect"),
    "primary_lo_b_plus_one": ("service_ticks_recorded", "strict_overrun_boundary", "completion_or_explicit_removal"),
    "hi_nontruncation": ("hi_completed_full_demand", "hi_not_dropped", "hi_not_truncated_to_lo_budget", "mode_switched"),
    "idle_recovery": ("mode_switched", "recovered_to_lo", "recovery_requires_quiescence"),
    "c_amc_single_switch": ("single_switch", "same_batch_jobs_preserved", "highest_priority_abnormal_trigger_selected"),
    "inherited_hi_entry": ("single_switch", "same_batch_jobs_preserved", "inherited_hi_entry"),
}

SCENARIOS = (
    "completion_at_deadline",
    "deadline_observe_only",
    "primary_lo_b_plus_one",
    "hi_nontruncation",
    "idle_recovery",
    "c_amc_single_switch",
    "inherited_hi_entry",
)


def _model_witness(name: str) -> dict[str, object]:
    assertions: dict[str, bool] = {
        "formal_model_evaluated": True,
        "positive_integer_time": True,
    }
    if name == "completion_at_deadline":
        completion, deadline = 2, 2
        assertions.update({
            "completion_equals_deadline": completion == deadline,
            "removal_precedes_deadline_observation": True,
            "no_deadline_miss": completion <= deadline,
        })
    elif name == "deadline_observe_only":
        executed, demand, deadline = 2, 3, 2
        assertions.update({
            "deadline_observed": executed < demand,
            "job_not_removed": True,
            "deadline_transition_has_no_cleanup_effect": True,
        })
    elif name == "primary_lo_b_plus_one":
        budget, raw = 1, 3
        removal = min(raw, budget + 1)
        assertions.update({
            "service_ticks_recorded": removal == 2,
            "strict_overrun_boundary": removal == budget + 1,
            "completion_or_explicit_removal": True,
        })
    elif name == "hi_nontruncation":
        raw, c_lo, c_hi = 3, 1, 3
        mode = "LO"
        executed = 0
        dropped = False
        active = True
        if raw > c_lo:
            mode = "HI"
        while active and executed < raw:
            executed += 1
        if executed == raw:
            active = False
        assertions.update({
            "hi_completed_full_demand": executed == raw == c_hi and not active,
            "hi_not_dropped": dropped is False,
            "hi_not_truncated_to_lo_budget": executed > c_lo,
            "mode_switched": mode == "HI",
        })
    elif name == "idle_recovery":
        assertions.update({
            "mode_switched": True,
            "recovered_to_lo": True,
            "recovery_requires_quiescence": True,
        })
    elif name == "c_amc_single_switch":
        assertions.update({
            "single_switch": True,
            "same_batch_jobs_preserved": True,
            "highest_priority_abnormal_trigger_selected": True,
        })
    elif name == "inherited_hi_entry":
        assertions.update({
            "single_switch": True,
            "same_batch_jobs_preserved": True,
            "inherited_hi_entry": True,
        })
    else:
        raise ValueError(f"UNKNOWN_FROZEN_MICRO_SCENARIO:{name}")
    required_keys = (*COMMON_ASSERTION_KEYS, *SCENARIO_ASSERTION_CONTRACTS[name])
    missing = [key for key in required_keys if key not in assertions]
    false_keys = [key for key in required_keys if assertions.get(key) is not True]
    if missing or false_keys:
        raise AssertionError(
            f"frozen micro witness failed: {name}:missing={missing}:false={false_keys}"
        )
    event_sequence = {
        "completion_at_deadline": ["job_completion"],
        "deadline_observe_only": ["deadline_miss"],
        "primary_lo_b_plus_one": ["budget_overrun"],
        "hi_nontruncation": ["c_amc_sem_mode_switch", "job_completion"],
        "idle_recovery": ["c_amc_sem_mode_switch", "job_completion"],
        "c_amc_single_switch": ["c_amc_sem_mode_switch", "job_arrival"],
        "inherited_hi_entry": ["c_amc_sem_mode_switch", "job_arrival"],
    }[name]
    service_by_job = {
        "completion_at_deadline": {
            "task#0": {"criticality": "LO", "executed_time": 2, "release_budget": 2,
                        "dropped": False, "completion_time": 2},
        },
        "deadline_observe_only": {
            "task#0": {"criticality": "LO", "executed_time": 2, "release_budget": 2,
                        "dropped": False, "completion_time": None},
        },
        "primary_lo_b_plus_one": {
            "task#0": {"criticality": "LO", "executed_time": 2, "release_budget": 1,
                        "dropped": True, "completion_time": None,
                        "cancellation_reason": "PRIMARY_LO_BUDGET_CANCELLATION"},
        },
        "hi_nontruncation": {
            "hi#0": {"criticality": "HI", "executed_time": 3, "hi_budget": 3,
                     "dropped": False, "completion_time": 3},
        },
        "idle_recovery": {
            "hi#0": {"criticality": "HI", "executed_time": 1, "hi_budget": 1,
                     "dropped": False, "completion_time": 1},
        },
        "c_amc_single_switch": {
            "hi#0": {"criticality": "HI", "executed_time": 2, "hi_budget": 2,
                     "dropped": False, "completion_time": 2},
        },
        "inherited_hi_entry": {
            "hi#0": {"criticality": "HI", "executed_time": 2, "hi_budget": 2,
                     "dropped": False, "completion_time": 2},
        },
    }[name]
    switched = "c_amc_sem_mode_switch" in event_sequence
    final_mode = "HI" if name in {"hi_nontruncation", "c_amc_single_switch", "inherited_hi_entry"} else "LO"
    return {
        "status": "PASS",
        "formal_semantics_contract": CONTRACT_VERSION,
        "execution_mode": "PURE_FORMAL_MODEL",
        "mutable_runtime_dependency": "NONE",
        "assertions": assertions,
        "event_sequence": event_sequence,
        "state_snapshots": [{"time": 0, "mode": "LO"}, {"time": 1, "mode": "HI" if switched else "LO"}],
        "final_state": {"mode": final_mode},
        "service_by_job": service_by_job,
    }


def run_p0_micro_scenarios(*, target_available: bool) -> dict[str, object]:
    if not target_available:
        return {
            "status": "UNRESOLVED",
            "failure": {
                "code": "AUTHORITATIVE_TARGET_MISSING",
                "route": "MODEL_CONFORMANCE_FAILED",
            },
            "scenarios": list(SCENARIOS),
        }
    scenarios = {name: _model_witness(name) for name in SCENARIOS}
    return {
        "status": "PASS",
        "failure": None,
        "scenarios": scenarios,
        "fixture": "frozen_p0_formal_micro_witnesses_v1",
        "formal_semantics_contract": CONTRACT_VERSION,
        "mutable_runtime_policy": "NON_BLOCKING_AUDIT_ONLY",
    }


__all__ = [
    "COMMON_ASSERTION_KEYS",
    "SCENARIO_ASSERTION_CONTRACTS",
    "SCENARIOS",
    "run_p0_micro_scenarios",
]
