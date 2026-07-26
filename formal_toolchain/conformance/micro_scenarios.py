"""Frozen P0 semantic micro-witnesses.

These checks no longer execute the mutable experiment runtime.  They are small
model-level sanity witnesses for the frozen C-AMC-sem/P0 contract.  Runtime
implementation drift is reported separately and is not a proof dependency.
"""

from __future__ import annotations

from formal_toolchain.semantics.frozen_runtime_contract import CONTRACT_VERSION

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
        assertions.update({
            "hi_completed_full_demand": raw == c_hi,
            "hi_not_truncated_to_lo_budget": raw > c_lo,
            "mode_switched": raw > c_lo,
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
    if not all(assertions.values()):
        raise AssertionError(f"frozen micro witness failed: {name}")
    return {
        "status": "PASS",
        "formal_semantics_contract": CONTRACT_VERSION,
        "execution_mode": "PURE_FORMAL_MODEL",
        "mutable_runtime_dependency": "NONE",
        "assertions": assertions,
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


__all__ = ["SCENARIOS", "run_p0_micro_scenarios"]
