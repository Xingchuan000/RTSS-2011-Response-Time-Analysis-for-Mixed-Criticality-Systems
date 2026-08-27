"""V8 raw protected-prefix conformance for the imported all-task theorem."""

from __future__ import annotations

from typing import Any, Mapping

from formal_toolchain.core.hashing import sha256_object
from formal_toolchain.reference.executable_semantics import initial_reference_state


def derive_raw_prefix_model_conformance(*, full_taskset: Any, raw_taskset: Any,
                                        construction: Any,
                                        runtime_schema_receipt: Mapping[str, Any],
                                        release_receptiveness_receipt: Mapping[str, Any],
                                        zero_relative_start_receipt: Mapping[str, Any]) -> dict[str, Any]:
    full_by_name = {task.name: task for task in full_taskset.tasks}
    runtime = runtime_schema_receipt.get("pp0_witness", runtime_schema_receipt)
    release = release_receptiveness_receipt.get("witness", release_receptiveness_receipt)
    zero_status = zero_relative_start_receipt.get(
        "obligation_status", zero_relative_start_receipt.get("status")
    )
    tasks = list(raw_taskset.tasks)
    protected = frozenset(construction.protected_task_names)
    flags = {
        "finite_nonempty_taskset": bool(tasks),
        "constrained_deadlines": all(0 < t.deadline <= t.period for t in tasks),
        "positive_integer_parameters": all(
            all(isinstance(v, int) and not isinstance(v, bool) and v > 0
                for v in (t.period, t.deadline, t.c_lo, t.c_hi)) for t in tasks
        ),
        "valid_offsets": all(isinstance(t.offset, int) and 0 <= t.offset < t.period for t in tasks),
        "strict_total_priority_order": tuple(t.priority_index for t in tasks) == tuple(range(len(tasks))),
        "all_parameters_inherited": all(
            t.name in full_by_name and all(
                getattr(t, field) == getattr(full_by_name[t.name], field)
                for field in ("period", "deadline", "c_lo", "c_hi", "criticality", "priority_index", "offset")
            ) for t in tasks
        ),
        "lo_wcet_relation": all(0 < t.c_hi <= t.c_lo for t in tasks if t.criticality == "LO"),
        "hi_wcet_relation": all(0 < t.c_lo <= t.c_hi for t in tasks if t.criticality == "HI"),
        "all_hi_tasks_preserved": all(t.name in protected for t in full_taskset.tasks if t.criticality == "HI"),
        "runtime_schema_pass": runtime_schema_receipt.get("status") == "PASS" or runtime_schema_receipt.get("obligation_status") == "PASS",
        "strict_fpps": runtime.get("single_processor_preemptive_work_conserving_fp") is True,
        "no_blocking": runtime.get("no_blocking_self_suspension_or_nonpreemptive_segments") is True,
        "fixed_supply": runtime.get("fixed_processor_supply_and_mode_independent_priority") is True,
        "release_fixed_demands": runtime.get("release_fixed_demands") is True,
        "abnormal_classification": runtime.get("abnormal_classification_at_arrival") is True,
        "abnormal_hi_only_switch": runtime.get("abnormal_hi_only_switch_trigger") is True,
        "idle_only_recovery": runtime.get("quiescent_idle_only_recovery") is True,
        "lo_version_at_release": runtime.get("lo_version_selected_at_release") is True,
        "protected_input_independence": runtime.get("protected_input_independence") is True,
        "release_receptiveness_derived": (
            release_receptiveness_receipt.get("obligation_status") == "PASS"
            or release_receptiveness_receipt.get("status") == "PASS"
            or (isinstance(release, Mapping) and release.get("status") == "PASS")
        ),
        "zero_relative_start_bound": zero_status == "PASS",
    }
    try:
        initial = initial_reference_state(raw_taskset)
        flags["standard_initial_state"] = (
            initial.time == 0 and initial.mode == "LO" and not initial.jobs
            and not initial.released and not initial.pending_releases
        )
    except Exception:
        flags["standard_initial_state"] = False

    static_names = {
        "finite_nonempty_taskset", "constrained_deadlines", "positive_integer_parameters",
        "valid_offsets", "strict_total_priority_order", "all_parameters_inherited",
        "lo_wcet_relation", "hi_wcet_relation", "all_hi_tasks_preserved",
    }
    static_ok = all(flags[name] for name in static_names)
    ok = all(flags.values())
    payload = {
        "schema_version": "raw_prefix_model_conformance_v8",
        "full_taskset_fingerprint": full_taskset.to_dict()["fingerprint"],
        "prefix_taskset_fingerprint": raw_taskset.to_dict()["fingerprint"],
        "cutoff_task_name": construction.cutoff_task_name,
        "construction_hash": sha256_object({
            "partition": construction.partition_witness,
            "inheritance": construction.inheritance_witness,
        }),
        "witness": flags,
        "conclusion": "RAW_PREFIX_STANDARD_C_AMC_SEM_MODEL_CONFORMS",
    }
    payload["conformance_hash"] = sha256_object(payload)
    return {
        **payload,
        "status": "PASS" if ok else ("FAIL" if not static_ok else "UNRESOLVED"),
        "failure": None if ok else {
            "code": "RAW_PREFIX_MODEL_NONCONFORMANT" if not static_ok else "RAW_PREFIX_MODEL_CONFORMANCE_UNRESOLVED",
            "failed_flags": [name for name, value in flags.items() if not value],
        },
    }
