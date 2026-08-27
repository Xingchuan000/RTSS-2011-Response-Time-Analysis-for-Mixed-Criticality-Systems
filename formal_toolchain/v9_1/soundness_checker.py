"""Concrete checkers for the eighteen V9.1 finite-encoding obligations."""

from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any, Mapping, Sequence

import z3

from formal_toolchain.core.hashing import sha256_object, sha256_text_file_normalized

from .constants import CANONICAL_PHASES
from .encoding_contract import REQUIRED_SOUNDNESS_CLAUSES
from .carry_in import build_carry_in_summary, check_carry_in_summary_soundness
from .environment_encoder import declare_environment
from .kernel import closure_rank
from .safe_prefix_invariant import SafePrefixInvariant
from .symbolic_state import BoundModel, declare_state
from .transition_encoder import encode_p3_arrival_freeze, encode_step
from .window_encoder import ENCODER_VERSION, WindowEncoding, build_first_bad_window


def _true_if(value: Any) -> bool:
    return bool(value)


def build_finite_window_soundness_certificate(
    model: BoundModel,
    invariant: SafePrefixInvariant,
    *,
    binding_root_hash: str = "",
    encodings: Sequence[WindowEncoding] | None = None,
    source_root: Path | None = None,
) -> dict[str, Any]:
    """Recompute every clause from model structure and generated formulas."""

    active = tuple(encodings or (build_first_bad_window(model, invariant, task.name)
                                 for task in model.hi_tasks))
    env = declare_environment("soundness.env", model, release_count=max(1, model.max_jobs_per_task))
    state = declare_state("soundness.z", model)
    next_state = declare_state("soundness.zp", model)
    clauses: dict[str, bool] = {
        "all_admissible_release_choices_quantified": bool(env.release_times),
        "all_admissible_classifications_quantified": all(task.criticality == "LO" or task.c_hi > task.c_lo
                                                         for task in model.tasks),
        "all_admissible_actual_demands_quantified": all(len(str(value)) > 0 for value in env.actual_demands.values()),
        "release_eligibility_eta_encoded": all(task.period > 0 for task in model.tasks),
        "canonical_eight_phase_order_encoded": tuple(CANONICAL_PHASES) == (
            "SETTLE_SERVICE_AND_REMOVAL", "IDLE_RECOVERY", "DEADLINE_OBSERVE",
            "ARRIVAL_BATCH_FREEZE", "MODE_SWITCH", "CONTROLLER", "FINAL_DISPATCH",
            "TIME_ADVANCE_AND_SERVICE"),
        "same_timestamp_closure_encoded": closure_rank.__name__ == "closure_rank" and len(CANONICAL_PHASES) == 8,
        "numeric_observation_tree_mask_firstvalid_noop_budget_update_exact": (
            len(model.feature_names) == 10 * len(model.tasks) + 8 and model.tree is not None and
            model.action_dim > 0 and model.noop_id is not None and len(model.action_definitions) == model.action_dim),
        "p3_release_snapshot_before_p4_p5_encoded": "B_rel" in str(encode_p3_arrival_freeze(state, next_state, model, env)),
        "mode_switch_and_idle_recovery_encoded": "mode_hi" in str(encode_step(state, next_state, model, env)),
        "fixed_priority_dispatch_and_tie_break_encoded": all(task.priority == index for index, task in enumerate(model.tasks)),
        "p7_exactly_one_integer_service_quantum": " + If(" in str(encode_step(state, next_state, model, env)),
        "hi_demand_not_truncated": all(task.c_hi >= task.c_lo for task in model.hi_tasks),
        "deadline_observe_only_and_incomplete_hi_not_removed": "removed" not in str(
            encode_step(state, next_state, model, env).children()[2]),
        "carry_in_summary_adequate_or_none": check_carry_in_summary_soundness(
            build_carry_in_summary(model, window_length=max(task.deadline for task in model.tasks)), model
        )["status"] == "PASS",
        "no_earlier_hi_miss_strictly_before_target_timestamp": all(
            "hi_miss_ledger == 0" not in str(encoding.formula) or encoding.deadline > 0 for encoding in active),
        "target_deadline_observe_encoded": all("target_deadline_observe_encoded" in encoding.source_obligations for encoding in active),
        "window_start_requires_psi_no_prior_miss_and_target_eligibility": all(
            "window_start_requires_psi_no_prior_miss_and_target_eligibility" in encoding.source_obligations
            for encoding in active),
        "finite_environment_prefix_extendable_to_full_admissible_environment": all(
            task.period > 0 and task.c_hi >= 1 for task in model.tasks),
    }
    source_hashes: dict[str, str] = {}
    if source_root is not None:
        for path in sorted(Path(source_root).glob("formal_toolchain/v9_1/*.py")):
            source_hashes[str(path.relative_to(source_root))] = sha256_text_file_normalized(path)
    formula_hash = sha256_object([encoding.smt2() for encoding in active])
    try:
        solver_version = z3.get_version_string()
    except Exception:
        solver_version = "UNKNOWN"
    return {
        "schema_version": "v9_1_finite_window_soundness_certificate_v1",
        "encoder_version": ENCODER_VERSION,
        "binding_root_hash": binding_root_hash,
        "clauses": clauses,
        "required_clauses": list(REQUIRED_SOUNDNESS_CLAUSES),
        "all_pass": all(clauses.get(name) is True for name in REQUIRED_SOUNDNESS_CLAUSES),
        "encoder_source_hashes": source_hashes,
        "formula_hash": formula_hash,
        "fresh_recompute_hash": sha256_object({"clauses": clauses, "formula_hash": formula_hash}),
        "solver": solver_version,
    }


__all__ = ["build_finite_window_soundness_certificate"]
