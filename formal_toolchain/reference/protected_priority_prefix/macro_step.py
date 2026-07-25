"""Canonical closed-boundary macro-step contracts.

Implements the eight required lemmas (L1–L8) for proving Rel_pp_close
preservation across canonical macro-step boundaries.

Each lemma is parameterized over the phase-indexed relation and batch
cursors; none use bounded replay or source-hash matching as proof.

Phase G updates (Section 9):
  L1 tail service exclusion — binds static FP priority proof
  L2 final dispatch correspondence — two-case parametric proof
  L3 protected service correspondence — two-case parametric proof
  L4 completion/removal correspondence — equivalence proof
  L5 deadline-batch correspondence — cursor induction with fold lemma
  L6 arrival-batch projection — cursor induction with fold lemma
  L7 mode/tail phase join — identity skip reconstruction
  L8 canonical macro-step preservation — composes L1-L7
"""

from __future__ import annotations

from typing import Any, Mapping

from formal_toolchain.core.hashing import sha256_object

from .types import ProtectedPrefixBuildResult
from .phase_relation import build_phase_relation
from .batch_cursor import (
    BatchCursor, BatchCursorProof, BatchCursorFoldLemma,
    construct_batch_cursor, verify_batch_cursor_proof,
    construct_fold_lemma, verify_fold_lemma,
    prove_parameterized_fold_kernel,
)


def _tasks(taskset: object) -> tuple[Any, ...]:
    return tuple(taskset.tasks)


def _attach_predecessors(
    payload: dict[str, Any], predecessors: Mapping[str, Any],
) -> dict[str, Any]:
    """Record exact predecessor theorem IDs and artifact/receipt hashes."""
    theorem_ids = {
        "TAIL_SERVICE_EXCLUSION": "PPP_L1_TAIL_SERVICE_EXCLUSION",
        "FINAL_DISPATCH_CORRESPONDENCE": "PPP_L2_FINAL_DISPATCH_CORRESPONDENCE",
        "PROTECTED_SERVICE_CORRESPONDENCE": "PPP_L3_SERVICE_CORRESPONDENCE",
        "COMPLETION_REMOVAL_CORRESPONDENCE": "PPP_L4_COMPLETION_REMOVAL_CORRESPONDENCE",
        "DEADLINE_BATCH_CORRESPONDENCE": "PPP_L5_DEADLINE_BATCH_FOLD",
        "ARRIVAL_BATCH_PROJECTION": "PPP_L6_ARRIVAL_BATCH_FOLD",
        "MODE_TAIL_PHASE_JOIN": "PPP_L7_CANONICAL_PHASE_JOIN",
    }
    payload.setdefault("theorem_id", theorem_ids.get(str(payload.get("lemma"))))
    payload.setdefault("relation_schema", "phase_relation_v4_close_at")
    if payload.get("theorem_id") == "PPP_L5_DEADLINE_BATCH_FOLD":
        payload.setdefault("preserved_relation_fields", [
            "job_key", "criticality", "release_time", "absolute_deadline",
            "actual_demand", "executed_service", "completed", "missed",
            "miss_job_keys",
        ])
        payload.setdefault("post_deadline_ledger_theorem", payload.get("status") == "PASS")
    payload["predecessor_theorem_ids"] = {
        name: value.get("theorem_id") or value.get("obligation_id")
        for name, value in predecessors.items() if isinstance(value, Mapping)
    }
    payload["predecessor_receipt_hashes"] = {
        name: str(value.get("artifact_hash") or value.get("receipt_hash") or "")
        for name, value in predecessors.items() if isinstance(value, Mapping)
    }
    payload["artifact_hash"] = sha256_object({
        "theorem_id": payload.get("theorem_id") or payload.get("lemma"),
        "predecessor_theorem_ids": payload["predecessor_theorem_ids"],
        "predecessor_receipt_hashes": payload["predecessor_receipt_hashes"],
        "status": payload.get("status"),
    })
    return payload


# ---------------------------------------------------------------------------
# L1: Tail service exclusion (Section 9)
# ---------------------------------------------------------------------------


def prove_tail_service_exclusion(
    *,
    full_taskset: object,
    construction: ProtectedPrefixBuildResult,
    scheduler_semantics_receipt: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """L1: When protected ready set is non-empty, full scheduler cannot serve tail.

    Section 9 L1 requirements:
      - tail priority index > every protected priority index
      - strict FP scheduler
      - no mode-dependent priority

    The proof is structural: protected tasks have smaller priority_index
    (higher priority) than tail tasks.  A strict FP scheduler always
    selects the highest-priority ready job, so when protected ready is
    non-empty, no tail job can be selected.
    """
    protected = frozenset(construction.protected_task_names)
    tail = frozenset(construction.tail_task_names)
    ordered = _tasks(full_taskset)
    priorities = {task.name: int(task.priority_index) for task in ordered}

    all_protected_higher_priority = all(
        priorities[p] < priorities[t]
        for p in protected for t in tail
    )
    order_preserved = (
        tuple(task.name for task in ordered)
        == construction.protected_task_names + construction.tail_task_names
    )
    scheduler_bound = (
        isinstance(scheduler_semantics_receipt, Mapping)
        and scheduler_semantics_receipt.get("status") == "PASS"
        and scheduler_semantics_receipt.get("strict_fixed_priority_dispatch") is True
        and scheduler_semantics_receipt.get("no_mode_dependent_priority") is True
    )
    structural_ok = all_protected_higher_priority and order_preserved
    ok = structural_ok and scheduler_bound
    return _attach_predecessors({
        "status": "PASS" if ok else ("FAIL" if not structural_ok else "UNRESOLVED"),
        "code": None if ok else (
            "TAIL_PRIORITY_PARTITION_INVALID" if not structural_ok
            else "STRICT_FP_SCHEDULER_SEMANTICS_NOT_PROVED"
        ),
        "lemma": "TAIL_SERVICE_EXCLUSION",
        # The priority partition establishes the structural implication; the
        # executable scheduler receipt is a separate prerequisite for PASS.
        "protected_ready_implies_protected_dispatch": structural_ok,
        "all_protected_higher_priority": all_protected_higher_priority,
        "tail_priority_gt_every_protected_priority": all_protected_higher_priority,
        "strict_fp_scheduler": scheduler_bound,
        "no_mode_dependent_priority": scheduler_bound,
        "structural_priority_order_valid": structural_ok,
        "protected_task_names": sorted(protected),
        "tail_task_names": sorted(tail),
        "phase_relation": "RelPP_SvcEnd",
        "excluded": "tail_jobs",
        "lemma_hash": sha256_object({
            "protected": sorted(protected),
            "tail": sorted(tail),
            "priorities": {k: v for k, v in sorted(priorities.items())},
        }),
    }, {
        "STRICT_FP_WORK_CONSERVING_DISPATCH": scheduler_semantics_receipt or {},
    })


# ---------------------------------------------------------------------------
# L2: Final dispatch correspondence (Section 9)
# ---------------------------------------------------------------------------


def prove_final_dispatch_correspondence(
    *,
    construction: ProtectedPrefixBuildResult,
    tail_service_exclusion_receipt: Mapping[str, Any] | None = None,
    dispatch_semantics_receipt: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """L2: Total final-dispatch correspondence.

    Two cases (Section 9 L2):
    - protected ready nonempty -> same protected job selected
    - protected ready empty -> full may run tail, prefix idle;
      protected running projection both none
    """
    protected = frozenset(construction.protected_task_names)
    tail_exclusion_ok = (
        isinstance(tail_service_exclusion_receipt, Mapping)
        and tail_service_exclusion_receipt.get("status") == "PASS"
    )

    two_case_proof = {
        "case_1_protected_ready_nonempty": {
            "description": "Protected ready non-empty: both sides select the same protected job",
            "requires": [
                "same protected ready set (from RelPP_PreDisp)",
                "strict FP scheduler (same priority_index ordering)",
                "same tie-break (release_time, task_name, release_index)",
            ],
        },
        "case_2_protected_ready_empty": {
            "description": "Protected ready empty: full may serve tail, prefix idle",
            "requires": [
                "full prefix ready empty => protected running projection none",
                "tail-only dispatch does not affect protected observable",
            ],
        },
    }

    dispatch_bound = (
        isinstance(dispatch_semantics_receipt, Mapping)
        and dispatch_semantics_receipt.get("status") == "PASS"
        and dispatch_semantics_receipt.get("same_protected_ready_set_implies_same_selection") is True
    )
    return _attach_predecessors({
        "status": "PASS" if (tail_exclusion_ok and dispatch_bound) else "UNRESOLVED",
        "lemma": "FINAL_DISPATCH_CORRESPONDENCE",
        "code": None if (tail_exclusion_ok and dispatch_bound) else "PARAMETRIC_TRANSITION_PROOF_MISSING",
        "phase_relation": "RelPP_PreDisp",
        "two_case_proof": two_case_proof,
        "requires_pp0_smt2": True,
        "transition_case": "FINAL_DISPATCH",
        "reason": (
            "The two-case proof is structurally established by tail priority ordering "
            "and strict FP scheduling.  Full SMT2 verification of the FINAL_DISPATCH "
            "transition schema confirms the parameterized form."
        ),
    }, {
        "PPP_L1_TAIL_SERVICE_EXCLUSION": tail_service_exclusion_receipt or {},
        "FINAL_DISPATCH": dispatch_semantics_receipt or {},
    })


# ---------------------------------------------------------------------------
# L3: Protected service correspondence (Section 9)
# ---------------------------------------------------------------------------


def prove_protected_service_correspondence(
    *,
    construction: ProtectedPrefixBuildResult,
    pp0_receipts: Mapping[str, Any] | None = None,
    idle_jump_receipt: Mapping[str, Any] | None = None,
    service_relation_receipt: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """L3: Protected service correspondence.

    Two cases (Section 9 L3):
    - protected ready non-empty: both sides add the same unit of service to the same job.
    - protected empty: full tail service and prefix idle are both protected stutter.
    """
    protected = frozenset(construction.protected_task_names)

    pp0 = pp0_receipts or {}
    service_ok = pp0.get("SERVICE_UNIT", {}).get("status") == "PASS"
    tail_ok = pp0.get("TAIL_ONLY_SERVICE", {}).get("status") == "PASS"
    idle_ok = (
        isinstance(idle_jump_receipt, Mapping)
        and idle_jump_receipt.get("status") == "PASS"
        and idle_jump_receipt.get("parameterized") is True
        and idle_jump_receipt.get("independent_of_complete_execution_witness") is True
    )
    relation_ok = (
        isinstance(service_relation_receipt, Mapping)
        and service_relation_receipt.get("status") == "PASS"
        and service_relation_receipt.get("same_protected_running_job") is True
        and service_relation_receipt.get("same_unit_service_increment") is True
        and service_relation_receipt.get("tail_service_is_protected_stutter") is True
        and service_relation_receipt.get("relation_schema")
            == "phase_relation_v4_close_at"
    )
    established = service_ok and tail_ok and idle_ok and relation_ok
    return _attach_predecessors({
        "status": "PASS" if established else "UNRESOLVED",
        "lemma": "PROTECTED_SERVICE_CORRESPONDENCE",
        "code": None if established else "PARAMETRIC_TRANSITION_OR_IDLE_STUTTER_PROOF_MISSING",
        "phase_relation": "RelPP_SvcEnd",
        "two_case_proof": {
            "case_1_protected_ready_nonempty": {
                "description": "Protected ready non-empty: both SERVICE_UNIT on same job",
                "requires": ["RelPP_Close gives same protected running job",
                             "SERVICE_UNIT adds exactly 1 service"],
            },
            "case_2_protected_empty": {
                "description": "Protected ready empty: full TAIL_ONLY_SERVICE, prefix idle => both stutter on ObsP",
                "requires": [
                    "TAIL_ONLY_SERVICE does not modify protected observable",
                    "prefix idle jump is expanded by the parameterized CloseAt stutter theorem",
                ],
            },
        },
        "requires_pp0_smt2": True,
        "transition_cases": ["SERVICE_UNIT", "TAIL_ONLY_SERVICE"],
        "idle_jump_stutter_theorem_consumed": idle_ok,
        "relational_kernel_consumed": relation_ok,
        "reason": (
            "This obligation requires verifying SERVICE_UNIT and TAIL_ONLY_SERVICE "
            "transition schemas via SMT2 queries over the compiled transition IR."
        ),
    }, {
        "SERVICE_UNIT": pp0.get("SERVICE_UNIT", {}),
        "TAIL_ONLY_SERVICE": pp0.get("TAIL_ONLY_SERVICE", {}),
        "IDLE_JUMP": idle_jump_receipt or {},
        "SERVICE_RELATION": service_relation_receipt or {},
    })


# ---------------------------------------------------------------------------
# L4: Completion/removal correspondence (Section 9)
# ---------------------------------------------------------------------------


def prove_completion_removal_correspondence(
    *,
    construction: ProtectedPrefixBuildResult,
    pp0_receipts: Mapping[str, Any] | None = None,
    removal_relation_receipt: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """L4: Completion/removal correspondence.

    Same fixed demand + same accumulated service => same guard and same-time removal.
    After removal, protected ledger relation holds.
    """
    pp0 = pp0_receipts or {}
    pp0_ok = pp0.get("REM_COMPLETION", {}).get("status") == "PASS"
    relation_ok = (
        isinstance(removal_relation_receipt, Mapping)
        and removal_relation_receipt.get("status") == "PASS"
        and removal_relation_receipt.get("same_removal_guard") is True
        and removal_relation_receipt.get("same_terminal_record") is True
        and removal_relation_receipt.get("relation_reestablished_after_removal") is True
        and removal_relation_receipt.get("relation_schema")
            == "phase_relation_v4_close_at"
    )
    ok = pp0_ok and relation_ok
    return _attach_predecessors({
        "status": "PASS" if ok else "UNRESOLVED",
        "lemma": "COMPLETION_REMOVAL_CORRESPONDENCE",
        "code": None if ok else "PARAMETRIC_TRANSITION_PROOF_MISSING",
        "phase_relation": "RelPP_AfterREM",
        "equivalence_proof": {
            "antecedent": (
                "same fixed_demand AND same accumulated_service "
                "(from RelPP_Close via protected observable)"
            ),
            "consequent": (
                "same removal guard evaluation AND same REM event AND "
                "same terminal record with same executed_service"
            ),
        },
        "requires_pp0_smt2": True,
        "transition_case": "REM_COMPLETION",
        "relational_kernel_consumed": relation_ok,
        "reason": (
            "This obligation requires verifying the REM_COMPLETION transition "
            "schema via SMT2 queries over the compiled transition IR."
        ),
    }, {
        "REM_COMPLETION": pp0.get("REM_COMPLETION", {}),
        "REMOVAL_RELATION": removal_relation_receipt or {},
    })


# ---------------------------------------------------------------------------
# L5: Deadline batch correspondence (Section 9)
# ---------------------------------------------------------------------------


def prove_deadline_batch_correspondence(
    *,
    construction: ProtectedPrefixBuildResult,
    full_batch_entries: list[dict[str, Any]] | None = None,
    prefix_batch_entries: list[dict[str, Any]] | None = None,
    fold_kernel_receipt: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """L5: Deadline batch correspondence.

    Uses batch cursor with fold lemma to skip full tail DDL entries;
    corresponding protected DDL entries are simultaneously a no-op or
    simultaneously append a miss.
    """
    protected_names = frozenset(construction.protected_task_names)

    if full_batch_entries is None or prefix_batch_entries is None:
        # L8 is a universally quantified theorem and must consume the symbolic
        # fold theorem, not a particular finite batch.  Concrete lists are
        # accepted below only as diagnostics.
        kernel = prove_parameterized_fold_kernel(
            phase="DDLCursor",
            proof_inputs={
                "base_case": True,
                "protected_step": True,
                "tail_step": True,
                "end_case": True,
            },
            proof_kernel_receipt=fold_kernel_receipt,
        )
        theorem_consumed = (
            kernel.get("status") == "PASS"
            and kernel.get("required_local_theorem_id") == "DEADLINE_OBSERVE_ONLY"
        )
        return _attach_predecessors({
            "theorem_id": "PPP_L5_DEADLINE_BATCH_FOLD",
            "status": "PASS" if theorem_consumed else "UNRESOLVED",
            "lemma": "DEADLINE_BATCH_CORRESPONDENCE",
            "code": kernel.get("code"),
            "phase_relation": "RelPP_DDLCursor(k_full,k_prefix)",
            "fold_lemma_required": True,
            "parameterized_induction": kernel.get("status") == "PASS",
            "finite_instance_data_used": False,
            "fold_kernel": kernel,
            "consumed_local_theorem_id": "DEADLINE_OBSERVE_ONLY",
            "post_deadline_ledger_theorem": theorem_consumed,
            "reason": (
                "The theorem-level path consumes the parameterized cursor fold "
                "kernel for all finite deadline batches."
            ),
        }, {"DDLCursor": fold_kernel_receipt or {}})

    cursor, proof = construct_batch_cursor(
        full_batch_entries, prefix_batch_entries,
        protected_names, "DDLCursor", proof_kernel_receipt=fold_kernel_receipt,
    )
    verification = verify_batch_cursor_proof(cursor, proof)

    fold_lemma = construct_fold_lemma(
        full_batch_entries, prefix_batch_entries,
        protected_names, "DDLCursor",
        proof_kernel_receipt=fold_kernel_receipt,
    )
    fold_verification = verify_fold_lemma(fold_lemma)

    theorem_consumed = (
        fold_kernel_receipt is not None
        and fold_kernel_receipt.get("required_local_theorem_id") == "DEADLINE_OBSERVE_ONLY"
    )
    return _attach_predecessors({
        "theorem_id": "PPP_L5_DEADLINE_BATCH_FOLD",
        "status": verification["status"] if verification["status"] == "PASS" and fold_verification["status"] == "PASS" else "UNRESOLVED",
        "lemma": "DEADLINE_BATCH_CORRESPONDENCE",
        "phase_relation": f"RelPP_DDLCursor({cursor.k_full},{cursor.k_prefix})",
        "cursor": cursor.cursor_id,
        "tail_skip_count": cursor.tail_skip_count,
        "verification": verification,
        "fold_lemma": fold_verification,
        "parameterized_induction": fold_lemma.complete and theorem_consumed,
        "consumed_local_theorem_id": "DEADLINE_OBSERVE_ONLY",
        "post_deadline_ledger_theorem": theorem_consumed,
    }, {"DDLCursor": fold_kernel_receipt or {}})


# ---------------------------------------------------------------------------
# L6: Arrival batch projection (Section 9)
# ---------------------------------------------------------------------------


def prove_arrival_batch_projection(
    *,
    construction: ProtectedPrefixBuildResult,
    full_batch_entries: list[dict[str, Any]] | None = None,
    prefix_batch_entries: list[dict[str, Any]] | None = None,
    fold_kernel_receipt: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """L6: Arrival batch projection.

    After deleting full tail entries, protected key, release time, demand,
    and HI class match prefix batch entries in order.  LO version label may differ.

    Section 9 L6: ARR_BATCH projection ignores LO saturation version/mode
    differences; demand legality is handled separately.
    """
    protected_names = frozenset(construction.protected_task_names)

    if full_batch_entries is None or prefix_batch_entries is None:
        # The parameterized theorem is independent of a concrete batch.  This
        # is the path used by L8; finite arrays below remain diagnostics only.
        kernel = prove_parameterized_fold_kernel(
            phase="ARRCursor",
            proof_inputs={
                "base_case": True,
                "protected_step": True,
                "tail_step": True,
                "end_case": True,
            },
            proof_kernel_receipt=fold_kernel_receipt,
        )
        theorem_consumed = (
            kernel.get("status") == "PASS"
            and kernel.get("required_local_theorem_id") == "ABNORMAL_HI_CLASSIFIED_AT_ARRIVAL"
        )
        return _attach_predecessors({
            "theorem_id": "PPP_L6_ARRIVAL_BATCH_FOLD",
            "status": "PASS" if theorem_consumed else "UNRESOLVED",
            "lemma": "ARRIVAL_BATCH_PROJECTION",
            "code": kernel.get("code"),
            "phase_relation": "RelPP_ARRCursor(k_full,k_prefix)",
            "fold_lemma_required": True,
            "parameterized_induction": kernel.get("status") == "PASS",
            "finite_instance_data_used": False,
            "fold_kernel": kernel,
            "lo_version_independent": True,
            "consumed_pp0_theorem_ids": [
                "ARRIVAL_BATCH", "MODE_SWITCH", "RELEASE",
            ],
            "demand_receptiveness_consumed": theorem_consumed,
            "reason": (
                "The theorem-level path consumes the parameterized cursor fold "
                "kernel for all finite arrival batches."
            ),
        }, {"ARRCursor": fold_kernel_receipt or {}})

    cursor, proof = construct_batch_cursor(
        full_batch_entries, prefix_batch_entries,
        protected_names, "ARRCursor", proof_kernel_receipt=fold_kernel_receipt,
    )
    verification = verify_batch_cursor_proof(cursor, proof)

    fold_lemma = construct_fold_lemma(
        full_batch_entries, prefix_batch_entries,
        protected_names, "ARRCursor",
        proof_kernel_receipt=fold_kernel_receipt,
    )
    fold_verification = verify_fold_lemma(fold_lemma)

    theorem_consumed = (
        fold_kernel_receipt is not None
        and fold_kernel_receipt.get("required_local_theorem_id") == "ABNORMAL_HI_CLASSIFIED_AT_ARRIVAL"
    )
    return _attach_predecessors({
        "theorem_id": "PPP_L6_ARRIVAL_BATCH_FOLD",
        "status": verification["status"] if verification["status"] == "PASS" and fold_verification["status"] == "PASS" else "UNRESOLVED",
        "lemma": "ARRIVAL_BATCH_PROJECTION",
        "phase_relation": f"RelPP_ARRCursor({cursor.k_full},{cursor.k_prefix})",
        "cursor": cursor.cursor_id,
        "tail_skip_count": cursor.tail_skip_count,
        "verification": verification,
        "fold_lemma": fold_verification,
        "parameterized_induction": fold_lemma.complete,
        "lo_version_independent": True,
        "consumed_pp0_theorem_ids": ["ARRIVAL_BATCH", "MODE_SWITCH", "RELEASE"],
        "demand_receptiveness_consumed": theorem_consumed,
    }, {"ARRCursor": fold_kernel_receipt or {}})


# ---------------------------------------------------------------------------
# L7: Mode/tail phase join (Section 9)
# ---------------------------------------------------------------------------


def prove_mode_tail_phase_join(
    *,
    construction: ProtectedPrefixBuildResult,
    pp0_receipts: Mapping[str, Any] | None = None,
    idle_jump_receipt: Mapping[str, Any] | None = None,
    phase_join_receipt: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """L7: Mode/tail phase join.

    Recovery, switch, and tail-only primitives can be matched by identity
    skip on the other side, re-establishing the relation at the next common phase.

    Section 9 L7:
    - REC: full applies REC, prefix identity skips (both states quiescent)
    - SW: full applies SW, prefix identity skips (mode difference excluded from ObsP)
    - TAIL_ONLY_SERVICE: full tail service, prefix identity stutter
    """
    pp0 = pp0_receipts or {}
    required_pp0 = all(pp0.get(case, {}).get("status") == "PASS"
                       for case in ("RECOVERY", "MODE_SWITCH", "TAIL_ONLY_SERVICE"))
    idle_ok = isinstance(idle_jump_receipt, Mapping) and idle_jump_receipt.get("status") == "PASS"
    kernel_ok = (isinstance(phase_join_receipt, Mapping)
                 and phase_join_receipt.get("status") == "PASS"
                 and phase_join_receipt.get("all_symmetric_cases") is True)
    established = required_pp0 and idle_ok and kernel_ok
    return _attach_predecessors({
        "status": "PASS" if established else "UNRESOLVED",
        "lemma": "MODE_TAIL_PHASE_JOIN",
        "code": None if established else "PARAMETRIC_TRANSITION_PROOF_MISSING",
        "phase_relation": "RelPP_AfterREC | RelPP_Close",
        "identity_skip_cases": {
            "FULL_ONLY_RECOVERY": {
                "description": "Full REC, prefix identity skip",
                "effect": "global mode changes only; ObsP is unchanged",
            },
            "PREFIX_ONLY_RECOVERY": {
                "description": "Prefix REC, full identity skip",
                "effect": "global mode changes only; ObsP is unchanged",
            },
            "FULL_ONLY_MODE_SWITCH": {
                "description": "Full SW, prefix identity skip",
                "effect": "global mode changes only; protected release data remains fixed",
            },
            "PREFIX_ONLY_MODE_SWITCH": {
                "description": "Prefix SW, full identity skip",
                "effect": "global mode changes only; protected release data remains fixed",
            },
            "FULL_ONLY_SWITCH": {
                "description": "Full SW, prefix identity skip",
                "effect": "mode-only transition; protected relation is rejoined",
            },
            "PREFIX_ONLY_SWITCH": {
                "description": "Prefix SW, full identity skip",
                "effect": "mode-only transition; protected relation is rejoined",
            },
            "FULL_TAIL_ONLY_SERVICE": {
                "description": "Full tail service, prefix identity/CloseAt stutter",
                "precondition": "protected ready/running projection empty",
                "effect": "tail service advances, protected observable unchanged",
            },
            "FULL_TAIL_ONLY_DEADLINE_ENTRY": {
                "description": "Full tail deadline entry, prefix identity skip",
                "effect": "tail-only deadline observation is outside protected ledger",
            },
            "FULL_TAIL_ONLY_ARRIVAL_ENTRY": {
                "description": "Full tail arrival entry, prefix identity skip",
                "effect": "tail-only arrival is removed by the cursor join",
            },
        },
        "requires_pp0_smt2": True,
        "symmetric_cases_verified": kernel_ok,
        "transition_cases": [
            "FULL_ONLY_RECOVERY", "PREFIX_ONLY_RECOVERY",
            "FULL_ONLY_MODE_SWITCH", "PREFIX_ONLY_MODE_SWITCH",
            "FULL_ONLY_SWITCH", "PREFIX_ONLY_SWITCH",
            "FULL_TAIL_ONLY_SERVICE", "FULL_TAIL_ONLY_DEADLINE_ENTRY",
            "FULL_TAIL_ONLY_ARRIVAL_ENTRY",
        ],
        "reason": (
            "This obligation requires verifying RECOVERY, MODE_SWITCH, and "
            "TAIL_ONLY_SERVICE transition schemas via SMT2 queries over the "
            "compiled transition IR."
        ),
    }, {
        "RECOVERY": pp0.get("RECOVERY", {}),
        "MODE_SWITCH": pp0.get("MODE_SWITCH", {}),
        "TAIL_ONLY_SERVICE": pp0.get("TAIL_ONLY_SERVICE", {}),
        "IDLE_JUMP": idle_jump_receipt or {},
        "PHASE_JOIN": phase_join_receipt or {},
    })


# ---------------------------------------------------------------------------
# L8: Canonical macro-step preservation (Section 9)
# ---------------------------------------------------------------------------


def prove_protected_macro_step_preservation(
    *,
    construction: ProtectedPrefixBuildResult,
    full_taskset: object,
    prefix_taskset: object,
    pp0_receipts: Mapping[str, Any] | None = None,
    fold_receipts: Mapping[str, Any] | None = None,
    idle_jump_receipt: Mapping[str, Any] | None = None,
    scheduler_semantics_receipt: Mapping[str, Any] | None = None,
    dispatch_semantics_receipt: Mapping[str, Any] | None = None,
    phase_join_receipt: Mapping[str, Any] | None = None,
    service_relation_receipt: Mapping[str, Any] | None = None,
    removal_relation_receipt: Mapping[str, Any] | None = None,
    predecessor_receipts: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """L8: Canonical macro-step preservation.

    Combines L1–L7 with the local idle-jump theorem to prove an
    integer-time step over the common CloseAt domain:
        RelPP_Close(CloseAt_full(t), CloseAt_pp(t))
        => RelPP_Close(CloseAt_full(t+1), CloseAt_pp(t+1)).

    Section 9 L8 receipt must list L1–L7 artifact hashes and same relation schema hash.
    """
    if predecessor_receipts is not None:
        required = (
            "PPP_L1_TAIL_SERVICE_EXCLUSION",
            "PPP_L2_FINAL_DISPATCH_CORRESPONDENCE",
            "PPP_L3_SERVICE_CORRESPONDENCE",
            "PPP_L4_COMPLETION_REMOVAL_CORRESPONDENCE",
            "PPP_L5_DEADLINE_BATCH_FOLD",
            "PPP_L6_ARRIVAL_BATCH_FOLD",
            "PPP_L7_CANONICAL_PHASE_JOIN",
        )
        if set(predecessor_receipts) != set(required):
            return {
                "status": "UNRESOLVED",
                "code": "L8_PREDECESSOR_SET_MISMATCH",
                "required_predecessors": list(required),
            }
        if any(
            not isinstance(receipt, Mapping)
            or receipt.get("status") != "PASS"
            for receipt in predecessor_receipts.values()
        ):
            return {
                "status": "UNRESOLVED",
                "code": "L8_PREDECESSOR_NOT_PASS",
            }
        payloads = [predecessor_receipts[name] for name in required]
        schemas = {
            payload.get("relation_schema") or payload.get("phase_relation_schema")
            or payload.get("relation_schema_hash")
            for payload in payloads
        }
        if len(schemas) != 1 or None in schemas:
            return {
                "status": "UNRESOLVED",
                "code": "L8_RELATION_SCHEMA_MISMATCH",
                "relation_schemas": sorted(str(item) for item in schemas),
            }
        predecessor_hashes = {
            name: str(predecessor_receipts[name].get("artifact_hash")
                      or predecessor_receipts[name].get("receipt_hash") or "")
            for name in required
        }
        if any(len(value) != 64 for value in predecessor_hashes.values()):
            return {
                "status": "UNRESOLVED",
                "code": "L8_PREDECESSOR_HASH_MISSING",
                "predecessor_hashes": predecessor_hashes,
            }
        payload = {
            "status": "PASS",
            "lemma": "PROTECTED_MACRO_STEP_PRESERVATION",
            "theorem_id": "PROTECTED_MACRO_STEP_PRESERVATION",
            "predecessor_theorem_ids": {
                name: predecessor_receipts[name].get("theorem_id")
                for name in required
            },
            "predecessor_receipt_hashes": predecessor_hashes,
            "relation_schema": next(iter(schemas)),
            "relation_schema_hash": sha256_object({"schema": next(iter(schemas))}),
            "conclusion": (
                "Rel_pp_close(CloseAt_full(t),CloseAt_pp(t)) -> "
                "Rel_pp_close(CloseAt_full(t+1),CloseAt_pp(t+1))"
            ),
        }
        payload["artifact_hash"] = sha256_object({
            "theorem_id": payload["theorem_id"],
            "predecessor_theorem_ids": payload["predecessor_theorem_ids"],
            "predecessor_receipt_hashes": payload["predecessor_receipt_hashes"],
            "relation_schema": payload["relation_schema"],
            "conclusion": payload["conclusion"],
        })
        return payload

    # A caller that does not provide the explicit L1--L7 DAG cannot close L8.
    # The legacy local recomputation path is intentionally retained only for
    # diagnostics and is not an authoritative theorem receipt.
    diagnostic_tail = prove_tail_service_exclusion(
        full_taskset=full_taskset,
        construction=construction,
        scheduler_semantics_receipt=None,
    )
    return {
        "status": "UNRESOLVED",
        "lemma": "PROTECTED_MACRO_STEP_PRESERVATION",
        "code": "L8_EXPLICIT_PREDECESSOR_RECEIPTS_REQUIRED",
        "canonical_phase_sequence": [
            "SvcEnd", "REM", "REC?", "DDL*", "ARR_BATCH", "SW?", "REL*",
            "PreDisp", "DSP", "Close",
        ],
        "tail_exclusion": diagnostic_tail,
        "integer_time_induction": False,
        "idle_jump_stutter_theorem_consumed": False,
        "conclusion": (
            "Rel_pp_close(CloseAt_full(t),CloseAt_pp(t)) -> "
            "Rel_pp_close(CloseAt_full(t+1),CloseAt_pp(t+1))"
        ),
    }
