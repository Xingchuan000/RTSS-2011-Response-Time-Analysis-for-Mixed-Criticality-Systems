"""Canonical closed-boundary macro-step contracts.

Implements the eight required lemmas (L1–L8) for proving Rel_pp_close
preservation across canonical macro-step boundaries.

Each lemma is parameterized over the phase-indexed relation and batch
cursors; none use bounded replay or source-hash matching as proof.
"""

from __future__ import annotations

from typing import Any, Mapping

from .types import ProtectedPrefixBuildResult
from .phase_relation import build_phase_relation
from .batch_cursor import BatchCursor, BatchCursorProof


def _tasks(taskset: object) -> tuple[Any, ...]:
    return tuple(taskset.tasks)


def prove_tail_service_exclusion(*, full_taskset: object, construction: ProtectedPrefixBuildResult) -> dict[str, Any]:
    """L1: When protected ready set is non-empty, full scheduler cannot serve tail."""
    protected = set(construction.protected_task_names)
    tail = set(construction.tail_task_names)
    ordered = _tasks(full_taskset)
    priorities = {task.name: int(task.priority_index) for task in ordered}
    ok = (all(priorities[p] < priorities[t] for p in protected for t in tail)
          and tuple(task.name for task in ordered) == construction.protected_task_names + construction.tail_task_names)
    return {"status": "PASS" if ok else "FAIL", "lemma": "TAIL_SERVICE_EXCLUSION",
            "protected_ready_implies_protected_dispatch": ok,
            "protected_task_names": sorted(protected), "tail_task_names": sorted(tail),
            "phase_relation": "RelPP_SvcEnd", "excluded": "tail_jobs"}


def prove_final_dispatch_correspondence(*, construction: ProtectedPrefixBuildResult) -> dict[str, Any]:
    """L2: Total final-dispatch correspondence.

    - Protected ready non-empty: both sides select the same protected job.
    - Protected ready empty: full may serve tail, prefix idle; protected running projection = none.
    """
    return {
        "status": "UNRESOLVED",
        "lemma": "FINAL_DISPATCH_CORRESPONDENCE",
        "code": "PARAMETRIC_TRANSITION_PROOF_MISSING",
        "phase_relation": "RelPP_PreDisp",
        "reason": (
            "This obligation requires a universally quantified proof over the "
            "phase-indexed full/prefix dispatch function; it must be verified by "
            "SMT2 queries over the FINAL_DISPATCH transition schema."
        ),
    }


def prove_protected_service_correspondence(*, construction: ProtectedPrefixBuildResult) -> dict[str, Any]:
    """L3: Protected service correspondence.

    - Protected ready non-empty: both sides add the same unit of service to the same job.
    - Protected empty: full tail service and prefix idle are both protected stutter.
    """
    return {
        "status": "UNRESOLVED",
        "lemma": "PROTECTED_SERVICE_CORRESPONDENCE",
        "code": "PARAMETRIC_TRANSITION_PROOF_MISSING",
        "phase_relation": "RelPP_SvcEnd",
        "reason": (
            "This obligation requires verifying SERVICE_UNIT and TAIL_ONLY_SERVICE "
            "transition schemas via SMT2 queries."
        ),
    }


def prove_completion_removal_correspondence(*, construction: ProtectedPrefixBuildResult) -> dict[str, Any]:
    """L4: Completion/removal correspondence.

    Same fixed demand + same accumulated service => same guard and same-time removal.
    After removal, protected ledger relation holds.
    """
    return {
        "status": "UNRESOLVED",
        "lemma": "COMPLETION_REMOVAL_CORRESPONDENCE",
        "code": "PARAMETRIC_TRANSITION_PROOF_MISSING",
        "phase_relation": "RelPP_AfterREM",
        "reason": (
            "This obligation requires verifying the REM_COMPLETION transition "
            "schema via SMT2 queries."
        ),
    }


def prove_deadline_batch_correspondence(
    *,
    construction: ProtectedPrefixBuildResult,
    full_batch_entries: list[dict[str, Any]] | None = None,
    prefix_batch_entries: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """L5: Deadline batch correspondence.

    Uses batch cursor to skip full tail DDL entries; corresponding protected
    DDL entries are simultaneously a no-op or simultaneously append a miss.
    """
    if full_batch_entries is None or prefix_batch_entries is None:
        return {
            "status": "UNRESOLVED",
            "lemma": "DEADLINE_BATCH_CORRESPONDENCE",
            "code": "BATCH_CURSOR_INPUT_MISSING",
            "reason": "Deadline batch entries must be provided from the concrete execution states.",
        }

    from .batch_cursor import construct_batch_cursor, verify_batch_cursor_proof
    cursor, proof = construct_batch_cursor(
        full_batch_entries, prefix_batch_entries,
        frozenset(construction.protected_task_names), "DDLCursor",
    )
    verification = verify_batch_cursor_proof(cursor, proof)

    return {
        "status": verification["status"],
        "lemma": "DEADLINE_BATCH_CORRESPONDENCE",
        "phase_relation": f"RelPP_DDLCursor({cursor.k_full},{cursor.k_prefix})",
        "cursor": cursor.cursor_id,
        "tail_skip_count": cursor.tail_skip_count,
        "verification": verification,
    }


def prove_arrival_batch_projection(
    *,
    construction: ProtectedPrefixBuildResult,
    full_batch_entries: list[dict[str, Any]] | None = None,
    prefix_batch_entries: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """L6: Arrival batch projection.

    After deleting full tail entries, protected key, release time, demand,
    and HI class match prefix batch entries in order.  LO version label may differ.
    """
    if full_batch_entries is None or prefix_batch_entries is None:
        return {
            "status": "UNRESOLVED",
            "lemma": "ARRIVAL_BATCH_PROJECTION",
            "code": "BATCH_CURSOR_INPUT_MISSING",
            "reason": "Arrival batch entries must be provided from the concrete execution states.",
        }

    from .batch_cursor import construct_batch_cursor, verify_batch_cursor_proof
    cursor, proof = construct_batch_cursor(
        full_batch_entries, prefix_batch_entries,
        frozenset(construction.protected_task_names), "ARRCursor",
    )
    verification = verify_batch_cursor_proof(cursor, proof)

    return {
        "status": verification["status"],
        "lemma": "ARRIVAL_BATCH_PROJECTION",
        "phase_relation": f"RelPP_ARRCursor({cursor.k_full},{cursor.k_prefix})",
        "cursor": cursor.cursor_id,
        "tail_skip_count": cursor.tail_skip_count,
        "verification": verification,
    }


def prove_mode_tail_phase_join(*, construction: ProtectedPrefixBuildResult) -> dict[str, Any]:
    """L7: Mode/tail phase join.

    Recovery, switch, and tail-only primitives can be matched by identity
    skip on the other side, re-establishing the relation at the next common phase.
    """
    return {
        "status": "UNRESOLVED",
        "lemma": "MODE_TAIL_PHASE_JOIN",
        "code": "PARAMETRIC_TRANSITION_PROOF_MISSING",
        "phase_relation": "RelPP_AfterREC ∨ RelPP_Close",
        "reason": (
            "This obligation requires verifying RECOVERY, MODE_SWITCH, and "
            "TAIL_ONLY_SERVICE transition schemas via SMT2 queries."
        ),
    }


def prove_protected_macro_step_preservation(
    *,
    construction: ProtectedPrefixBuildResult,
    full_taskset: object,
    prefix_taskset: object,
) -> dict[str, Any]:
    """L8: Canonical macro-step preservation.

    Combines L1–L7 to prove:
        RelPP_Close(full_t, prefix_t) => RelPP_Close(full_{t+1}, prefix_{t+1})

    where t+1 is the next canonical macro boundary.
    """
    tail = prove_tail_service_exclusion(full_taskset=full_taskset, construction=construction)
    lemmas = [
        prove_final_dispatch_correspondence(construction=construction),
        prove_protected_service_correspondence(construction=construction),
        prove_completion_removal_correspondence(construction=construction),
        prove_deadline_batch_correspondence(construction=construction),
        prove_arrival_batch_projection(construction=construction),
        prove_mode_tail_phase_join(construction=construction),
    ]

    all_pass = tail["status"] == "PASS" and all(item["status"] == "PASS" for item in lemmas)
    return {
        "status": "PASS" if all_pass else "UNRESOLVED",
        "lemma": "PROTECTED_MACRO_STEP_PRESERVATION",
        "canonical_phase_sequence": [
            "SvcEnd", "REM", "REC?", "DDL*", "ARR_BATCH", "SW?", "REL*", "PreDisp", "DSP", "Close",
        ],
        "tail_exclusion": tail,
        "lemmas": lemmas,
        "conclusion": "Rel_pp_close(Close(t)) -> Rel_pp_close(Close(t+1))",
        "phase_relation_schema": "phase_relation_v1",
    }
