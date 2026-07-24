"""Canonical closed-boundary macro-step contracts.

These functions discharge parameterized local obligations.  They do not use a
bounded replay as a substitute for the induction theorem.
"""

from __future__ import annotations

from typing import Any, Mapping

from .state_relation import rel_pp_close
from .types import ProtectedPrefixBuildResult


def _tasks(taskset: object) -> tuple[Any, ...]:
    return tuple(taskset.tasks)


def prove_tail_service_exclusion(*, full_taskset: object, construction: ProtectedPrefixBuildResult) -> dict[str, Any]:
    protected = set(construction.protected_task_names)
    tail = set(construction.tail_task_names)
    ordered = _tasks(full_taskset)
    priorities = {task.name: int(task.priority_index) for task in ordered}
    ok = (all(priorities[p] < priorities[t] for p in protected for t in tail)
          and tuple(task.name for task in ordered) == construction.protected_task_names + construction.tail_task_names)
    return {"status": "PASS" if ok else "FAIL", "lemma": "TAIL_SERVICE_EXCLUSION",
            "protected_ready_implies_protected_dispatch": ok,
            "protected_task_names": sorted(protected), "tail_task_names": sorted(tail)}


def prove_final_dispatch_correspondence(*, construction: ProtectedPrefixBuildResult) -> dict[str, Any]:
    return {
        "status": "UNRESOLVED",
        "lemma": "FINAL_DISPATCH_CORRESPONDENCE",
        "code": "PARAMETRIC_TRANSITION_PROOF_MISSING",
        "reason": (
            "This obligation requires a universally quantified proof over the "
            "phase-indexed full/prefix transition relation; construction metadata "
            "alone is insufficient."
        ),
    }
def prove_protected_service_correspondence(*, construction: ProtectedPrefixBuildResult) -> dict[str, Any]:
    return {
        "status": "UNRESOLVED",
        "lemma": "PROTECTED_SERVICE_CORRESPONDENCE",
        "code": "PARAMETRIC_TRANSITION_PROOF_MISSING",
        "reason": (
            "This obligation requires a universally quantified proof over the "
            "phase-indexed full/prefix transition relation; construction metadata "
            "alone is insufficient."
        ),
    }
def prove_completion_removal_correspondence(*, construction: ProtectedPrefixBuildResult) -> dict[str, Any]:
    return {
        "status": "UNRESOLVED",
        "lemma": "COMPLETION_REMOVAL_CORRESPONDENCE",
        "code": "PARAMETRIC_TRANSITION_PROOF_MISSING",
        "reason": (
            "This obligation requires a universally quantified proof over the "
            "phase-indexed full/prefix transition relation; construction metadata "
            "alone is insufficient."
        ),
    }
def prove_deadline_batch_correspondence(*, construction: ProtectedPrefixBuildResult) -> dict[str, Any]:
    return {
        "status": "UNRESOLVED",
        "lemma": "DEADLINE_BATCH_CORRESPONDENCE",
        "code": "PARAMETRIC_TRANSITION_PROOF_MISSING",
        "reason": (
            "This obligation requires a universally quantified proof over the "
            "phase-indexed full/prefix transition relation; construction metadata "
            "alone is insufficient."
        ),
    }
def prove_arrival_batch_projection(*, construction: ProtectedPrefixBuildResult) -> dict[str, Any]:
    return {
        "status": "UNRESOLVED",
        "lemma": "ARRIVAL_BATCH_PROJECTION",
        "code": "PARAMETRIC_TRANSITION_PROOF_MISSING",
        "reason": (
            "This obligation requires a universally quantified proof over the "
            "phase-indexed full/prefix transition relation; construction metadata "
            "alone is insufficient."
        ),
    }
def prove_mode_tail_phase_join(*, construction: ProtectedPrefixBuildResult) -> dict[str, Any]:
    return {
        "status": "UNRESOLVED",
        "lemma": "MODE_TAIL_PHASE_JOIN",
        "code": "PARAMETRIC_TRANSITION_PROOF_MISSING",
        "reason": (
            "This obligation requires a universally quantified proof over the "
            "phase-indexed full/prefix transition relation; construction metadata "
            "alone is insufficient."
        ),
    }
def prove_protected_macro_step_preservation(*, construction: ProtectedPrefixBuildResult, full_taskset: object, prefix_taskset: object) -> dict[str, Any]:
    tail = prove_tail_service_exclusion(full_taskset=full_taskset, construction=construction)
    lemmas = [prove_final_dispatch_correspondence(construction=construction), prove_protected_service_correspondence(construction=construction),
              prove_completion_removal_correspondence(construction=construction), prove_deadline_batch_correspondence(construction=construction),
              prove_arrival_batch_projection(construction=construction), prove_mode_tail_phase_join(construction=construction)]
    ok = tail["status"] == "PASS" and all(item["status"] == "PASS" for item in lemmas)
    return {"status": "PASS" if ok else "UNRESOLVED", "lemma": "PROTECTED_MACRO_STEP_PRESERVATION",
            "canonical_phase_sequence": ["SvcEnd", "REM", "REC?", "DDL*", "ARR_BATCH", "SW?", "REL*", "PreDisp", "DSP", "Close"],
            "tail_exclusion": tail, "lemmas": lemmas,
            "conclusion": "Rel_pp_close(Close(t)) -> Rel_pp_close(Close(t+1))"}
