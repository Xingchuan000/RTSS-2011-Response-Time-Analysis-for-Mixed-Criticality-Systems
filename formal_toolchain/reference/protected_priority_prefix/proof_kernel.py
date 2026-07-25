"""Parameterized proof kernel receipts for the PP route mathematical chain.

Each function in this module produces a PARAMETERIZED proof receipt.
These receipts are universally quantified over all tasksets satisfying the
construction preconditions (PPC1-PPC7, FPR1-FPR7, PE1-PE4).  They are
NOT per-seed instance certificates -- those are produced by the route
checkers after binding the kernel receipts to specific fingerprints.

Proof ordering follows the V6 dependency DAG:
  PP1-PP4 (structural lemmas)
  -> PP5 series (transition-level lemmas L1-L8)
  -> PP6 (weak forward simulation)
  -> PP6-A/B (HI bad-prefix reflection)
  -> PP7-A/B (imported theorem discharge)
  -> PP8 (full-reference HI safety)
  -> final contradiction (with N6)
"""

from __future__ import annotations

from typing import Any, Mapping

from formal_toolchain.core.hashing import sha256_object


# ---------------------------------------------------------------------------
# Canonical proof schema version
# ---------------------------------------------------------------------------

PROOF_KERNEL_VERSION = "pp_proof_kernel_v1"
RELATION_SCHEMA_HASH = sha256_object({"schema": "phase_relation_v4_close_at"})


# ---------------------------------------------------------------------------
# Shared proof infrastructure
# ---------------------------------------------------------------------------

def _kernel_receipt(
    theorem_id: str,
    status: str,
    fields: dict[str, Any],
    *,
    code: str | None = None,
    reason: str = "",
) -> dict[str, Any]:
    """Construct a standard proof kernel receipt."""
    payload = {
        "proof_kernel_version": PROOF_KERNEL_VERSION,
        "theorem_id": theorem_id,
        "status": status,
        "code": code,
        "relation_schema_hash": RELATION_SCHEMA_HASH,
        "parameterized": True,
        **fields,
    }
    payload["receipt_hash"] = sha256_object(payload)
    if reason:
        payload["reason"] = reason
    return payload


def _structural_lemma_status(
    conditions: dict[str, bool],
) -> tuple[str, str | None]:
    """Evaluate a structural lemma: all conditions must hold."""
    if all(conditions.values()):
        return "PASS", None
    failed = [k for k, v in conditions.items() if not v]
    return "FAIL", f"STRUCTURAL_CONDITIONS_FAILED:{','.join(failed[:3])}"


# ---------------------------------------------------------------------------
# PP1: Protected prefix closure (structural)
# ---------------------------------------------------------------------------

def prove_pp1_prefix_closure(
    *,
    construction_result: Any,
    full_taskset: Any,
) -> dict[str, Any]:
    """PP1: Protected prefix closure lemma.

    Proves:
      1. All HI tasks belong to P (protected set)
      2. L^{tail} contains only LO tasks
      3. Every tail task has strictly lower priority than every protected task
      4. P is priority-closed (higher-priority-of-protected is also protected)
    """
    ordered = list(full_taskset.tasks)
    cutoff_idx = construction_result.cutoff_priority_index
    protected_names = frozenset(construction_result.protected_task_names)
    tail_names = frozenset(construction_result.tail_task_names)

    conditions = {
        "all_hi_protected": all(
            t.criticality != "HI" or t.name in protected_names
            for t in ordered
        ),
        "tail_all_lo": all(
            t.criticality == "LO"
            for t in ordered
            if t.name in tail_names
        ),
        "order_preserved":
            tuple(t.name for t in ordered) ==
            construction_result.protected_task_names + construction_result.tail_task_names,
        "hi_subset_protected":
            all(t.name in protected_names
                for t in ordered if t.criticality == "HI"),
        "tail_priority_lower": all(
            ordered[i].priority_index > ordered[j].priority_index
            for i, t_i in enumerate(ordered)
            for j, t_j in enumerate(ordered)
            if t_i.name in tail_names and t_j.name in protected_names
        ),
    }

    status, code = _structural_lemma_status(conditions)
    return _kernel_receipt(
        "PP1_PROTECTED_PREFIX_CLOSURE",
        status,
        {
            "lemma": "PROTECTED_PREFIX_CLOSURE",
            "conditions": conditions,
            "protected_task_names": sorted(protected_names),
            "tail_task_names": sorted(tail_names),
            "cutoff_task_name": construction_result.cutoff_task_name,
        },
        code=code,
        reason="Structural priority-closure proof from partition definition PPC1-PPC4.",
    )


# ---------------------------------------------------------------------------
# PP2: Saturated prefix C-AMC-sem parameter legality
# ---------------------------------------------------------------------------

def prove_pp2_saturation_legality(
    *,
    prefix_taskset: Any,
) -> dict[str, Any]:
    """PP2: Saturated prefix satisfies C-AMC-sem WCET monotonicity.

    For LO tasks: 0 < C_pp(HI) = C_pp(LO) = C_ref(LO)
    For HI tasks: 0 < C_pp(LO) <= C_pp(HI) [unchanged from full]
    """
    conditions = {}
    for t in prefix_taskset.tasks:
        if t.criticality == "LO":
            conditions[f"{t.name}_lo_saturated"] = (
                t.c_hi == t.c_lo and t.c_lo > 0
            )
            conditions[f"{t.name}_lo_wcet_order"] = (
                0 < t.c_hi <= t.c_lo
            )
        else:
            conditions[f"{t.name}_hi_wcet_order"] = (
                0 < t.c_lo <= t.c_hi
            )
            conditions[f"{t.name}_hi_positive"] = (
                t.c_lo > 0 and t.c_hi > 0
            )

    status, code = _structural_lemma_status(conditions)
    return _kernel_receipt(
        "PP2_SATURATION_LEGALITY",
        status,
        {
            "lemma": "SATURATED_PREFIX_C_AMC_SEM_LEGALITY",
            "conditions": conditions,
            "prefix_taskset_fingerprint": prefix_taskset.to_dict().get("fingerprint"),
        },
        code=code,
        reason="LO saturation preserves WCET ordering required by C-AMC-sem model.",
    )


# ---------------------------------------------------------------------------
# PP3: Protected LO demand receptiveness
# ---------------------------------------------------------------------------

def prove_pp3_lo_demand_receptiveness(
    *,
    full_taskset: Any,
    prefix_taskset: Any,
    construction_result: Any,
) -> dict[str, Any]:
    """PP3: Any full-reference protected LO demand is legal in prefix.

    Because C_pp(LO) = C_pp(HI) = C_ref(LO), any protected LO demand
    that was legal in full (<= C_ref(LO) or <= C_ref(HI)) is <= C_pp(X)
    for both prefix modes.
    """
    protected_lo = [
        t for t in full_taskset.tasks
        if t.name in frozenset(construction_result.protected_task_names)
        and t.criticality == "LO"
    ]
    full_by_name = {t.name: t for t in full_taskset.tasks}
    prefix_by_name = {t.name: t for t in prefix_taskset.tasks}

    conditions = {}
    for t in protected_lo:
        pt = prefix_by_name[t.name]
        ft = full_by_name[t.name]
        conditions[f"{t.name}_max_demand"] = max(ft.c_lo, ft.c_hi) <= pt.c_lo
        conditions[f"{t.name}_equal_lo"] = pt.c_lo == ft.c_lo
        conditions[f"{t.name}_equal_hi"] = pt.c_hi == ft.c_lo
        conditions[f"{t.name}_saturated"] = pt.c_lo == pt.c_hi

    status, code = _structural_lemma_status(conditions)
    return _kernel_receipt(
        "PP3_LO_DEMAND_RECEPTIVENESS",
        status,
        {
            "lemma": "PROTECTED_LO_DEMAND_RECEPTIVENESS",
            "conditions": conditions,
            "mode_independent": True,
            "saturation_bound": "C_pp(LO) == C_pp(HI) == C_ref(LO)",
        },
        code=code,
        reason="Saturation ensures any full-reference protected LO demand "
               "is legal in prefix regardless of prefix release mode.",
    )


# ---------------------------------------------------------------------------
# PP4: HI job demand projection
# ---------------------------------------------------------------------------

def prove_pp4_hi_demand_projection(
    *,
    full_taskset: Any,
    prefix_taskset: Any,
    construction_result: Any,
) -> dict[str, Any]:
    """PP4: HI job demand and classification projects identically.

    All HI tasks are protected; their WCETs are unchanged.  HI
    normal/abnormal class is determined solely by the HI job's
    release-fixed demand and the (unchanged) C_i(LO) threshold.
    """
    protected_hi = [
        t for t in full_taskset.tasks
        if t.name in frozenset(construction_result.protected_task_names)
        and t.criticality == "HI"
    ]
    full_by_name = {t.name: t for t in full_taskset.tasks}
    prefix_by_name = {t.name: t for t in prefix_taskset.tasks}

    conditions = {}
    for t in protected_hi:
        pt = prefix_by_name[t.name]
        ft = full_by_name[t.name]
        conditions[f"{t.name}_lo_equal"] = pt.c_lo == ft.c_lo
        conditions[f"{t.name}_hi_equal"] = pt.c_hi == ft.c_hi
        conditions[f"{t.name}_threshold"] = pt.c_lo == ft.c_lo  # classification threshold preserved

    conditions["all_hi_protected"] = all(
        t.name in frozenset(construction_result.protected_task_names)
        for t in full_taskset.tasks if t.criticality == "HI"
    )

    status, code = _structural_lemma_status(conditions)
    return _kernel_receipt(
        "PP4_HI_DEMAND_PROJECTION",
        status,
        {
            "lemma": "HI_DEMAND_PROJECTION",
            "conditions": conditions,
            "hi_tasks_preserved": len(protected_hi),
            "classification_threshold_preserved": True,
        },
        code=code,
        reason="All HI tasks are protected with unchanged WCETs; "
               "normal/abnormal classification is preserved.",
    )


# ---------------------------------------------------------------------------
# PP5 L1: Tail service exclusion
# ---------------------------------------------------------------------------

def prove_l1_tail_service_exclusion(
    *,
    construction_result: Any,
    full_taskset: Any,
) -> dict[str, Any]:
    """L1: When any protected job is ready, tail cannot be served.

    Proof: Tail priority index > every protected priority index.
    Strict FP scheduler always selects highest-priority ready job.
    """
    ordered = list(full_taskset.tasks)
    protected = frozenset(construction_result.protected_task_names)
    tail = frozenset(construction_result.tail_task_names)
    priorities = {t.name: int(t.priority_index) for t in ordered}

    all_higher = all(
        priorities[p] < priorities[t]
        for p in protected for t in tail
    )
    order_ok = (
        tuple(t.name for t in ordered)
        == construction_result.protected_task_names + construction_result.tail_task_names
    )

    conditions = {
        "all_protected_higher_priority": all_higher,
        "order_preserved": order_ok,
        "protected_ready_implies_protected_dispatch": all_higher and order_ok,
    }
    structural_status, structural_code = _structural_lemma_status(conditions)
    # Priority partitioning is proved structurally here.  The service-exclusion
    # conclusion additionally needs a source-bound strict-FP dispatch theorem,
    # which this function does not consume.  Do not promote the structural
    # ordering fact into a scheduler theorem.
    status = "FAIL" if structural_status == "FAIL" else "UNRESOLVED"
    code = structural_code if structural_status == "FAIL" else "STRICT_FP_SCHEDULER_SEMANTICS_RECEIPT_REQUIRED"

    return _kernel_receipt(
        "PP5_L1_TAIL_SERVICE_EXCLUSION",
        status,
        {
            "lemma": "TAIL_SERVICE_EXCLUSION",
            "conditions": conditions,
            "protected_task_names": sorted(protected),
            "tail_task_names": sorted(tail),
            "proof_type": "structural_priority_ordering",
            "phase": "SvcEnd",
        },
        code=code,
        reason="Static priority ordering: every tail task has strictly lower "
               "priority than every protected task.  Strict FP => tail never "
               "runs when protected ready.",
    )


# ---------------------------------------------------------------------------
# PP5 L2: Final dispatch correspondence
# ---------------------------------------------------------------------------

def prove_l2_final_dispatch_correspondence(
    *,
    construction_result: Any,
) -> dict[str, Any]:
    """L2: Total final-dispatch correspondence.

    Two cases:
    - Protected ready nonempty: both sides select the same protected job
      (same ready set, same strict-FP tie-break via Ord)
    - Protected ready empty: full may run tail/idle, prefix idle;
      both projected running key = none.
    """
    protected = frozenset(construction_result.protected_task_names)

    return _kernel_receipt(
        "PP5_L2_FINAL_DISPATCH_CORRESPONDENCE",
        "UNRESOLVED",
        {
            "lemma": "FINAL_DISPATCH_CORRESPONDENCE",
            "phase": "PreDisp",
            "case_1": {
                "name": "protected_ready_nonempty",
                "result": "same_protected_dispatch",
                "requires": [
                    "same protected ready set (from RelPP_PreDisp)",
                    "strict FP scheduler (same priority_index ordering)",
                    "same job-level tie-break (release_index FIFO)",
                ],
            },
            "case_2": {
                "name": "protected_ready_empty",
                "result": "both_projected_running_none",
                "requires": [
                    "prefix has no tail jobs",
                    "full tail dispatch does not affect protected observables",
                    "prefix idle is protected-observable stutter",
                ],
            },
            "total": True,
            "proof_type": "two_case_structural",
        },
        code="SOURCE_BOUND_DISPATCH_RELATION_REQUIRED",
        reason="The two-case outline is valid only after the executable dispatch "
               "transition and tie-break are bound to the relation.  Narrative "
               "case coverage is not a proof receipt.",
    )


# ---------------------------------------------------------------------------
# PP5 L3: Protected service correspondence
# ---------------------------------------------------------------------------

def prove_l3_protected_service_correspondence(
    *,
    construction_result: Any,
) -> dict[str, Any]:
    """L3: Protected service correspondence.

    Two cases:
    - Protected ready nonempty: closed relation gives same running key;
      one SERVICE_UNIT adds exactly 1 to the same job.
    - Protected ready empty: full tail service and prefix idle both leave
      protected service vector unchanged.
    """
    return _kernel_receipt(
        "PP5_L3_PROTECTED_SERVICE_CORRESPONDENCE",
        "UNRESOLVED",
        {
            "lemma": "PROTECTED_SERVICE_CORRESPONDENCE",
            "phase": "Close",
            "case_1": {
                "name": "protected_ready_nonempty",
                "description": "Same protected running job gets same unit service",
                "guarantee": "SERVICE_UNIT adds exactly 1 to running protected job",
            },
            "case_2": {
                "name": "protected_ready_empty",
                "description": "Tail service (full) or idle (prefix) changes no protected service",
                "guarantee": "protected service vector unchanged",
            },
            "service_rate": "unit_quantum_per_tick",
            "proof_type": "two_case_structural",
        },
        code="SOURCE_BOUND_SERVICE_RELATION_REQUIRED",
        reason="Requires a relational proof for SERVICE_UNIT and the tail/idle "
               "stutter branch over the actual transition relation.",
    )


# ---------------------------------------------------------------------------
# PP5 L4: Completion/removal correspondence
# ---------------------------------------------------------------------------

def prove_l4_completion_removal_correspondence(
    *,
    construction_result: Any,
) -> dict[str, Any]:
    """L4: Protected completion/removal correspondence.

    Guard: service >= fixed_demand (G3).  Same demand + same service =>
    same guard evaluation => same-time removal on both sides.
    After removal: same active/ready projection, same completion ledger.
    """
    return _kernel_receipt(
        "PP5_L4_COMPLETION_REMOVAL_CORRESPONDENCE",
        "UNRESOLVED",
        {
            "lemma": "COMPLETION_REMOVAL_CORRESPONDENCE",
            "phase": "SvcEnd",
            "guard": "service >= fixed_demand",
            "guard_independence": "reads only job-local demand and service (G3)",
            "equivalence": "same demand + same service => same guard => same removal",
            "post_removal": "same active/ready projection, same completion ledger",
            "proof_type": "equivalence_from_relation",
        },
        code="SOURCE_BOUND_REMOVAL_RELATION_REQUIRED",
        reason="Requires a code-bound proof that the executable removal guard "
               "and all ledger updates are exactly the stated job-local relation.",
    )


# ---------------------------------------------------------------------------
# PP5 L5: Deadline batch correspondence
# ---------------------------------------------------------------------------

def prove_l5_deadline_batch_correspondence(
    *,
    construction_result: Any,
) -> dict[str, Any]:
    """L5: Deadline batch correspondence.

    Uses (DDL,k) proof cursor.  For each protected due-key:
    - Due guard: d_k = t (preserved by relation)
    - Miss guard: not Finished(k) (preserved by relation)
    - Effect: append miss record (observe-only, G4)

    Tail due-keys: prefix executes SKIP_DDL_entry (identity).
    Batch fold lemma proves protection by induction on cursor.
    """
    return _kernel_receipt(
        "PP5_L5_DEADLINE_BATCH_CORRESPONDENCE",
        "UNRESOLVED",
        {
            "lemma": "DEADLINE_BATCH_CORRESPONDENCE",
            "phase": "DDLCursor",
            "cursor_type": "(DDL,k)",
            "protected_entry": {
                "due_guard": "absolute_deadline == current_time",
                "miss_guard": "not finished (service < fixed_demand)",
                "effect": "observe_only_append_miss",
                "guard_fields": ["absolute_deadline", "finished", "service", "fixed_demand"],
            },
            "tail_entry": {
                "prefix_action": "SKIP_DDL_entry (identity)",
                "effect": "tail miss ledger only, no effect on protected observable",
            },
            "fold_lemma": {
                "base_case": "empty batch => trivially equal",
                "inductive_step": "protected => PP4-B; tail => skip",
                "termination": "finite due-key set",
            },
            "proof_type": "cursor_induction",
        },
        code="SOURCE_BOUND_DEADLINE_FOLD_REQUIRED",
        reason="Requires the parameterized cursor fold and a code-bound deadline "
               "observation transition; the displayed fold is only an outline.",
    )


# ---------------------------------------------------------------------------
# PP5 L6: Arrival batch projection
# ---------------------------------------------------------------------------

def prove_l6_arrival_batch_projection(
    *,
    construction_result: Any,
) -> dict[str, Any]:
    """L6: Arrival batch projection.

    Uses (ARR,k) proof cursor.  For each protected release:
    - Same key, release time, actual demand, HI class
    - LO version label may differ (excluded from Obs_P)

    Tail arrivals: prefix executes SKIP_ARR_entry (identity).
    SW trigger: only from protected abnormal HI jobs (tail is all LO).
    """
    return _kernel_receipt(
        "PP5_L6_ARRIVAL_BATCH_PROJECTION",
        "UNRESOLVED",
        {
            "lemma": "ARRIVAL_BATCH_PROJECTION",
            "phase": "ARRCursor",
            "cursor_type": "(ARR,k)",
            "protected_entry": {
                "matched_fields": ["job_key", "release_time", "actual_demand", "hi_class"],
                "unmatched_fields": ["lo_version_label"],
                "requires_lo_saturation": True,
            },
            "tail_entry": {
                "prefix_action": "SKIP_ARR_entry (identity)",
                "effect": "tail releases only, no effect on protected observable",
            },
            "switch_trigger": {
                "source": "protected abnormal HI jobs only",
                "tail_independent": True,
            },
            "fold_lemma": {
                "base_case": "empty batch => trivially equal",
                "inductive_step": "protected => PP4/PP4-A; tail => skip",
                "termination": "finite arrival batch",
            },
            "proof_type": "cursor_induction",
        },
        code="SOURCE_BOUND_ARRIVAL_FOLD_REQUIRED",
        reason="Requires a parameterized cursor fold over the executable arrival, "
               "switch and release handlers.  Saturation proves demand legality "
               "but not transition correspondence by itself.",
    )


# ---------------------------------------------------------------------------
# PP5 L7: Mode/tail phase join
# ---------------------------------------------------------------------------

def prove_l7_mode_tail_phase_join(
    *,
    construction_result: Any,
) -> dict[str, Any]:
    """L7: Mode/tail phase join with identity skip.

    REC, SW, and tail-only primitives on one side matched by
    identity skip on the other side.  Re-establishes phase relation
    at the next common phase boundary.

    G6: SW/REC effect does not change protected derived guards.
    G1: protected ready eligibility independent of mode.
    """
    return _kernel_receipt(
        "PP5_L7_MODE_TAIL_PHASE_JOIN",
        "UNRESOLVED",
        {
            "lemma": "MODE_TAIL_PHASE_JOIN",
            "identity_skip_cases": {
                "FULL_ONLY_RECOVERY": {
                    "full_action": "REC (mode HI->LO)",
                    "prefix_action": "SKIP_REC (identity)",
                    "effect_on_Obs_P": "none (global mode excluded)",
                    "requires": "G6 + PP0-J (mode-transparent lifecycle)",
                },
                "PREFIX_ONLY_RECOVERY": {
                    "prefix_action": "REC (mode HI->LO)",
                    "full_action": "SKIP_REC (identity)",
                    "effect_on_Obs_P": "none",
                },
                "FULL_ONLY_MODE_SWITCH": {
                    "full_action": "SW (mode LO->HI)",
                    "prefix_action": "SKIP_SW (identity)",
                    "effect_on_Obs_P": "none",
                },
                "PREFIX_ONLY_MODE_SWITCH": {
                    "prefix_action": "SW (mode LO->HI)",
                    "full_action": "SKIP_SW (identity)",
                    "effect_on_Obs_P": "none",
                },
                "FULL_TAIL_ONLY_SERVICE": {
                    "full_action": "Tail service",
                    "prefix_action": "SKIP (identity/idle stutter)",
                    "precondition": "protected ready/running empty",
                },
            },
            "symmetric": True,
            "proof_type": "identity_skip_reconstruction",
        },
        code="SOURCE_BOUND_PHASE_JOIN_REQUIRED",
        reason="Requires symmetric source-bound REC/SW/tail-only stutter proofs "
               "and an explicit phase-indexed join theorem.",
    )


# ---------------------------------------------------------------------------
# PP5 L8: Canonical macro-step preservation
# ---------------------------------------------------------------------------

def prove_l8_macro_step_preservation(
    *,
    construction_result: Any,
    full_taskset: Any,
) -> dict[str, Any]:
    """L8: Canonical macro-step preservation.

    Composes L1-L7 to prove:
        Rel_pp_close(Close_full(t), Close_pp(t))
        => Rel_pp_close(Close_full(t+1), Close_pp(t+1))

    Phase sequence (8b):
        SvcEnd -> REM -> REC? -> DDL* -> ARR_BATCH[SW?;REL*] -> PreDisp -> DSP -> Close
    """
    # Run all lemmas to capture their hashes
    lemmas = {
        "L1": prove_l1_tail_service_exclusion(
            construction_result=construction_result,
            full_taskset=full_taskset,
        ),
        "L2": prove_l2_final_dispatch_correspondence(
            construction_result=construction_result,
        ),
        "L3": prove_l3_protected_service_correspondence(
            construction_result=construction_result,
        ),
        "L4": prove_l4_completion_removal_correspondence(
            construction_result=construction_result,
        ),
        "L5": prove_l5_deadline_batch_correspondence(
            construction_result=construction_result,
        ),
        "L6": prove_l6_arrival_batch_projection(
            construction_result=construction_result,
        ),
        "L7": prove_l7_mode_tail_phase_join(
            construction_result=construction_result,
        ),
    }

    all_pass = all(
        lm.get("status") == "PASS"
        for lm in lemmas.values()
    )

    return _kernel_receipt(
        "PP5_L8_MACRO_STEP_PRESERVATION",
        "PASS" if all_pass else "UNRESOLVED",
        {
            "lemma": "PROTECTED_MACRO_STEP_PRESERVATION",
            "canonical_phase_sequence": [
                "SvcEnd", "REM", "REC?", "DDL*",
                "ARR_BATCH", "SW?", "REL*", "PreDisp", "DSP", "Close",
            ],
            "lemmas": lemmas,
            "lemma_hashes": {
                name: sha256_object(lm)
                for name, lm in lemmas.items()
            },
            "conclusion": (
                "Rel_pp_close(Close_full(t),Close_pp(t)) => "
                "Rel_pp_close(Close_full(t+1),Close_pp(t+1))"
            ),
            "integer_time_induction": all_pass,
        },
        code=None if all_pass else "MACRO_STEP_LEMMA_CHAIN_INCOMPLETE",
        reason="L1-L7 provide complete case analysis for one unit macro-step. "
               "Phase sequence (8b) is exhaustive by PP5-E.",
    )


# ---------------------------------------------------------------------------
# Same-time closure termination (PE3 component)
# ---------------------------------------------------------------------------

def prove_same_time_closure_termination_kernel() -> dict[str, Any]:
    """Prove same-timestamp closure terminates.

    Uses 7-dimensional lexicographic measure:
      (#REM_t, #REC_t, #DDL_t, #ARR_t, #SW_t, #REL_t, #DSP_t)

    Each primitive transition strictly decreases exactly one component.
    SERVICE_UNIT and TAIL_ONLY_SERVICE advance time, exiting the loop.
    The measure is bounded below by (0,0,0,0,0,0,0) and well-founded.
    """
    measure_order = ("remaining_REM", "REC_enabled", "remaining_DDL_entries",
                     "remaining_ARR_entries", "remaining_SW_entries", "remaining_REL_entries",
                     "DSP_enabled")
    from .executable_transition_compiler import compile_all_transitions
    compiled = {ir.case_id: ir for ir in compile_all_transitions()}
    expected = set(("REM_COMPLETION", "RECOVERY", "DEADLINE_OBSERVATION",
                    "ARRIVAL_BATCH", "MODE_SWITCH", "RELEASE",
                    "FINAL_DISPATCH", "SERVICE_UNIT", "TAIL_ONLY_SERVICE"))
    transitions = {}
    for case_id, ir in compiled.items():
        transitions[case_id] = {
            "compiled_ir_hash": ir.ir_hash(),
            "all_paths_covered": ir.is_compiled(),
            "consumed_frontier_event": case_id not in {"SERVICE_UNIT", "TAIL_ONLY_SERVICE"},
            "exits_loop": case_id in {"SERVICE_UNIT", "TAIL_ONLY_SERVICE"},
        }
    all_bound = expected == set(compiled) and all(
        item.get("all_paths_covered") is True for item in transitions.values()
    )
    # Compilation status alone does not prove a lexicographic decrease.
    # A path-level measure-delta theorem is still required for every primitive.
    all_decrease = False

    return _kernel_receipt(
        "SAME_TIME_CLOSURE_TERMINATES",
        "PASS" if all_decrease else "UNRESOLVED",
        {
            "lemma": "CLOSURE_TERMINATION",
            "measure": list(measure_order),
            "measure_cardinality": 7,
            "measure_lower_bound": (0, 0, 0, 0, 0, 0, 0),
            "measure_well_founded": True,
            "transitions": transitions,
            "no_earlier_component_increased": all_decrease,
            "generated_events_only_in_later_phase": all_decrease,
            "all_decrease_proved": all_decrease,
            "all_generated_only_later_phase": all_decrease,
            "proof_scope": "ALL_LEGAL_SAME_TIME_CLOSURES",
            "proof_type": "lexicographic_measure",
        },
        code=None if all_decrease else "SOURCE_BOUND_CLOSURE_MEASURE_DECREASE_REQUIRED",
        reason="The measure definition is well-founded, but strict decrease for "
               "every executable micro-step has not been established from total "
               "compiled transition semantics.",
    )


# ---------------------------------------------------------------------------
# Canonical successor totality (PE3 component)
# ---------------------------------------------------------------------------

def prove_canonical_successor_total_kernel() -> dict[str, Any]:
    """Prove canonical successor is total over all legal closed states.

    After closure termination, the post-closure state has:
    - Either a ready job (service branch: one SERVICE_UNIT advances time by 1)
    - Or no ready job (idle branch: jump to earliest future event)
    Both branches strictly advance time.
    """
    from .executable_transition_compiler import compile_all_transitions
    compiled = tuple(compile_all_transitions())
    compiled_ok = len(compiled) == 9 and all(ir.is_compiled() for ir in compiled)
    totality_proved = False
    return _kernel_receipt(
        "PROTECTED_PREFIX_CANONICAL_SUCCESSOR_TOTAL",
        "PASS" if totality_proved else "UNRESOLVED",
        {
            "lemma": "CANONICAL_SUCCESSOR_TOTAL",
            "all_legal_closed_states_covered": totality_proved,
            "same_time_closure_termination_consumed": totality_proved,
            "future_event_or_service_branch_total": totality_proved,
            "projected_oracle_contract_consumed": totality_proved,
            "branches": {
                "service_branch": {
                    "condition": "ready nonempty",
                    "action": "SERVICE_UNIT or TAIL_ONLY_SERVICE",
                    "time_delta": 1,
                },
                "idle_branch": {
                    "condition": "ready empty",
                    "action": "close_timestamp jump to earliest future event",
                    "time_delta": "> 0 (strictly future)",
                },
            },
            "case_split_total": totality_proved,
            "proof_type": "two_branch_totality",
        },
        code=None if totality_proved else "SOURCE_BOUND_CANONICAL_SUCCESSOR_TOTALITY_REQUIRED",
        reason="The branch outline is not a totality proof.  It still requires "
               "source-bound closure termination, legal-state preservation, "
               "projected-oracle input totality and the no-future-event case.",
    )


# ---------------------------------------------------------------------------
# Time divergence (PE4)
# ---------------------------------------------------------------------------

def prove_time_divergence_kernel() -> dict[str, Any]:
    """Prove time diverges under repeated successor application.

    Each macro-step advances time by >= 1 (service branch: exactly 1;
    idle branch: strictly positive).  Induction => unbounded time growth.
    No infinite zero-time chain (non-Zeno).
    """
    from .executable_transition_compiler import compile_all_transitions
    compiled = {ir.case_id: ir for ir in compile_all_transitions()}
    service_ir_available = all(compiled.get(case) is not None and compiled[case].is_compiled()
                               for case in ("SERVICE_UNIT", "TAIL_ONLY_SERVICE"))
    divergence_proved = False
    return _kernel_receipt(
        "PROTECTED_PREFIX_TIME_DIVERGENCE",
        "PASS" if divergence_proved else "UNRESOLVED",
        {
            "lemma": "TIME_DIVERGENCE",
            "service_ir_available": service_ir_available,
            "service_branch_advances_by_1": divergence_proved,
            "idle_branch_jumps_to_future_event": divergence_proved,
            "unbounded_iteration_proved": divergence_proved,
            "non_zeno": divergence_proved,
            "minimum_progress_per_step": 1,
            "proof_type": "induction_on_time",
        },
        code=None if divergence_proved else "SOURCE_BOUND_TIME_DIVERGENCE_REQUIRED",
        reason="Time divergence requires the proved total successor and an "
               "explicit infinite-idle/terminal extension when no future event exists.",
    )


# ---------------------------------------------------------------------------
# Idle jump stutter expansion (PP5-B/D component)
# ---------------------------------------------------------------------------

def prove_idle_jump_stutter_kernel() -> dict[str, Any]:
    """Prove idle jump is a protected-observable stutter.

    When the protected ready set is empty, the prefix idle jump
    (time advance to next event by close_timestamp) does not change
    any protected observable field.  Time advances; no protected job
    gets service, completes, or misses a deadline during the jump.

    This is a LOCAL theorem about the prefix transition system,
    independent of complete execution existence (no circularity).
    """
    from .executable_transition_compiler import compile_all_transitions
    compiled = {ir.case_id: ir for ir in compile_all_transitions()}
    source_ir_available = all(
        compiled.get(case) is not None and compiled[case].is_compiled()
        for case in ("FINAL_DISPATCH", "SERVICE_UNIT", "TAIL_ONLY_SERVICE")
    )
    # The no-skipped-release/deadline frame theorem is not derivable from
    # source availability alone.
    source_bound = False
    return _kernel_receipt(
        "PROTECTED_PREFIX_IDLE_JUMP_STUTTER_EXPANSION",
        "PASS" if source_bound else "UNRESOLVED",
        {
            "theorem_id": "PROTECTED_PREFIX_IDLE_JUMP_STUTTER_EXPANSION",
            "parameterized": source_bound,
            "independent_of_complete_execution_witness": True,
            "all_integer_times_observable": source_bound,
            "close_at_defined_for_every_intermediate_integer": source_bound,
            "source_ir_available": source_ir_available,
            "source_bound_transition_relation": source_bound,
            "protected_observable_frame_proved": source_bound,
            "lemma": "IDLE_JUMP_STUTTER_EXPANSION",
            "description": (
                "When protected ready set is empty, time advance by "
                "close_timestamp does not modify protected observable. "
                "No protected job gets service, completes, or misses deadline."
            ),
            "preserved_observable_fields": [
                "release_time", "absolute_deadline", "criticality",
                "fixed_demand", "service", "active", "ready",
                "priority_index", "completed", "missed",
            ],
            "proof_type": "source_bound_local_frame_theorem",
            "scope": "ALL_INTEGER_TIME_JUMPS",
        },
        code=None if source_bound else "SOURCE_BOUND_IDLE_JUMP_FRAME_THEOREM_REQUIRED",
        reason="The frame statement must be proved against close_timestamp and "
               "the event frontier, including the guarantee that no protected "
               "deadline or release is skipped during the jump.",
    )


# ---------------------------------------------------------------------------
# Complete execution existence (PP0-T, PE1-PE4)
# ---------------------------------------------------------------------------

def prove_complete_execution_exists_kernel(
    *,
    prefix_taskset: Any,
) -> dict[str, Any]:
    """Prove one complete time-divergent execution exists for a fixed input stream.

    PE1: Standard empty LO initial state
    PE2: Input-total (every release accepted)
    PE3: Canonical successor total (closure terminates + branch total)
    PE4: Time-divergent

    Uses dependent choice: from initial state, repeatedly apply total
    successor function.  The result is a single infinite execution that
    is a true prefix of all its finite approximations.
    """
    return _kernel_receipt(
        "PROTECTED_PREFIX_COMPLETE_EXECUTION_EXISTS",
        "UNRESOLVED",
        {
            "lemma": "COMPLETE_EXECUTION_EXISTS",
            "theorem_id": "PROTECTED_PREFIX_COMPLETE_EXECUTION_EXISTS",
            "dependent_choice_construction_verified": False,
            "canonical_successor_total_consumed": False,
            "recurring_history_preserved": False,
            "same_fixed_oracle": False,
            "pe1_standard_initial_state": False,
            "pe2_input_totality": False,
            "pe3_canonical_successor_total": False,
            "pe4_time_divergence": False,
            "quantifier_order": "forall-full-exists-one-prefix-forall-boundaries",
            "complete_execution_constructed": False,
            "construction": "dependent_choice_on_total_successor",
            "proof_type": "dependent_choice_construction",
        },
        code="SOURCE_BOUND_COMPLETE_EXECUTION_CONSTRUCTION_REQUIRED",
        reason="A dependent-choice outline is not an executable proof.  The "
               "single fixed oracle, initial state, total successor and all finite "
               "prefix compatibility must be constructed and bound explicitly.",
    )


# ---------------------------------------------------------------------------
# Weak forward simulation (PP6)
# ---------------------------------------------------------------------------

def prove_weak_forward_simulation_kernel(
    *,
    macro_step_receipt: Mapping[str, Any] | None = None,
    execution_receipt: Mapping[str, Any] | None = None,
    base_case_receipt: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Prove full-to-prefix weak forward simulation.

    Quantifier order:
      forall full execution, exists one complete prefix execution,
      forall closed boundaries t:
        Obs_P(Close_full(t)) = Obs_P(Close_pp(t))

    Proof structure:
      1. Fix arbitrary full execution
      2. Project fixed protected input stream (PP0-K, PP3, PP4)
      3. Construct complete prefix execution via dependent choice (PP0-T)
      4. Base case: t=0, standard empty LO states (PP5.4/PP5.5)
      5. Induction: L8 macro-step for all t (PP5-F)
      6. Conclusion: relation holds at all Close(t)
    """
    from .executable_transition_compiler import compile_all_transitions
    from .phase_relation import JOB_FIELDS, PENDING_RELEASE_FIELDS
    compiled = tuple(compile_all_transitions())
    executable_bound = len(compiled) == 9 and all(item.is_compiled() for item in compiled)
    def passed(value: Mapping[str, Any] | None) -> bool:
        if not isinstance(value, Mapping):
            return False
        return value.get("status", value.get("obligation_status")) == "PASS" or (
            isinstance(value.get("witness"), Mapping)
            and value["witness"].get("status") == "PASS"
        )
    predecessors_bound = all(passed(value) for value in (
        macro_step_receipt, execution_receipt, base_case_receipt,
    ))
    # PASS-labelled predecessors do not themselves constitute the natural-
    # number induction object or establish witness/oracle identity.
    established = False
    return _kernel_receipt(
        "PROTECTED_PREFIX_WEAK_FORWARD_SIMULATION",
        "PASS" if established else "UNRESOLVED",
        {
            "theorem_id": "PROTECTED_PREFIX_WEAK_FORWARD_SIMULATION",
            "quantifier_order": "forall-full-exists-one-prefix-forall-boundaries",
            "induction_on_t_complete": established,
            "fixed_oracle_identity_checked": established,
            "witness_identity_checked": established,
            "relation_schema": "phase_relation_v4_close_at",
            "base_case_proved": passed(base_case_receipt),
            "complete_execution_witness_proved": passed(execution_receipt),
            "macro_step_L1_L8_proved": passed(macro_step_receipt),
            "observation": "Obs_P preserves: job key, criticality, release, "
                          "deadline, priority, actual demand, HI class, "
                          "service, completion, miss ledger.  Excludes: "
                          "global mode, LO version label, tail state.",
            "induction_proved": established,
            "source_transition_ir_hashes": [item.ir_hash() for item in compiled],
            "preserved_job_fields": list(JOB_FIELDS),
            "preserved_pending_release_fields": list(PENDING_RELEASE_FIELDS),
            "predecessor_receipt_hashes": {
                "macro_step": sha256_object(macro_step_receipt or {}),
                "execution": sha256_object(execution_receipt or {}),
                "base_case": sha256_object(base_case_receipt or {}),
            },
            "proof_type": "induction_over_closed_boundaries",
        },
        code=None if established else "SOURCE_BOUND_WEAK_SIMULATION_INDUCTION_REQUIRED",
        reason="Requires verified base relation, one compatible complete prefix "
               "execution, and a source-bound L8 induction step for every t.",
    )


# ---------------------------------------------------------------------------
# HI bad-prefix reflection (PP6-A/B)
# ---------------------------------------------------------------------------

def prove_hi_bad_prefix_reflection_kernel(
    *,
    simulation_receipt: Mapping[str, Any] | None = None,
    deadline_batch_receipt: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Prove full-reference HI miss reflects to prefix HI miss.

    If full reference execution has an HI job miss deadline at time t*,
    then the PP6-constructed prefix execution has the same HI job miss
    at the same time t*.

    Reflection fields preserved:
      - job key (task_name, release_index)
      - task criticality (= HI)
      - release time
      - absolute deadline (= t*)
      - actual demand
      - accumulated service at t* (< actual_demand)
      - completion flag (= False)
      - miss ledger entry

    N4PPBoundaryAlignment ensures PreClosed(t*) projects to Close(t*).
    """
    def passed(value: Mapping[str, Any] | None) -> bool:
        if not isinstance(value, Mapping):
            return False
        return value.get("status", value.get("obligation_status")) == "PASS" or (
            isinstance(value.get("witness"), Mapping)
            and value["witness"].get("status") == "PASS"
        )
    predecessor_statuses_available = passed(simulation_receipt) and passed(deadline_batch_receipt)
    source_bound = False
    preserved = (
        "job_key", "criticality", "release_time", "absolute_deadline",
        "actual_demand", "executed_service", "completed", "missed",
    )
    return _kernel_receipt(
        "PROTECTED_PREFIX_HI_BAD_PREFIX_REFLECTION",
        "PASS" if source_bound else "UNRESOLVED",
        {
            "theorem_id": "PROTECTED_PREFIX_HI_BAD_PREFIX_REFLECTION",
            "reflection_fields": [
                "job_key", "criticality", "release_time",
                "absolute_deadline", "actual_demand", "service",
                "completed", "missed",
            ],
            "preserved_by": "PP6 weak forward simulation (Obs_P equality)",
            "boundary_alignment": "N4PPBoundaryAlignment: PreClosed -> Close",
            "reflection_lemma": "PP6-A",
            "finite_bad_prefix_lemma": "PP6-B",
            "all_reflection_fields_derived": source_bound,
            "predecessor_statuses_available": predecessor_statuses_available,
            "source_bound": source_bound,
            "preserved_job_fields": list(preserved),
            "predecessor_receipt_hashes": {
                "simulation": sha256_object(simulation_receipt or {}),
                "deadline_batch": sha256_object(deadline_batch_receipt or {}),
            },
            "requires": [
                "PP6 weak forward simulation",
                "HI jobs are all protected",
                "deadline observation is observe-only (PP0-E/G4)",
                "miss ledger included in Obs_P",
            ],
            "proof_type": "simulation_projection",
        },
        code=None if source_bound else "SOURCE_BOUND_BAD_PREFIX_DERIVATION_REQUIRED",
        reason="The implication follows only after the weak simulation receipt "
               "exports the protected field equalities and the deadline batch "
               "relation is source-bound.",
    )


# ---------------------------------------------------------------------------
# Prefix imported assumption discharge (PP7-A)
# ---------------------------------------------------------------------------

def prove_pp7a_imported_assumptions_kernel() -> dict[str, Any]:
    """PP7-A: Prefix satisfies imported C-AMC-sem theorem assumptions.

    Discharges: finite independent sporadic taskset, constrained deadlines,
    single-processor FPPS, no blocking, WCET monotonicity (PP2), release-fixed
    bounded demands (PP3/PP4), classification/switch/recovery semantics (FPR),
    complete recurring history, candidate domain completeness, discrete-tick model.
    """
    return _kernel_receipt(
        "PP7A_IMPORTED_ASSUMPTIONS_DISCHARGED",
        "UNRESOLVED",
        {
            "lemma": "PREFIX_IMPORTED_ASSUMPTIONS_DISCHARGED",
            "imported_theorem": "C_AMC_SEM_ALL_TASK_SCHEDULABILITY_SUFFICIENCY",
            "discharged_conditions": {
                "finite_independent_taskset": "PPC1-PPC4, PPC5",
                "constrained_deadlines": "PPC5 (period/deadline inherited)",
                "single_processor_fpps": "FPR1, FPR4",
                "no_blocking": "PP0-B, FPR5",
                "wcer_monotonicity": "PP2",
                "release_fixed_bounded_demands": "PP3, PP4, PP0-D, PP0-H",
                "classification_at_arrival": "PP0-G, FPR2",
                "abnormal_switch_trigger": "PP0-G, FPR2",
                "idle_recovery": "PP0-G, FPR2",
                "lo_version_selection": "FPR2",
                "hi_primary_semantics": "FPR2",
                "standard_empty_lo_initial": "PE1",
                "complete_recurring_history": "PP0-T, PE1-PE4",
                "candidate_domain_completeness": "separate RTA checker obligation",
                "discrete_tick_submodel": "PP0-I, PP0-L",
            },
            "proof_type": "assumption_discharge",
        },
        code="PREFIX_MODEL_CONFORMANCE_RECEIPT_REQUIRED",
        reason="Assumption names are an outline.  A prefix-specific model "
               "conformance receipt must discharge each imported theorem premise.",
    )


# ---------------------------------------------------------------------------
# Prefix all-task RTA to mathematical inequalities (PP7-B)
# ---------------------------------------------------------------------------

def prove_pp7b_rta_to_inequalities_kernel() -> dict[str, Any]:
    """PP7-B: Checker PASS lifts to mathematical RTA inequalities.

    If the checker correctly enumerates all candidates and the arithmetic
    kernel is sound, then PrefixAllTaskCheckerPASS implies:
      forall tau_i in Gamma_pp: R_i(LO) <= D_i and R_i(HI) <= D_i.
    """
    return _kernel_receipt(
        "PP7B_RTA_TO_INEQUALITIES",
        "UNRESOLVED",
        {
            "lemma": "CHECKER_PASS_LIFTS_TO_MATHEMATICAL_INEQUALITIES",
            "requires": [
                "PrefixRTAEnumerationCompleteness",
                "PrefixRTAArithmeticKernelSoundness",
                "PrefixRTAResultInterpretationSoundness",
                "ImportedWCRTTheoremBinding",
                "PrefixInstanceEvidenceBinding",
            ],
            "conclusion": "forall tau_i: R_i(LO) <= D_i AND R_i(HI) <= D_i",
            "proof_type": "verifier_soundness_composition",
        },
        code="RTA_VERIFIER_SOUNDNESS_RECEIPTS_REQUIRED",
        reason="The implication needs actual enumeration, arithmetic, theorem "
               "binding and instance-fingerprint receipts; listing them is not proof.",
    )


# ---------------------------------------------------------------------------
# Final contradiction (PP8): prefix schedulability => full-reference HI safety
# ---------------------------------------------------------------------------

def prove_pp8_reference_hi_safety_from_prefix_kernel(
    *,
    bad_prefix_reflection_receipt: Mapping[str, Any] | None = None,
    mathematical_conformance_receipt: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """PP8: Prefix all-task schedulability implies full-reference HI safety.

    By contradiction:
      Assume full reference has HI miss at t*.
      PP6-A/B: prefix has same HI miss at t*.
      PP7-A/B + imported theorem + prefix all-task PASS:
        prefix has NO misses (all tasks schedulable).
      Contradiction => full reference has NO HI miss.
    """
    def passed(value: Mapping[str, Any] | None) -> bool:
        return isinstance(value, Mapping) and value.get("status") == "PASS"

    predecessor_statuses_available = passed(bad_prefix_reflection_receipt) and passed(
        mathematical_conformance_receipt
    )
    source_bound = False
    predecessor_hashes = {
        "PROTECTED_PREFIX_HI_BAD_PREFIX_REFLECTION": sha256_object(
            bad_prefix_reflection_receipt or {}
        ),
        "PROTECTED_PREFIX_MATHEMATICAL_CONFORMANCE": sha256_object(
            mathematical_conformance_receipt or {}
        ),
    }
    return _kernel_receipt(
        "REFERENCE_HI_SAFETY_FROM_PROTECTED_PREFIX",
        "PASS" if source_bound else "UNRESOLVED",
        {
            "theorem_id": "REFERENCE_HI_SAFETY_FROM_PROTECTED_PREFIX",
            "proof_by_contradiction": {
                "assume": "full reference has HI miss",
                "pp6_consequence": "prefix has same HI miss",
                "imported_theorem_consequence": "prefix has NO misses",
                "contradiction": "prefix both has and does not have HI miss",
                "conclusion": "full reference has NO HI miss",
            },
            "requires": [
                "PP6-A/B: HI bad-prefix reflection",
                "PP7-A: imported assumptions discharged",
                "PP7-B: RTA PASS => mathematical inequalities",
                "Imported theorem: all-task schedulability",
                "Prefix all-task RTA PASS",
            ],
            "proof_type": "contradiction",
            "predecessor_statuses_available": predecessor_statuses_available,
            "source_bound": source_bound,
            "predecessor_receipt_hashes": predecessor_hashes,
        },
        code=None if source_bound else "SOURCE_BOUND_SAFETY_COMPOSITION_REQUIRED",
        reason="The contradiction is valid only after consuming verified bad-prefix "
               "reflection and prefix all-task schedulability receipts.",
    )


# ---------------------------------------------------------------------------
# Master proof bundle: all parameterized proofs in one call
# ---------------------------------------------------------------------------

def generate_protected_prefix_proof_bundle(
    *,
    full_taskset: Any,
    prefix_taskset: Any,
    construction_result: Any,
) -> dict[str, Any]:
    """Generate the complete parameterized proof bundle.

    This produces ALL proof kernel receipts needed by the PP route.
    Individual route checkers bind these receipts to their specific
    taskset fingerprints and configuration hashes.
    """
    proofs = {
        # Structural PP1-PP4
        "PP1": prove_pp1_prefix_closure(
            construction_result=construction_result,
            full_taskset=full_taskset,
        ),
        "PP2": prove_pp2_saturation_legality(
            prefix_taskset=prefix_taskset,
        ),
        "PP3": prove_pp3_lo_demand_receptiveness(
            full_taskset=full_taskset,
            prefix_taskset=prefix_taskset,
            construction_result=construction_result,
        ),
        "PP4": prove_pp4_hi_demand_projection(
            full_taskset=full_taskset,
            prefix_taskset=prefix_taskset,
            construction_result=construction_result,
        ),

        # PP5 L1-L8 macro-step lemmas
        "PP5_L8": prove_l8_macro_step_preservation(
            construction_result=construction_result,
            full_taskset=full_taskset,
        ),

        # Execution existence
        "SAME_TIME_CLOSURE": prove_same_time_closure_termination_kernel(),
        "CANONICAL_SUCCESSOR": prove_canonical_successor_total_kernel(),
        "TIME_DIVERGENCE": prove_time_divergence_kernel(),
        "IDLE_JUMP_STUTTER": prove_idle_jump_stutter_kernel(),
        "COMPLETE_EXECUTION": prove_complete_execution_exists_kernel(
            prefix_taskset=prefix_taskset,
        ),

        # PP6-PP8
        "WEAK_SIMULATION": prove_weak_forward_simulation_kernel(),
        "HI_BAD_PREFIX_REFLECTION": prove_hi_bad_prefix_reflection_kernel(),
        "PP7A": prove_pp7a_imported_assumptions_kernel(),
        "PP7B": prove_pp7b_rta_to_inequalities_kernel(),
        "PP8": prove_pp8_reference_hi_safety_from_prefix_kernel(),
    }

    all_pass = all(
        p.get("status") == "PASS"
        for p in proofs.values()
    )

    return {
        "proof_bundle_version": PROOF_KERNEL_VERSION,
        "status": "PASS" if all_pass else "UNRESOLVED",
        "proofs": proofs,
        "full_taskset_fingerprint": full_taskset.to_dict().get("fingerprint"),
        "prefix_taskset_fingerprint": prefix_taskset.to_dict().get("fingerprint"),
        "all_proofs_pass": all_pass,
        "bundle_hash": sha256_object({
            name: p.get("receipt_hash", sha256_object(p))
            for name, p in proofs.items()
        }),
    }
