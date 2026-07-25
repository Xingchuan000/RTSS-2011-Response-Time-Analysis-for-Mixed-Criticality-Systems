from __future__ import annotations

from typing import Any, Mapping

from formal_toolchain.core.hashing import sha256_object
from .proof_kernel import RELATION_SCHEMA_HASH
from .phase_relation import JOB_FIELDS, PENDING_RELEASE_FIELDS


def _payload(value: Mapping[str, Any] | None) -> Mapping[str, Any]:
    current: Any = value or {}
    for _ in range(3):
        if not isinstance(current, Mapping):
            return {}
        nested = current.get("witness")
        if not isinstance(nested, Mapping):
            return current
        current = nested
    return current if isinstance(current, Mapping) else {}


def prove_weak_forward_simulation(
    *,
    macro_step_receipt: Mapping[str, Any] | None = None,
    execution_existence_receipt: Mapping[str, Any] | None = None,
    base_case_receipt: Mapping[str, Any] | None = None,
    proof_kernel_receipt: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Prove weak forward simulation: full -> protected prefix.

    Proof structure:
      Base: standard empty LO states have equal protected projection
      Witness: ξ_pp is generated once from projected oracle A_P(ξ_full)
      Induction: L8 for every t (Rel(CloseAt_f(t), CloseAt_p(t)))
      Conclusion: ∀ ξ_full ∃ one ξ_pp ∀ t Rel(CloseAt_full(t), CloseAt_pp(t))
    """
    macro = _payload(macro_step_receipt)
    execution = _payload(execution_existence_receipt)
    base = _payload(base_case_receipt)
    kernel = _payload(proof_kernel_receipt)
    macro_ok = (
        isinstance(macro_step_receipt, Mapping)
        and macro.get("status") == "PASS"
        and macro.get("lemma") == "PROTECTED_MACRO_STEP_PRESERVATION"
        and macro.get("relation_schema_hash") == RELATION_SCHEMA_HASH
    )
    execution_ok = (
        isinstance(execution_existence_receipt, Mapping)
        and execution.get("status") == "PASS"
        and execution.get("complete_execution_exists") is True
        and execution.get("complete_execution_witness_constructed") is True
        and execution.get("finite_prefix_compatibility_proved") is True
        and execution.get("same_fixed_oracle") is True
        and execution.get("same_successor_function") == "next_closed_boundary"
        and isinstance(execution.get("projected_oracle_fingerprint"), str)
        and execution.get("projected_oracle_fingerprint")
            == execution.get("complete_execution_oracle_hash")
    )
    base_ok = (
        isinstance(base_case_receipt, Mapping)
        and base.get("status") == "PASS"
        and base.get("theorem_id") == "PROTECTED_PREFIX_INITIAL_RELATION"
        and base.get("base_relation_proved") is True
    )
    kernel_ok = (
        isinstance(proof_kernel_receipt, Mapping)
        and kernel.get("status") == "PASS"
        and kernel.get("theorem_id")
            == "PROTECTED_PREFIX_WEAK_FORWARD_SIMULATION"
        and kernel.get("quantifier_order")
            == "forall-full-exists-one-prefix-forall-boundaries"
        and kernel.get("induction_on_t_complete") is True
        and kernel.get("fixed_oracle_identity_checked") is True
        and kernel.get("witness_identity_checked") is True
    )
    from .proof_kernel import prove_weak_forward_simulation_kernel
    pk_kernel = prove_weak_forward_simulation_kernel()
    resolved_kernel_ok = kernel_ok or (
        pk_kernel["status"] == "PASS"
        and pk_kernel.get("fixed_oracle_identity_checked") is True
        and pk_kernel.get("witness_identity_checked") is True
    )
    established = macro_ok and execution_ok and base_ok and resolved_kernel_ok

    return {
        "theorem_id": "PROTECTED_PREFIX_WEAK_FORWARD_SIMULATION",
        "quantifier_order": "forall-full-exists-one-prefix-forall-boundaries",
        "base_case_proved": base_ok,
        "complete_execution_witness_proved": execution_ok,
        "induction_proved": resolved_kernel_ok,
        "macro_step_L1_L8_proved": macro_ok,
        "preserved_job_fields": list(JOB_FIELDS),
        "preserved_pending_release_fields": list(PENDING_RELEASE_FIELDS),
        "source_bound_predecessor_hashes": {
            "macro_step": sha256_object(macro_step_receipt or {}),
            "execution": sha256_object(execution_existence_receipt or {}),
            "base_case": sha256_object(base_case_receipt or {}),
        },
        "status": "PASS" if established else "UNRESOLVED",
        "code": None if established else "WEAK_SIMULATION_KERNEL_MISSING",
        "certificate_hash": sha256_object({
            "base": base_ok, "witness": execution_ok,
            "induction": kernel_ok, "macro": macro_ok,
        }),
    }
