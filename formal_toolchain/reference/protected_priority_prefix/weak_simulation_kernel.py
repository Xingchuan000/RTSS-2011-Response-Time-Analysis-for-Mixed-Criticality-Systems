from __future__ import annotations

from typing import Any, Mapping

from formal_toolchain.core.hashing import sha256_object


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
    macro_ok = (
        isinstance(macro_step_receipt, Mapping)
        and macro_step_receipt.get("status") == "PASS"
        and macro_step_receipt.get("lemma") == "PROTECTED_MACRO_STEP_PRESERVATION"
    )
    execution_ok = (
        isinstance(execution_existence_receipt, Mapping)
        and execution_existence_receipt.get("status") == "PASS"
        and execution_existence_receipt.get("witness", {}).get("single_complete_execution") is True
    )
    base_ok = (
        isinstance(base_case_receipt, Mapping)
        and base_case_receipt.get("status") == "PASS"
        and base_case_receipt.get("base_relation_proved") is True
    )
    kernel_ok = (
        isinstance(proof_kernel_receipt, Mapping)
        and proof_kernel_receipt.get("status") == "PASS"
        and proof_kernel_receipt.get("theorem_id") == "PROTECTED_PREFIX_WEAK_FORWARD_SIMULATION"
        and proof_kernel_receipt.get("quantifier_order")
            == "forall-full-exists-one-prefix-forall-boundaries"
        and proof_kernel_receipt.get("induction_on_t_complete") is True
    )
    established = macro_ok and execution_ok and base_ok and kernel_ok

    return {
        "theorem_id": "PROTECTED_PREFIX_WEAK_FORWARD_SIMULATION",
        "quantifier_order": "forall-full-exists-one-prefix-forall-boundaries",
        "base_case_proved": base_ok,
        "single_witness_proved": execution_ok,
        "induction_proved": kernel_ok,
        "macro_step_L1_L8_proved": macro_ok,
        "status": "PASS" if established else "UNRESOLVED",
        "code": None if established else "WEAK_SIMULATION_KERNEL_MISSING",
        "certificate_hash": sha256_object({
            "base": base_ok, "witness": execution_ok,
            "induction": kernel_ok, "macro": macro_ok,
        }),
    }
