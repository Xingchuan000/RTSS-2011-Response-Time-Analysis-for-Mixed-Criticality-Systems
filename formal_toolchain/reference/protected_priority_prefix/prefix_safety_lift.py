from __future__ import annotations

from typing import Any, Mapping

from formal_toolchain.core.hashing import sha256_object


def prove_reference_hi_safety_from_prefix(
    *,
    bad_prefix_reflection_receipt: Mapping[str, Any] | None = None,
    mathematical_conformance_receipt: Mapping[str, Any] | None = None,
    proof_kernel_receipt: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Prove full-reference HI safety from protected prefix.

    Prerequisites:
      PROTECTED_PREFIX_HI_BAD_PREFIX_REFLECTION = PASS
      PROTECTED_PREFIX_MATHEMATICAL_CONFORMANCE = PASS (incl. prefix model + RTA)

    Proof by contradiction:
      Assume full reference has a HI miss
      -> PP6: corresponding prefix execution has a HI miss
      -> prefix all-task RTA: no miss possible
      -> contradiction
      -> full reference has no HI miss

    Conclusion: full-reference HI safety only (no claim about LO tail safety).
    """
    bad_prefix_ok = (
        isinstance(bad_prefix_reflection_receipt, Mapping)
        and bad_prefix_reflection_receipt.get("status") == "PASS"
        and bad_prefix_reflection_receipt.get("bad_prefix_reflection_hash") is not None
    )
    math_ok = (
        isinstance(mathematical_conformance_receipt, Mapping)
        and mathematical_conformance_receipt.get("status") == "PASS"
    )
    kernel_ok = (
        isinstance(proof_kernel_receipt, Mapping)
        and proof_kernel_receipt.get("status") == "PASS"
        and proof_kernel_receipt.get("theorem_id") == "PROTECTED_PREFIX_HI_SAFETY_LIFT"
        and proof_kernel_receipt.get("bad_prefix_reflection_consumed") is True
        and proof_kernel_receipt.get("prefix_all_task_schedulability_consumed") is True
        and proof_kernel_receipt.get("contradiction_proved") is True
        and proof_kernel_receipt.get("conclusion_is_full_reference_hi_safety_only") is True
    )
    established = bad_prefix_ok and math_ok and kernel_ok

    return {
        "theorem_id": "REFERENCE_HI_SAFETY_FROM_PROTECTED_PREFIX",
        "conclusion": "ALL_REFERENCE_HI_JOBS_MEET_DEADLINES",
        "scope": "FULL_REFERENCE_HI_SAFETY_ONLY",
        "lo_tail_safety_not_claimed": True,
        "bad_prefix_reflection_consumed": bad_prefix_ok,
        "prefix_mathematical_conformance_consumed": math_ok,
        "contradiction_proved": kernel_ok,
        "status": "PASS" if established else "UNRESOLVED",
        "code": None if established else "HI_SAFETY_LIFT_KERNEL_MISSING",
        "certificate_hash": sha256_object({
            "bad_prefix": bad_prefix_ok,
            "math_conformance": math_ok,
            "kernel": kernel_ok,
        }),
    }
