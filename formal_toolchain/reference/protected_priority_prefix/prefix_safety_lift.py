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
    math_witness = {}
    if isinstance(mathematical_conformance_receipt, Mapping):
        math_witness = mathematical_conformance_receipt.get(
            "witness", mathematical_conformance_receipt
        )
        if isinstance(math_witness, Mapping) and isinstance(math_witness.get("witness"), Mapping):
            math_witness = math_witness["witness"]
    predecessor_hashes = {
        "PROTECTED_PREFIX_HI_BAD_PREFIX_REFLECTION": (
            bad_prefix_reflection_receipt.get("artifact_hash")
            if isinstance(bad_prefix_reflection_receipt, Mapping) else None
        ),
        "PROTECTED_PREFIX_MATHEMATICAL_CONFORMANCE": (
            mathematical_conformance_receipt.get("artifact_hash")
            if isinstance(mathematical_conformance_receipt, Mapping) else None
        ),
    }
    composition_bound = (
        isinstance(math_witness, Mapping)
        and math_witness.get("theorem_id") == "PROTECTED_PREFIX_MATHEMATICAL_CONFORMANCE"
        and math_witness.get("proof_partition") == ["PP7-A1", "PP7-A2", "PP7-B"]
        and isinstance(math_witness.get("pp7_a1_model_conformance_hash"), str)
        and isinstance(math_witness.get("pp7_a2_imported_theorem_binding_hash"), str)
        and isinstance(math_witness.get("pp7_b_rta_soundness_hash"), str)
    )
    source_kernel_ok = (
        isinstance(proof_kernel_receipt, Mapping)
        and proof_kernel_receipt.get("status") == "PASS"
        and proof_kernel_receipt.get("theorem_id")
            == "REFERENCE_HI_SAFETY_FROM_PROTECTED_PREFIX"
        and proof_kernel_receipt.get("source_bound") is True
        and proof_kernel_receipt.get("contradiction_proved") is True
        and proof_kernel_receipt.get("predecessor_receipt_hashes")
            == predecessor_hashes
    )
    # PP8 is a trusted contradiction-composition step only after the exact
    # verified predecessor artifacts have been bound by the source proof
    # kernel.  Two PASS labels or shape-compatible dictionaries are not enough.
    established = bad_prefix_ok and math_ok and composition_bound and source_kernel_ok
    if not established:
        return {
            "theorem_id": "REFERENCE_HI_SAFETY_FROM_PROTECTED_PREFIX",
            "conclusion": "ALL_REFERENCE_HI_JOBS_MEET_DEADLINES",
            "scope": "FULL_REFERENCE_HI_SAFETY_ONLY",
            "lo_tail_safety_not_claimed": True,
            "bad_prefix_reflection_consumed": bad_prefix_ok,
            "prefix_mathematical_conformance_consumed": math_ok,
            "contradiction_proved": False,
            "source_bound_composition": source_kernel_ok,
            "status": "UNRESOLVED",
            "code": "PP8_PREDECESSOR_COMPOSITION_UNRESOLVED",
            "predecessor_theorem_ids": list(predecessor_hashes),
            "predecessor_receipt_hashes": predecessor_hashes,
            "certificate_hash": sha256_object({"predecessors": predecessor_hashes}),
        }

    return {
        "theorem_id": "REFERENCE_HI_SAFETY_FROM_PROTECTED_PREFIX",
        "conclusion": "ALL_REFERENCE_HI_JOBS_MEET_DEADLINES",
        "scope": "FULL_REFERENCE_HI_SAFETY_ONLY",
        "lo_tail_safety_not_claimed": True,
        "bad_prefix_reflection_consumed": bad_prefix_ok,
        "prefix_mathematical_conformance_consumed": math_ok,
        "contradiction_proved": True,
        "source_bound_composition": source_kernel_ok,
        "predecessor_theorem_ids": list(predecessor_hashes),
        "predecessor_receipt_hashes": predecessor_hashes,
        "pp7_composition_receipt_hash": math_witness.get("receipt_hash"),
        "status": "PASS",
        "code": None,
        "certificate_hash": sha256_object({
            "bad_prefix": bad_prefix_ok,
            "math_conformance": math_ok,
            "composition": composition_bound,
            "predecessors": predecessor_hashes,
        }),
    }
