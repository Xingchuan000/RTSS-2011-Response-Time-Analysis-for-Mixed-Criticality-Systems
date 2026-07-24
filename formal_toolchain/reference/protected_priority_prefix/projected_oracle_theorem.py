"""Parameterized theorem receipts for full-to-protected input projection."""

from __future__ import annotations

from typing import Any, Mapping

from formal_toolchain.core.hashing import sha256_object


def prove_projected_oracle_theorem(
    *, full_view: Any, projected_oracle: Any, protected_task_names: frozenset[str],
    prefix_taskset_fingerprint: str, saturation_certificate_hash: str,
    proof_kernel_receipt: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Verify the quantified projection contract, never a finite q sample."""
    full_fp = getattr(full_view, "oracle_fingerprint", lambda: None)()
    projected_fp = getattr(projected_oracle, "oracle_fingerprint", lambda: None)()
    kernel_ok = (
        isinstance(proof_kernel_receipt, Mapping)
        and proof_kernel_receipt.get("status") == "PASS"
        and proof_kernel_receipt.get("theorem_id") == "PROTECTED_INPUT_STREAM_PROJECTION"
        and proof_kernel_receipt.get("forall_release_indices") is True
        and proof_kernel_receipt.get("finite_instance_data_used") is False
        and proof_kernel_receipt.get("full_oracle_fingerprint") == full_fp
        and proof_kernel_receipt.get("projected_oracle_fingerprint") == projected_fp
    )
    payload = {
        "theorem_id": "PROTECTED_INPUT_STREAM_PROJECTION",
        "status": "PASS" if kernel_ok else "UNRESOLVED",
        "forall_release_indices": kernel_ok,
        "finite_instance_data_used": False,
        "protected_task_names": sorted(protected_task_names),
        "full_oracle_fingerprint": full_fp,
        "projected_oracle_fingerprint": projected_fp,
        "prefix_taskset_fingerprint": prefix_taskset_fingerprint,
        "saturation_certificate_hash": saturation_certificate_hash,
        "protected_record_fields_preserved": kernel_ok,
        "tail_entries_deleted_only": kernel_ok,
    }
    payload["receipt_hash"] = sha256_object(payload)
    return payload
