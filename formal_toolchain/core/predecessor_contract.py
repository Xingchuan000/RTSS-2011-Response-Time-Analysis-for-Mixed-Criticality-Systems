from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .artifact import verify_obligation_certificate
from .contexts import expected_context_for_obligation


def validate_verified_predecessor(
    *, predecessors: Mapping[str, Mapping[str, Any]], obligation_id: str,
    contexts: Mapping[str, Mapping[str, Any]],
) -> Mapping[str, Any]:
    certificate = predecessors.get(obligation_id)
    if not isinstance(certificate, Mapping):
        raise ValueError(f"missing predecessor: {obligation_id}")
    if certificate.get("obligation_id") != obligation_id:
        raise ValueError(f"wrong predecessor id: {obligation_id}")
    if certificate.get("obligation_status") != "PASS":
        raise ValueError(f"predecessor not PASS: {obligation_id}")
    if not verify_obligation_certificate(certificate):
        raise ValueError(f"invalid predecessor hash: {obligation_id}")
    expected_context_hash = expected_context_for_obligation(obligation_id, contexts)
    if certificate.get("certificate_context_hash") != expected_context_hash:
        raise ValueError(f"predecessor context mismatch: {obligation_id}")
    return certificate
