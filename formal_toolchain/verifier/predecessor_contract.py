from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from formal_toolchain.core.artifact import verify_obligation_certificate
from formal_toolchain.core.contexts import expected_context_for_obligation


class PredecessorContractError(ValueError):
    pass


def require_verified_predecessor(
    *,
    predecessors: Mapping[str, Mapping[str, Any]],
    obligation_id: str,
    contexts: Mapping[str, Mapping[str, Any]],
) -> Mapping[str, Any]:
    cert = predecessors.get(obligation_id)
    if not isinstance(cert, Mapping):
        raise PredecessorContractError(f"missing predecessor: {obligation_id}")
    if cert.get("obligation_id") != obligation_id:
        raise PredecessorContractError(f"wrong predecessor id: {obligation_id}")
    if cert.get("obligation_status") != "PASS":
        raise PredecessorContractError(f"predecessor not PASS: {obligation_id}")
    if not verify_obligation_certificate(cert):
        raise PredecessorContractError(f"invalid predecessor hash: {obligation_id}")
    expected = expected_context_for_obligation(obligation_id, contexts)
    if cert.get("certificate_context_hash") != expected:
        raise PredecessorContractError(f"predecessor context mismatch: {obligation_id}")
    return cert


def require_exact_predecessor_set(
    *,
    predecessors: Mapping[str, Mapping[str, Any]],
    expected_ids: set[str],
) -> None:
    actual = set(predecessors)
    if actual != expected_ids:
        missing = sorted(expected_ids - actual)
        extra = sorted(actual - expected_ids)
        raise PredecessorContractError(
            f"predecessor set mismatch: missing={missing}, extra={extra}"
        )
