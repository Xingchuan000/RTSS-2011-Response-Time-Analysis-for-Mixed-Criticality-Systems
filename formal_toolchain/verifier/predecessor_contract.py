from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from formal_toolchain.core.artifact import verify_obligation_certificate
from formal_toolchain.core.contexts import expected_context_for_obligation
from formal_toolchain.core.predecessor_contract import validate_verified_predecessor


class PredecessorContractError(ValueError):
    pass


def require_verified_predecessor(
    *,
    predecessors: Mapping[str, Mapping[str, Any]],
    obligation_id: str,
    contexts: Mapping[str, Mapping[str, Any]],
) -> Mapping[str, Any]:
    try:
        return validate_verified_predecessor(
            predecessors=predecessors, obligation_id=obligation_id, contexts=contexts,
        )
    except ValueError as exc:
        raise PredecessorContractError(str(exc)) from exc


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
