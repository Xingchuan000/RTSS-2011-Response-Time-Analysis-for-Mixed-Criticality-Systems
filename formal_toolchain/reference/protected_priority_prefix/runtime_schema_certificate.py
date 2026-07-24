"""Obligation-envelope helpers for PP0 runtime conformance.

Uses the PP0 transition schema checker instead of the old AST-based checks.
"""

from __future__ import annotations

from typing import Any, Mapping

from formal_toolchain.core.artifact import obligation_certificate
from formal_toolchain.reference.protected_priority_prefix.pp0_checker import (
    build_pp0_transition_certificate,
)
from formal_toolchain.reference.protected_priority_prefix.runtime_schema import (
    build_runtime_schema_certificate,
    verify_runtime_schema_certificate,
)


OBLIGATION_ID = "PROTECTED_PREFIX_RUNTIME_SCHEMA_CONFORMANCE"


def build_runtime_schema_obligation_certificate(*, context_hash: str, predecessor: Mapping[str, Any]) -> dict[str, Any]:
    pp0 = build_pp0_transition_certificate()
    pred_ok = predecessor.get("obligation_status") == "PASS"
    status = "PASS" if pred_ok and pp0["status"] == "PASS" else pp0["status"]
    return obligation_certificate(
        obligation_id=OBLIGATION_ID, status=status, context_hash=context_hash,
        inputs={"predecessor_ids": ["SATURATED_PROTECTED_PREFIX_REFERENCE"],
                "pp0_certificate_hash": pp0["certificate_hash"]},
        witness=pp0, checker_id=__name__, checker_version="protected-prefix-pp0-v1",
        failure=None if status == "PASS" else {
            "code": pp0.get("code", "RUNTIME_SCHEMA_CONFORMANCE_UNRESOLVED"),
            "route": "UNRESOLVED",
            "pp0_certificate": pp0,
        },
    )


__all__ = ["build_runtime_schema_certificate", "verify_runtime_schema_certificate",
           "build_runtime_schema_obligation_certificate",
           "build_pp0_transition_certificate"]
