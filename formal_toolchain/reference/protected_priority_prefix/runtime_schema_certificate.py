"""Obligation-envelope helpers for PP0 runtime conformance."""

from __future__ import annotations

from typing import Any, Mapping

from formal_toolchain.core.artifact import obligation_certificate
from .runtime_schema import build_runtime_schema_certificate, verify_runtime_schema_certificate


OBLIGATION_ID = "PROTECTED_PREFIX_RUNTIME_SCHEMA_CONFORMANCE"


def build_runtime_schema_obligation_certificate(*, context_hash: str, predecessor: Mapping[str, Any]) -> dict[str, Any]:
    runtime = build_runtime_schema_certificate()
    status = "PASS" if predecessor.get("obligation_status") == "PASS" and runtime["status"] == "PASS" else runtime["status"]
    return obligation_certificate(
        obligation_id=OBLIGATION_ID, status=status, context_hash=context_hash,
        inputs={"predecessor_ids": ["SATURATED_PROTECTED_PREFIX_REFERENCE"], "runtime_certificate_hash": runtime["certificate_hash"]},
        witness=runtime, checker_id=__name__, checker_version="protected-prefix-runtime-v1",
        failure=None if status == "PASS" else {"code": "RUNTIME_SCHEMA_CONFORMANCE_FAILED", "route": "UNRESOLVED", "runtime": runtime},
    )


__all__ = ["build_runtime_schema_certificate", "verify_runtime_schema_certificate", "build_runtime_schema_obligation_certificate"]
