"""Non-blocking audit of the mutable experiment runtime.

This audit is intentionally not a proof obligation.  It records whether the
current implementation tree differs from the implementation hash frozen into a
request, while the C-AMC-sem proof remains bound to the frozen formal model.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from formal_toolchain.adapters.source_manifest import build_source_manifest


def audit_mutable_runtime(
    source_root: str | Path,
    *,
    expected_implementation_audit_hash: str | None = None,
) -> dict[str, Any]:
    manifest = build_source_manifest(Path(source_root))
    actual = str(manifest.get("implementation_audit_hash", ""))
    drift = bool(expected_implementation_audit_hash and actual != expected_implementation_audit_hash)
    return {
        "schema_version": "mutable_runtime_drift_audit_v1",
        "status": "WARN" if drift else "PASS",
        "blocking": False,
        "policy": "NON_BLOCKING_AUDIT_ONLY",
        "expected_implementation_audit_hash": expected_implementation_audit_hash,
        "actual_implementation_audit_hash": actual,
        "drift_detected": drift,
        "formal_semantics_hash": manifest.get("semantic_hash"),
        "guarantee_note": (
            "The proof covers the frozen C-AMC-sem/P0 model and exported target "
            "parameters; mutable runtime conformance is reported separately."
        ),
    }


__all__ = ["audit_mutable_runtime"]
