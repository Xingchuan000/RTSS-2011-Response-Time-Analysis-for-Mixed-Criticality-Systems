from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def verify_early_stop_configuration_gate(*, raw_inputs=None, candidate_evidence=None,
                                         expected_context_hash=None, **kwargs: Any) -> dict[str, Any]:
    if raw_inputs is None:
        return {"status": "UNRESOLVED", "route": "UNRESOLVED", "code": "OBLIGATION_EVIDENCE_MISSING"}
    enabled = bool(getattr(raw_inputs.target.runtime_config, "stop_at_first_miss", False))
    witness = {"stop_at_first_miss": enabled, "closure_completion_required": enabled}
    if enabled:
        closure = kwargs.get("closure_completion_certificate")
        if not isinstance(closure, Mapping) or closure.get("obligation_status") != "PASS":
            return {"status": "UNRESOLVED", "route": "MODEL_CONFORMANCE_FAILED", "code": "EARLY_STOP_CLOSURE_COMPLETION_MISSING", "witness": witness}
    return {"status": "PASS", "route": None, "code": None, "witness": witness}
