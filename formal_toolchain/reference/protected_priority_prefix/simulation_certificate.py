"""Certificate builder for the parameterized weak-forward-simulation receipt."""

from __future__ import annotations

from typing import Any

from formal_toolchain.core.hashing import sha256_object
from .macro_step import prove_protected_macro_step_preservation
from .runtime_schema import build_runtime_schema_certificate


def build_simulation_certificate(*, full_taskset: object, prefix_taskset: object, construction: Any, domain_witness: dict[str, Any]) -> dict[str, Any]:
    macro = prove_protected_macro_step_preservation(construction=construction, full_taskset=full_taskset, prefix_taskset=prefix_taskset)
    payload = {
        "schema_version": "protected-prefix-weak-forward-simulation-v1",
        "quantification": "forall full execution exists one prefix execution forall natural-number closed boundaries",
        "relation": "protected observable equality at close boundaries",
        "domain_witness": domain_witness,
        "runtime_schema_certificate_hash": build_runtime_schema_certificate()["certificate_hash"],
        "macro_step_receipt": macro,
        "full_taskset_fingerprint": full_taskset.to_dict()["fingerprint"],
        "prefix_taskset_fingerprint": prefix_taskset.to_dict()["fingerprint"],
    }
    domain_ok = domain_witness.get("status", True) in (True, "PASS")
    if domain_ok and macro["status"] == "PASS":
        status = "PASS"
    elif macro["status"] == "UNRESOLVED":
        status = "UNRESOLVED"
    else:
        status = "FAIL"
    return {**payload, "status": status,
            "certificate_hash": sha256_object(payload)}
