"""Certificate builder for the parameterized weak-forward-simulation receipt."""

from __future__ import annotations

from typing import Any

from formal_toolchain.core.hashing import sha256_object
from .macro_step import prove_protected_macro_step_preservation
from .runtime_schema import build_runtime_schema_certificate
from .execution_builder import prove_complete_execution_exists


def build_simulation_certificate(
    *,
    full_taskset: object,
    prefix_taskset: object,
    construction: Any,
    domain_witness: dict[str, Any],
    prefix_initial_state: object | None = None,
    protected_oracle: Any = None,
    transition_totality_receipt: dict[str, Any] | None = None,
    base_case_receipt: dict[str, Any] | None = None,
    proof_kernel_receipt: dict[str, Any] | None = None,
) -> dict[str, Any]:
    macro = prove_protected_macro_step_preservation(
        construction=construction, full_taskset=full_taskset, prefix_taskset=prefix_taskset,
    )
    full_fp = full_taskset.to_dict()["fingerprint"]
    prefix_fp = prefix_taskset.to_dict()["fingerprint"]

    execution_existence = prove_complete_execution_exists(
        canonical_successor_receipt=transition_totality_receipt,
        time_divergence_receipt=transition_totality_receipt,
        input_projection_receipt=domain_witness,
        demand_receptiveness_receipt=domain_witness,
        prefix_taskset=prefix_taskset,
        protected_oracle=protected_oracle,
        prefix_initial_state=prefix_initial_state,
    ) if prefix_initial_state is not None and protected_oracle is not None else {
        "status": "UNRESOLVED",
        "code": "PREFIX_EXECUTION_TOTALITY_UNRESOLVED",
    }

    payload = {
        "schema_version": "protected-prefix-weak-forward-simulation-v1",
        "quantification": "forall full execution exists one prefix execution forall natural-number closed boundaries",
        "quantifier_order": "forall-full-exists-one-prefix-forall-boundaries",
        "relation": "protected observable equality at close boundaries",
        "domain_witness": domain_witness,
        "runtime_schema_certificate_hash": build_runtime_schema_certificate()["certificate_hash"],
        "macro_step_receipt": macro,
        "execution_existence": execution_existence,
        "full_taskset_fingerprint": full_fp,
        "prefix_taskset_fingerprint": prefix_fp,
        "dependencies": {
            "full_input_projection_theorem": domain_witness,
            "prefix_execution_existence_theorem": execution_existence,
            "base_case": base_case_receipt or {"status": "UNRESOLVED", "code": "BASE_CASE_NOT_GENERATED"},
            "macro_step_induction": {"status": macro["status"], "macro_step": macro},
            "complete_execution_witness": execution_existence,
        "proof_kernel": proof_kernel_receipt or {"status": "UNRESOLVED", "code": "WEAK_SIMULATION_SOURCE_BOUND_RECEIPTS_REQUIRED"},
        },
    }

    dependency_statuses = {
        "domain": "PASS" if domain_witness.get("status") in (True, "PASS") else str(domain_witness.get("status", "UNRESOLVED")),
        "execution_existence": str(execution_existence.get("status", "UNRESOLVED")),
        "base_case": str((base_case_receipt or {}).get("status", "UNRESOLVED")),
        "macro_step": str(macro.get("status", "UNRESOLVED")),
        "complete_execution_witness": str(execution_existence.get("status", "UNRESOLVED")),
        "proof_kernel": (
            "PASS" if isinstance(proof_kernel_receipt, dict)
            and proof_kernel_receipt.get("status") == "PASS"
            and proof_kernel_receipt.get("theorem_id") == "PROTECTED_PREFIX_WEAK_FORWARD_SIMULATION"
            else "UNRESOLVED"
        ),
    }
    if all(value == "PASS" for value in dependency_statuses.values()):
        status = "PASS"
    elif any(value == "FAIL" for value in dependency_statuses.values()):
        status = "FAIL"
    else:
        status = "UNRESOLVED"

    payload["dependency_statuses"] = dependency_statuses
    return {
        **payload,
        "status": status,
        "certificate_hash": sha256_object(payload),
    }
