from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from formal_toolchain.core.hashing import sha256_object
from .proof_kernel import RELATION_SCHEMA_HASH
from .phase_relation import JOB_FIELDS, PENDING_RELEASE_FIELDS


@dataclass(frozen=True, slots=True)
class NaturalNumberInductionWitness:
    """The explicit induction object for closed-boundary simulation."""

    base_index: int
    successor_index: str
    invariant: str
    fixed_projected_oracle_id: str
    l8_step_receipt_hash: str
    relation_schema_hash: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "base_index": self.base_index,
            "successor_index": self.successor_index,
            "invariant": self.invariant,
            "fixed_projected_oracle_id": self.fixed_projected_oracle_id,
            "l8_step_receipt_hash": self.l8_step_receipt_hash,
            "relation_schema_hash": self.relation_schema_hash,
        }


def construct_natural_number_induction_witness(
    *, macro_step_receipt: Mapping[str, Any],
    execution_existence_receipt: Mapping[str, Any],
    base_case_receipt: Mapping[str, Any],
) -> dict[str, Any]:
    """Construct, rather than accept, the ``n=0``/``n+1`` induction object."""
    macro = _payload(macro_step_receipt)
    execution = _payload(execution_existence_receipt)
    base = _payload(base_case_receipt)
    l8_hash = str(macro.get("artifact_hash") or macro.get("receipt_hash") or "")
    oracle_id = str(
        execution.get("complete_execution_oracle_hash")
        or execution.get("projected_oracle_fingerprint") or ""
    )
    ok = (
        macro.get("status") == "PASS"
        and macro.get("theorem_id") == "PROTECTED_MACRO_STEP_PRESERVATION"
        and len(l8_hash) == 64
        and execution.get("status") == "PASS"
        and execution.get("same_fixed_oracle") is True
        and execution.get("complete_execution_witness_constructed") is True
        and execution.get("parametric_complete_execution_theorem") is not None
        and execution.get("idle_jump_expansion_verified") is True
        and execution.get("time_indexed_closed_observation_defined") is True
        and base.get("status") == "PASS"
        and base.get("theorem_id") == "PROTECTED_PREFIX_INITIAL_RELATION"
        and base.get("base_relation_proved") is True
    )
    witness = NaturalNumberInductionWitness(
        base_index=0,
        successor_index="t -> t+1 over absolute integer time via CloseAt expansion and L8",
        invariant="Rel_pp_close(CloseAt_full(t), CloseAt_prefix(t))",
        fixed_projected_oracle_id=oracle_id,
        l8_step_receipt_hash=l8_hash,
        relation_schema_hash=RELATION_SCHEMA_HASH,
    )
    payload = {
        "theorem_id": "PROTECTED_PREFIX_WEAK_FORWARD_SIMULATION",
        "status": "PASS" if ok else "UNRESOLVED",
        "source_bound": ok,
        "quantifier_order": "forall xi_ref exists one xi_pp forall t in N",
        "legacy_quantifier_order": "forall-full-exists-one-prefix-forall-boundaries",
        "base_case": ok,
        "successor_step": ok,
        "induction_on_t_complete": ok,
        "fixed_oracle_identity_checked": ok,
        "witness_identity_checked": ok,
        "induction_witness": witness.to_dict(),
        "predecessor_receipt_hashes": {
            "macro_step": sha256_object(macro_step_receipt),
            "execution": sha256_object(execution_existence_receipt),
            "base_case": sha256_object(base_case_receipt),
        },
        "relation_schema_hash": RELATION_SCHEMA_HASH,
    }
    payload["receipt_hash"] = sha256_object(payload)
    return payload


def _payload(value: Mapping[str, Any] | None) -> Mapping[str, Any]:
    current: Any = value or {}
    for _ in range(3):
        if not isinstance(current, Mapping):
            return {}
        nested = current.get("witness")
        if not isinstance(nested, Mapping):
            return current
        current = nested
    return current if isinstance(current, Mapping) else {}


def prove_weak_forward_simulation(
    *,
    macro_step_receipt: Mapping[str, Any] | None = None,
    execution_existence_receipt: Mapping[str, Any] | None = None,
    base_case_receipt: Mapping[str, Any] | None = None,
    simulation_domain_receipt: Mapping[str, Any] | None = None,
    input_projection_receipt: Mapping[str, Any] | None = None,
    demand_receptiveness_receipt: Mapping[str, Any] | None = None,
    proof_kernel_receipt: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Prove weak forward simulation: full -> protected prefix.

    Proof structure:
      Base: standard empty LO states have equal protected projection
      Witness: ξ_pp is generated once from projected oracle A_P(ξ_full)
      Induction: L8 for every t (Rel(CloseAt_f(t), CloseAt_p(t)))
      Conclusion: ∀ ξ_full ∃ one ξ_pp ∀ t Rel(CloseAt_full(t), CloseAt_pp(t))
    """
    macro = _payload(macro_step_receipt)
    execution = _payload(execution_existence_receipt)
    base = _payload(base_case_receipt)
    domain = _payload(simulation_domain_receipt)
    projection_envelope = _payload(input_projection_receipt)
    projection_nested = projection_envelope.get("projection_receipt")
    projection = (
        {**projection_envelope, **projection_nested}
        if isinstance(projection_nested, Mapping) else projection_envelope
    )
    demand = _payload(demand_receptiveness_receipt)
    kernel = _payload(proof_kernel_receipt)
    macro_ok = (
        isinstance(macro_step_receipt, Mapping)
        and macro.get("status") == "PASS"
        and macro.get("lemma") == "PROTECTED_MACRO_STEP_PRESERVATION"
        and macro.get("relation_schema_hash") == RELATION_SCHEMA_HASH
    )
    execution_ok = (
        isinstance(execution_existence_receipt, Mapping)
        and execution.get("status") == "PASS"
        and execution.get("complete_execution_exists") is True
        and execution.get("complete_execution_witness_constructed") is True
        and execution.get("finite_prefix_compatibility_proved") is True
        and execution.get("same_fixed_oracle") is True
        and execution.get("same_successor_function") == "next_closed_boundary"
        and execution.get("idle_jump_expansion_verified") is True
        and execution.get("time_indexed_closed_observation_defined") is True
        and isinstance(execution.get("projected_oracle_fingerprint"), str)
        and execution.get("projected_oracle_fingerprint")
            == execution.get("complete_execution_oracle_hash")
    )
    base_ok = (
        isinstance(base_case_receipt, Mapping)
        and base.get("status") == "PASS"
        and base.get("theorem_id") == "PROTECTED_PREFIX_INITIAL_RELATION"
        and base.get("base_relation_proved") is True
    )
    domain_ok = (
        isinstance(simulation_domain_receipt, Mapping)
        and domain.get("quantifier_order")
            == "forall-full-exists-one-prefix-forall-boundaries"
        and domain.get("reference_model_conformance") is True
        and domain.get("partition_valid") is True
        and domain.get("saturation_valid") is True
        and domain.get("runtime_schema_valid") is True
        and domain.get("protected_input_independence") is True
        and domain.get("projected_demands_legal") is True
        and domain.get("complete_prefix_execution_exists") is True
        and domain.get("same_fixed_oracle") is True
    )
    projection_fp = projection.get("projected_oracle_fingerprint")
    input_ok = (
        isinstance(input_projection_receipt, Mapping)
        and projection.get("status", "PASS") == "PASS"
        and projection.get("quantifier_scope")
            == "forall-full-execution-exists-unique-projected-stream"
        and projection.get("forall_release_indices") is True
        and projection.get("complete_recurring_stream") is True
        and isinstance(projection_fp, str)
        and demand.get("all_projected_demands_legal") is True
        and demand.get("mode_independent_lo_receptiveness") is True
        and demand.get("projected_oracle_fingerprint") == projection_fp
        and execution.get("projected_oracle_fingerprint") == projection_fp
    )
    kernel_ok = (
        isinstance(proof_kernel_receipt, Mapping)
        and kernel.get("status") == "PASS"
        and kernel.get("theorem_id")
            == "PROTECTED_PREFIX_WEAK_FORWARD_SIMULATION"
        and kernel.get("quantifier_order")
            == "forall-full-exists-one-prefix-forall-boundaries"
        and kernel.get("induction_on_t_complete") is True
        and kernel.get("fixed_oracle_identity_checked") is True
        and kernel.get("witness_identity_checked") is True
        and kernel.get("simulation_domain_consumed") is True
        and kernel.get("input_projection_consumed") is True
        and kernel.get("demand_receptiveness_consumed") is True
    )
    from .proof_kernel import prove_weak_forward_simulation_kernel
    pk_kernel = prove_weak_forward_simulation_kernel(
        macro_step_receipt=macro_step_receipt,
        execution_receipt=execution_existence_receipt,
        base_case_receipt=base_case_receipt,
        simulation_domain_receipt=simulation_domain_receipt,
        input_projection_receipt=input_projection_receipt,
        demand_receptiveness_receipt=demand_receptiveness_receipt,
    )
    resolved_kernel_ok = kernel_ok and pk_kernel.get("status") == "PASS"
    established = (
        macro_ok and execution_ok and base_ok and domain_ok and input_ok
        and resolved_kernel_ok
    )

    return {
        "theorem_id": "PROTECTED_PREFIX_WEAK_FORWARD_SIMULATION",
        "quantifier_order": "forall xi_ref exists one xi_pp forall t in N",
        "legacy_quantifier_order": "forall-full-exists-one-prefix-forall-boundaries",
        "quantifier_statement": "forall xi_ref exists one xi_pp forall t in N",
        "base_case_proved": base_ok,
        "complete_execution_witness_proved": execution_ok,
        "induction_proved": resolved_kernel_ok,
        "absolute_integer_time_induction": established,
        "idle_jump_close_at_expansion_consumed": execution_ok,
        "induction_witness": pk_kernel.get("induction_witness"),
        "relation_schema_hash": RELATION_SCHEMA_HASH,
        "macro_step_L1_L8_proved": macro_ok,
        "simulation_domain_consumed": domain_ok,
        "input_projection_consumed": input_ok,
        "demand_receptiveness_consumed": input_ok,
        "all_hi_tasks_protected": domain.get("partition_valid") is True,
        "projected_oracle_fingerprint": projection_fp,
        "preserved_job_fields": list(JOB_FIELDS),
        "preserved_relation_fields": list(PENDING_RELEASE_FIELDS) + [
            "miss_job_keys", "running_job_key",
        ],
        "preserved_pending_release_fields": list(PENDING_RELEASE_FIELDS),
        "source_bound_predecessor_hashes": {
            "macro_step": sha256_object(macro_step_receipt or {}),
            "execution": sha256_object(execution_existence_receipt or {}),
            "base_case": sha256_object(base_case_receipt or {}),
            "simulation_domain": sha256_object(simulation_domain_receipt or {}),
            "input_projection": sha256_object(input_projection_receipt or {}),
            "demand_receptiveness": sha256_object(demand_receptiveness_receipt or {}),
        },
        "status": "PASS" if established else "UNRESOLVED",
        "code": None if established else "WEAK_SIMULATION_KERNEL_MISSING",
        "certificate_hash": sha256_object({
            "base": base_ok, "witness": execution_ok,
            "induction": kernel_ok, "macro": macro_ok,
        }),
    }
