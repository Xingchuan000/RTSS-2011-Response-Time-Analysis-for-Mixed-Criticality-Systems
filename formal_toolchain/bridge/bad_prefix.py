"""Phase K 参数化 HI bad-prefix reflection certificate builder。"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from formal_toolchain.core.artifact import obligation_certificate, verify_obligation_certificate
from formal_toolchain.core.predecessor_contract import validate_verified_predecessor
from formal_toolchain.core.hashing import sha256_object
from formal_toolchain.bridge.state_relation import (
    N6_REQUIRED_QUANTITIES,
    parameterized_state_relation_schema_hash,
    validate_n6_relation_interface,
)


@dataclass(frozen=True, slots=True)
class FirstHIMissWitness:
    job_key: tuple[str, int]
    miss_time: int
    release_time: int
    absolute_deadline: int
    executed_service: int
    concrete_prefix_hash: str
    reference_prefix_hash: str
    priority_index: int


@dataclass(frozen=True, slots=True)
class FirstHIMissSet:
    miss_time: int
    jobs: tuple[dict[str, Any], ...]


def select_first_hi_miss(snapshot, priority_map: dict[str, int] | None = None) -> FirstHIMissWitness | None:
    miss_ledger = list(snapshot.miss_ledger) if hasattr(snapshot, "miss_ledger") else []
    hi_misses = []
    for m in miss_ledger:
        raw = getattr(m, "criticality", None)
        if raw is None:
            raise ValueError(f"MISS_CRITICALITY_MISSING:{getattr(m, 'job_key', m)}")
        if str(raw) == "HI":
            hi_misses.append(m)
    if not hi_misses:
        return None
    pmap = priority_map or {}
    earliest = min(
        hi_misses,
        key=lambda m: (
            m.miss_time,
            pmap.get((m.job_key[0] if hasattr(m, "job_key") else ""), 0),
            m.job_key[0] if hasattr(m, "job_key") else "",
            m.job_key[1] if hasattr(m, "job_key") else 0,
        ),
    )
    return FirstHIMissWitness(
        job_key=earliest.job_key if hasattr(earliest, "job_key") else ("", 0),
        miss_time=earliest.miss_time if hasattr(earliest, "miss_time") else 0,
        release_time=getattr(earliest, "release_time", 0),
        absolute_deadline=earliest.absolute_deadline if hasattr(earliest, "absolute_deadline") else 0,
        executed_service=getattr(earliest, "executed_at_miss", 0),
        concrete_prefix_hash="",
        reference_prefix_hash="",
        priority_index=pmap.get(
            (earliest.job_key[0] if hasattr(earliest, "job_key") else ""), 0
        ) if hasattr(earliest, "job_key") else 0,
    )


def select_first_hi_miss_set(snapshot, priority_map: dict[str, int] | None = None) -> FirstHIMissSet | None:
    """返回最早 miss_time 上的全部 HI miss jobs。"""
    miss_ledger = list(snapshot.miss_ledger) if hasattr(snapshot, "miss_ledger") else []
    if not miss_ledger:
        raise ValueError("BAD_PREFIX_WITNESS_REQUIRES_NONEMPTY_MISS_LEDGER")
    hi = []
    for m in miss_ledger:
        raw = getattr(m, "criticality", None)
        if raw is None:
            raise ValueError(f"MISS_CRITICALITY_MISSING:{getattr(m, 'job_key', m)}")
        if str(raw) == "HI":
            hi.append(m)
    if not hi:
        return None
    t = min(m.miss_time for m in hi)
    pmap = priority_map or {}
    jobs = tuple(sorted(
        (
            {
                "job_key": m.job_key,
                "miss_time": m.miss_time,
                "priority_index": pmap.get((m.job_key[0] if hasattr(m, "job_key") else ""), 0),
                "absolute_deadline": m.absolute_deadline if hasattr(m, "absolute_deadline") else 0,
                "executed_at_miss": m.executed_at_miss if hasattr(m, "executed_at_miss") else 0,
            }
            for m in hi if m.miss_time == t
        ),
        key=lambda x: (x["priority_index"], x["job_key"]),
    ))
    return FirstHIMissSet(t, jobs)


def _theory(theorem_id: str) -> dict[str, str]:
    return json.loads((Path(__file__).resolve().parents[1] / "theory" / "hashes.json").read_text(encoding="utf-8"))["statements"][theorem_id]


def _validate_n6_theorem_receipt(
    theorem_statement: Mapping[str, Any],
    receipt: Mapping[str, Any],
) -> None:
    proof_object = theorem_statement.get("proof_object", {})
    expected_body = {
        "backend_id": "finite-hi-bad-prefix-z3-v1",
        "proof_object_hash": proof_object.get("sha256"),
        "theorem_statement_hash": theorem_statement.get("statement_hash"),
        "theorem_assumption_hash": theorem_statement.get("assumption_hash"),
        "source_bindings": receipt.get("source_bindings"),
        "relation_interface": "n6_closed_prefix_relation_interface_v2",
        "parameterized_relation_schema_hash": parameterized_state_relation_schema_hash(),
        "required_quantities": list(N6_REQUIRED_QUANTITIES),
        "solver_obligations": receipt.get("solver_obligations"),
        "z3_version": receipt.get("z3_version"),
    }
    if receipt.get("status") != "PASS":
        raise ValueError("N6_THEOREM_RECEIPT_INVALID")
    if any(value is None for value in expected_body.values()):
        raise ValueError("N6_THEOREM_RECEIPT_INCOMPLETE")
    for key, value in expected_body.items():
        if receipt.get(key) != value:
            raise ValueError(f"N6_THEOREM_RECEIPT_BINDING_MISMATCH:{key}")
    if receipt.get("receipt_hash") != sha256_object(expected_body):
        raise ValueError("N6_THEOREM_RECEIPT_HASH_INVALID")


def _closed_prefix_relation_interface(
    certificate: Mapping[str, Any],
) -> Mapping[str, Any]:
    witness = certificate.get(
        "witness",
        {},
    )

    if not isinstance(witness, Mapping):
        raise ValueError(
            "N6_CLOSED_PREFIX_WITNESS_MISSING"
        )

    if (
        witness.get(
            "pointwise_closed_prefix_relation"
        )
        is not True
    ):
        raise ValueError(
            "N6_POINTWISE_PREFIX_RELATION_MISSING"
        )

    interface = witness.get(
        "n6_relation_interface"
    )

    if not isinstance(interface, Mapping):
        raise ValueError(
            "N6_RELATION_INTERFACE_MISSING"
        )

    validate_n6_relation_interface(
        interface
    )

    return interface


def build_hi_bad_prefix_reflection_certificate(
    *, verified_predecessors: Mapping[str, Mapping[str, Any]],
    contexts: Mapping[str, Mapping[str, Any]], context_hash: str,
    theorem_statement: Mapping[str, Any], theorem_proof_receipt: Mapping[str, Any],
    concrete_snapshot: Any = None, reference_snapshot: Any = None,
) -> dict[str, Any]:
    expected_ids = {
        "CLOSED_PREFIX_REFINEMENT", "REFERENCE_PREFIX_EXTENSION",
        "RELEASE_FIXED_REMOVAL_MAPPING", "DEADLINE_OBSERVATION", "HI_NONTRUNCATION",
        "EFFECTIVE_EVENT_FRONTIER_RELATION", "EARLY_STOP_CONFIGURATION_GATE",
    }
    if set(verified_predecessors) != expected_ids:
        raise ValueError("N6_PREDECESSOR_SET_MISMATCH")
    for obligation_id in sorted(expected_ids):
        validate_verified_predecessor(
            predecessors=verified_predecessors, obligation_id=obligation_id, contexts=contexts,
        )
    closed_witness = (
        verified_predecessors[
            "CLOSED_PREFIX_REFINEMENT"
        ].get("witness", {})
    )

    if (
        closed_witness.get(
            "reference_transition_system_id"
        )
        != "FIXED_EXECUTABLE_REFERENCE_P0_V3"
    ):
        raise ValueError(
            "N6_REFERENCE_TRANSITION_SYSTEM_ID_MISMATCH"
        )

    relation_interface = (
        _closed_prefix_relation_interface(
            verified_predecessors[
                "CLOSED_PREFIX_REFINEMENT"
            ]
        )
    )

    prefix_extension_certificate = verified_predecessors[
        "REFERENCE_PREFIX_EXTENSION"
    ]
    prefix_extension_inputs = prefix_extension_certificate.get(
        "inputs", {}
    )
    reference_taskset_fingerprint = (
        prefix_extension_inputs.get(
            "reference_taskset_fingerprint"
        )
        if isinstance(prefix_extension_inputs, Mapping)
        else None
    )
    if (
        not isinstance(reference_taskset_fingerprint, str)
        or not reference_taskset_fingerprint
    ):
        raise ValueError(
            "N6_REFERENCE_TASKSET_FINGERPRINT_MISSING"
        )

    deadline_witness = (
        verified_predecessors[
            "DEADLINE_OBSERVATION"
        ].get("witness", {})
    )

    if (
        deadline_witness.get(
            "deadline_is_observation_only"
        )
        is not True
        or deadline_witness.get(
            "completion_precedes_equal_deadline"
        )
        is not True
    ):
        raise ValueError(
            "N6_DEADLINE_OBSERVATION_INTERFACE_INVALID"
        )

    nontruncation_witness = (
        verified_predecessors[
            "HI_NONTRUNCATION"
        ].get("witness", {})
    )

    contract = nontruncation_witness.get(
        "contract",
        {},
    )

    if (
        not isinstance(contract, Mapping)
        or contract.get("hi_nontruncation")
        is not True
    ):
        raise ValueError(
            "N6_HI_NONTRUNCATION_INTERFACE_INVALID"
        )
    if theorem_statement.get("theorem_id") != "FINITE_HI_BAD_PREFIX_REFLECTION":
        raise ValueError("N6_THEOREM_STATEMENT_REQUIRED")
    proof_object = theorem_statement.get("proof_object", {})
    if proof_object.get("backend") != "finite-hi-bad-prefix-z3-v1":
        raise ValueError("N6_THEOREM_BACKEND_INVALID")
    _validate_n6_theorem_receipt(
        theorem_statement,
        theorem_proof_receipt,
    )
    first_miss = None
    if concrete_snapshot is not None:
        first_miss = select_first_hi_miss_set(
            concrete_snapshot, getattr(concrete_snapshot, "priority_map", {}))
    witness = {
        "schema_version": "finite_hi_bad_prefix_reflection_v2",
        "quantification": "FOR_ALL_FIRST_FINITE_CONCRETE_HI_BAD_CLOSED_PREFIXES",
        "theorem_id": theorem_statement["theorem_id"],
        "theorem_statement_hash": theorem_statement["statement_hash"],
        "theorem_assumption_hash": theorem_statement["assumption_hash"],
        "theorem_proof_object_hash": proof_object["sha256"],
        "backend_receipt_hash": theorem_proof_receipt["receipt_hash"],
        "preserved_quantities": [
            "job_key", "release_time", "absolute_deadline", "executed_service",
            "removal_demand", "miss_time",
        ],
        "logical_steps": [
            "select_finite_first_concrete_hi_miss_set", "apply_closed_prefix_refinement",
            "specialize_state_relation", "derive_reference_service_deficit",
            "apply_reference_deadline_observation", "preserve_first_miss_time_and_set",
        ],
        "trace_diagnostic": {"status": "NOT_RUN", "reason": "UNIVERSAL_THEOREM_INSTANCE"},
        "first_miss_set": first_miss,
        "closed_prefix_relation_interface":
            dict(relation_interface),
        "reference_transition_system_id":
            "FIXED_EXECUTABLE_REFERENCE_P0_V3",
        "reference_taskset_fingerprint":
            reference_taskset_fingerprint,
        "proof_decomposition": {
            "pointwise_relation_source":
                "CLOSED_PREFIX_REFINEMENT",

            "demand_source":
                "RELEASE_FIXED_REMOVAL_MAPPING"
                "+HI_NONTRUNCATION",

            "miss_observation_source":
                "DEADLINE_OBSERVATION",

            "firstness_argument":
                "POINTWISE_NONMISS_REFLECTION_OVER_ALL_EARLIER_CLOSED_PREFIXES",
        },
    }
    return obligation_certificate(
        obligation_id="HI_BAD_CLOSED_PREFIX_REFLECTION", status="PASS", context_hash=context_hash,
        inputs={"theorem_id": theorem_statement["theorem_id"],
                "theorem_statement_hash": theorem_statement["statement_hash"],
                "theorem_assumption_hash": theorem_statement["assumption_hash"]},
        witness=witness,
        direct_predecessor_hashes={key: value["artifact_hash"] for key, value in verified_predecessors.items()},
        checker_id=__name__, checker_version="finite-hi-bad-prefix-reflection-v2")
