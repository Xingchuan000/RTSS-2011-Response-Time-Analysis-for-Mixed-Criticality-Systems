from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from formal_toolchain.core.hashing import sha256_file, sha256_object
from formal_toolchain.bridge.state_relation import (
    N6_REQUIRED_QUANTITIES,
    parameterized_state_relation_schema_hash,
)

ROOT = Path(__file__).resolve().parents[2]
EXPECTED_N6_SOLVER_OBLIGATIONS = (
    "HI_SERVICE_DEFICIT_REFLECTS",
    "HI_DEADLINE_TIME_REFLECTS",
    "HI_BAD_JOB_POINTWISE_REFLECTS",
    "EARLIER_CLOSED_PREFIX_NONMISS_REFLECTS",
    "FIRST_MISS_SET_MEMBERSHIP_REFLECTS",
)
N6_SOURCE_FILES = {
    "formal_toolchain/bridge/state_relation.py": ROOT / "bridge" / "state_relation.py",
    "formal_toolchain/bridge/case_templates.py": ROOT / "bridge" / "case_templates.py",
    "formal_toolchain/bridge/transition_cases.py": ROOT / "bridge" / "transition_cases.py",
    "formal_toolchain/bridge/prefix_refinement.py": ROOT / "bridge" / "prefix_refinement.py",
    "formal_toolchain/bridge/event_projection.py": ROOT / "bridge" / "event_projection.py",
    "formal_toolchain/bridge/bad_prefix.py": ROOT / "bridge" / "bad_prefix.py",
    "formal_toolchain/adapters/formal_runtime_snapshot.py":
        ROOT / "adapters" / "formal_runtime_snapshot.py",
    "formal_toolchain/reference/executable_semantics.py":
        ROOT / "reference" / "executable_semantics.py",
    "formal_toolchain/reference/p0_transition_contract.py":
        ROOT / "reference" / "p0_transition_contract.py",
    "formal_toolchain/reference/p0_projection.py":
        ROOT / "reference" / "p0_projection.py",
    "formal_toolchain/reference/transition_identity.py":
        ROOT / "reference" / "transition_identity.py",
}


def current_n6_source_bindings() -> dict[str, str]:
    return {relative: sha256_file(path) for relative, path in N6_SOURCE_FILES.items()}


def _build_n6_obligations(
    z3: Any,
    context: Any,
) -> dict[str, Any]:
    c_time = z3.Int(
        "c_time",
        ctx=context,
    )
    r_time = z3.Int(
        "r_time",
        ctx=context,
    )

    c_present = z3.Int(
        "c_present",
        ctx=context,
    )
    r_present = z3.Int(
        "r_present",
        ctx=context,
    )

    c_key = z3.Int(
        "c_key",
        ctx=context,
    )
    r_key = z3.Int(
        "r_key",
        ctx=context,
    )
    target_key = z3.Int(
        "target_key",
        ctx=context,
    )

    c_criticality = z3.Int(
        "c_criticality",
        ctx=context,
    )
    r_criticality = z3.Int(
        "r_criticality",
        ctx=context,
    )

    c_release = z3.Int(
        "c_release",
        ctx=context,
    )
    r_release = z3.Int(
        "r_release",
        ctx=context,
    )

    c_deadline = z3.Int(
        "c_deadline",
        ctx=context,
    )
    r_deadline = z3.Int(
        "r_deadline",
        ctx=context,
    )

    c_service = z3.Int(
        "c_service",
        ctx=context,
    )
    r_service = z3.Int(
        "r_service",
        ctx=context,
    )

    c_demand = z3.Int(
        "c_demand",
        ctx=context,
    )
    r_demand = z3.Int(
        "r_demand",
        ctx=context,
    )

    c_hi_miss = z3.Int(
        "c_hi_miss",
        ctx=context,
    )
    r_hi_miss = z3.Int(
        "r_hi_miss",
        ctx=context,
    )

    domain = z3.And(
        c_present >= 0,
        c_present <= 1,
        r_present >= 0,
        r_present <= 1,

        c_criticality >= 0,
        c_criticality <= 1,
        r_criticality >= 0,
        r_criticality <= 1,

        c_hi_miss >= 0,
        c_hi_miss <= 1,
        r_hi_miss >= 0,
        r_hi_miss <= 1,

        c_time >= 0,
        r_time >= 0,

        c_deadline >= c_release,
        r_deadline >= r_release,

        c_service >= 0,
        r_service >= 0,

        c_demand > 0,
        r_demand > 0,
    )

    closed_prefix_relation = z3.And(
        c_time == r_time,
        c_present == r_present,
        c_key == r_key,
        c_criticality == r_criticality,
        c_release == r_release,
        c_deadline == r_deadline,
        c_service == r_service,
        c_demand == r_demand,
        c_hi_miss == r_hi_miss,
    )

    concrete_hi_bad = z3.And(
        c_present == 1,
        c_criticality == 1,
        c_hi_miss == 1,
        c_time == c_deadline,
        c_service < c_demand,
    )

    reference_hi_bad = z3.And(
        r_present == 1,
        r_criticality == 1,
        r_hi_miss == 1,
        r_time == r_deadline,
        r_service < r_demand,
    )

    concrete_member = z3.And(
        concrete_hi_bad,
        c_key == target_key,
    )

    reference_member = z3.And(
        reference_hi_bad,
        r_key == target_key,
    )

    return {
        "HI_SERVICE_DEFICIT_REFLECTS":
            z3.Implies(
                z3.And(
                    domain,
                    closed_prefix_relation,
                    c_service < c_demand,
                ),
                r_service < r_demand,
            ),

        "HI_DEADLINE_TIME_REFLECTS":
            z3.Implies(
                z3.And(
                    domain,
                    closed_prefix_relation,
                    c_time == c_deadline,
                ),
                r_time == r_deadline,
            ),

        "HI_BAD_JOB_POINTWISE_REFLECTS":
            z3.Implies(
                z3.And(
                    domain,
                    closed_prefix_relation,
                    concrete_hi_bad,
                ),
                reference_hi_bad,
            ),

        "EARLIER_CLOSED_PREFIX_NONMISS_REFLECTS":
            z3.Implies(
                z3.And(
                    domain,
                    closed_prefix_relation,
                    c_hi_miss == 0,
                ),
                r_hi_miss == 0,
            ),

        "FIRST_MISS_SET_MEMBERSHIP_REFLECTS":
            z3.Implies(
                z3.And(
                    domain,
                    closed_prefix_relation,
                    concrete_member,
                ),
                reference_member,
            ),
    }


def verify_finite_hi_bad_prefix_math() -> dict[str, Any]:
    import z3

    context = z3.Context()
    obligations = _build_n6_obligations(z3, context)
    receipts = {}
    for obligation_id in EXPECTED_N6_SOLVER_OBLIGATIONS:
        proposition = obligations[obligation_id]
        solver = z3.Solver(ctx=context)
        solver.add(z3.Not(proposition))
        result = solver.check()
        if result != z3.unsat:
            return {"status": "FAIL", "code": f"N6_Z3_OBLIGATION_NOT_PROVED:{obligation_id}",
                    "model": str(solver.model()) if result == z3.sat else None}
        receipts[obligation_id] = {"result": "UNSAT", "smt2_hash": sha256_object({
            "obligation_id": obligation_id, "smt2": solver.to_smt2(),
        })}
    return {"status": "PASS", "obligations": receipts, "z3_version": z3.get_version_string()}


class FiniteHIBadPrefixBackend:
    def verify(self, proof_path: Path, *, theorem: dict[str, Any]) -> dict[str, Any]:
        proof_path = Path(proof_path).resolve(strict=True)
        proof = json.loads(proof_path.read_text(encoding="utf-8"))
        if proof.get("schema_version") != "finite_hi_bad_prefix_reflection_proof_v2":
            return {"status": "FAIL", "code": "PROOF_SCHEMA_VERSION_MISMATCH"}
        if proof.get("theorem_id") != "FINITE_HI_BAD_PREFIX_REFLECTION":
            return {"status": "FAIL", "code": "PROOF_THEOREM_ID_MISMATCH"}
        if proof.get("theorem_statement_hash") != theorem.get("statement_hash"):
            return {"status": "FAIL", "code": "THEOREM_STATEMENT_HASH_MISMATCH"}
        if proof.get("theorem_assumption_hash") != theorem.get("assumption_hash"):
            return {"status": "FAIL", "code": "THEOREM_ASSUMPTION_HASH_MISMATCH"}
        if proof.get("relation_interface") != "n6_closed_prefix_relation_interface_v2":
            return {"status": "FAIL", "code": "RELATION_INTERFACE_MISMATCH"}
        if proof.get("parameterized_relation_schema_hash") != parameterized_state_relation_schema_hash():
            return {"status": "FAIL", "code": "PARAMETERIZED_RELATION_SCHEMA_MISMATCH"}
        if proof.get("required_quantities") != list(N6_REQUIRED_QUANTITIES):
            return {"status": "FAIL", "code": "N6_REQUIRED_QUANTITIES_MISMATCH"}
        if proof.get("proof_scope") != "POINTWISE_RELATION_SPECIALIZATION_OVER_FINITE_CLOSED_PREFIXES":
            return {"status": "FAIL", "code": "PROOF_SCOPE_MISMATCH"}
        if proof.get("source_bindings") != current_n6_source_bindings():
            return {"status": "FAIL", "code": "SOURCE_BINDINGS_MISMATCH"}
        math = verify_finite_hi_bad_prefix_math()
        if math.get("status") != "PASS":
            return math
        if set(proof.get("solver_obligation_receipts", {})) != set(EXPECTED_N6_SOLVER_OBLIGATIONS):
            return {"status": "FAIL", "code": "SOLVER_OBLIGATION_IDS_MISMATCH"}
        if proof.get("solver_obligation_receipts") != math["obligations"]:
            return {"status": "FAIL", "code": "SOLVER_RECEIPTS_MISMATCH", "fresh": math}
        proof_hash = sha256_file(proof_path)
        if theorem.get("proof_object", {}).get("sha256") != proof_hash:
            return {"status": "FAIL", "code": "PROOF_OBJECT_HASH_MISMATCH"}
        receipt_body = {
            "backend_id": "finite-hi-bad-prefix-z3-v1",
            "proof_object_hash": proof_hash,
            "theorem_statement_hash": theorem["statement_hash"],
            "theorem_assumption_hash": theorem["assumption_hash"],
            "source_bindings": proof["source_bindings"],
            "relation_interface": proof["relation_interface"],
            "parameterized_relation_schema_hash": proof["parameterized_relation_schema_hash"],
            "required_quantities": proof["required_quantities"],
            "solver_obligations": math["obligations"],
            "z3_version": math["z3_version"],
        }
        return {"status": "PASS", **receipt_body, "receipt_hash": sha256_object(receipt_body)}
