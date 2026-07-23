from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from formal_toolchain.core.hashing import sha256_file, sha256_object


PREFIX_EXTENSION_SOURCE_FILES = {
    "formal_toolchain/reference/reference_state.py":
        (
            Path(__file__).resolve().parents[2]
            / "reference"
            / "reference_state.py"
        ),

    "formal_toolchain/reference/executable_semantics.py":
        (
            Path(__file__).resolve().parents[2]
            / "reference"
            / "executable_semantics.py"
        ),

    "formal_toolchain/reference/c_amc_sem_semantics.py":
        (
            Path(__file__).resolve().parents[2]
            / "reference"
            / "c_amc_sem_semantics.py"
        ),

    "formal_toolchain/bridge/logical_events.py":
        (
            Path(__file__).resolve().parents[2]
            / "bridge"
            / "logical_events.py"
        ),
}


def current_prefix_extension_source_bindings(
) -> dict[str, str]:
    return {
        relative_path: sha256_file(path)
        for relative_path, path
        in PREFIX_EXTENSION_SOURCE_FILES.items()
    }


EXPECTED_CASE_IDS = (
    "SAME_TIMESTAMP_CLOSURE",
    "READY_SERVICE_OR_EARLIER_BOUNDARY",
    "IDLE_JUMP_TO_MINIMUM_FUTURE_EVENT",
)
EXPECTED_SOLVER_OBLIGATIONS = (
    "CLOSED_STATE_CASE_PARTITION_EXHAUSTIVE",
    "CLOSED_STATE_CASE_PARTITION_EXCLUSIVE",
    "SAME_TIMESTAMP_CLOSURE_LEXICOGRAPHIC_DECREASE",
    "LEAST_FUTURE_RELEASE_STRICT",
    "LEAST_FUTURE_RELEASE_CONGRUENT",
    "LEAST_FUTURE_RELEASE_INDEX_NONNEGATIVE",
)


def _prove_unsat(
    *,
    z3: Any,
    obligation_id: str,
    proposition: Any,
    context: Any,
) -> dict[str, Any]:
    solver = z3.Solver(ctx=context)
    solver.add(z3.Not(proposition))
    result = solver.check()
    if result != z3.unsat:
        return {
            "status": "FAIL",
            "code": f"Z3_OBLIGATION_NOT_PROVED:{obligation_id}",
            "solver_result": str(result),
            "model": str(solver.model()) if result == z3.sat else None,
        }
    return {
        "status": "PASS",
        "receipt": {
            "result": "UNSAT",
            "smt2_hash": sha256_object({
                "obligation_id": obligation_id,
                "smt2": solver.to_smt2(),
            }),
        },
    }


def _verify_case_partition() -> dict[str, Any]:
    try:
        import z3
    except ImportError:
        return {"status": "FAIL", "code": "Z3_NOT_AVAILABLE"}

    context = z3.Context()
    same_time_count = z3.Int("same_time_count", ctx=context)
    active_count = z3.Int("active_count", ctx=context)
    future_count = z3.Int("future_count", ctx=context)
    task_count = z3.Int("task_count", ctx=context)
    domain = z3.And(
        same_time_count >= 0,
        active_count >= 0,
        future_count >= 0,
        task_count > 0,
        z3.Implies(
            z3.And(same_time_count == 0, active_count == 0),
            future_count > 0,
        ),
    )
    cases = (
        same_time_count > 0,
        z3.And(same_time_count == 0, active_count > 0),
        z3.And(same_time_count == 0, active_count == 0, future_count > 0),
    )
    exhaustive = z3.Implies(domain, z3.Or(*cases))
    exclusive_terms = [
        z3.Not(z3.And(left, right))
        for index, left in enumerate(cases)
        for right in cases[index + 1:]
    ]
    exclusive = z3.Implies(domain, z3.And(*exclusive_terms))
    receipts = {}
    for obligation_id, proposition in (
        ("CLOSED_STATE_CASE_PARTITION_EXHAUSTIVE", exhaustive),
        ("CLOSED_STATE_CASE_PARTITION_EXCLUSIVE", exclusive),
    ):
        result = _prove_unsat(
            z3=z3,
            obligation_id=obligation_id,
            proposition=proposition,
            context=context,
        )
        if result["status"] != "PASS":
            return result
        receipts[obligation_id] = result["receipt"]
    return {
        "status": "PASS",
        "obligations": receipts,
        "z3_version": z3.get_version_string(),
    }


def _lexicographically_less(*, z3: Any, before: list[Any], after: list[Any]) -> Any:
    alternatives = []
    for index in range(len(before)):
        alternatives.append(z3.And(
            *[after[prefix] == before[prefix] for prefix in range(index)],
            after[index] < before[index],
        ))
    return z3.Or(*alternatives)


def _verify_closure_rank_decrease() -> dict[str, Any]:
    try:
        import z3
    except ImportError:
        return {"status": "FAIL", "code": "Z3_NOT_AVAILABLE"}

    context = z3.Context()
    rank_count = 7
    before = [z3.Int(f"before_{index}", ctx=context) for index in range(rank_count)]
    after = [z3.Int(f"after_{index}", ctx=context) for index in range(rank_count)]
    selected_rank = z3.Int("selected_rank", ctx=context)
    domain = z3.And(
        selected_rank >= 0,
        selected_rank < rank_count,
        *[value >= 0 for value in before],
        *[value >= 0 for value in after],
        z3.Or(*[
            z3.And(
                selected_rank == rank,
                before[rank] >= 1,
                *[after[lower] == before[lower] for lower in range(rank)],
                after[rank] == before[rank] - 1,
            )
            for rank in range(rank_count)
        ]),
    )
    result = _prove_unsat(
        z3=z3,
        obligation_id="SAME_TIMESTAMP_CLOSURE_LEXICOGRAPHIC_DECREASE",
        proposition=z3.Implies(
            domain,
            _lexicographically_less(z3=z3, before=before, after=after),
        ),
        context=context,
    )
    if result["status"] != "PASS":
        return result
    return {
        "status": "PASS",
        "obligations": {
            "SAME_TIMESTAMP_CLOSURE_LEXICOGRAPHIC_DECREASE": result["receipt"],
        },
        "z3_version": z3.get_version_string(),
    }


def verify_reference_prefix_extension_math() -> dict[str, Any]:
    components = (
        _verify_case_partition(),
        _verify_closure_rank_decrease(),
        _verify_periodic_arithmetic(),
    )
    obligations = {}
    versions = set()
    for component in components:
        if component.get("status") != "PASS":
            return component
        obligations.update(component["obligations"])
        versions.add(component["z3_version"])
    if set(obligations) != set(EXPECTED_SOLVER_OBLIGATIONS):
        return {
            "status": "FAIL",
            "code": "REFERENCE_PREFIX_SOLVER_OBLIGATION_SET_MISMATCH",
        }
    return {
        "status": "PASS",
        "obligations": obligations,
        "z3_version": sorted(versions)[0],
    }


def _verify_periodic_arithmetic() -> dict[str, Any]:
    try:
        import z3
    except ImportError:
        return {"status": "FAIL", "code": "Z3_NOT_AVAILABLE"}

    context = z3.Context()
    time = z3.Int("time", ctx=context)
    offset = z3.Int("offset", ctx=context)
    period = z3.Int("period", ctx=context)
    k = z3.If(time < offset, z3.IntVal(0, ctx=context), (time - offset) / period + 1)
    next_release = offset + k * period
    domain = z3.And(time >= 0, period > 0, offset >= 0, offset < period)
    obligations = {
        "LEAST_FUTURE_RELEASE_STRICT": z3.Implies(domain, next_release > time),
        "LEAST_FUTURE_RELEASE_CONGRUENT": z3.Implies(domain, (next_release - offset) % period == 0),
        "LEAST_FUTURE_RELEASE_INDEX_NONNEGATIVE": z3.Implies(domain, k >= 0),
    }
    receipts = {}
    for obligation_id, proposition in obligations.items():
        result = _prove_unsat(
            z3=z3,
            obligation_id=obligation_id,
            proposition=proposition,
            context=context,
        )
        if result["status"] != "PASS":
            return result
        receipts[obligation_id] = result["receipt"]
    return {"status": "PASS", "obligations": receipts, "z3_version": z3.get_version_string()}


class ReferencePrefixExtensionBackend:
    def verify(self, proof_path: Path, *, theorem: Mapping[str, Any]) -> dict[str, Any]:
        proof_path = Path(proof_path).resolve(strict=True)
        proof = json.loads(proof_path.read_text(encoding="utf-8"))
        if proof.get("schema_version") != "reference_prefix_extension_proof_v3":
            return {"status": "FAIL", "code": "PROOF_SCHEMA_VERSION_MISMATCH"}
        if proof.get("theorem_id") != "REFERENCE_PREFIX_EXTENSION":
            return {"status": "FAIL", "code": "PROOF_THEOREM_ID_MISMATCH"}
        if proof.get("case_ids") != list(EXPECTED_CASE_IDS):
            return {"status": "FAIL", "code": "PROOF_CASE_IDS_MISMATCH"}
        source_bindings = proof.get("source_bindings")
        expected_bindings = (
            current_prefix_extension_source_bindings()
        )
        if source_bindings != expected_bindings:
            return {
                "status": "FAIL",
                "code":
                    "SOURCE_BINDINGS_MISMATCH",
                "expected": expected_bindings,
                "actual": source_bindings,
            }
        if proof.get("theorem_statement_hash") != theorem.get("statement_hash"):
            return {"status": "FAIL", "code": "THEOREM_STATEMENT_HASH_MISMATCH"}
        if proof.get("theorem_assumption_hash") != theorem.get("assumption_hash"):
            return {"status": "FAIL", "code": "THEOREM_ASSUMPTION_HASH_MISMATCH"}
        if proof.get("solver_backend") != "z3":
            return {"status": "FAIL", "code": "SOLVER_BACKEND_NOT_Z3"}
        if proof.get("runtime_checked_contracts") != [
            "validate_reference_state",
            "_append_generated_event",
            "_checked_successor",
        ]:
            return {"status": "FAIL", "code": "RUNTIME_CHECKED_CONTRACTS_MISMATCH"}
        math = verify_reference_prefix_extension_math()
        if math.get("status") != "PASS":
            return math
        if proof.get("solver_obligation_receipts") != math.get("obligations"):
            return {"status": "FAIL", "code": "SOLVER_RECEIPTS_MISMATCH", "fresh": math}
        proof_hash = sha256_file(proof_path)
        declared_proof = theorem.get("proof_object", {})
        if declared_proof.get("sha256") and declared_proof.get("sha256") != proof_hash:
            return {"status": "FAIL", "code": "PROOF_OBJECT_HASH_MISMATCH"}
        receipt_body = {
            "backend_id": "reference-prefix-extension-z3-v3",
            "proof_object_hash": proof_hash,
            "theorem_statement_hash": theorem["statement_hash"],
            "theorem_assumption_hash": theorem["assumption_hash"],
            "source_bindings": source_bindings,
            "case_ids": list(EXPECTED_CASE_IDS),
            "solver_obligations": math["obligations"],
            "z3_version": math["z3_version"],
        }
        return {"status": "PASS", **receipt_body, "receipt_hash": sha256_object(receipt_body)}
