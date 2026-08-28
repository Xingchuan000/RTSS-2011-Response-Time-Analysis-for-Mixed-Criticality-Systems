from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from formal_toolchain.core.hashing import sha256_file_by_mode, sha256_object, sha256_text_file_normalized


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

    "formal_toolchain/reference/p0_transition_contract.py":
        (
            Path(__file__).resolve().parents[2]
            / "reference"
            / "p0_transition_contract.py"
        ),

    "formal_toolchain/reference/p0_projection.py":
        (
            Path(__file__).resolve().parents[2]
            / "reference"
            / "p0_projection.py"
        ),
}


def current_prefix_extension_source_bindings(
) -> dict[str, str]:
    return {
        relative_path: sha256_text_file_normalized(path)
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




def _prove_smt2_unsat(*, obligation_id: str, smt2: str) -> dict[str, Any]:
    """Prove one closed SMT-LIB2 obligation using z3py or system libz3.

    The runtime-refinement theorem layer carries a narrow ctypes fallback for
    environments where the shared Z3 library is installed but the Python
    package is not.  Reusing the same universal solver path keeps proof-object
    regeneration independent of a particular Python environment; it does not
    replace SMT solving with finite testing.
    """
    from formal_toolchain.theory.smt_solver import solve_closed_smt2

    result, error = solve_closed_smt2(smt2)
    if result != "UNSAT":
        return {
            "status": "FAIL" if result in {"SAT", "UNKNOWN"} else "UNRESOLVED",
            "code": f"Z3_OBLIGATION_NOT_PROVED:{obligation_id}",
            "solver_result": result,
            "detail": error,
        }
    return {
        "status": "PASS",
        "receipt": {
            "result": "UNSAT",
            "smt2_hash": sha256_object({
                "obligation_id": obligation_id,
                "smt2": smt2,
            }),
        },
        "z3_version": "libz3-or-z3py",
    }


def _fallback_case_partition() -> dict[str, Any]:
    declarations = """(set-logic QF_LIA)
(declare-const same_time_count Int)
(declare-const active_count Int)
(declare-const future_count Int)
(declare-const task_count Int)
(assert (>= same_time_count 0))
(assert (>= active_count 0))
(assert (>= future_count 0))
(assert (> task_count 0))
(assert (=> (and (= same_time_count 0) (= active_count 0)) (> future_count 0)))
"""
    cases = "(or (> same_time_count 0) (and (= same_time_count 0) (> active_count 0)) (and (= same_time_count 0) (= active_count 0) (> future_count 0)))"
    pair_overlap = "(or (and (> same_time_count 0) (and (= same_time_count 0) (> active_count 0))) (and (> same_time_count 0) (and (= same_time_count 0) (= active_count 0) (> future_count 0))) (and (and (= same_time_count 0) (> active_count 0)) (and (= same_time_count 0) (= active_count 0) (> future_count 0))))"
    queries = {
        "CLOSED_STATE_CASE_PARTITION_EXHAUSTIVE": declarations + f"(assert (not {cases}))\n(check-sat)\n",
        "CLOSED_STATE_CASE_PARTITION_EXCLUSIVE": declarations + f"(assert {pair_overlap})\n(check-sat)\n",
    }
    receipts = {}
    for obligation_id, smt2 in queries.items():
        result = _prove_smt2_unsat(obligation_id=obligation_id, smt2=smt2)
        if result["status"] != "PASS":
            return result
        receipts[obligation_id] = result["receipt"]
    return {"status": "PASS", "obligations": receipts, "z3_version": "libz3-or-z3py"}


def _fallback_closure_rank_decrease() -> dict[str, Any]:
    rank_count = 7
    lines = ["(set-logic QF_LIA)", "(declare-const selected_rank Int)"]
    for index in range(rank_count):
        lines.extend([
            f"(declare-const before_{index} Int)",
            f"(declare-const after_{index} Int)",
            f"(assert (>= before_{index} 0))",
            f"(assert (>= after_{index} 0))",
        ])
    lines.extend(["(assert (>= selected_rank 0))", f"(assert (< selected_rank {rank_count}))"])
    branches = []
    lex = []
    for rank in range(rank_count):
        prefix_eq = " ".join(f"(= after_{i} before_{i})" for i in range(rank))
        prefix = f" {prefix_eq}" if prefix_eq else ""
        branches.append(
            f"(and (= selected_rank {rank}) (>= before_{rank} 1){prefix} (= after_{rank} (- before_{rank} 1)))"
        )
        lex.append(f"(and{prefix} (< after_{rank} before_{rank}))")
    lines.append(f"(assert (or {' '.join(branches)}))")
    lines.append(f"(assert (not (or {' '.join(lex)})))")
    lines.append("(check-sat)")
    smt2 = "\n".join(lines) + "\n"
    obligation_id = "SAME_TIMESTAMP_CLOSURE_LEXICOGRAPHIC_DECREASE"
    result = _prove_smt2_unsat(obligation_id=obligation_id, smt2=smt2)
    if result["status"] != "PASS":
        return result
    return {"status": "PASS", "obligations": {obligation_id: result["receipt"]}, "z3_version": "libz3-or-z3py"}


def _fallback_periodic_arithmetic() -> dict[str, Any]:
    common = """(set-logic QF_NIA)
(declare-const time Int)
(declare-const offset Int)
(declare-const period Int)
(define-fun k () Int (ite (< time offset) 0 (+ (div (- time offset) period) 1)))
(define-fun next_release () Int (+ offset (* k period)))
(assert (>= time 0))
(assert (> period 0))
(assert (>= offset 0))
(assert (< offset period))
"""
    negated = {
        "LEAST_FUTURE_RELEASE_STRICT": "(assert (not (> next_release time)))",
        "LEAST_FUTURE_RELEASE_CONGRUENT": "(assert (not (= (mod (- next_release offset) period) 0)))",
        "LEAST_FUTURE_RELEASE_INDEX_NONNEGATIVE": "(assert (not (>= k 0)))",
    }
    receipts = {}
    for obligation_id, assertion in negated.items():
        smt2 = common + assertion + "\n(check-sat)\n"
        result = _prove_smt2_unsat(obligation_id=obligation_id, smt2=smt2)
        if result["status"] != "PASS":
            return result
        receipts[obligation_id] = result["receipt"]
    return {"status": "PASS", "obligations": receipts, "z3_version": "libz3-or-z3py"}


def _verify_case_partition() -> dict[str, Any]:
    try:
        import z3
    except ImportError:
        return _fallback_case_partition()

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
        return _fallback_closure_rank_decrease()

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


def _canonical_reference_prefix_extension_smt2(obligation_id: str) -> str:
    """Return a solver/frontend-independent SMT-LIB2 obligation.

    ``Solver.to_smt2()`` is not a stable receipt format: its whitespace,
    auxiliary declarations and arithmetic pretty-printing can change across
    z3py/libz3 versions.  The theorem receipt therefore commits to these
    canonical closed formulas and every environment solves exactly the same
    text.
    """
    if obligation_id in {
        "CLOSED_STATE_CASE_PARTITION_EXHAUSTIVE",
        "CLOSED_STATE_CASE_PARTITION_EXCLUSIVE",
    }:
        declarations = """(set-logic QF_LIA)
(declare-const same_time_count Int)
(declare-const active_count Int)
(declare-const future_count Int)
(declare-const task_count Int)
(assert (>= same_time_count 0))
(assert (>= active_count 0))
(assert (>= future_count 0))
(assert (> task_count 0))
(assert (=> (and (= same_time_count 0) (= active_count 0)) (> future_count 0)))
"""
        cases = "(or (> same_time_count 0) (and (= same_time_count 0) (> active_count 0)) (and (= same_time_count 0) (= active_count 0) (> future_count 0)))"
        overlap = "(or (and (> same_time_count 0) (and (= same_time_count 0) (> active_count 0))) (and (> same_time_count 0) (and (= same_time_count 0) (= active_count 0) (> future_count 0))) (and (and (= same_time_count 0) (> active_count 0)) (and (= same_time_count 0) (= active_count 0) (> future_count 0))))"
        assertion = (
            f"(assert (not {cases}))"
            if obligation_id == "CLOSED_STATE_CASE_PARTITION_EXHAUSTIVE"
            else f"(assert {overlap})"
        )
        return declarations + assertion + "\n(check-sat)\n"

    if obligation_id == "SAME_TIMESTAMP_CLOSURE_LEXICOGRAPHIC_DECREASE":
        rank_count = 7
        lines = ["(set-logic QF_LIA)", "(declare-const selected_rank Int)"]
        for index in range(rank_count):
            lines.extend([
                f"(declare-const before_{index} Int)",
                f"(declare-const after_{index} Int)",
                f"(assert (>= before_{index} 0))",
                f"(assert (>= after_{index} 0))",
            ])
        lines.extend(["(assert (>= selected_rank 0))", f"(assert (< selected_rank {rank_count}))"])
        branches: list[str] = []
        lex: list[str] = []
        for rank in range(rank_count):
            prefix_eq = " ".join(f"(= after_{i} before_{i})" for i in range(rank))
            prefix = f" {prefix_eq}" if prefix_eq else ""
            branches.append(
                f"(and (= selected_rank {rank}) (>= before_{rank} 1){prefix} (= after_{rank} (- before_{rank} 1)))"
            )
            lex.append(f"(and{prefix} (< after_{rank} before_{rank}))")
        lines.append(f"(assert (or {' '.join(branches)}))")
        lines.append(f"(assert (not (or {' '.join(lex)})))")
        lines.append("(check-sat)")
        return "\n".join(lines) + "\n"

    if obligation_id in {
        "LEAST_FUTURE_RELEASE_STRICT",
        "LEAST_FUTURE_RELEASE_CONGRUENT",
        "LEAST_FUTURE_RELEASE_INDEX_NONNEGATIVE",
    }:
        common = """(set-logic QF_NIA)
(declare-const time Int)
(declare-const offset Int)
(declare-const period Int)
(define-fun k () Int (ite (< time offset) 0 (+ (div (- time offset) period) 1)))
(define-fun next_release () Int (+ offset (* k period)))
(assert (>= time 0))
(assert (> period 0))
(assert (>= offset 0))
(assert (< offset period))
"""
        negated = {
            "LEAST_FUTURE_RELEASE_STRICT": "(assert (not (> next_release time)))",
            "LEAST_FUTURE_RELEASE_CONGRUENT": "(assert (not (= (mod (- next_release offset) period) 0)))",
            "LEAST_FUTURE_RELEASE_INDEX_NONNEGATIVE": "(assert (not (>= k 0)))",
        }
        return common + negated[obligation_id] + "\n(check-sat)\n"

    raise KeyError(f"UNKNOWN_REFERENCE_PREFIX_EXTENSION_OBLIGATION:{obligation_id}")


def verify_reference_prefix_extension_math() -> dict[str, Any]:
    """Freshly solve the six canonical obligations.

    Both z3py and the ctypes/system-libz3 fallback consume the same canonical
    SMT-LIB2 text, so the stored receipt is independent of the local Z3
    frontend and version.
    """
    obligations: dict[str, dict[str, str]] = {}
    for obligation_id in EXPECTED_SOLVER_OBLIGATIONS:
        smt2 = _canonical_reference_prefix_extension_smt2(obligation_id)
        result = _prove_smt2_unsat(obligation_id=obligation_id, smt2=smt2)
        if result.get("status") != "PASS":
            return result
        obligations[obligation_id] = result["receipt"]
    return {
        "status": "PASS",
        "obligations": obligations,
        "z3_version": "libz3-or-z3py",
    }


def _verify_periodic_arithmetic() -> dict[str, Any]:
    try:
        import z3
    except ImportError:
        return _fallback_periodic_arithmetic()

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
        if proof.get("source_binding_hash_mode") != "canonical_text_v1":
            return {
                "status": "FAIL",
                "code": "SOURCE_BINDING_HASH_MODE_INVALID",
            }
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
        declared_proof = theorem.get("proof_object", {})
        try:
            proof_hash = sha256_file_by_mode(
                proof_path, declared_proof.get("hash_mode", "raw_bytes_v1")
            )
        except ValueError as exc:
            return {"status": "FAIL", "code": "PROOF_OBJECT_HASH_MODE_INVALID",
                    "detail": str(exc)}
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
