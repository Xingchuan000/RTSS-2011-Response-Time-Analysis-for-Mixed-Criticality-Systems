"""Idle-jump protected-observable stuttering obligations.

Finite executions are useful counterexample/diagnostic inputs, but they cannot
prove the universal local theorem required by PP5.  A PASS theorem receipt must
come from a source-bound transition proof kernel and quantify over every legal
idle jump.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from formal_toolchain.core.hashing import sha256_object
from .time_indexed_close import close_at, verify_time_indexed_domain


def diagnose_idle_jump_stutter_on_finite_execution(
    *,
    execution: Sequence[Any] | None,
    observable_projector: Any = None,
) -> dict[str, Any]:
    if execution is None:
        return {"status": "UNRESOLVED", "code": "IDLE_EXPANSION_EXECUTION_MISSING"}
    domain = verify_time_indexed_domain(execution, observable_projector=observable_projector)
    if domain.get("status") != "PASS":
        return domain

    inserted: list[int] = []
    for left, right in zip(execution, execution[1:]):
        left_time = int(left.time)
        right_time = int(right.time)
        if right_time <= left_time + 1:
            continue
        baseline = close_at(
            execution, left_time, observable_projector=observable_projector
        ).protected_observable
        for t in range(left_time + 1, right_time):
            obs = close_at(execution, t, observable_projector=observable_projector)
            if obs.protected_observable != baseline:
                return {
                    "status": "FAIL",
                    "code": "FINITE_IDLE_EXPANSION_PROTECTED_OBSERVABLE_CHANGED",
                    "parameterized": False,
                }
            inserted.append(t)

    payload = {
        "status": "PASS",
        "scope": "FINITE_EXECUTION_DIAGNOSTIC",
        "parameterized": False,
        "finite_horizon_only": True,
        "inserted_idle_ticks": inserted,
        "finite_sample_stutters": True,
        "execution_time_fingerprint": domain["execution_time_fingerprint"],
    }
    payload["receipt_hash"] = sha256_object(payload)
    return payload


def prove_idle_jump_stutter_expansion(
    *,
    execution: Sequence[Any] | None = None,
    observable_projector: Any = None,
    proof_kernel_receipt: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Discharge the universal local idle-jump theorem.

    Required theorem shape:
      for every legal closed idle state ``s`` and executable jump ``s -> s'``
      from time ``t`` to ``u``, every ``v`` with ``t < v < u`` has a defined
      ``CloseAt`` observation whose protected projection equals that of ``s``.
    """
    diagnostic = diagnose_idle_jump_stutter_on_finite_execution(
        execution=execution,
        observable_projector=observable_projector,
    ) if execution is not None else None

    kernel_ok = (
        isinstance(proof_kernel_receipt, Mapping)
        and proof_kernel_receipt.get("status") == "PASS"
        and proof_kernel_receipt.get("theorem_id")
            == "PROTECTED_PREFIX_IDLE_JUMP_STUTTER_EXPANSION"
        and proof_kernel_receipt.get("proof_scope")
            == "ALL_LEGAL_CLOSED_IDLE_JUMPS"
        and proof_kernel_receipt.get("source_bound_transition_relation") is True
        and proof_kernel_receipt.get("close_at_defined_for_every_intermediate_integer") is True
        and proof_kernel_receipt.get("protected_observable_frame_proved") is True
        and proof_kernel_receipt.get("independent_of_complete_execution_witness") is True
    )

    payload = {
        "theorem_id": "PROTECTED_PREFIX_IDLE_JUMP_STUTTER_EXPANSION",
        "status": "PASS" if kernel_ok else "UNRESOLVED",
        "code": None if kernel_ok else "IDLE_JUMP_PARAMETERIZED_PROOF_KERNEL_MISSING",
        "proof_scope": "ALL_LEGAL_CLOSED_IDLE_JUMPS" if kernel_ok else None,
        "parameterized": kernel_ok,
        "source_bound_transition_relation": kernel_ok,
        "all_integer_times_observable": kernel_ok,
        "time_indexed_closed_observation_defined": kernel_ok,
        "protected_observable_stutters_on_expanded_idle_ticks": kernel_ok,
        "independent_of_complete_execution_witness": kernel_ok,
        "finite_diagnostic": diagnostic,
    }
    payload["receipt_hash"] = sha256_object(payload)
    return payload
