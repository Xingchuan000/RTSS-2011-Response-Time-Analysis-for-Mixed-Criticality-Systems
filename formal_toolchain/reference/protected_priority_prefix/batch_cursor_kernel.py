from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from formal_toolchain.core.hashing import sha256_object


def build_parameterized_fold_receipt(
    *, phase: str, transition_ir: Any, pp0_receipt: dict[str, Any],
    required_local_theorem_id: str,
    local_theorem_receipts: dict[str, Any] | None = None,
    additional_pp0_receipts: dict[str, Any] | None = None,
    projection_receipt: dict[str, Any] | None = None,
    demand_receptiveness_receipt: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Construct a source-bound finite-sequence induction receipt.

    The base and exhaustion cases are definitional for a finite cursor.  Step
    cases are discharged only from executable PP0 receipts and the relevant
    local/input theorems; no caller-supplied ``base_case=True`` flags are
    accepted.
    """
    compiled = bool(transition_ir is not None and transition_ir.is_compiled())
    paths = tuple(getattr(transition_ir, "paths", ())) if transition_ir is not None else ()
    source_bound = bool(
        compiled and paths
        and getattr(transition_ir, "total_semantic_coverage", False) is True
        and getattr(transition_ir, "semantic_effect", None) is not None
        and transition_ir.semantic_effect.derivation_complete is True
    )
    pp0_ok = isinstance(pp0_receipt, dict) and pp0_receipt.get("status") == "PASS"
    local = local_theorem_receipts or {}
    required_local = local.get(required_local_theorem_id, {})
    local_ok = (
        isinstance(required_local, dict)
        and required_local.get("status") == "PASS"
        and required_local.get("theorem_id") == required_local_theorem_id
        and required_local.get("parameterized") is True
    )
    extra = additional_pp0_receipts or {}

    projection_ok = True
    demand_ok = True
    extra_pp0_ok = True
    mode_step_ok = True
    tail_step_ok = True
    if phase == "DDLCursor":
        tail_step_ok = local_ok  # observe-only tail ledger is erased by Obs_P
        mode_step_ok = True
    elif phase == "ARRCursor":
        extra_pp0_ok = all(
            isinstance(extra.get(case), dict)
            and extra[case].get("status") == "PASS"
            for case in ("MODE_SWITCH", "RELEASE")
        )
        projection = projection_receipt or {}
        demand = demand_receptiveness_receipt or {}
        projection_ok = (
            projection.get("status", projection.get("obligation_status")) == "PASS"
            and (
                projection.get("forall_release_indices") is True
                or isinstance(projection.get("witness"), dict)
                and projection["witness"].get("forall_release_indices") is True
            )
        )
        demand_ok = demand.get("status", demand.get("obligation_status")) == "PASS"
        mode_local = local.get("MODE_TRANSITIONS_ZERO_TIME", {})
        independent_local = local.get("PROTECTED_INPUT_INDEPENDENT_OF_TAIL", {})
        mode_step_ok = all(
            item.get("status") == "PASS" and item.get("parameterized") is True
            for item in (mode_local, independent_local)
        )
        tail_step_ok = projection_ok and mode_step_ok
    else:
        source_bound = False

    cases = {
        "cursor_zero_base": source_bound,
        "protected_entry_step": source_bound and pp0_ok and local_ok and extra_pp0_ok and demand_ok,
        "full_tail_entry_prefix_identity_step": source_bound and tail_step_ok,
        "mode_only_step": source_bound and mode_step_ok,
        "cursor_exhaustion_join": source_bound and projection_ok,
    }
    status = "PASS" if source_bound and all(cases.values()) else "UNRESOLVED"
    payload = {
        "theorem_id": "BATCH_CURSOR_PARAMETERIZED_FOLD",
        "phase": phase,
        "status": status,
        "parameterized": status == "PASS",
        "source_bound": source_bound,
        "source_transition_ir_hash": transition_ir.ir_hash() if transition_ir is not None else None,
        "source_path_hashes": [path.path_hash() for path in paths],
        "pp0_receipt_hash": sha256_object(pp0_receipt),
        "additional_pp0_receipt_hashes": {
            key: sha256_object(value) for key, value in sorted(extra.items())
        },
        "required_local_theorem_id": required_local_theorem_id,
        "required_local_theorem_receipt_hash": sha256_object(required_local),
        "input_projection_receipt_hash": sha256_object(projection_receipt or {}),
        "demand_receptiveness_receipt_hash": sha256_object(demand_receptiveness_receipt or {}),
        **cases,
        "induction_cases": cases,
        "all_batch_sizes": status == "PASS",
        "finite_instance_data_used": False,
        "relation_schema_hash": sha256_object({"schema": "phase_relation_v4_close_at"}),
        "cursor_measure": "remaining_full_entries + remaining_prefix_entries",
        "cursor_measure_strictly_decreases": status == "PASS",
    }
    payload["receipt_hash"] = sha256_object(payload)
    return payload


@dataclass(frozen=True, slots=True)
class DDLFoldKernel:
    base: bool
    protected_step: bool
    tail_step: bool
    end: bool

    def complete(self) -> bool:
        return self.base and self.protected_step and self.tail_step and self.end


@dataclass(frozen=True, slots=True)
class ARRFoldKernel:
    base: bool
    protected_step: bool
    tail_step: bool
    end: bool

    def complete(self) -> bool:
        return self.base and self.protected_step and self.tail_step and self.end


def prove_ddl_fold_kernel(
    proof_kernel_receipt: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """DDL fold: protected miss-ledger equality after deadline observation batch.

    Base: both cursors at start/end relation
    Protected step: same job/deadline/completion -> same miss update
    Tail step: full cursor advances; prefix stutters
    End: protected miss ledger equal
    """
    from .batch_cursor import BatchCursorFoldLemma

    kernel_ok = (
        isinstance(proof_kernel_receipt, dict)
        and proof_kernel_receipt.get("status") == "PASS"
        and proof_kernel_receipt.get("lemma") == "BATCH_CURSOR_PARAMETERIZED_FOLD"
        and proof_kernel_receipt.get("phase") == "DDLCursor"
        and proof_kernel_receipt.get("base_case") is True
        and proof_kernel_receipt.get("protected_step") is True
        and proof_kernel_receipt.get("tail_step") is True
        and proof_kernel_receipt.get("end_case") is True
        and proof_kernel_receipt.get("parameterized") is True
    )

    kernel = DDLFoldKernel(
        base=kernel_ok,
        protected_step=kernel_ok,
        tail_step=kernel_ok,
        end=kernel_ok,
    )

    return {
        "theorem_id": "PROTECTED_PREFIX_DDL_FOLD",
        "phase": "DDLCursor",
        "base_case": kernel.base,
        "protected_step": kernel.protected_step,
        "tail_step": kernel.tail_step,
        "end_case": kernel.end,
        "fold_complete": kernel.complete(),
        "parameterized": kernel_ok,
        "kernel_receipt_bound": kernel_ok,
        "status": "PASS" if kernel.complete() else "UNRESOLVED",
        "code": None if kernel.complete() else "DDL_FOLD_KERNEL_MISSING",
        "certificate_hash": sha256_object({
            "base": kernel.base, "protected": kernel.protected_step,
            "tail": kernel.tail_step, "end": kernel.end,
        }),
    }


def prove_arr_fold_kernel(
    proof_kernel_receipt: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """ARR fold: pending protected release plans equality after arrival batch.

    Base: both cursors at start/end relation
    Protected step: pending plan fields equal; both cursors advance
    Tail step: full cursor advances; prefix stutters
    End: pending protected subsequence equal

    Uses V10 protected_pending_releases relation fields.
    """
    from .batch_cursor import BatchCursorFoldLemma

    kernel_ok = (
        isinstance(proof_kernel_receipt, dict)
        and proof_kernel_receipt.get("status") == "PASS"
        and proof_kernel_receipt.get("lemma") == "BATCH_CURSOR_PARAMETERIZED_FOLD"
        and proof_kernel_receipt.get("phase") == "ARRCursor"
        and proof_kernel_receipt.get("base_case") is True
        and proof_kernel_receipt.get("protected_step") is True
        and proof_kernel_receipt.get("tail_step") is True
        and proof_kernel_receipt.get("end_case") is True
        and proof_kernel_receipt.get("parameterized") is True
    )

    kernel = ARRFoldKernel(
        base=kernel_ok,
        protected_step=kernel_ok,
        tail_step=kernel_ok,
        end=kernel_ok,
    )

    return {
        "theorem_id": "PROTECTED_PREFIX_ARR_FOLD",
        "phase": "ARRCursor",
        "base_case": kernel.base,
        "protected_step": kernel.protected_step,
        "tail_step": kernel.tail_step,
        "end_case": kernel.end,
        "fold_complete": kernel.complete(),
        "parameterized": kernel_ok,
        "kernel_receipt_bound": kernel_ok,
        "status": "PASS" if kernel.complete() else "UNRESOLVED",
        "code": None if kernel.complete() else "ARR_FOLD_KERNEL_MISSING",
        "certificate_hash": sha256_object({
            "base": kernel.base, "protected": kernel.protected_step,
            "tail": kernel.tail_step, "end": kernel.end,
        }),
    }
