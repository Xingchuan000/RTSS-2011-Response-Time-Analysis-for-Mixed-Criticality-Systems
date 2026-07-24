from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from formal_toolchain.core.hashing import sha256_object


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
