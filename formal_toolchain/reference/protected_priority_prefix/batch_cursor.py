"""Batch cursors for deadline and arrival batch correspondence.

The DDLCursor and ARRCursor track positions within deadline batches and
arrival batches respectively.  They allow the full execution to skip
tail entries (which have no prefix counterpart) while maintaining exact
correspondence for protected entries.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class BatchCursor:
    phase: str  # "DDLCursor" or "ARRCursor"
    k_full: int
    k_prefix: int
    full_batch_size: int
    prefix_batch_size: int
    tail_skip_count: int

    @property
    def cursor_id(self) -> str:
        return f"{self.phase}({self.k_full},{self.k_prefix})"


@dataclass(frozen=True, slots=True)
class BatchCursorProof:
    """Proof object for batch cursor correspondence.

    Verifies that after skipping tail entries, the protected entries
    in the full batch match the prefix batch in key, time, demand,
    HI class, and order.
    """
    base_case: bool
    head_step: bool
    tail_induction: bool
    protected_entry_correspondence: bool


def construct_batch_cursor(
    full_batch_entries: list[dict[str, Any]],
    prefix_batch_entries: list[dict[str, Any]],
    protected_task_names: frozenset[str],
    phase: str,
) -> tuple[BatchCursor, BatchCursorProof]:
    """Construct a batch cursor and proof object.

    Skips full-batch entries whose task names are not in the protected
    set, then verifies that the remaining protected entries correspond
    one-to-one with the prefix batch.
    """
    protected_full = [
        entry for entry in full_batch_entries
        if entry.get("task_name") in protected_task_names
    ]
    tail_count = len(full_batch_entries) - len(protected_full)

    cursor = BatchCursor(
        phase=phase,
        k_full=len(protected_full),
        k_prefix=len(prefix_batch_entries),
        full_batch_size=len(full_batch_entries),
        prefix_batch_size=len(prefix_batch_entries),
        tail_skip_count=tail_count,
    )

    correspondence = len(protected_full) == len(prefix_batch_entries)
    if correspondence and protected_full:
        for pf, pp in zip(protected_full, prefix_batch_entries):
            if (
                pf.get("job_key") != pp.get("job_key")
                or pf.get("release_time") != pp.get("release_time")
                or pf.get("actual_demand") != pp.get("actual_demand")
                or pf.get("hi_class") != pp.get("hi_class")
            ):
                correspondence = False
                break

    proof = BatchCursorProof(
        base_case=True,
        head_step=correspondence,
        tail_induction=tail_count >= 0,
        protected_entry_correspondence=correspondence,
    )

    return cursor, proof


def verify_batch_cursor_proof(
    cursor: BatchCursor,
    proof: BatchCursorProof,
) -> dict[str, Any]:
    """Verify that a batch cursor proof is valid."""
    ok = all((
        proof.base_case,
        proof.head_step,
        proof.tail_induction,
        proof.protected_entry_correspondence,
        cursor.k_full == cursor.k_prefix,
        cursor.prefix_batch_size >= 0,
    ))

    return {
        "status": "PASS" if ok else "FAIL",
        "cursor": cursor.cursor_id,
        "tail_skip_count": cursor.tail_skip_count,
        "proof": {
            "base_case": proof.base_case,
            "head_step": proof.head_step,
            "tail_induction": proof.tail_induction,
            "protected_entry_correspondence": proof.protected_entry_correspondence,
        },
    }
