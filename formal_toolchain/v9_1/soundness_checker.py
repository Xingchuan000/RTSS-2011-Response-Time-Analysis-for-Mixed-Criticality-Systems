"""Fail-closed interface for the V9.1 finite-encoding soundness theorem.

The eighteen clauses in ``REQUIRED_SOUNDNESS_CLAUSES`` are proof obligations,
not source-code coverage checks.  They must not be discharged by string
matching, field presence, or a manifest boolean.  Until a checker/proof
constructor proves those clauses from the encoder definitions, this interface
returns an explicit unresolved certificate and cannot be used to open the route.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

from formal_toolchain.core.hashing import sha256_object, sha256_text_file_normalized

from .encoding_contract import REQUIRED_SOUNDNESS_CLAUSES
from .readiness import blocker_rows
from .safe_prefix_invariant import SafePrefixInvariant
from .symbolic_state import BoundModel
from .window_encoder import ENCODER_VERSION, WindowEncoding


def build_finite_window_soundness_certificate(
    model: BoundModel,
    invariant: SafePrefixInvariant,
    *,
    binding_root_hash: str = "",
    encodings: Sequence[WindowEncoding] | None = None,
    source_root: Path | None = None,
) -> dict[str, Any]:
    """Return a non-promoting certificate until the soundness theorem exists."""

    del model, invariant
    active = tuple(encodings or ())
    clauses = {name: False for name in REQUIRED_SOUNDNESS_CLAUSES}
    source_hashes: dict[str, str] = {}
    if source_root is not None:
        for path in sorted(Path(source_root).glob("formal_toolchain/v9_1/*.py")):
            source_hashes[str(path.relative_to(source_root))] = sha256_text_file_normalized(path)
    formula_hash = sha256_object([encoding.smt2() for encoding in active])
    return {
        "schema_version": "v9_1_finite_window_soundness_certificate_v2_unresolved",
        "encoder_version": ENCODER_VERSION,
        "binding_root_hash": binding_root_hash,
        "clauses": clauses,
        "required_clauses": list(REQUIRED_SOUNDNESS_CLAUSES),
        "all_pass": False,
        "status": "UNRESOLVED",
        "failure_code": "V9_1_FINITE_ENCODING_SOUNDNESS_THEOREM_UNBOUND",
        "implementation_gaps": blocker_rows(),
        "encoder_source_hashes": source_hashes,
        "formula_hash": formula_hash,
        "fresh_recompute_hash": sha256_object({"clauses": clauses, "formula_hash": formula_hash}),
    }


__all__ = ["build_finite_window_soundness_certificate"]
