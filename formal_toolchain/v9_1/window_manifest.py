"""Manifest generation for per-HI first-bad-window formulas."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from formal_toolchain.core.hashing import sha256_file, sha256_object

from .constants import REQUIRED_SOUNDNESS_CLAUSES
from .safe_prefix_invariant import SafePrefixInvariant
from .symbolic_state import BoundModel
from .window_encoder import ENCODER_VERSION, WindowEncoding, build_first_bad_window, write_first_bad_window


def build_window_manifest(
    model: BoundModel,
    invariant: SafePrefixInvariant,
    output_dir: Path,
    *,
    binding_root_hash: str,
    soundness_coverage: dict[str, bool] | None = None,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for task in model.hi_tasks:
        encoding = build_first_bad_window(model, invariant, task.name)
        filename = f"first_bad_window__{task.name}.smt2"
        path = output_dir / filename
        formula_hash = write_first_bad_window(encoding, path)
        rows.append({"task": task.name, "deadline": task.deadline, "smt2": filename,
                     "formula_hash": formula_hash, "source_obligations": list(encoding.source_obligations)})
    coverage = dict(soundness_coverage or {name: False for name in REQUIRED_SOUNDNESS_CLAUSES})
    return {
        "schema_version": "v9_1_proof_manifest_v1",
        "binding_root_hash": binding_root_hash,
        "window_encoder_version": ENCODER_VERSION,
        "safe_prefix_inductiveness_condition": "NOT NewHIMiss(z,z')",
        "carry_in_summary": "EXPLICIT",
        "finite_window_soundness_clauses": coverage,
        "hi_windows": rows,
        "formula_root_hash": sha256_object({row["task"]: row["formula_hash"] for row in rows}),
    }


__all__ = ["build_window_manifest"]
