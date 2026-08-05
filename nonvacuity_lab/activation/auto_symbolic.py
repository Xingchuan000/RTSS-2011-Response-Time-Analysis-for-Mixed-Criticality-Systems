"""Automatic Phase 6 symbolic activation with mandatory concrete replay."""

from __future__ import annotations

import importlib
import json
from pathlib import Path

from .schema import ActivationResult
from .symbolic_model_builder import build_symbolic_problem
from .smt_receipt import finalize_replay, solve_and_write
from .witness_replay import replay_symbolic_witness
from ..schema import ActivationStatus


def _load_object(spec: str):
    module_name, separator, name = str(spec).partition(":")
    if not separator or not module_name or not name:
        raise ValueError("binding spec must be module:object")
    return getattr(importlib.import_module(module_name), name)


def run_auto_symbolic_activation(
    *, mutation_id: str, activation: dict, resolved_target: dict,
    clean_source_root: Path, overlay_source_root: Path, output_dir: Path,
) -> ActivationResult:
    """Build, solve, and replay one witness; SAT alone is never activation."""
    problem = build_symbolic_problem(
        resolved_target, dict(activation["binding"]), str(activation["formula_kind"])
    )
    smt_dir = output_dir / "smt"
    receipt = solve_and_write(problem, smt_dir)
    if receipt["solver_status"] != "SAT":
        return ActivationResult(
            mutation_id=mutation_id, status=ActivationStatus.NOT_ACTIVATED,
            evidence_modes=("SYMBOLIC",), guard_satisfiable=False, details=receipt,
        )
    model = json.loads((smt_dir / "activation_model.json").read_text(encoding="utf-8"))
    factory = _load_object(activation["runtime_binding_factory"])
    runtime_binding = factory(
        clean_source_root=clean_source_root, overlay_source_root=overlay_source_root,
        resolved_target=resolved_target, binding=dict(activation["binding"]),
    )
    replay = replay_symbolic_witness(
        model, binding=runtime_binding.state_binding,
        clean_runtime=runtime_binding.clean_runtime,
        overlay_runtime=runtime_binding.overlay_runtime,
        expected=problem.metadata,
    )
    final = finalize_replay(smt_dir / "activation_receipt.json", replay)
    activated = final.get("activated") is True and replay.get("status") == "MATCHED"
    return ActivationResult(
        mutation_id=mutation_id,
        status=ActivationStatus.ACTIVATED if activated else ActivationStatus.ACTIVATION_SETUP_INVALID,
        evidence_modes=("SYMBOLIC", "CONCRETE_REPLAY"),
        leaf_id=int(problem.metadata["leaf_id"]), action_id=int(problem.metadata["action_id"]),
        guard_satisfiable=True, illegal_action_witness=str(smt_dir / "activation_model.json"),
        post_invariant_violation=str(activation["formula_kind"]) == "B_RAW_TOP1_BREAKS_INVARIANT",
        details={"solver": final, "replay": replay},
    )
