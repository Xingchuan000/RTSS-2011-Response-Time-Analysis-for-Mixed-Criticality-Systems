from __future__ import annotations

import json
from pathlib import Path

from ..canonical import file_hash


def solve_and_write(problem, output_dir: Path) -> dict:
    import z3
    output_dir.mkdir(parents=True, exist_ok=False)
    smt2 = output_dir / "activation.smt2"
    smt2.write_text(problem.solver.to_smt2(), encoding="utf-8")
    result = problem.solver.check()
    receipt = {"solver_status": str(result).upper(), "activated": False, "runtime_replay_status": "NOT_RUN", "smt2_sha256": file_hash(smt2), "metadata": problem.metadata}
    if result == z3.sat:
        model = problem.solver.model()
        model_data = {"features": {str(k): model.eval(v, model_completion=True).as_long() for k, v in problem.feature_vars.items()}, "budgets": {str(k): model.eval(v, model_completion=True).as_long() for k, v in problem.budget_vars.items()}}
        model_path = output_dir / "activation_model.json"
        model_path.write_text(json.dumps(model_data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        receipt["activated"] = None
        receipt["model_sha256"] = file_hash(model_path)
    (output_dir / "activation_receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return receipt


def finalize_replay(receipt_path: Path, replay: dict) -> dict:
    """Only concrete replay may turn a SAT model into an activated witness."""
    receipt = json.loads(Path(receipt_path).read_text(encoding="utf-8"))
    receipt["runtime_replay_status"] = replay.get("status", "ACTIVATION_MODEL_RUNTIME_MISMATCH")
    receipt["activated"] = bool(replay.get("activated", False))
    Path(receipt_path).write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return receipt
