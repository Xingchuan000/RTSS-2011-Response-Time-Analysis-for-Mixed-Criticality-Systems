from __future__ import annotations

import ast
import json
from pathlib import Path

from formal_toolchain.workflow.seed_workspace import freeze_seed_workspace

ROOT = Path(__file__).resolve().parents[3]
FIXTURE = ROOT / "tests" / "formal" / "fixtures" / "synthetic_p0"

FORBIDDEN_IDENTIFIERS = {
    "nonvacuity_profile",
    "nonvacuity_params",
    "nonvacuity_disabled_guards",
    "unchecked_apply",
    "unchecked_if_invalid",
    "first_valid_else_top1",
    "top1_or_noop",
}


def _identifiers(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            names.add(node.id)
        elif isinstance(node, ast.Attribute):
            names.add(node.attr)
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            names.add(node.value)
    return names


def test_proof_and_runtime_core_have_no_experiment_profiles():
    offenders: list[tuple[str, list[str]]] = []
    for package in ("amc_py", "formal_toolchain"):
        for path in (ROOT / package).rglob("*.py"):
            found = sorted(FORBIDDEN_IDENTIFIERS & _identifiers(path))
            if found:
                offenders.append((str(path.relative_to(ROOT)), found))
    assert offenders == []


def test_clean_request_contains_no_experiment_fields(tmp_path: Path):
    imported = freeze_seed_workspace(
        FIXTURE,
        "best_overall",
        tmp_path / "workspace",
        code_root=ROOT,
        refresh_phase_k_map=True,
        proof_route="protected_prefix",
    )
    request = json.loads(Path(imported["request"]).read_text(encoding="utf-8"))
    text = json.dumps(request, ensure_ascii=False).lower()
    assert "nonvacuity" not in text
    assert "mutation_id" not in text
    assert "expected_failure" not in text
