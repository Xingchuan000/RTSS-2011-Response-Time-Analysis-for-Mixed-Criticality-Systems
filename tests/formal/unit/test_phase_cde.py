"""Phase C/D/E 的确定性和 fail-closed 验收测试。"""

import json
import shutil
from dataclasses import dataclass
from pathlib import Path

import pytest

from formal_toolchain.adapters.immutable_inputs import build_immutable_input_certificate
from formal_toolchain.adapters.runtime_config import export_effective_config
from formal_toolchain.adapters.runtime_manifest import build_dependency_manifest, build_runtime_environment_manifest
from formal_toolchain.adapters.source_manifest import TARGET_FILES, build_source_manifest
from formal_toolchain.binding.python_ast_ir import function_to_ir
from formal_toolchain.core.hashing import sha256_object
from formal_toolchain.verifier.theory_verifier import verify_theory_library

ROOT = Path(__file__).parents[3]


def test_theory_manifest_is_verified_and_has_no_instance_data():
    result = verify_theory_library(ROOT / "formal_toolchain/theory")
    assert result["status"] == "PASS"
    assert result["theorem_count"] == 10


def test_theory_verifier_rejects_seed_specific_tcb(tmp_path: Path):
    theory = tmp_path / "theory"
    (theory / "statements").mkdir(parents=True)
    (theory / "theory_manifest.json").write_text((ROOT / "formal_toolchain/theory/theory_manifest.json").read_text(), encoding="utf-8")
    shutil.copy2(ROOT / "formal_toolchain/theory/assurance_policy.json", theory / "assurance_policy.json")
    shutil.copy2(ROOT / "formal_toolchain/theory/hashes.json", theory / "hashes.json")
    for statement in (ROOT / "formal_toolchain/theory/statements").glob("*.json"):
        shutil.copy2(statement, theory / "statements" / statement.name)
    original = json.loads((ROOT / "formal_toolchain/theory/statements/FPPS_LO_RTA_POSTFIXED_SUFFICIENCY.json").read_text())
    original["exact_statement"] += " seed 185"
    original["statement_hash"] = sha256_object({key: original[key] for key in ("theorem_id", "exact_statement", "conclusion", "source_reference", "assurance_level", "version")})
    (theory / "statements/FPPS_LO_RTA_POSTFIXED_SUFFICIENCY.json").write_text(json.dumps(original), encoding="utf-8")
    with pytest.raises(ValueError, match="seed-specific"):
        verify_theory_library(theory)


def test_source_manifest_contains_fixed_targets_and_changes_with_token(tmp_path: Path):
    for relative in TARGET_FILES:
        destination = tmp_path / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, destination)
    manifest = build_source_manifest(tmp_path)
    assert len(manifest["files"]) >= 14
    assert all("__pycache__" not in item["path"] for item in manifest["files"])
    target = tmp_path / TARGET_FILES[0]
    target.write_text(target.read_text(encoding="utf-8") + "\n# mutation\n", encoding="utf-8")
    mutated = build_source_manifest(tmp_path)
    assert manifest["semantic_hash"] != mutated["semantic_hash"]


def test_runtime_manifest_has_no_json_float_and_config_has_provenance():
    runtime = build_runtime_environment_manifest()
    assert isinstance(runtime["float_info"]["max"], str)
    @dataclass
    class Config:
        semantics: str = "C_AMC_SEM"
        capture_trace: bool = False
    @dataclass
    class Env:
        capture_trace: bool = True
        agent_period: int = 10
    config = export_effective_config(Config(), Env())
    assert config["fields"]["capture_trace"]["value"] is True
    assert config["fields"]["capture_trace"]["origin"] == "environment_wrapper"
    assert build_dependency_manifest()["schema_version"] == "dependency_manifest_v1"


def test_immutable_hash_excludes_downstream_certificate():
    args = dict(source_manifest={"x": "1"}, runtime_manifest={"x": "2"}, dependency_manifest={},
                checker_manifest={}, taskset={}, priority=[], tree={}, features={}, actions={},
                fixed_point={}, effective_config={}, theory={}, specs={})
    first = build_immutable_input_certificate(**args)
    args["specs"] = {"schema": "changed"}
    second = build_immutable_input_certificate(**args)
    assert first["witness"]["immutable_input_hash"] != second["witness"]["immutable_input_hash"]


def test_ast_subset_passes_simple_function_and_rejects_while_and_eval():
    assert function_to_ir("def f(x):\n    return min(x, 3)\n", "f")["status"] == "PASS"
    rejected = function_to_ir("def f(x):\n    while x:\n        x -= 1\n    return x\n", "f")
    assert rejected["status"] == "UNRESOLVED"
    assert rejected["failure"]["code"] == "UNSUPPORTED_AST_NODE"
    rejected_call = function_to_ir("def f(x):\n    return eval(x)\n", "f")
    assert rejected_call["failure"]["code"] == "UNSUPPORTED_AST_NODE"
