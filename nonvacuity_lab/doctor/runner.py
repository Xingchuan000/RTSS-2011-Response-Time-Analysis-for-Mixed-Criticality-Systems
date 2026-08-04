from __future__ import annotations

import ast
import importlib
import json
from pathlib import Path
import sys

from ..canonical import tree_hash
from ..config_io import validate_config_kind, verify_config_hash
from .schema import DoctorCheck, DoctorReceipt, DoctorStatus
from .checks import check_output_isolation, recursively_find_placeholders, check_pair_graph, check_hout_profile, check_patch_binding
from .check_isolation import check_ordinary_cli_surface
from .check_registry import check_expected_obligations, load_obligation_ids


def _core_findings(source_root: Path):
    findings = []
    for package in ("amc_py", "formal_toolchain"):
        root = source_root / package
        if not root.is_dir():
            findings.append(str(root))
            continue
        for path in root.rglob("*.py"):
            try: tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            except (OSError, SyntaxError):
                continue
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    if any(alias.name.startswith("nonvacuity_lab") for alias in node.names): findings.append(str(path))
                elif isinstance(node, ast.ImportFrom) and (node.module or "").startswith("nonvacuity_lab"):
                    findings.append(str(path))
    return findings


def _check_z3():
    try:
        z3 = importlib.import_module("z3")
        x = z3.Int("doctor_x")
        solver = z3.Solver(); solver.add(x > 3, x < 5)
        ok = solver.check() == z3.sat and solver.model()[x].as_long() == 4
        return DoctorCheck.pass_("z3", "Z3 smoke test passed", version=z3.get_version_string()) if ok else DoctorCheck.fail("z3", "Z3 smoke test failed")
    except Exception as exc:
        return DoctorCheck.fail("z3", "Z3 unavailable", error=repr(exc))


def run_doctor(config_path: Path, output_path: Path | None = None) -> DoctorReceipt:
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    validate_config_kind(config)
    verify_config_hash(config)
    source = Path(config["source_binding"]["clean_source_root"]).resolve()
    expected_source_hash = config["source_binding"].get("clean_source_root_sha256", "")
    actual_source_hash = tree_hash(source)
    checks = [
        DoctorCheck.pass_("python_version", "Python version supported", version=sys.version.split()[0]) if sys.version_info >= (3, 11) else DoctorCheck.fail("python_version", "Python >= 3.11 required"),
        DoctorCheck.pass_("jsonschema", "jsonschema import ok") if _importable("jsonschema") else DoctorCheck.fail("jsonschema", "jsonschema import failed"),
        _check_z3(),
        DoctorCheck.pass_("source_binding", "source root hash matches", sha256=actual_source_hash) if expected_source_hash == actual_source_hash else DoctorCheck.fail("source_binding", "source root hash mismatch", expected=expected_source_hash, actual=actual_source_hash),
        DoctorCheck.pass_("core_isolation", "formal/runtime core is lab-blind") if not _core_findings(source) else DoctorCheck.fail("core_isolation", "core imports nonvacuity_lab", findings=_core_findings(source)),
        DoctorCheck.pass_("resolved_targets", "resolved config loaded", count=len(config.get("mutations", []))),
        check_pair_graph(config.get("mutations", [])),
        check_output_isolation(config),
    ]
    checks.extend(check_ordinary_cli_surface(config))
    try:
        checks.append(check_expected_obligations(config, load_obligation_ids(source)))
    except Exception as exc:
        checks.append(DoctorCheck.fail("obligation_registry", "obligation registry unavailable", error=repr(exc)))
    placeholders = recursively_find_placeholders(config)
    checks.append(DoctorCheck.fail("placeholders", "resolved config contains placeholders", findings=placeholders) if placeholders else DoctorCheck.pass_("placeholders", "no unresolved placeholders"))
    for profile_id, profile in config.get("hout_profiles", {}).items():
        profile = {"profile_id": profile_id, **profile}
        checks.append(check_hout_profile(profile))
    for mutation in config.get("mutations", []):
        mutator = mutation.get("mutator", {})
        parameters = mutator.get("parameters", {}) if isinstance(mutator, dict) else {}
        patches = parameters.get("patches", []) if isinstance(parameters, dict) else []
        for patch in patches:
            if isinstance(patch, dict):
                checks.append(check_patch_binding(source, patch))
    overall = DoctorStatus.FAIL if any(c.status is DoctorStatus.FAIL for c in checks) else DoctorStatus.PASS
    receipt = DoctorReceipt("nonvacuity_doctor_receipt_v1", str(config["campaign_id"]), str(config.get("config_sha256")), actual_source_hash, overall, tuple(checks))
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(receipt.to_dict(), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return receipt


def _importable(name):
    try: importlib.import_module(name)
    except Exception: return False
    return True
