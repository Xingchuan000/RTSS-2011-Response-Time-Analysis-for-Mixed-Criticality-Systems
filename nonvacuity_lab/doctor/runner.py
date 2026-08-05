from __future__ import annotations

import ast
import importlib
import json
from pathlib import Path
import sys

from ..canonical import tree_hash
from ..config_io import validate_config_kind, verify_config_hash
from ..v2_runner import _v2_mutation_to_v1, _resolve_hout_profile_paths
from ..schema import MutationManifest
from .schema import DoctorCheck, DoctorReceipt, DoctorStatus
from .checks import check_output_isolation, recursively_find_placeholders, check_pair_graph, check_hout_profile, check_patch_binding
from .check_isolation import check_ordinary_cli_surface
from .check_registry import check_expected_obligations, load_obligation_ids
from ..analysis.rta_slack import scan_rta_slack, select_minimum_slack


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



def _check_proof_dependency_lock(source_root: Path) -> DoctorCheck:
    """Check the exact dependency policy used by ordinary prove_seed."""

    try:
        from formal_toolchain.adapters.runtime_manifest import (
            build_dependency_manifest, check_dependency_policy,
        )
        lock_path = source_root / "formal_toolchain/specs/proof_dependency_lock.json"
        if not lock_path.is_file():
            return DoctorCheck.fail(
                "proof_dependency_lock", "proof dependency lock missing",
                path=str(lock_path),
            )
        lock = json.loads(lock_path.read_text(encoding="utf-8"))
        result = check_dependency_policy(build_dependency_manifest(), lock=lock)
        if result.get("status") == "PASS":
            return DoctorCheck.pass_(
                "proof_dependency_lock",
                "ordinary proof dependency lock satisfied",
                python=result.get("python_actual"),
                lock=str(lock_path),
            )
        return DoctorCheck.fail(
            "proof_dependency_lock",
            "ordinary proof dependency lock not satisfied",
            result=result,
        )
    except (OSError, ValueError, TypeError, ImportError) as exc:
        return DoctorCheck.fail(
            "proof_dependency_lock", "proof dependency check failed",
            error=repr(exc),
        )


def _check_ordinary_source_bindings(source_root: Path) -> DoctorCheck:
    """Run the source binders that ordinary PPP compilation consumes."""

    try:
        from formal_toolchain.binding.action_binding import bind_action_runtime
        from formal_toolchain.binding.event_runtime_binding import bind_event_runtime
        from formal_toolchain.binding.removal_binding import bind_removal_runtime

        results = {
            "event": bind_event_runtime(source_root),
            "action": bind_action_runtime(source_root),
            "removal": bind_removal_runtime(source_root),
        }
        failures = {
            name: value for name, value in results.items()
            if value.get("status") != "PASS"
        }
        if failures:
            return DoctorCheck.fail(
                "ordinary_source_bindings",
                "ordinary PPP source binding failed",
                failures=failures,
            )
        return DoctorCheck.pass_(
            "ordinary_source_bindings",
            "ordinary PPP source bindings passed",
            selection_semantics=results["action"].get("order_evidence", {}).get("selection_semantics"),
        )
    except (OSError, ValueError, TypeError, ImportError, SyntaxError) as exc:
        return DoctorCheck.fail(
            "ordinary_source_bindings", "ordinary source binding check failed",
            error=repr(exc),
        )


def _enabled_mutations(config: dict) -> list[dict]:
    return [
        item for item in config.get("mutations", [])
        if isinstance(item, dict) and bool(item.get("enabled", False))
    ]


def _active_config_view(config: dict, mutations: list[dict]) -> dict:
    active = dict(config)
    active["mutations"] = mutations
    used_profiles = {
        str(item.get("hout_profile_id"))
        for item in mutations
        if item.get("hout_profile_id")
    }
    active["hout_profiles"] = {
        str(profile_id): profile
        for profile_id, profile in config.get("hout_profiles", {}).items()
        if str(profile_id) in used_profiles
    }
    return active


def _requires_z3(mutations: list[dict]) -> bool:
    return any(
        "symbolic" in str(item.get("activation", {}).get("mode", "")).lower()
        for item in mutations
    )

def run_doctor(config_path: Path, output_path: Path | None = None) -> DoctorReceipt:
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    validate_config_kind(config)
    verify_config_hash(config)
    source = Path(config["source_binding"]["clean_source_root"]).resolve()
    expected_source_hash = config["source_binding"].get("clean_source_root_sha256", "")
    actual_source_hash = tree_hash(source)
    enabled_mutations = _enabled_mutations(config)
    active_config = _active_config_view(config, enabled_mutations)
    z3_check = (
        _check_z3()
        if _requires_z3(enabled_mutations)
        else DoctorCheck.skip("z3", "Z3 not required by enabled mutations")
    )
    findings = _core_findings(source)
    checks = [
        DoctorCheck.pass_("python_version", "Python version supported", version=sys.version.split()[0]) if sys.version_info >= (3, 11) else DoctorCheck.fail("python_version", "Python >= 3.11 required"),
        DoctorCheck.pass_("jsonschema", "jsonschema import ok") if _importable("jsonschema") else DoctorCheck.fail("jsonschema", "jsonschema import failed"),
        _check_proof_dependency_lock(source),
        _check_ordinary_source_bindings(source),
        z3_check,
        DoctorCheck.pass_("source_binding", "source root hash matches", sha256=actual_source_hash) if expected_source_hash == actual_source_hash else DoctorCheck.fail("source_binding", "source root hash mismatch", expected=expected_source_hash, actual=actual_source_hash),
        DoctorCheck.pass_("core_isolation", "formal/runtime core is lab-blind") if not findings else DoctorCheck.fail("core_isolation", "core imports nonvacuity_lab", findings=findings),
        DoctorCheck.pass_("resolved_targets", "enabled mutation plan loaded", count=len(enabled_mutations)),
        check_pair_graph(enabled_mutations),
        check_output_isolation(config),
    ]
    checks.extend(check_ordinary_cli_surface(config))
    try:
        checks.append(check_expected_obligations(active_config, load_obligation_ids(source)))
    except Exception as exc:
        checks.append(DoctorCheck.fail("obligation_registry", "obligation registry unavailable", error=repr(exc)))
    placeholders = recursively_find_placeholders(active_config)
    checks.append(DoctorCheck.fail("placeholders", "enabled experiment plan contains placeholders", findings=placeholders) if placeholders else DoctorCheck.pass_("placeholders", "no unresolved placeholders in enabled experiment plan"))
    config_base = Path(config_path).resolve().parent
    for profile_id, profile in active_config.get("hout_profiles", {}).items():
        profile = _resolve_hout_profile_paths(
            {"profile_id": profile_id, **profile}, base_dir=config_base
        )
        checks.append(check_hout_profile(profile))
    for mutation in enabled_mutations:
        mutator = mutation.get("mutator", {})
        parameters = mutator.get("parameters", {}) if isinstance(mutator, dict) else {}
        patches = parameters.get("patches", []) if isinstance(parameters, dict) else []
        for patch in patches:
            if isinstance(patch, dict):
                checks.append(check_patch_binding(source, patch))
    for mutation in enabled_mutations:
        try:
            v1 = _v2_mutation_to_v1(
                mutation, config=config, base_dir=Path(config_path).resolve().parent
            )
            manifest = MutationManifest.from_mapping(
                v1, base_dir=Path(config_path).resolve().parent
            )
            from ..preflight import audit_mutation
            result = audit_mutation(manifest, source_root=source)
            if result["status"] == "PASS":
                checks.append(DoctorCheck.pass_(f"mutation:{mutation['mutation_id']}", "preflight passed"))
            else:
                checks.append(DoctorCheck.fail(f"mutation:{mutation['mutation_id']}", "preflight failed", issues=result["issues"]))
        except (OSError, ValueError, KeyError, TypeError) as exc:
            checks.append(DoctorCheck.fail(f"mutation:{mutation.get('mutation_id')}", "preflight setup failed", error=repr(exc)))
        if str(mutation.get("mutation_id", "")).split("_", 1)[0] == "D1":
            metadata = mutation.get("metadata", {})
            roots = metadata.get("bundle_roots", [])
            minimum_population_size = int(metadata.get("minimum_population_size", 20))
            try:
                rows = scan_rta_slack([Path(str(item)) for item in roots])
                identities = {(row.get("seed"), row.get("variant")) for row in rows}
                if len(identities) < minimum_population_size:
                    raise ValueError(
                        f"fewer than {minimum_population_size} proof artifacts: {len(identities)}"
                    )
                selected = select_minimum_slack(rows)
                if not selected.get("envelope_target_file") or not selected.get("envelope_json_pointer"):
                    raise ValueError("minimum-slack record lacks target adapter fields")
                checks.append(DoctorCheck.pass_("d1_population", "D1 proof artifact population valid", count=len(identities), minimum=minimum_population_size))
            except (OSError, ValueError, TypeError, KeyError) as exc:
                checks.append(DoctorCheck.fail("d1_population", "D1 proof artifact population invalid", error=str(exc)))
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
