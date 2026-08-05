from __future__ import annotations

from pathlib import Path
import re
import ast

from ..canonical import python_symbol_hash

from .schema import DoctorCheck


PLACEHOLDER_PATTERNS = ("TODO", "TBD", "<PATH>", "REPLACE_ME", "PLACEHOLDER", ".../")
ALLOWED_COMMAND_PLACEHOLDERS = {"seed_dir", "tree_path", "scenario_file", "runtime_config", "taskset", "output_dir"}


def recursively_find_placeholders(value, path="$", findings=None):
    findings = [] if findings is None else findings
    if isinstance(value, dict):
        for key, child in value.items():
            recursively_find_placeholders(child, f"{path}.{key}", findings)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            recursively_find_placeholders(child, f"{path}[{index}]", findings)
    elif isinstance(value, str):
        for pattern in PLACEHOLDER_PATTERNS:
            if pattern in value:
                findings.append({"path": path, "pattern": pattern, "value": value})
    return findings


def command_placeholders(command):
    return set(re.findall(r"\{([A-Za-z_][A-Za-z0-9_]*)\}", " ".join(str(x) for x in command)))


def check_output_isolation(config):
    roots = config.get("output_roots", {})
    formal = Path(roots.get("formal_proofs", "outputs/formal_proofs")).resolve()
    lab = Path(roots.get("nonvacuity_lab", config.get("output_root", "outputs/nonvacuity_lab"))).resolve()
    overlap = formal == lab or formal in lab.parents or lab in formal.parents
    return DoctorCheck.fail("output_isolation", "formal proof root and lab root overlap", formal_root=str(formal), lab_root=str(lab)) if overlap else DoctorCheck.pass_("output_isolation", "formal and lab output roots are disjoint")


def check_pair_graph(mutations):
    rows = {str(item["mutation_id"]): item for item in mutations}
    order = {str(item["mutation_id"]): index for index, item in enumerate(mutations)}
    errors = []
    edges = {}
    for row in mutations:
        mutation_id = str(row["mutation_id"])
        producer = row.get("pair_with")
        if not producer:
            continue
        producer = str(producer)
        edges[mutation_id] = producer
        if producer not in rows:
            errors.append((mutation_id, producer, "missing"))
            continue
        if order[producer] >= order[mutation_id]:
            errors.append((mutation_id, producer, "producer must appear first"))
        if rows[producer].get("seed") != row.get("seed") or rows[producer].get("tree_variant") != row.get("tree_variant"):
            errors.append((mutation_id, producer, "seed/variant mismatch"))
    for start in edges:
        seen = set()
        current = start
        while current in edges:
            if current in seen:
                errors.append((start, current, "cycle"))
                break
            seen.add(current)
            current = edges[current]
    return DoctorCheck.fail("pair_graph", "pair dependencies invalid", errors=errors) if errors else DoctorCheck.pass_("pair_graph", "pair dependencies valid")


def check_hout_profile(profile):
    import json

    errors = []
    for name in ("base_command", "mutated_command"):
        command = profile.get(name, ())
        if not isinstance(command, list) or not command:
            errors.append(f"{name}: command missing")
            continue
        unknown = command_placeholders(command) - ALLOWED_COMMAND_PLACEHOLDERS
        if unknown:
            errors.append(f"{name}: unknown placeholders {sorted(unknown)}")
    scenarios = profile.get("scenario_seeds", ())
    if not isinstance(scenarios, list) or not scenarios:
        errors.append("scenario_seeds must be a non-empty array")
    for key in ("horizon", "worker_count", "random_seed"):
        if profile.get(key) is None:
            errors.append(f"missing field: {key}")
    for key in ("taskset_path", "runtime_config_path"):
        value = profile.get(key)
        if value is None or not Path(value).is_file():
            errors.append(f"missing file: {value}")
    runtime_path = profile.get("runtime_config_path")
    if runtime_path is not None and Path(runtime_path).is_file():
        try:
            runtime = json.loads(Path(runtime_path).read_text(encoding="utf-8"))
            factory = runtime.get("ordinary_hout_factory") if isinstance(runtime, dict) else None
            if not isinstance(factory, str) or ":" not in factory:
                errors.append("runtime_config ordinary_hout_factory must be module:function")
            else:
                try:
                    import importlib
                    module_name, _, function_name = factory.partition(":")
                    candidate = getattr(importlib.import_module(module_name), function_name)
                    if not callable(candidate):
                        errors.append("runtime_config ordinary_hout_factory is not callable")
                except (ImportError, AttributeError, TypeError, ValueError) as exc:
                    errors.append(f"runtime_config ordinary_hout_factory import failed: {exc}")
        except (OSError, ValueError, TypeError) as exc:
            errors.append(f"runtime_config invalid: {exc}")
    if set(profile.get("required_scenarios", ())) - set(scenarios if isinstance(scenarios, list) else ()):
        errors.append("required_scenarios not contained in scenario_seeds")
    return DoctorCheck.fail(f"hout:{profile.get('profile_id', 'unknown')}", "HOUT profile invalid", findings=errors) if errors else DoctorCheck.pass_(f"hout:{profile.get('profile_id', 'unknown')}", "HOUT profile valid")


def check_patch_binding(source_root: Path, patch: dict):
    from .schema import DoctorCheck
    path = source_root / str(patch.get("target_file", ""))
    if not path.is_file() or path.is_symlink():
        return DoctorCheck.fail("patch_binding", "patch target missing", path=str(path))
    try:
        source = path.read_text(encoding="utf-8")
        ast.parse(source, filename=str(path))
        if patch.get("before_ast_hash"):
            actual_hash = python_symbol_hash(source, str(patch["target_symbol"]))
            if actual_hash != str(patch["before_ast_hash"]):
                return DoctorCheck.fail("patch_binding", "symbol AST hash mismatch", path=str(path), expected=patch["before_ast_hash"], actual=actual_hash)
        before = str(patch["before_snippet"])
        count = source.count(before)
        expected = int(patch.get("occurrence", 1))
        if count != expected:
            return DoctorCheck.fail("patch_binding", "before snippet is not unique", path=str(path), expected=expected, actual=count)
        after = source.replace(before, str(patch["after_snippet"]), 1)
        ast.parse(after, filename=str(path))
    except (OSError, KeyError, TypeError, SyntaxError, ValueError) as exc:
        return DoctorCheck.fail("patch_binding", "patch does not parse", path=str(path), error=repr(exc))
    return DoctorCheck.pass_("patch_binding", "patch binds and parses", path=str(path))
