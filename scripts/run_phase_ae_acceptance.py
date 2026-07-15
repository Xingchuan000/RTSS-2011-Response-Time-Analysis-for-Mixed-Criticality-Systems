#!/usr/bin/env python3
"""Phase A-E synthetic/golden 总验收入口，不接触真实 seed。"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))

from formal_toolchain.adapters.target_factory import build_target
from formal_toolchain.adapters.runtime_config import export_formal_target_config
from formal_toolchain.binding.action_binding import bind_action_runtime
from formal_toolchain.binding.event_runtime_binding import bind_event_runtime
from formal_toolchain.binding.observation_binding import bind_observation_runtime
from formal_toolchain.binding.recovery_binding import bind_recovery_runtime
from formal_toolchain.binding.removal_binding import bind_removal_runtime
from formal_toolchain.binding.quantization_binding import bind_quantization_runtime
from formal_toolchain.adapters.amc_taskset import derive_feature_task_order
from formal_toolchain.conformance.micro_scenarios import run_p0_micro_scenarios
from formal_toolchain.conformance.preflight import preflight_formal_target
from formal_toolchain.core.registry import interface_coverage, load_registry, validate_registry
from formal_toolchain.core.registry import check_registry_migration, registry_fingerprint
from formal_toolchain.verifier.artifact_verifier import verify_certificate
from formal_toolchain.verifier.aggregator import aggregate_phase_ae_local


def main() -> int:
    target = build_target("tests.formal.fixtures.synthetic_p0.target:build_target")
    artifact = ROOT / "tests/formal/fixtures/synthetic_p0"
    preflight = preflight_formal_target(target, artifact)
    config = export_formal_target_config(target)
    registry = load_registry(ROOT / "formal_toolchain/specs/obligation_registry.json")
    validate_registry(registry)
    coverage = interface_coverage(registry, specs_root=ROOT / "formal_toolchain/specs")
    migration = json.loads((ROOT / "formal_toolchain/specs/migration_manifest.json").read_text())
    current_registry_hash = registry_fingerprint(registry)
    if migration.get("registry_fingerprint") != current_registry_hash:
        raise ValueError("migration manifest 未记录当前 registry fingerprint")
    check_registry_migration(current_registry_hash, current_registry_hash, migration)
    certificate = {"artifact_schema_version": "common_certificate_v1", "obligation_id": "X",
                   "obligation_status": "PASS", "certificate_context_hash": "a" * 64,
                   "direct_predecessor_hashes": {}, "checker_id": "acceptance",
                   "checker_version": "v1", "inputs": {}, "witness": {"fixture": "synthetic_p0"},
                   "evidence": [{"kind": "golden", "status": "PASS"}], "failure": None}
    schema = verify_certificate(certificate, schema_name="common.schema.json")
    binders = [bind_event_runtime(ROOT), bind_action_runtime(ROOT), bind_removal_runtime(ROOT),
               bind_recovery_runtime(ROOT)]
    feature_order = derive_feature_task_order(target.feature_names)
    binders.append(bind_observation_runtime(ROOT, artifact / "feature_names.json",
                   ordered_tasks=[task.name for task in target.ordered_tasks],
                   feature_task_order=feature_order))
    binders.append(bind_quantization_runtime(ROOT, artifact / "fixed_point_config.json"))
    micro = run_p0_micro_scenarios(target_available=True)
    phase_result = aggregate_phase_ae_local(binders)
    formal_tests = subprocess.run([sys.executable, "-m", "pytest", "-q", "tests/formal"], cwd=ROOT).returncode
    baseline = subprocess.run([sys.executable, "scripts/run_formal_baseline.py",
                               "--output", str(ROOT / ".phase_ae_baseline.json")], cwd=ROOT).returncode
    result = {"preflight": preflight.get("obligation_status"), "effective_config": config.get("status"),
              "registry_coverage": coverage, "schema": schema.get("status"),
              "binders": [item.get("status") for item in binders], "micro_scenarios": micro.get("status"),
              "formal_tests_returncode": formal_tests, "baseline_returncode": baseline,
              "migration": "PASS", "phase_result": phase_result,
              "workflow_status": "VERIFIED" if phase_result == "PHASE_AE_ACCEPTED" else phase_result,
              "final_safety_claim": "NOT_EVALUATED",
              "unimplemented_later_phases": ["F", "G", "H", "I", "J", "K"]}
    print(json.dumps(result, ensure_ascii=False, indent=2))
    passed = (result["preflight"] == "PASS" and result["effective_config"] == "PASS" and
              not any(coverage.values()) and result["schema"] == "PASS" and
              all(item == "PASS" for item in result["binders"]) and result["micro_scenarios"] == "PASS" and
              formal_tests == 0 and baseline == 0 and phase_result == "PHASE_AE_ACCEPTED")
    (ROOT / ".phase_ae_baseline.json").unlink(missing_ok=True)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
