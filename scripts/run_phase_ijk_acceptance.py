"""Phase I-K 正式 synthetic 编排入口。

Phase F-H certificate 在本次进程中由 fresh verifier 生成；Phase K proof
则由当前源码 branch map 和固定模板现场编译，fixture 不保存 PASS proof。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import tempfile
import subprocess
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from formal_toolchain.bridge.deadline_removal import (
    build_release_fixed_removal_certificate, map_release_fixed_job,
    verify_release_fixed_removal_certificate,
)
from formal_toolchain.bridge.job_mapping import (
    build_parameterized_release_mapping_certificate,
    verify_parameterized_release_mapping_certificate,
)
from formal_toolchain.core.hashing import sha256_object
from formal_toolchain.core.canonical_json import canonical_dumps
from formal_toolchain.bridge.runtime_branch_map import build_runtime_branch_map
from formal_toolchain.bridge.compile_bridge import compile_phase_k
from formal_toolchain.bridge.p0_case_manifest import p0_case_manifest_hash
from formal_toolchain.reference.rta_production import protected_hi_rta
from formal_toolchain.reference.rta_replay import replay_rta
from formal_toolchain.reference.recurring_hi import build_recurring_hi_instances
from formal_toolchain.reference.protected_hi import protected_hi_safety_corollary
from formal_toolchain.reference.task_mapping import build_reference_taskset
from formal_toolchain.verifier.reference_mapping_verifier import verify_reference_mapping
from formal_toolchain.adapters.target_factory import build_target


def _unresolved(code: str, **details: Any) -> tuple[int, dict[str, Any]]:
    return 1, {"workflow_status": "UNRESOLVED", "phase": "I-K",
               "phase_result": "PHASE_IJK_UNRESOLVED",
               "final_safety_claim": "NOT_EVALUATED",
               "unimplemented_later_phases": ["L", "M"],
               "failure": {"code": code, "route": "UNRESOLVED", **details}}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixture", default="synthetic_p0")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)
    root = ROOT
    fixture = root / "tests/formal/fixtures" / args.fixture
    required = ("target_recipe.json", "phase_ijk_inputs.json", "phase_k_case_map.json")
    if not fixture.is_dir():
        code, result = _unresolved("FIXTURE_NOT_FOUND")
    elif any(not (fixture / name).is_file() for name in required):
        code, result = _unresolved("AUTHORITATIVE_PHASE_IK_INPUT_MISSING",
                                   required=list(required))
    else:
        try:
            recipe = json.loads((fixture / "target_recipe.json").read_text(encoding="utf-8"))
            inputs = json.loads((fixture / "phase_ijk_inputs.json").read_text(encoding="utf-8"))
            # Phase F-H certificate 必须在本次正式流程中由 fresh verifier 生成；
            # 不把预先制作的 envelope 当作 Phase K 正向输入。
            with tempfile.TemporaryDirectory(prefix="phase_fh_") as fresh_dir:
                fresh = subprocess.run([sys.executable, str(ROOT / "scripts/run_phase_fh_acceptance.py"),
                                        "--fixture", args.fixture, "--out", fresh_dir],
                                       cwd=ROOT, capture_output=True, text=True, check=False)
                if fresh.returncode != 0:
                    raise ValueError("fresh Phase F-H verifier 未通过: " + fresh.stdout[-1000:])
                envelope = json.loads((Path(fresh_dir) / "verified/certified_envelope.json").read_text(encoding="utf-8"))
                envelope_certificate = json.loads((Path(fresh_dir) / "verified/certified_envelope_certificate.json").read_text(encoding="utf-8"))
            preservation = envelope.get("preservation_certificate")
            if (envelope.get("preservation_certificate_hash") != sha256_object(envelope_certificate)
                    or preservation != envelope_certificate
                    or envelope_certificate.get("obligation_status") != "PASS"
                    or envelope_certificate.get("evidence", [{}])[0].get("fresh_process") is not True):
                raise ValueError("certified envelope 未绑定 fresh Phase F-H verifier certificate")
            case_map = json.loads((fixture / "phase_k_case_map.json").read_text(encoding="utf-8"))
            if case_map.get("source_hash") != inputs.get("source_hash"):
                raise ValueError("Phase K case map source hash mismatch")
            if case_map.get("schema_version") != "phase_k_transition_path_map_v1" or not isinstance(case_map.get("paths"), dict):
                raise ValueError("Phase K case map schema invalid")
            expected_theorems = {
                "casewise_simulation": "CASEWISE_SIMULATION_IMPLIES_PREFIX_REFINEMENT",
                "prefix_extension": "REFERENCE_PREFIX_EXTENSION",
                "bad_prefix_reflection": "FINITE_HI_BAD_PREFIX_REFLECTION",
            }
            if inputs.get("theorem_ids") != expected_theorems:
                raise ValueError("Phase K theorem_ids 必须精确绑定 theory manifest")
            target = build_target(recipe["factory"], recipe.get("kwargs", {}))
            reference = build_reference_taskset(
                target.ordered_tasks, inputs["budget_by_task"], xf=inputs["xf"],
                certified_envelope=envelope,
                semantic_context_hash=inputs["semantic_context_hash"],
                effective_runtime_config_hash=inputs["effective_runtime_config_hash"],
            )
            mapping = verify_reference_mapping(
                reference=reference, ordered_tasks=target.ordered_tasks,
                budget_by_task=inputs["budget_by_task"], certified_envelope=envelope,
                xf=inputs["xf"], semantic_context_hash=inputs["semantic_context_hash"],
                effective_runtime_config_hash=inputs["effective_runtime_config_hash"],
            )
            if mapping.get("obligation_status") != "PASS":
                code, result = _unresolved("REFERENCE_MAPPING_FAILED", mapping=mapping)
            else:
                rta = protected_hi_rta(reference)
                replay = replay_rta(reference, rta)
                if rta.get("obligation_status") != "PASS" or replay.get("status") != "PASS":
                    code, result = _unresolved("RTA_OR_REPLAY_FAILED", rta=rta, replay=replay)
                else:
                    recurring = build_recurring_hi_instances(reference, rta_certificate=rta)
                    corollary = protected_hi_safety_corollary(recurring)
                    branch_map = build_runtime_branch_map(
                        root, source_hash=inputs["source_hash"],
                        path_map=case_map,
                    )
                    if corollary.get("status") != "PASS" or branch_map.get("status") != "PASS":
                        code, result = _unresolved("THEOREM_OR_BRANCH_BINDING_UNRESOLVED",
                                                   corollary=corollary, branch_map=branch_map)
                    else:
                        parameterized = build_parameterized_release_mapping_certificate(
                            source_context_hash=reference.source_context_hash)
                        if not verify_parameterized_release_mapping_certificate(parameterized):
                            code, result = _unresolved("PARAMETERIZED_RELEASE_MAPPING_INVALID")
                        else:
                            finite = None
                            if "release_mappings" in inputs:
                                mappings = [map_release_fixed_job(**row) for row in inputs["release_mappings"]]
                                finite_certificate = build_release_fixed_removal_certificate(
                                    mappings, source_context_hash=reference.source_context_hash)
                                finite = {"certificate": finite_certificate,
                                          "verified": verify_release_fixed_removal_certificate(finite_certificate)}
                            bridge_context_hash = sha256_object({
                                "schema_version": "bridge_context_v1",
                                "reference_context_hash": reference.source_context_hash,
                                "source_hash": branch_map["source_hash"],
                                "branch_map_hash": branch_map["path_map_hash"],
                                "p0_case_manifest_hash": p0_case_manifest_hash(),
                            })
                            if inputs.get("bridge_context_hash") != bridge_context_hash:
                                raise ValueError("bridge_context_hash 与当前 reference/branch/context 不一致")
                            bridge = compile_phase_k(source_root=root, branch_map=branch_map,
                                                     reference_taskset=reference.to_dict(),
                                                     bridge_context_hash=bridge_context_hash)
                            if bridge.get("status") == "PASS":
                                code, result = 0, {"workflow_status": "ACCEPTED", "phase": "I-K",
                                    "phase_result": "PHASE_IJK_ACCEPTED", "final_safety_claim": "NOT_EVALUATED",
                                    "mapping": mapping, "rta": rta, "replay": replay,
                                    "recurring": recurring, "corollary": corollary,
                                    "branch_map": branch_map, "parameterized_release_mapping": parameterized,
                                    "finite_boundary_evidence": finite, "bridge": bridge}
                            else:
                                code, result = _unresolved("PARAMETERIZED_BRIDGE_COMPILATION_FAILED",
                                    parameterized_release_mapping=parameterized,
                                    finite_boundary_evidence=finite, bridge=bridge,
                                    required=["source_code", "phase_k_case_map", "z3"])
        except (KeyError, TypeError, ValueError, OSError) as exc:
            code, result = _unresolved("PHASE_IK_INPUT_INVALID", message=str(exc))
    if args.out is not None:
        output_file = args.out if args.out.suffix == ".json" else args.out / "phase_ijk_result.json"
        output_dir = output_file.parent
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file.write_text(canonical_dumps(result), encoding="utf-8")
        bridge = result.get("bridge")
        if isinstance(bridge, dict) and bridge.get("status") == "PASS":
            generated = {
                "branch_map.json": result["branch_map"],
                "transition_case_proofs.json": bridge["transition_cases"],
                "base_relation_certificate.json": bridge["prerequisites"]["base_relation"],
                "same_timestamp_closure_certificate.json": bridge["prerequisites"]["same_timestamp"],
                "positive_time_service_certificate.json": bridge["prerequisites"]["positive_time"],
                "controller_postclosure_certificate.json": bridge["prerequisites"]["controller_postclosure"],
                "event_projection_certificate.json": bridge["prerequisites"]["event_projection"],
                "closed_prefix_refinement_certificate.json": bridge["closed_prefix"],
                "reference_prefix_extension_certificate.json": bridge["reference_extension"],
                "hi_bad_prefix_reflection_certificate.json": bridge["bad_prefix_reflection"],
            }
            for name, artifact in generated.items():
                (output_dir / name).write_text(canonical_dumps(artifact), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
