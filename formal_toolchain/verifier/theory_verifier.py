"""TheoryManifest 的独立校验器。

该 verifier 不读取 candidate、taskset、tree 或 runtime artifact，因此 theorem
库不会吸收 seed-specific 实例结论。
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from formal_toolchain.core.hashing import sha256_object

ALLOWED_LEVELS = frozenset({"DECLARED_AXIOM_TCB", "PUBLISHED_THEORY_TCB", "MACHINE_CHECKED_PROJECT_LEMMA"})
REQUIRED_FIELDS = frozenset({"theorem_id", "exact_statement", "assumptions", "conclusion",
                             "source_reference", "assurance_level", "version",
                             "statement_hash", "assumption_hash"})


def _load(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        data = json.load(stream)
    if not isinstance(data, dict):
        raise ValueError(f"理论对象必须是 object: {path}")
    return data


def verify_theory_library(theory_dir: Path) -> dict[str, Any]:
    """校验 manifest、每个 theorem 的两类 hash 以及必需 theorem 集合。"""
    manifest = _load(theory_dir / "theory_manifest.json")
    policy = _load(theory_dir / "assurance_policy.json")
    policy_levels = set(policy.get("allowed_levels", []))
    if not policy_levels or not policy_levels <= ALLOWED_LEVELS:
        raise ValueError("assurance_policy.allowed_levels 非法")
    if not set(manifest.get("proof_schema_compatible", [])) & {"proof_request_v2", "common_certificate_v1", "common_certificate_v2"}:
        raise ValueError("theory manifest 与当前 proof schema 不兼容")
    registry_path = theory_dir.parents[0] / "specs/obligation_registry.json"
    if manifest.get("proof_schema_version") not in {"common_certificate_v1", "common_certificate_v2"}:
        raise ValueError("theory manifest 未精确绑定当前 proof schema")
    registry_available = registry_path.is_file()
    from formal_toolchain.core.registry import load_registry, registry_fingerprint
    if registry_available and manifest.get("registry_fingerprint") != registry_fingerprint(load_registry(registry_path)):
        raise ValueError("theory manifest 与当前 Registry 不匹配")
    required = manifest.get("required_theorems", [])
    if len(required) != len(set(required)):
        raise ValueError("required_theorems 存在重复 ID")
    statements_dir = theory_dir / "statements"
    objects = [_load(path) for path in sorted(statements_dir.glob("*.json"))]
    hashes = _load(theory_dir / "hashes.json").get("statements", {})
    ids = [obj.get("theorem_id") for obj in objects]
    if len(ids) != len(set(ids)):
        raise ValueError("theorem_id 必须唯一")
    if set(required) - set(ids):
        raise ValueError(f"缺少 required theorem: {sorted(set(required) - set(ids))}")
    for obj in objects:
        if REQUIRED_FIELDS - obj.keys():
            raise ValueError(f"{obj.get('theorem_id')} 缺少 theorem 字段")
        if obj["assurance_level"] not in policy_levels:
            raise ValueError(f"{obj['theorem_id']} assurance level 非法")
        if not str(obj["exact_statement"]).strip() or not str(obj["conclusion"]).strip() or not str(obj["source_reference"]).strip():
            raise ValueError(f"{obj['theorem_id']} 的 statement/conclusion/source_reference 不能为空")
        if not isinstance(obj["assumptions"], list) or not obj["assumptions"]:
            raise ValueError(f"{obj['theorem_id']} assumptions 必须是非空列表")
        if obj["assurance_level"] == "MACHINE_FORMALIZED_KERNEL":
            proof_object = obj.get("proof_object")
            if not isinstance(proof_object, dict) or not proof_object.get("path") or not proof_object.get("sha256"):
                raise ValueError(f"{obj['theorem_id']} 缺少 proof_object")
        statement = {key: obj[key] for key in ("theorem_id", "exact_statement", "conclusion", "source_reference", "assurance_level", "version")}
        assumptions = {"theorem_id": obj["theorem_id"], "assumptions": obj["assumptions"], "version": obj["version"]}
        if obj["statement_hash"] != sha256_object(statement):
            raise ValueError(f"{obj['theorem_id']} statement_hash 不匹配")
        if obj["assumption_hash"] != sha256_object(assumptions):
            raise ValueError(f"{obj['theorem_id']} assumption_hash 不匹配")
        # 防止把实例结论伪装成通用理论输入。
        # 通用术语如 priority、taskset 不构成实例结论；只拒绝带实例值/字段的
        # 形式，例如 ``seed 185``、``taskset_seed`` 或具体 artifact 字段。
        forbidden = (r"\bseed\s*\d+", "taskset_seed", "priority_order",
                     "integer_tree.json", "candidate_envelope:", "runtime_config:")
        text = json.dumps(obj, ensure_ascii=False).lower()
        if obj["assurance_level"] == "DECLARED_AXIOM_TCB" and any(
                (re.search(word, text) if word.startswith(r"\b") else word in text)
                for word in forbidden):
            raise ValueError(f"{obj['theorem_id']} 含有 seed-specific TCB 数据")
        declared = hashes.get(obj["theorem_id"])
        if declared != {"statement_hash": obj["statement_hash"], "assumption_hash": obj["assumption_hash"]}:
            raise ValueError(f"{obj['theorem_id']} 与 theory/hashes.json 不一致")
    if not registry_available:
        raise ValueError("theory manifest 校验缺少当前 Registry")
    return {"status": "PASS", "library_version": manifest.get("library_version"),
            "theorem_count": len(objects), "theorem_ids": sorted(ids)}


def write_theory_certificate(theory_dir: Path, output_path: Path) -> dict[str, Any]:
    """将独立校验结果落盘为标准 certificate envelope。"""
    result = verify_theory_library(theory_dir)
    manifest = _load(Path(theory_dir) / "theory_manifest.json")
    certificate = {
        "artifact_schema_version": "common_certificate_v1",
        "obligation_id": "THEORY_LIBRARY_VERSION",
        "obligation_status": "PASS",
        "certificate_context_hash": sha256_object({"theory_manifest": manifest}),
        # THEORY_LIBRARY_VERSION 是理论输入根；其 predecessor 集合必须与
        # registry 的 depends_on 精确一致，不能凭空增加未注册的 hash 名称。
        "direct_predecessor_hashes": {},
        "checker_id": "formal_toolchain.verifier.theory_verifier",
        "checker_version": "phase-c-v1",
        "inputs": {"library_version": result["library_version"], "theorem_count": result["theorem_count"]},
        "witness": {"theorem_ids": result["theorem_ids"]},
        "evidence": [{"kind": "independent_theory_recomputation", "status": "PASS"}],
        "failure": None,
    }
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(certificate, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return certificate
