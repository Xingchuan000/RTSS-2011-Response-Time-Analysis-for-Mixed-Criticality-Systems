"""release-fixed mapping 的独立源码绑定 checker。"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from formal_toolchain.binding.removal_binding import bind_removal_runtime
from formal_toolchain.bridge.job_mapping import verify_parameterized_release_mapping_certificate


def verify_release_mapping(*, candidate_certificate: Mapping[str, Any],
                           source_root: Path,
                           bridge_context_hash: str) -> dict[str, Any]:
    """同时验证参数化公式、源码边界、context hash 和 P0 contract。"""

    if not verify_parameterized_release_mapping_certificate(candidate_certificate):
        return {"status": "FAIL", "route": "PROOF_BUNDLE_INVALID",
                "code": "RELEASE_MAPPING_CERTIFICATE_INVALID"}
    binding = bind_removal_runtime(Path(source_root))
    if binding.get("status") != "PASS":
        return {"status": "FAIL", "route": "MODEL_CONFORMANCE_FAILED",
                "code": "REMOVAL_RUNTIME_BINDING_FAILED", "witness": binding}
    contract = binding.get("p0_contract", {})
    required = {"completion_precedes_deadline_observation": True, "hi_nontruncation": True}
    if any(contract.get(key) is not expected for key, expected in required.items()):
        return {"status": "FAIL", "route": "MODEL_CONFORMANCE_FAILED",
                "code": "RELEASE_MAPPING_SOURCE_CONTRACT_FAILED", "witness": contract}
    certificate_context = candidate_certificate.get("certificate_context_hash")
    if certificate_context != bridge_context_hash:
        return {"status": "FAIL", "route": "PROOF_BUNDLE_INVALID",
                "code": "RELEASE_MAPPING_CONTEXT_MISMATCH"}
    return {"status": "PASS", "route": None, "code": None,
            "witness": {"source_binding": binding, "formula_verified": True}}
