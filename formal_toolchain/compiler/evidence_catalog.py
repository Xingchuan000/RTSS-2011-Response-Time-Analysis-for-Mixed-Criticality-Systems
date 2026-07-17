"""candidate compiler 的 evidence catalog。

该 catalog 只描述 compiler 产出的原始对象名称，不包含 verifier 的 checker
实现；两个 catalog 物理分离是逻辑独立性的最低边界。
"""

from __future__ import annotations


COMPILER_EVIDENCE_KEYS = {
    "TREE_WELLFORMEDNESS": "TREE",
    "LEAF_GUARD_PARTITION": "LEAF_GUARD_PARTITION",
    "FEATURE_QUANTIZATION": "QUANTIZATION",
    "ACTION_TRANSITION": "ACTION",
    "MASK_FALLBACK": "MASK",
    "EXECUTABLE_POLICY_SEMANTICS": "EXECUTABLE",
    "CANDIDATE_ENVELOPE": "CANDIDATE",
    "COMMON_TRANSITION_PRESERVATION": "COMMON",
    "DEPLOYED_POLICY_PRESERVATION": "DEPLOYED",
    "BUDGET_DOMAIN": "DOMAIN",
    "CERTIFIED_ENVELOPE": "CERTIFIED",
    "CODE_REFERENCE_UPPER_BOUND_MAPPING": "MAPPING",
    "REFERENCE_TASKSET": "REFERENCE",
    "PROTECTED_HI_RTA_ARITHMETIC": "RTA_COMPOSITE",
    "PER_HI_TASK_INDUCTIVE_WCRT": "RECURRING",
    "PROTECTED_HI_SAFETY_COROLLARY": "COROLLARY",
}


def evidence_key_for(obligation_id: str) -> str | None:
    """返回该 obligation 的专属 key；结构和 bridge 不走诊断回退。"""

    if obligation_id in {"ARTIFACT_MANIFEST", "COMPONENT_CONTEXT_INTEGRITY",
                          "DIRECT_PREDECESSOR_HASHES", "STATUS_EVIDENCE",
                          "OUTER_BUNDLE_ROOT", "INDEPENDENT_BUNDLE_VERIFICATION",
                          "CLAIM_AGGREGATION_RESULT",
                          "CLOSED_PREFIX_REFINEMENT", "REFERENCE_PREFIX_EXTENSION",
                          "HI_BAD_CLOSED_PREFIX_REFLECTION"}:
        return None
    return COMPILER_EVIDENCE_KEYS.get(obligation_id, obligation_id)
