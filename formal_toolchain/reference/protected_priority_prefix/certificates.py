from __future__ import annotations

from typing import Any, Mapping

from formal_toolchain.core.artifact import obligation_certificate
from formal_toolchain.core.obligation_ids import (
    PROTECTED_PRIORITY_PREFIX_PARTITION, SATURATED_PROTECTED_PREFIX_REFERENCE,
)

from .types import ProtectedPrefixBuildResult


def build_partition_certificate(result: ProtectedPrefixBuildResult, *, context_hash: str) -> dict[str, Any]:
    return obligation_certificate(
        obligation_id=PROTECTED_PRIORITY_PREFIX_PARTITION, status="PASS",
        context_hash=context_hash, inputs={"full_taskset_fingerprint": result.full_taskset_fingerprint},
        witness=dict(result.partition_witness), checker_id=__name__, checker_version="protected-prefix-v1",
    )


def build_saturation_certificate(result: ProtectedPrefixBuildResult, *, context_hash: str) -> dict[str, Any]:
    return obligation_certificate(
        obligation_id=SATURATED_PROTECTED_PREFIX_REFERENCE, status="PASS",
        context_hash=context_hash,
        inputs={"prefix_taskset_fingerprint": result.prefix_taskset.to_dict()["fingerprint"]},
        witness=dict(result.saturation_witness), checker_id=__name__, checker_version="protected-prefix-v1",
    )


def verify_construction_witness(result: ProtectedPrefixBuildResult, witness: Mapping[str, Any]) -> bool:
    return (witness.get("full_fingerprint") == result.full_taskset_fingerprint
            and witness.get("cutoff_task_name") == result.cutoff_task_name
            and witness.get("cutoff_priority_index") == result.cutoff_priority_index
            and list(result.protected_task_names) == witness.get("protected_task_names")
            and list(result.tail_task_names) == witness.get("tail_task_names"))
