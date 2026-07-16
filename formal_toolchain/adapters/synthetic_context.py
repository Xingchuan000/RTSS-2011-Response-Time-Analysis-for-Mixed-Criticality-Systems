"""Phase F-H synthetic target 的唯一 canonical context。"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from formal_toolchain.core.hashing import sha256_file, sha256_object
from formal_toolchain.adapters.runtime_config import export_effective_config


def build_synthetic_context(target: Any, inventory: Mapping[str, Any], domain: Mapping[str, Any]) -> dict[str, Any]:
    """把 F/G/H 共同消费的对象序列化为一个不可由调用方任意指定的 context。"""
    artifact_hashes = dict(inventory["files"])
    fixed_hash = str(inventory["fixed_point_config_hash"])
    task_records = [
        {"name": task.name, "criticality": task.criticality.value, "period": task.period,
         "deadline": task.deadline, "c_lo": task.c_lo, "c_hi": task.c_hi}
        for task in target.ordered_tasks
    ]
    config = export_effective_config(target.runtime_config, target.environment)
    body = {"schema_version": "synthetic_fh_context_v1", "tasks": task_records,
            "effective_config": config, "artifact_hashes": artifact_hashes,
            "fixed_point_semantic_hash": fixed_hash,
            "budget_domain_hash": sha256_object(domain)}
    return {"context_hash": sha256_object(body), "body": body}

