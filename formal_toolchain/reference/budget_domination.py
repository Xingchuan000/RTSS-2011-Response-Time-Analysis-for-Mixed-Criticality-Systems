from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping

from formal_toolchain.core.artifact import obligation_certificate
from formal_toolchain.core.hashing import sha256_object


@dataclass(frozen=True, slots=True)
class DominationRow:
    task_name: str
    criticality: str
    code_c_lo: int
    code_c_hi: int
    envelope_upper: int | None
    envelope_lower: int | None
    reference_c_lo: int
    reference_c_hi: int
    normal_release_bound: int
    degraded_release_bound: int | None
    normal_release_dominated: bool
    degraded_release_dominated: bool
    hi_release_exact: bool
    hi_normal_bound: int | None = None
    hi_abnormal_switch_bound: int | None = None
    lo_primary_normal_bound: int | None = None
    lo_same_batch_switch_bound: int | None = None
    lo_degraded_hi_mode_bound: int | None = None
    hi_normal_dominated: bool | None = None
    hi_abnormal_dominated: bool | None = None
    lo_primary_normal_dominated: bool | None = None
    lo_same_batch_dominated: bool | None = None
    lo_degraded_dominated: bool | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_budget_to_reference_domination(
    *,
    reference_taskset: Mapping[str, Any],
    certified_envelope: Mapping[str, Any],
    reference_context_hash: str,
    direct_predecessor_hashes: Mapping[str, str],
) -> dict[str, Any]:
    upper = certified_envelope.get("upper", {})
    lower = certified_envelope.get("lower", {})
    rows: list[DominationRow] = []

    tasks = reference_taskset.get("tasks")
    if not isinstance(tasks, list) or not tasks:
        raise ValueError("reference taskset missing")

    for task in tasks:
        name = str(task["name"])
        criticality = str(task["criticality"])
        code_c_lo = int(task["code_c_lo"])
        code_c_hi = int(task["code_c_hi"])
        reference_c_lo = int(task["c_lo"])
        reference_c_hi = int(task["c_hi"])

        if criticality == "LO":
            b_upper = int(upper[name])
            normal_bound = min(code_c_lo, b_upper + 1)
            degraded = int(task["degraded_cost"])
            lo_primary_normal = min(code_c_lo, b_upper + 1)
            lo_same_batch_switch = min(code_c_lo, b_upper + 1)
            lo_degraded_hi = degraded
            row = DominationRow(
                task_name=name,
                criticality=criticality,
                code_c_lo=code_c_lo,
                code_c_hi=code_c_hi,
                envelope_upper=b_upper,
                envelope_lower=None,
                reference_c_lo=reference_c_lo,
                reference_c_hi=reference_c_hi,
                normal_release_bound=normal_bound,
                degraded_release_bound=degraded,
                normal_release_dominated=normal_bound <= reference_c_lo,
                degraded_release_dominated=degraded <= reference_c_lo,
                hi_release_exact=True,
                lo_primary_normal_bound=lo_primary_normal,
                lo_same_batch_switch_bound=lo_same_batch_switch,
                lo_degraded_hi_mode_bound=lo_degraded_hi,
                lo_primary_normal_dominated=lo_primary_normal <= reference_c_lo,
                lo_same_batch_dominated=lo_same_batch_switch <= reference_c_lo,
                lo_degraded_dominated=lo_degraded_hi <= reference_c_lo,
            )
        elif criticality == "HI":
            b_lower = int(lower[name])
            hi_normal = code_c_lo
            hi_abnormal_switch = code_c_hi
            row = DominationRow(
                task_name=name,
                criticality=criticality,
                code_c_lo=code_c_lo,
                code_c_hi=code_c_hi,
                envelope_upper=None,
                envelope_lower=b_lower,
                reference_c_lo=reference_c_lo,
                reference_c_hi=reference_c_hi,
                normal_release_bound=code_c_lo,
                degraded_release_bound=None,
                normal_release_dominated=code_c_lo <= reference_c_lo,
                degraded_release_dominated=True,
                hi_release_exact=(reference_c_hi == code_c_hi),
                hi_normal_bound=hi_normal,
                hi_abnormal_switch_bound=hi_abnormal_switch,
                hi_normal_dominated=hi_normal <= reference_c_lo,
                hi_abnormal_dominated=hi_abnormal_switch <= reference_c_hi,
            )
        else:
            raise ValueError(f"unknown criticality: {criticality}")
        rows.append(row)

    def _row_pass(row: DominationRow) -> bool:
        if row.criticality == "HI":
            return all([
                row.hi_normal_dominated is not False,
                row.hi_abnormal_dominated is not False,
                row.hi_release_exact,
            ])
        else:
            return all([
                row.lo_primary_normal_dominated is not False,
                row.lo_same_batch_dominated is not False,
                row.lo_degraded_dominated is not False,
            ])

    status = "PASS" if rows and all(_row_pass(row) for row in rows) else "FAIL"

    witness = {
        "schema_version": "budget_reference_domination_v1",
        "reference_taskset_fingerprint": reference_taskset.get("fingerprint"),
        "rows": [row.to_dict() for row in rows],
        "task_count": len(rows),
    }
    witness["domination_hash"] = sha256_object(witness)

    return obligation_certificate(
        obligation_id="BUDGET_ENVELOPE_TO_REFERENCE_DOMINATION",
        status=status,
        context_hash=reference_context_hash,
        inputs={
            "reference_taskset_fingerprint": reference_taskset.get("fingerprint"),
            "certified_envelope_hash": certified_envelope.get("artifact_hash")
            or sha256_object(certified_envelope),
        },
        witness=witness,
        direct_predecessor_hashes=dict(direct_predecessor_hashes),
        checker_id=__name__,
        checker_version="budget-reference-domination-v1",
        failure=None if status == "PASS" else {
            "route": "MODEL_CONFORMANCE_FAILED",
            "code": "BUDGET_REFERENCE_DOMINATION_FAILED",
        },
    )
