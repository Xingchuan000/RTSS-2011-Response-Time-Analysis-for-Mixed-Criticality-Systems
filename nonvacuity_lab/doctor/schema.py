from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import StrEnum


class DoctorStatus(StrEnum):
    PASS = "PASS"
    FAIL = "FAIL"
    SKIP = "SKIP"


@dataclass(frozen=True)
class DoctorCheck:
    check_id: str
    status: DoctorStatus
    summary: str
    details: dict

    @classmethod
    def pass_(cls, check_id, summary, **details): return cls(check_id, DoctorStatus.PASS, summary, details)
    @classmethod
    def fail(cls, check_id, summary, **details): return cls(check_id, DoctorStatus.FAIL, summary, details)
    @classmethod
    def skip(cls, check_id, summary, **details): return cls(check_id, DoctorStatus.SKIP, summary, details)


@dataclass(frozen=True)
class DoctorReceipt:
    schema_version: str
    campaign_id: str
    config_sha256: str
    source_root_sha256: str
    overall_status: DoctorStatus
    checks: tuple[DoctorCheck, ...]

    def to_dict(self):
        return {**asdict(self), "overall_status": self.overall_status.value, "checks": [{**asdict(c), "status": c.status.value} for c in self.checks]}
