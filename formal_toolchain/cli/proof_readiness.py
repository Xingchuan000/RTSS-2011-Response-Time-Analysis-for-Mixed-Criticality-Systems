"""Print machine-readable readiness of the single V9.2 proof route."""

from __future__ import annotations

import importlib.util
import json

from formal_toolchain.v9_2.constants import PROOF_ROUTE
from formal_toolchain.v9_2.readiness import blocker_rows, proof_pipeline_ready


def readiness_report() -> dict[str, object]:
    z3_available = importlib.util.find_spec("z3") is not None
    blockers = blocker_rows()
    return {
        "schema_version": "v9_2_proof_readiness_v1",
        "proof_route": PROOF_ROUTE,
        "formal_dependency_z3_available": z3_available,
        "proof_pipeline_ready": proof_pipeline_ready() and z3_available,
        "implementation_blockers": blockers,
        "implementation_blocker_count": len(blockers),
        "test_level": (
            "FORMAL_PROOF_READY"
            if proof_pipeline_ready() and z3_available
            else "ENCODER_DEVELOPMENT_ONLY"
        ),
    }


def main() -> int:
    report = readiness_report()
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["proof_pipeline_ready"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
