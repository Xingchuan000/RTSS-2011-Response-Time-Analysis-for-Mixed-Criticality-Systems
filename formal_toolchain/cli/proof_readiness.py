"""Machine-readable readiness for the active V10.16 implementation."""

from __future__ import annotations

import importlib.util
import json

from formal_toolchain.v10_1.constants import FRAMEWORK_REVISION, PROOF_ROUTE


def readiness_report() -> dict[str, object]:
    z3_available = importlib.util.find_spec("z3") is not None
    blockers = [] if z3_available else ["PYTHON_Z3_SOLVER_NOT_AVAILABLE"]
    return {
        "schema_version": "v10_1_proof_readiness_v1",
        "proof_route": PROOF_ROUTE,
        "framework_revision": FRAMEWORK_REVISION,
        "proof_pipeline_ready": z3_available,
        "formal_dependency_z3_available": z3_available,
        "implementation_blockers": blockers,
        "implementation_blocker_count": len(blockers),
        "terminal_routes": {
            "BASE_C_AMC_SEM": {
                "ready": True,
                "requires_z3": False,
                "analysis": "exact Zhang-Zheng-Gu 2024 Section 4.1 Equations (4),(6),(11)-(17)",
            },
            "PCSSC": {
                "ready": z3_available,
                "event_graph_required": False,
                "canonical_case_consistent_terminal": True,
                "pre_hi_direct_v10_16_phase_blocks": True,
                "lo_entry_v10_13_refinement": True,
                "deadline_canonical_case_domain": True,
            },
        },
    }


def main() -> int:
    report = readiness_report()
    print(json.dumps(report, ensure_ascii=False))
    return 0 if report["proof_pipeline_ready"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
