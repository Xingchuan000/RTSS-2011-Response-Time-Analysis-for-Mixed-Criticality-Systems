"""fresh-process verifier CLI。"""

import argparse
import json
from pathlib import Path

from formal_toolchain.verifier.recompute import verify_bundle


EXIT_CODES = {"DEPLOYED_TREE_PROVED": 0, "MODEL_CONFORMANCE_FAILED": 10,
              "POLICY_CONTRACT_VIOLATION": 11, "REFERENCE_CERTIFICATE_FAILED": 12,
              "CONCRETE_TIMING_COUNTEREXAMPLE": 13, "REFERENCE_COUNTEREXAMPLE": 14,
              "UNRESOLVED": 20, "PROOF_BUNDLE_INVALID": 30}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="独立验证 candidate proof bundle")
    parser.add_argument("--request", required=True, type=Path)
    parser.add_argument("--bundle", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        summary = verify_bundle(args.request, args.bundle, args.out)
    except Exception as exc:
        print(json.dumps({"workflow_status": "FAILED", "result_status": "PROOF_BUNDLE_INVALID",
                          "failure_route": "PROOF_BUNDLE_INVALID", "failure_code": "INTERNAL_VERIFY_ERROR",
                          "failure_message": str(exc)}, ensure_ascii=False))
        return 70
    print(json.dumps(summary, ensure_ascii=False))
    return EXIT_CODES.get(str(summary.get("result_status")), 70)

if __name__ == "__main__":
    raise SystemExit(main())
