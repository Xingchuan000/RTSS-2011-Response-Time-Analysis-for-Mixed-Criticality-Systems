"""V9.1 fresh-process verifier CLI."""

import argparse
import json
from pathlib import Path

from formal_toolchain.v9_1.constants import (
    RESULT_CONCRETE_COUNTEREXAMPLE, RESULT_INVALID, RESULT_PROVED, RESULT_UNRESOLVED,
)
from formal_toolchain.v9_1.verifier import verify_bundle_v9_1

EXIT_CODES = {
    RESULT_PROVED: 0,
    RESULT_CONCRETE_COUNTEREXAMPLE: 13,
    RESULT_UNRESOLVED: 20,
    RESULT_INVALID: 30,
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="independently verify a V9.1 candidate proof bundle")
    parser.add_argument("--request", required=True, type=Path)
    parser.add_argument("--bundle", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--timeout-ms", type=int, default=120_000,
                        help="fresh Z3 timeout per proof obligation")
    parser.add_argument("--max-boot-replay-ticks", type=int, default=2_000,
                        help="maximum boot-prefix ticks used only for SAT witness classification")
    args = parser.parse_args(argv)
    try:
        summary = verify_bundle_v9_1(
            args.request, args.bundle, args.out, source_root=args.source_root,
            timeout_ms=args.timeout_ms, max_boot_replay_ticks=args.max_boot_replay_ticks,
        )
    except (OSError, ValueError, KeyError) as exc:
        print(json.dumps({"workflow_status": "FAILED", "result_status": RESULT_INVALID,
                          "failure_route": RESULT_INVALID, "failure_code": "V9_1_VERIFIER_INPUT_ERROR",
                          "failure_message": str(exc)}, ensure_ascii=False))
        return 30
    print(json.dumps(summary, ensure_ascii=False))
    return EXIT_CODES.get(str(summary.get("result_status")), 30)


if __name__ == "__main__":
    raise SystemExit(main())
