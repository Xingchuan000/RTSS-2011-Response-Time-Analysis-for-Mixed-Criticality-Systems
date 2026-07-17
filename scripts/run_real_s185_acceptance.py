"""真实 s185 单命令接入验收。

验收只判断标准入口是否给出合法结果和准确失败原因；没有 authoritative
formal_inputs 时，``UNRESOLVED/AUTHORITATIVE_TARGET_MISSING`` 是合规结果，
绝不以 HOUT 或默认 taskset 补齐输入。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from formal_toolchain.workflow.prove_seed import EXIT_CODES, prove_seed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed-dir", required=True, type=Path)
    parser.add_argument("--code-root", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)
    code, result = prove_seed(seed_dir=args.seed_dir, tree_variant="best_overall",
                              code_root=args.code_root, out=args.out, overwrite=True)
    allowed = set(EXIT_CODES)
    if result.get("result_status") not in allowed:
        print(json.dumps({"result_status": "PROOF_BUNDLE_INVALID",
                          "failure_code": "REAL_S185_INVALID_RESULT"}, ensure_ascii=False))
        return EXIT_CODES["PROOF_BUNDLE_INVALID"]
    print(json.dumps(result, ensure_ascii=False))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
