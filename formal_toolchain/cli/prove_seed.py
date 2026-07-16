"""Phase L/M 顶层单命令入口。"""

import argparse
import json
from pathlib import Path

from formal_toolchain.workflow.prove_seed import prove_seed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="导入、编译、独立验证并报告一个 seed folder")
    parser.add_argument("--seed-dir", required=True, type=Path)
    parser.add_argument("--tree-variant", required=True,
                        choices=("best_overall", "best_balanced", "best_performance"))
    parser.add_argument("--code-root", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--target-recipe", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    code, result = prove_seed(seed_dir=args.seed_dir, tree_variant=args.tree_variant,
                               code_root=args.code_root, out=args.out,
                               target_recipe=args.target_recipe, overwrite=args.overwrite)
    if args.json:
        print(json.dumps(result, ensure_ascii=False))
    else:
        print(f"{result.get('workflow_status', 'FAILED')}: "
              f"{result.get('result_status', 'PROOF_BUNDLE_INVALID')}")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
