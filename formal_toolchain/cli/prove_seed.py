"""Top-level V10.1 proof command.  Module entry is intentionally unchanged."""

import argparse
import json
from pathlib import Path

from formal_toolchain.workflow.prove_seed import prove_seed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="prove one seed with the V10.1 BASE/PCSSC route")
    parser.add_argument("--seed-dir", required=True, type=Path)
    parser.add_argument("--tree-variant", default="best_overall",
                        choices=("best_overall", "best_balanced", "best_performance"))
    parser.add_argument("--code-root", type=Path, default=Path.cwd())
    parser.add_argument("--out", type=Path, help="default: seed-dir/.formal_proof_v10_1")
    parser.add_argument("--target-recipe", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--solver-timeout-ms", type=int, default=120_000,
                        help="fresh Z3 timeout for V10.1 source/controller obligations")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    output_dir = args.out or (args.seed_dir / ".formal_proof_v10_1")
    code, result = prove_seed(
        seed_dir=args.seed_dir,
        tree_variant=args.tree_variant,
        code_root=args.code_root,
        out=output_dir,
        target_recipe=args.target_recipe,
        overwrite=args.overwrite,
        solver_timeout_ms=args.solver_timeout_ms,
    )
    if args.json:
        print(json.dumps(result, ensure_ascii=False))
    else:
        print(f"{result.get('workflow_status', 'FAILED')}: {result.get('result_status', 'PROOF_BUNDLE_INVALID')}")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
