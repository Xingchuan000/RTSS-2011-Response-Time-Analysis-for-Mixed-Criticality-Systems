"""Phase L/M 顶层单命令入口。"""

import argparse
import json
from pathlib import Path

from formal_toolchain.workflow.prove_seed import prove_seed
from amc_py.nonvacuity import SUPPORTED_PROFILES, SUPPORTED_DISABLED_GUARDS


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="导入、编译、独立验证并报告一个 seed folder")
    parser.add_argument("--seed-dir", required=True, type=Path)
    parser.add_argument("--tree-variant", required=True,
                        choices=("best_overall", "best_balanced", "best_performance"))
    parser.add_argument("--code-root", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--target-recipe", type=Path)
    parser.add_argument("--proof-route", choices=("protected_prefix", "strict_full"),
                        default="protected_prefix")
    parser.add_argument(
        "--nonvacuity-profile",
        choices=SUPPORTED_PROFILES,
        default="off",
        help="Opt-in non-vacuity experiment profile. Default: off.",
    )
    parser.add_argument(
        "--nonvacuity-disabled-guard",
        action="append",
        default=[],
        choices=SUPPORTED_DISABLED_GUARDS,
        help="Guard to disable for b4_disable_guard; may be repeated.",
    )
    parser.add_argument("--nonvacuity-action-ratio", type=float)
    parser.add_argument("--nonvacuity-min-budget-delta", type=int)
    parser.add_argument("--nonvacuity-controller-overhead-ticks", type=int)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--refresh-phase-k-map",
        action="store_true",
        help="Regenerate the Phase K path map from the current --code-root while freezing the request.",
    )
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    nonvacuity_params = {
        key: value
        for key, value in {
            "disabled_guards": args.nonvacuity_disabled_guard,
            "action_ratio": args.nonvacuity_action_ratio,
            "min_budget_delta": args.nonvacuity_min_budget_delta,
            "controller_overhead_ticks": args.nonvacuity_controller_overhead_ticks,
        }.items()
        if value not in (None, [], ())
    }
    code, result = prove_seed(
        seed_dir=args.seed_dir,
        tree_variant=args.tree_variant,
        code_root=args.code_root,
        out=args.out,
        target_recipe=args.target_recipe,
        overwrite=args.overwrite,
        nonvacuity_profile=args.nonvacuity_profile,
        nonvacuity_params=nonvacuity_params,
        refresh_phase_k_map=args.refresh_phase_k_map,
        proof_route=args.proof_route,
    )
    if args.json:
        print(json.dumps(result, ensure_ascii=False))
    else:
        print(f"{result.get('workflow_status', 'FAILED')}: "
              f"{result.get('result_status', 'PROOF_BUNDLE_INVALID')}")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
