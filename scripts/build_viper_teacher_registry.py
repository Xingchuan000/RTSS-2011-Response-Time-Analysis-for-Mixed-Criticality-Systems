"""构建 teacher registry CSV。"""

from __future__ import annotations

import argparse
import csv
import sys as _sys
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from amc_py.viper.registry import build_teacher_registry_row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher-root", type=Path, required=True)
    parser.add_argument("--seeds", type=str, required=True)
    parser.add_argument("--checkpoint-name", type=str, default="model_best_qos_recovery_stable.pt")
    parser.add_argument("--fallback-checkpoint-name", type=str, default="model_final.pt")
    parser.add_argument("--run-dir-template", type=str, default="r0_s{seed}")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--teacher-id-prefix", type=str, default="teacher")
    parser.add_argument("--runtime-semantics", type=str, default="AMC_PLUS")
    parser.add_argument("--c-amc-sem-xf", type=float, default=0.5)
    parser.add_argument("--reward-mode", type=str, default="mendes")
    parser.add_argument("--action-space", type=str, default="single")
    parser.add_argument("--observation-mode", type=str, default="v11_full_10d")
    parser.add_argument("--agent-period", type=int, default=1000)
    parser.add_argument("--budget-increase-ratio", type=float, default=0.10)
    parser.add_argument("--budget-decrease-ratio", type=float, default=0.05)
    parser.add_argument("--budget-floor-ratio", type=float, default=0.0)
    parser.add_argument("--forbid-decreasing-hi-budgets", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--enable-deploy-cap-mask", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--deploy-cap-mask-ratio", type=float, default=4.0)
    parser.add_argument("--deploy-cap-mask-criticality", type=str, default="lo")
    args = parser.parse_args()
    rows: list[dict[str, object]] = []
    for seed_text in args.seeds.split(","):
        seed = int(seed_text.strip())
        candidate_dir = args.teacher_root / args.run_dir_template.format(seed=seed)
        if not candidate_dir.exists():
            candidate_dir = args.teacher_root / str(seed)
        if not candidate_dir.exists():
            raise FileNotFoundError(f"teacher 输出目录不存在: {args.teacher_root} (seed={seed})")
        model_path = candidate_dir / args.checkpoint_name
        if not model_path.exists():
            model_path = candidate_dir / args.fallback_checkpoint_name
        config_path = candidate_dir / "config.json"
        if not config_path.exists():
            print(
                f"warning: config.json 不存在，将在 CSV 中写空值: {config_path}",
                file=_sys.stderr,
            )
        rows.append(
            build_teacher_registry_row(
                teacher_id=f"{args.teacher_id_prefix}_{seed}",
                taskset_seed=seed,
                model_path=model_path,
                config_path=(config_path if config_path.exists() else None),
                train_output_dir=candidate_dir,
                runtime_semantics=args.runtime_semantics,
                c_amc_sem_xf=args.c_amc_sem_xf,
                reward_mode=args.reward_mode,
                action_space=args.action_space,
                observation_mode=args.observation_mode,
                agent_period=args.agent_period,
                budget_increase_ratio=args.budget_increase_ratio,
                budget_decrease_ratio=args.budget_decrease_ratio,
                budget_floor_ratio=args.budget_floor_ratio,
                forbid_decreasing_hi_budgets=args.forbid_decreasing_hi_budgets,
                enable_deploy_cap_mask=args.enable_deploy_cap_mask,
                deploy_cap_mask_ratio=args.deploy_cap_mask_ratio,
                deploy_cap_mask_criticality=args.deploy_cap_mask_criticality,
                checkpoint_kind=("primary" if model_path.name == args.checkpoint_name else "fallback"),
            )
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
