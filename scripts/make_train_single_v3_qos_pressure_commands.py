"""根据 QoS pressure manifest 生成 single_v3 训练命令脚本。"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def parse_seed_spec(raw: str) -> str:
    """保留原字符串形式，直接写回命令参数。"""

    return raw.strip()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=str, required=True)
    parser.add_argument("--output-script", type=str, required=True)
    parser.add_argument("--output-dir-prefix", type=str, required=True)
    parser.add_argument("--episodes", type=int, default=120)
    parser.add_argument("--end-time", type=int, default=1000000)
    parser.add_argument("--agent-period", type=int, default=50000)
    parser.add_argument("--validation-seeds", type=str, default="200:229")
    parser.add_argument("--validation-end-time", type=int, default=1000000)
    parser.add_argument("--qos-stable-mode-delta", type=float, default=0.05)
    return parser


def main() -> None:
    args = build_parser().parse_args()

    with Path(args.manifest).open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError("manifest 为空")

    first = rows[0]
    required_mc_fields = [
        "mc_fairgen_mode",
        "mc_fairgen_num_tasks",
        "mc_fairgen_hi_ratio",
        "mc_fairgen_period_source",
        "mc_fairgen_period_scale",
        "mc_fairgen_u_hi_lo_min",
        "mc_fairgen_u_hi_lo_max",
        "mc_fairgen_u_hi_hi_min",
        "mc_fairgen_u_hi_hi_max",
        "mc_fairgen_u_lo_lo_min",
        "mc_fairgen_u_lo_lo_max",
        "mc_fairgen_hi_budget_rho_min",
        "mc_fairgen_hi_budget_rho_max",
        "mc_fairgen_lo_budget_rho_min",
        "mc_fairgen_lo_budget_rho_max",
        "mc_fairgen_hi_overrun_prob",
        "mc_fairgen_lo_overrun_prob",
        "mc_fairgen_hi_overrun_factor_min",
        "mc_fairgen_hi_overrun_factor_max",
        "mc_fairgen_lo_overrun_factor_min",
        "mc_fairgen_lo_overrun_factor_max",
    ]
    missing = [field for field in required_mc_fields if field not in first]
    if missing:
        print(f"WARNING: manifest 缺少字段，将使用 train_dqn_amc.py 默认值: {', '.join(missing)}", flush=True)

    output_script = Path(args.output_script)
    output_script.parent.mkdir(parents=True, exist_ok=True)

    lines: list[str] = ["#!/usr/bin/env bash", "set -euo pipefail", ""]
    for row in rows:
        seed = int(float(row["candidate_seed"]))
        out_dir = f"{args.output_dir_prefix}_seed{seed}_e{args.episodes}_inc0025_dec0015_floor09"

        cmd_parts = [
            "KMP_DUPLICATE_LIB_OK=TRUE conda run --no-capture-output -n amc-repro env PYTHONPATH=. python -u scripts/train_dqn_amc.py",
            "--workload mc_fairgen",
            f"--fixed-taskset-seed {seed}",
            f"--episodes {args.episodes}",
            f"--end-time {args.end_time}",
            f"--agent-period {args.agent_period}",
            f"--validation-seeds {parse_seed_spec(args.validation_seeds)}",
            f"--validation-end-time {args.validation_end_time}",
            "--train-seed-mode per-episode",
            "--validate-every 10",
            "--validation-workers 1",
            "--checkpoint 10",
            "--reward-mode interval_v1",
            "--action-space single",
            "--budget-increase-ratio 0.025",
            "--budget-decrease-ratio 0.015",
            "--include-explicit-noop",
            "--budget-floor-ratio 0.9",
            "--observation-mode v11_full_10d",
            "--ema-alpha 0.2",
            "--overrun-ema-alpha 0.1",
            "--history-k 8",
            "--event-window 10",
            "--max-cost-weight 0.7",
            "--risk-max-scale 3.0",
            "--include-safety-margin",
            "--save-best-by qos_stable",
            f"--qos-stable-mode-delta {args.qos_stable_mode_delta}",
            "--save-all-best-types",
            f"--output-dir {out_dir}",
        ]

        for field in required_mc_fields:
            if field in row and str(row[field]).strip() != "":
                cli_name = "--" + field.replace("_", "-")
                cmd_parts.append(f"{cli_name} {row[field]}")

        lines.append(" \\\n  ".join(cmd_parts))
        lines.append(f"KMP_DUPLICATE_LIB_OK=TRUE conda run --no-capture-output -n amc-repro env PYTHONPATH=. python -u scripts/select_qos_best_from_validation.py --run-dir {out_dir} --qos-stable-mode-delta {args.qos_stable_mode_delta}")
        lines.append("")

    output_script.write_text("\n".join(lines), encoding="utf-8")
    output_script.chmod(0o755)
    print(f"训练命令脚本已生成: {output_script}", flush=True)


if __name__ == "__main__":
    main()
