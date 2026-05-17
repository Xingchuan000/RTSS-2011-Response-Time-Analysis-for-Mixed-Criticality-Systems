"""根据 QoS pressure manifest 生成 compare_dqn_training_runs 命令脚本。"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


BEST_TYPES = ["qos_stable", "conservative_qos", "qos_best"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=str, required=True)
    parser.add_argument("--output-script", type=str, required=True)
    parser.add_argument("--output-dir-prefix", "--run-dir-prefix", dest="output_dir_prefix", type=str, required=True)
    parser.add_argument("--episodes", type=int, default=120)
    parser.add_argument("--compare-output-prefix", "--output-prefix", dest="compare_output_prefix", type=str, required=True)
    parser.add_argument("--qos-stable-mode-delta", type=float, default=0.05)
    return parser


def main() -> None:
    args = build_parser().parse_args()

    with Path(args.manifest).open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError("manifest 为空")

    run_dirs = [
        f"{args.output_dir_prefix}_seed{int(float(row['candidate_seed']))}_e{args.episodes}_inc0025_dec0015_floor09"
        for row in rows
    ]

    script_path = Path(args.output_script)
    script_path.parent.mkdir(parents=True, exist_ok=True)

    lines: list[str] = ["#!/usr/bin/env bash", "set -euo pipefail", ""]
    for best_type in BEST_TYPES:
        output_csv = f"{args.compare_output_prefix}_{best_type}_top{len(rows)}_e{args.episodes}.csv"
        cmd_parts = [
            "KMP_DUPLICATE_LIB_OK=TRUE conda run --no-capture-output -n amc-repro env PYTHONPATH=. python -u scripts/compare_dqn_training_runs.py",
            f"--best-type {best_type}",
            f"--output {output_csv}",
            f"--runs {','.join(run_dirs)}",
            f"--qos-stable-mode-delta {args.qos_stable_mode_delta}",
        ]
        lines.append(" \\\n  ".join(cmd_parts))
        lines.append("")

    script_path.write_text("\n".join(lines), encoding="utf-8")
    script_path.chmod(0o755)
    print(f"对比命令脚本已生成: {script_path}", flush=True)


if __name__ == "__main__":
    main()
