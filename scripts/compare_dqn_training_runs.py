"""对比多个 DQN 训练目录的验证结果。"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def _parse_runs(raw_value: str) -> list[Path]:
    """解析 `--runs` 参数。"""

    runs = [Path(part.strip()) for part in raw_value.split(",") if part.strip()]
    if not runs:
        raise ValueError("--runs 不能为空")
    return runs


def _load_csv(path: Path) -> list[dict[str, str]]:
    """读取 CSV 文件。"""

    if not path.exists():
        raise FileNotFoundError(f"缺少文件: {path}")
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _best_validation_row(rows: list[dict[str, str]]) -> dict[str, str]:
    """按文档规则选择 best validation 行。"""

    if not rows:
        raise ValueError("validation_metrics.csv 为空")
    zero_miss_rows = [row for row in rows if int(float(row["deadline_misses_sum"])) == 0]
    if not zero_miss_rows:
        raise ValueError("validation_metrics.csv 中不存在 deadline_misses_sum == 0 的行")
    return min(zero_miss_rows, key=lambda row: float(row["lo_cancellations_mean"]))


def compare_runs(run_dirs: list[Path], output_path: Path) -> None:
    """聚合多个训练目录并输出对比 CSV。"""

    out_rows: list[dict[str, str | float | int]] = []
    for run_dir in run_dirs:
        validation_rows = _load_csv(run_dir / "validation_metrics.csv")
        best_row = _best_validation_row(validation_rows)
        config_path = run_dir / "config.json"
        if not config_path.exists():
            raise FileNotFoundError(f"缺少文件: {config_path}")
        with config_path.open("r", encoding="utf-8") as f:
            config = json.load(f)

        out_rows.append(
            {
                "run": run_dir.name,
                "total_util": float(config["total_util"]),
                "reward_mode": str(config["reward_mode"]),
                "action_space": str(config["action_space"]),
                "episodes": int(config["train_seeds"] and len(config["train_seeds"])),
                "end_time": int(config["runtime_config"]["end_time"]),
                "train_seed_mode": str(config["train_seed_mode"]),
                "best_episode": int(float(best_row["episode"])),
                "best_deadline_misses_sum": int(float(best_row["deadline_misses_sum"])),
                "best_mode_changes_mean": float(best_row["mode_changes_mean"]),
                "best_lo_cancellations_mean": float(best_row["lo_cancellations_mean"]),
                "baseline_mode_changes_mean": float(best_row["baseline_mode_changes_mean"]),
                "baseline_lo_cancellations_mean": float(best_row["baseline_lo_cancellations_mean"]),
                "dqn_mode_delta": float(best_row["dqn_mode_changes_delta_mean"]),
                "dqn_lo_cancel_delta": float(best_row["dqn_lo_cancellations_delta_mean"]),
                "valid_action_count_mean": float(best_row["valid_action_count_mean"]),
                "masked_action_count_mean": float(best_row["masked_action_count_mean"]),
                "no_safe_action_steps": float(best_row["no_safe_action_steps_mean"]),
            }
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "run",
        "total_util",
        "reward_mode",
        "action_space",
        "episodes",
        "end_time",
        "train_seed_mode",
        "best_episode",
        "best_deadline_misses_sum",
        "best_mode_changes_mean",
        "best_lo_cancellations_mean",
        "baseline_mode_changes_mean",
        "baseline_lo_cancellations_mean",
        "dqn_mode_delta",
        "dqn_lo_cancel_delta",
        "valid_action_count_mean",
        "masked_action_count_mean",
        "no_safe_action_steps",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(out_rows)


def build_parser() -> argparse.ArgumentParser:
    """构建命令行参数解析器。"""

    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=str, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> None:
    """执行 run 对比。"""

    args = build_parser().parse_args()
    run_dirs = _parse_runs(args.runs)
    compare_runs(run_dirs=run_dirs, output_path=args.output)


if __name__ == "__main__":
    main()
