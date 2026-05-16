"""从已有 validation_metrics.csv 中重选 QoS best checkpoint。"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path

from amc_py.model_selection import best_row_for_type


BEST_TYPES = ("conservative_qos", "qos_stable", "qos_best")


def _load_csv(path: Path) -> list[dict[str, str]]:
    """读取 validation CSV。"""

    if not path.exists():
        raise FileNotFoundError(f"缺少文件: {path}")
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _copy_checkpoint_if_exists(run_dir: Path, episode: int, best_type: str) -> bool:
    """若对应 episode checkpoint 存在，则复制到标准 best 文件名。"""

    checkpoint_path = run_dir / "checkpoints" / f"model_episode_{episode:04d}.pt"
    if not checkpoint_path.exists():
        return False
    shutil.copy2(checkpoint_path, run_dir / f"model_best_{best_type}.pt")
    return True


def select_best(run_dir: Path, qos_stable_mode_delta: float) -> None:
    """对单个训练目录重选三类 QoS best。"""

    validation_rows = _load_csv(run_dir / "validation_metrics.csv")
    summary_rows: list[dict[str, object]] = []

    for best_type in BEST_TYPES:
        best_row = best_row_for_type(validation_rows, best_type=best_type, delta=qos_stable_mode_delta)
        metadata_path = run_dir / f"best_model_metadata_{best_type}.json"
        if best_row is None:
            metadata = {
                "best_type": best_type,
                "qos_stable_mode_delta": qos_stable_mode_delta,
                "found_valid_checkpoint": False,
                "reason": "No checkpoint satisfied HI safety and mode stability constraints.",
                "checkpoint_file_available": False,
            }
            summary_rows.append(
                {
                    "best_type": best_type,
                    "found_valid_checkpoint": False,
                    "best_episode": "",
                    "checkpoint_file_available": False,
                }
            )
        else:
            episode = int(float(best_row["episode"]))
            checkpoint_file_available = _copy_checkpoint_if_exists(run_dir, episode, best_type)
            metadata = {
                "best_type": best_type,
                "qos_stable_mode_delta": qos_stable_mode_delta,
                "found_valid_checkpoint": True,
                "best_validation_episode": episode,
                "hi_deadline_misses_sum": int(float(best_row.get("hi_deadline_misses_sum", best_row["deadline_misses_sum"]))),
                "lc_service_loss_mean": float(best_row["lc_service_loss_mean"]),
                "baseline_lc_service_loss_mean": float(best_row["baseline_lc_service_loss_mean"]),
                "relative_lc_loss_reduction": best_row.get("relative_lc_loss_reduction"),
                "lc_qos_mean": float(best_row["lc_qos_mean"]),
                "min_lc_service_mean": best_row.get("min_lc_service_mean"),
                "mode_changes_mean": float(best_row["mode_changes_mean"]),
                "baseline_mode_changes_mean": float(best_row["baseline_mode_changes_mean"]),
                "mode_change_delta_ratio": float(best_row["mode_change_delta_ratio"]),
                "budget_adjust_count_mean": float(best_row["budget_adjust_count_mean"]),
                "checkpoint_file_available": checkpoint_file_available,
            }
            summary_rows.append(
                {
                    "best_type": best_type,
                    "found_valid_checkpoint": True,
                    "best_episode": episode,
                    "checkpoint_file_available": checkpoint_file_available,
                }
            )
        with metadata_path.open("w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)

    output_path = run_dir / "qos_best_selection_summary.csv"
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["best_type", "found_valid_checkpoint", "best_episode", "checkpoint_file_available"],
        )
        writer.writeheader()
        writer.writerows(summary_rows)


def build_parser() -> argparse.ArgumentParser:
    """构建命令行参数。"""

    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--qos-stable-mode-delta", type=float, default=0.05)
    return parser


def main() -> None:
    """执行 QoS best 后处理。"""

    args = build_parser().parse_args()
    select_best(args.run_dir, args.qos_stable_mode_delta)


if __name__ == "__main__":
    main()
