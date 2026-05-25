"""对比多个 DQN 训练目录的验证结果。"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from amc_py.model_selection import best_row_for_type


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


def _require_new_qos_fields(rows: list[dict[str, str]]) -> None:
    """检查新 QoS 字段是否存在。"""

    if not rows:
        raise ValueError("validation_metrics.csv 为空")
    required_fields = ["lc_service_loss_mean", "baseline_lc_service_loss_mean", "lc_qos_mean"]
    missing_fields = [field for field in required_fields if field not in rows[0]]
    if missing_fields:
        raise ValueError(
            "validation_metrics.csv does not contain "
            f"{', '.join(missing_fields)}; rerun training/evaluation with new metrics enabled."
        )


def _best_validation_row(
    rows: list[dict[str, str]],
    *,
    best_type: str,
    delta: float,
) -> dict[str, str] | None:
    """按指定 best_type 选择 best validation 行。"""

    if best_type != "legacy_lo_cancel":
        _require_new_qos_fields(rows)
    best_row = best_row_for_type(rows, best_type=best_type, delta=delta)
    if best_row is None:
        return None
    return dict(best_row)


def compare_runs(run_dirs: list[Path], output_path: Path, *, best_type: str, qos_stable_mode_delta: float) -> None:
    """聚合多个训练目录并输出对比 CSV。"""

    out_rows: list[dict[str, str | float | int]] = []
    for run_dir in run_dirs:
        validation_rows = _load_csv(run_dir / "validation_metrics.csv")
        best_row = _best_validation_row(validation_rows, best_type=best_type, delta=qos_stable_mode_delta)
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
                "q_network_type": str(config.get("q_network_type", "mlp")),
                "action_feature_mode": str(config.get("action_feature_mode", "")),
                "action_feature_dim": int(config.get("action_feature_dim", 0) or 0),
                "episodes": int(config["train_seeds"] and len(config["train_seeds"])),
                "end_time": int(config["runtime_config"]["end_time"]),
                "train_seed_mode": str(config["train_seed_mode"]),
                "best_type": best_type,
                "found_valid_checkpoint": best_row is not None,
                "best_episode": "" if best_row is None else int(float(best_row["episode"])),
                "best_deadline_misses_sum": "" if best_row is None else int(float(best_row["deadline_misses_sum"])),
                "best_hi_deadline_misses_sum": "" if best_row is None else int(float(best_row.get("hi_deadline_misses_sum", best_row["deadline_misses_sum"]))),
                "best_mode_changes_mean": "" if best_row is None else float(best_row["mode_changes_mean"]),
                "best_lo_cancellations_mean": "" if best_row is None else float(best_row["lo_cancellations_mean"]),
                "baseline_mode_changes_mean": "" if best_row is None else float(best_row["baseline_mode_changes_mean"]),
                "baseline_lo_cancellations_mean": "" if best_row is None else float(best_row["baseline_lo_cancellations_mean"]),
                "best_lc_service_loss_mean": "" if best_row is None else float(best_row["lc_service_loss_mean"]),
                "baseline_lc_service_loss_mean": "" if best_row is None else float(best_row["baseline_lc_service_loss_mean"]),
                "relative_lc_loss_reduction": "" if best_row is None else best_row.get("relative_lc_loss_reduction", ""),
                "best_lc_qos_mean": "" if best_row is None else float(best_row["lc_qos_mean"]),
                "best_min_lc_service_mean": "" if best_row is None else best_row.get("min_lc_service_mean", ""),
                "best_mode_change_delta_ratio": "" if best_row is None else float(best_row["mode_change_delta_ratio"]),
                "best_budget_adjust_count_mean": "" if best_row is None else float(best_row["budget_adjust_count_mean"]),
                "dqn_mode_delta": "" if best_row is None else float(best_row["dqn_mode_changes_delta_mean"]),
                "dqn_lo_cancel_delta": "" if best_row is None else float(best_row["dqn_lo_cancellations_delta_mean"]),
                "valid_action_count_mean": "" if best_row is None else float(best_row["valid_action_count_mean"]),
                "masked_action_count_mean": "" if best_row is None else float(best_row["masked_action_count_mean"]),
                "no_safe_action_steps": "" if best_row is None else float(best_row["no_safe_action_steps_mean"]),
            }
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "run",
        "total_util",
        "reward_mode",
        "action_space",
        "q_network_type",
        "action_feature_mode",
        "action_feature_dim",
        "episodes",
        "end_time",
        "train_seed_mode",
        "best_type",
        "found_valid_checkpoint",
        "best_episode",
        "best_deadline_misses_sum",
        "best_hi_deadline_misses_sum",
        "best_mode_changes_mean",
        "best_lo_cancellations_mean",
        "baseline_mode_changes_mean",
        "baseline_lo_cancellations_mean",
        "best_lc_service_loss_mean",
        "baseline_lc_service_loss_mean",
        "relative_lc_loss_reduction",
        "best_lc_qos_mean",
        "best_min_lc_service_mean",
        "best_mode_change_delta_ratio",
        "best_budget_adjust_count_mean",
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
    parser.add_argument(
        "--best-type",
        choices=["qos_stable", "conservative_qos", "qos_best", "legacy_lo_cancel"],
        default="qos_stable",
    )
    parser.add_argument("--qos-stable-mode-delta", type=float, default=0.05)
    return parser


def main() -> None:
    """执行 run 对比。"""

    args = build_parser().parse_args()
    run_dirs = _parse_runs(args.runs)
    compare_runs(
        run_dirs=run_dirs,
        output_path=args.output,
        best_type=args.best_type,
        qos_stable_mode_delta=args.qos_stable_mode_delta,
    )


if __name__ == "__main__":
    main()
