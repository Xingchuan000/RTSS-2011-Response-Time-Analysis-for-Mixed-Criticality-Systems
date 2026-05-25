"""按 count 列汇总 validation_policy_actions 的策略动作统计。"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


def _parse_runs(raw: str) -> list[Path]:
    runs = [Path(part.strip()) for part in raw.split(",") if part.strip()]
    if not runs:
        raise ValueError("--runs 不能为空")
    return runs


def _load_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"缺少文件: {path}")
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _best_row(validation_rows: list[dict[str, str]], selection_type: str) -> dict[str, str]:
    candidates = [row for row in validation_rows if str(row.get("selection_type", "")) == selection_type]
    if not candidates:
        raise ValueError(f"validation_metrics.csv 不包含 selection_type={selection_type}")
    selected = max(candidates, key=lambda row: int(float(row["episode"])))
    return selected


def _safe_float(row: dict[str, str], key: str) -> float:
    text = str(row.get(key, "")).strip()
    if text == "":
        return 0.0
    return float(text)


def summarize_run(run_dir: Path, selection_type: str) -> dict[str, str | int | float]:
    validation_rows = _load_csv(run_dir / "validation_metrics.csv")
    policy_rows = _load_csv(run_dir / "validation_policy_actions.csv")
    selected = _best_row(validation_rows, selection_type)
    selected_episode = int(float(selected["episode"]))
    episode_rows = [row for row in policy_rows if int(float(row["episode"])) == selected_episode]

    counts: dict[int, int] = {}
    for row in episode_rows:
        action_id = int(float(row["action_id"]))
        count = int(float(row.get("count", "0") or 0))
        counts[action_id] = counts.get(action_id, 0) + count
    total = sum(counts.values())

    increase_total = 0
    decrease_total = 0
    noop_total = 0
    for row in episode_rows:
        count = int(float(row.get("count", "0") or 0))
        action_type = str(row.get("action_type", "")).strip()
        action_name = str(row.get("action_name", "")).strip()
        if action_type.startswith("increase"):
            increase_total += count
        elif action_type.startswith("decrease"):
            decrease_total += count
        elif action_type == "noop" or action_name == "noop":
            noop_total += count

    entropy = 0.0
    if total > 0:
        for value in counts.values():
            p = float(value) / float(total)
            if p > 0.0:
                entropy -= p * math.log(p)

    top_actions = sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:5]
    return {
        "run": run_dir.name,
        "selection_type": selection_type,
        "selected_episode": selected_episode,
        "relative_lc_loss_reduction": _safe_float(selected, "relative_lc_loss_reduction"),
        "lc_qos_delta_mean": _safe_float(selected, "lc_qos_delta_mean"),
        "mode_change_delta_ratio": _safe_float(selected, "mode_change_delta_ratio"),
        "hi_deadline_misses_sum": int(float(selected.get("hi_deadline_misses_sum", "0") or 0)),
        "policy_action_total": total,
        "increase_rate": (float(increase_total) / float(total)) if total > 0 else 0.0,
        "decrease_rate": (float(decrease_total) / float(total)) if total > 0 else 0.0,
        "noop_rate": (float(noop_total) / float(total)) if total > 0 else 0.0,
        "action7_rate": (float(counts.get(7, 0)) / float(total)) if total > 0 else 0.0,
        "action8_11_rate": (
            float(sum(counts.get(idx, 0) for idx in (8, 9, 10, 11))) / float(total)
            if total > 0
            else 0.0
        ),
        "action_entropy": entropy,
        "unique_actions": len([v for v in counts.values() if v > 0]),
        "top_actions": json.dumps(top_actions, ensure_ascii=False),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=str, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--selection-type", type=str, default="qos_stable")
    args = parser.parse_args()

    rows = [summarize_run(run_dir, args.selection_type) for run_dir in _parse_runs(args.runs)]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "run",
                "selection_type",
                "selected_episode",
                "relative_lc_loss_reduction",
                "lc_qos_delta_mean",
                "mode_change_delta_ratio",
                "hi_deadline_misses_sum",
                "policy_action_total",
                "increase_rate",
                "decrease_rate",
                "noop_rate",
                "action7_rate",
                "action8_11_rate",
                "action_entropy",
                "unique_actions",
                "top_actions",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
