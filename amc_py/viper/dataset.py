"""VIPER 数据集 schema 与 JSONL IO。

第一版严格按计划使用 JSONL + manifest，不引入额外的列式存储依赖。
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
import json
from pathlib import Path

import numpy as np


@dataclass(frozen=True, slots=True)
class ViperSample:
    """单个 teacher 标注样本。"""

    teacher_id: str
    taskset_seed: int | None
    scenario_seed: int
    scenario_split: str
    horizon: int
    decision_index: int
    time: int
    state_vector: tuple[float, ...]
    valid_action_mask: tuple[bool, ...]
    teacher_action_id: int | None
    teacher_action_valid: bool
    raw_q_values: tuple[float, ...]
    q_best: float | None
    q_second_best: float | None
    q_worst: float | None
    q_margin_second: float | None
    viper_weight: float | None
    behavior_policy: str
    behavior_action_id: int | None
    tree_iteration: int | None
    raw_budgets_json: str
    raw_recent_costs_json: str
    mask_reject_reasons_json: str


def _sample_to_json_dict(sample: ViperSample) -> dict[str, object]:
    row = asdict(sample)
    row["state_vector"] = list(sample.state_vector)
    row["valid_action_mask"] = list(sample.valid_action_mask)
    row["raw_q_values"] = list(sample.raw_q_values)
    return row


def _sample_from_json_dict(row: dict[str, object]) -> ViperSample:
    return ViperSample(
        teacher_id=str(row["teacher_id"]),
        taskset_seed=row.get("taskset_seed"),  # type: ignore[arg-type]
        scenario_seed=int(row["scenario_seed"]),
        scenario_split=str(row["scenario_split"]),
        horizon=int(row["horizon"]),
        decision_index=int(row["decision_index"]),
        time=int(row["time"]),
        state_vector=tuple(float(v) for v in row["state_vector"]),  # type: ignore[arg-type]
        valid_action_mask=tuple(bool(v) for v in row["valid_action_mask"]),  # type: ignore[arg-type]
        teacher_action_id=(None if row["teacher_action_id"] is None else int(row["teacher_action_id"])),
        teacher_action_valid=bool(row["teacher_action_valid"]),
        raw_q_values=tuple(float(v) for v in row["raw_q_values"]),  # type: ignore[arg-type]
        q_best=(None if row["q_best"] is None else float(row["q_best"])),
        q_second_best=(None if row["q_second_best"] is None else float(row["q_second_best"])),
        q_worst=(None if row["q_worst"] is None else float(row["q_worst"])),
        q_margin_second=(None if row["q_margin_second"] is None else float(row["q_margin_second"])),
        viper_weight=(None if row["viper_weight"] is None else float(row["viper_weight"])),
        behavior_policy=str(row["behavior_policy"]),
        behavior_action_id=(None if row["behavior_action_id"] is None else int(row["behavior_action_id"])),
        tree_iteration=(None if row["tree_iteration"] is None else int(row["tree_iteration"])),
        raw_budgets_json=str(row["raw_budgets_json"]),
        raw_recent_costs_json=str(row["raw_recent_costs_json"]),
        mask_reject_reasons_json=str(row["mask_reject_reasons_json"]),
    )


def write_viper_dataset(output_dir: Path, samples: Iterable[ViperSample], manifest: dict) -> None:
    """把 VIPER 数据集写成固定目录结构。"""

    output_dir.mkdir(parents=True, exist_ok=True)
    sample_list = list(samples)
    with (output_dir / "samples.jsonl").open("w", encoding="utf-8") as handle:
        for sample in sample_list:
            handle.write(json.dumps(_sample_to_json_dict(sample), ensure_ascii=False) + "\n")
    with (output_dir / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)


def read_viper_dataset(dataset_dir: Path) -> tuple[list[ViperSample], dict]:
    """读取 dataset 目录中的样本与 manifest。"""

    samples: list[ViperSample] = []
    with (dataset_dir / "samples.jsonl").open("r", encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if not text:
                continue
            samples.append(_sample_from_json_dict(json.loads(text)))
    with (dataset_dir / "manifest.json").open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    return samples, manifest


def samples_to_xyw(
    samples: Sequence[ViperSample],
    *,
    weight_mode: str,
    weight_epsilon: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """把样本投影成 sklearn 训练需要的 `X/y/w`。

    这里默认排除 `teacher_action_id is None` 的样本，因为它们无法作为分类标签训练 CART。
    """

    labeled_samples = [sample for sample in samples if sample.teacher_action_id is not None]
    if not labeled_samples:
        raise ValueError("没有可用于训练的 teacher labeled samples")
    x = np.asarray([sample.state_vector for sample in labeled_samples], dtype=np.float32)
    y = np.asarray([int(sample.teacher_action_id) for sample in labeled_samples], dtype=np.int64)
    if weight_mode == "uniform":
        w = np.ones(len(labeled_samples), dtype=np.float64)
    elif weight_mode == "viper_q_span":
        w = np.asarray(
            [
                max(float(sample.viper_weight if sample.viper_weight is not None else 0.0), weight_epsilon)
                for sample in labeled_samples
            ],
            dtype=np.float64,
        )
    elif weight_mode == "q_margin_second":
        w = np.asarray(
            [
                max(float(sample.q_margin_second if sample.q_margin_second is not None else 0.0), weight_epsilon)
                for sample in labeled_samples
            ],
            dtype=np.float64,
        )
    else:
        raise ValueError(f"不支持的 weight_mode: {weight_mode}")
    if not np.isfinite(w).all():
        raise ValueError("样本权重中出现非有限值")
    if float(w.sum()) <= 0.0:
        w = np.ones(len(labeled_samples), dtype=np.float64)
    return x, y, w
