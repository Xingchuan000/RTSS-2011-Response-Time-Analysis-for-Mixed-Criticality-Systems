#!/usr/bin/env python3
"""
Select plateau-stable DQN checkpoints from existing multi-run training outputs.

Expected input layout:
    outputs/er085_mr_20/
        r0_s185/
            validation_metrics.csv
            checkpoints/model_episode_0010.pt
            checkpoints/model_episode_0020.pt
            ...
        r1_s185/
        r2_s185/
        ...

The script does not retrain models. It scans validation_metrics.csv for each run,
computes neighborhood stability around each checkpoint, selects one plateau
checkpoint per run, then selects one best plateau checkpoint per taskset seed.
It also builds candidate directories compatible with downstream evaluation by
copying the selected checkpoint to cands/s<seed>/model_best.pt.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Iterable

RUN_DIR_RE = re.compile(r"^r(?P<run>\d+)_s(?P<seed>\d+)$")


@dataclass(frozen=True)
class Row:
    raw: dict[str, str]
    episode: int
    reduction: float | None
    safe: bool
    checkpoint_path: Path | None


@dataclass(frozen=True)
class Candidate:
    row: Row
    run_dir: Path
    run_id: int
    seed: int
    selection_type: str
    selection_rank: int
    plateau_score: float
    window_mean: float | None
    window_std: float | None
    window_min: float | None
    window_max: float | None
    window_count: int
    window_safe_count: int
    neighbor_ratio_count: int
    neighbor_ratio_threshold: float | None
    point_best_episode: int | None
    point_best_reduction: float | None
    reason: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Select plateau checkpoints from r<run>_s<seed> DQN outputs and "
            "build candidate directories for long-horizon evaluation."
        )
    )
    parser.add_argument(
        "--runs-root",
        type=Path,
        required=True,
        help="Root directory containing r<run>_s<seed> subdirectories, e.g. outputs/er085_mr_20.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
        help="Output directory for scan CSVs and cands/s<seed> candidate folders.",
    )
    parser.add_argument(
        "--window-episodes",
        type=int,
        default=20,
        help="Half-window in episodes for plateau scoring. Default: 20 means e-20,e-10,e,e+10,e+20 if validation step is 10.",
    )
    parser.add_argument(
        "--min-neighbors",
        type=int,
        default=3,
        help="Minimum safe checkpoints in the window for strict plateau selection. Default: 3.",
    )
    parser.add_argument(
        "--relaxed-min-neighbors",
        type=int,
        default=2,
        help="Minimum safe checkpoints in the window for relaxed fallback. Default: 2.",
    )
    parser.add_argument(
        "--std-penalty",
        type=float,
        default=0.5,
        help="Plateau score penalty: window_mean - std_penalty * window_std. Default: 0.5.",
    )
    parser.add_argument(
        "--neighbor-ratio",
        type=float,
        default=0.85,
        help="Diagnostic count threshold relative to per-run point best. Default: 0.85.",
    )
    parser.add_argument(
        "--min-window-reduction",
        type=float,
        default=0.0,
        help="Strict plateau requires window_min > this value. Default: 0.0.",
    )
    parser.add_argument(
        "--min-current-reduction",
        type=float,
        default=0.0,
        help="Selected center checkpoint must have current reduction > this value. Default: 0.0.",
    )
    parser.add_argument(
        "--max-mode-delta",
        type=float,
        default=0.05,
        help="QoS-stable mode delta threshold. Default: 0.05.",
    )
    parser.add_argument(
        "--allow-lo-deadline-miss",
        action="store_true",
        help="By default any LO deadline miss field >0 marks a checkpoint unsafe when the column exists.",
    )
    parser.add_argument(
        "--allow-no-safe-action-steps",
        action="store_true",
        help="By default no_safe_action_steps_mean >0 marks a checkpoint unsafe when the column exists.",
    )
    parser.add_argument(
        "--copy-extra-files",
        action="store_true",
        help="Copy config.json, validation/action logs, and elite logs from source run dirs when present.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Compute selections and write CSV/JSON metadata without copying model files.",
    )
    return parser.parse_args()


def parse_float(value: Any) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if text == "" or text.lower() in {"nan", "none", "null"}:
        return None
    try:
        val = float(text)
    except ValueError:
        return None
    if math.isnan(val) or math.isinf(val):
        return None
    return val


def parse_int(value: Any) -> int | None:
    val = parse_float(value)
    if val is None:
        return None
    return int(round(val))


def truthy(value: Any) -> bool | None:
    if value is None:
        return None
    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "y"}:
        return True
    if text in {"false", "0", "no", "n"}:
        return False
    return None


def is_row_safe(raw: dict[str, str], args: argparse.Namespace) -> bool:
    # Require no HI deadline misses. Prefer the explicit QoS field, fall back to total deadline_misses_sum.
    hi_miss = parse_float(raw.get("hi_deadline_misses_sum"))
    baseline_hi_miss = parse_float(raw.get("baseline_hi_deadline_misses_sum"))
    if hi_miss is not None and hi_miss > 0:
        return False
    if baseline_hi_miss is not None and baseline_hi_miss > 0:
        # This should usually be zero in your current controlled setting. Treat nonzero as unsafe.
        return False

    total_miss = parse_float(raw.get("deadline_misses_sum"))
    if hi_miss is None and total_miss is not None and total_miss > 0:
        return False

    if not args.allow_lo_deadline_miss:
        lo_miss = parse_float(raw.get("lo_deadline_misses_sum"))
        if lo_miss is not None and lo_miss > 0:
            return False

    mode_delta = parse_float(raw.get("mode_change_delta_ratio"))
    if mode_delta is not None and abs(mode_delta) > float(args.max_mode_delta):
        return False

    if not args.allow_no_safe_action_steps:
        no_safe = parse_float(raw.get("no_safe_action_steps_mean"))
        if no_safe is not None and no_safe > 0:
            return False

    # If the training script already wrote a Pareto-valid flag, use it as an additional safety gate.
    pareto = truthy(raw.get("is_pareto_valid"))
    if pareto is False:
        return False

    reduction = parse_float(raw.get("relative_lc_loss_reduction"))
    if reduction is None:
        return False
    return True


def find_checkpoint(run_dir: Path, episode: int) -> Path | None:
    ckpt_dir = run_dir / "checkpoints"
    candidates = [
        ckpt_dir / f"model_episode_{episode:04d}.pt",
        ckpt_dir / f"model_episode_{episode}.pt",
        run_dir / f"model_episode_{episode:04d}.pt",
        run_dir / f"model_episode_{episode}.pt",
    ]
    for path in candidates:
        if path.exists():
            return path

    # Robust fallback for unusual naming, e.g. checkpoint_ep_0010.pt.
    if ckpt_dir.exists():
        episode_texts = {str(episode), f"{episode:04d}"}
        for path in sorted(ckpt_dir.glob("*.pt")):
            name = path.name
            if any(text in name for text in episode_texts):
                return path
    return None


def read_run_rows(run_dir: Path, args: argparse.Namespace) -> list[Row]:
    metrics_path = run_dir / "validation_metrics.csv"
    if not metrics_path.exists():
        raise FileNotFoundError(f"Missing validation_metrics.csv: {metrics_path}")

    rows: list[Row] = []
    with metrics_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError(f"Empty validation_metrics.csv: {metrics_path}")
        if "episode" not in reader.fieldnames:
            raise ValueError(f"Missing episode column in {metrics_path}")
        if "relative_lc_loss_reduction" not in reader.fieldnames:
            raise ValueError(f"Missing relative_lc_loss_reduction column in {metrics_path}")
        for raw in reader:
            ep = parse_int(raw.get("episode"))
            if ep is None:
                continue
            reduction = parse_float(raw.get("relative_lc_loss_reduction"))
            ckpt = find_checkpoint(run_dir, ep)
            safe = is_row_safe(raw, args) and ckpt is not None
            rows.append(Row(raw=raw, episode=ep, reduction=reduction, safe=safe, checkpoint_path=ckpt))
    rows.sort(key=lambda r: r.episode)
    return rows


def run_info_from_dir(run_dir: Path) -> tuple[int, int]:
    match = RUN_DIR_RE.match(run_dir.name)
    if not match:
        raise ValueError(f"Run directory name must look like r<run>_s<seed>: {run_dir}")
    return int(match.group("run")), int(match.group("seed"))


def point_best(rows: Iterable[Row]) -> Row | None:
    valid = [r for r in rows if r.safe and r.reduction is not None]
    if valid:
        return max(valid, key=lambda r: float(r.reduction))
    fallback = [r for r in rows if r.checkpoint_path is not None and r.reduction is not None]
    if fallback:
        return max(fallback, key=lambda r: float(r.reduction))
    return None


def window_rows(rows: list[Row], center_episode: int, half_window: int) -> list[Row]:
    lo = center_episode - half_window
    hi = center_episode + half_window
    return [r for r in rows if lo <= r.episode <= hi]


def build_candidate_for_row(
    *,
    row: Row,
    rows: list[Row],
    run_dir: Path,
    run_id: int,
    seed: int,
    point: Row | None,
    args: argparse.Namespace,
    selection_type: str,
    selection_rank: int,
    reason: str,
) -> Candidate:
    win = window_rows(rows, row.episode, int(args.window_episodes))
    safe_vals = [float(r.reduction) for r in win if r.safe and r.reduction is not None]
    all_safe_vals = safe_vals
    w_mean = mean(all_safe_vals) if all_safe_vals else None
    w_std = pstdev(all_safe_vals) if len(all_safe_vals) > 1 else 0.0 if len(all_safe_vals) == 1 else None
    w_min = min(all_safe_vals) if all_safe_vals else None
    w_max = max(all_safe_vals) if all_safe_vals else None
    if w_mean is not None and w_std is not None:
        plateau_score = float(w_mean) - float(args.std_penalty) * float(w_std)
    else:
        plateau_score = float(row.reduction) if row.reduction is not None else -math.inf
    best_reduction = float(point.reduction) if point is not None and point.reduction is not None else None
    threshold = None if best_reduction is None else float(args.neighbor_ratio) * best_reduction
    neighbor_count = 0
    if threshold is not None:
        neighbor_count = sum(1 for r in win if r.safe and r.reduction is not None and float(r.reduction) >= threshold)
    return Candidate(
        row=row,
        run_dir=run_dir,
        run_id=run_id,
        seed=seed,
        selection_type=selection_type,
        selection_rank=selection_rank,
        plateau_score=plateau_score,
        window_mean=w_mean,
        window_std=w_std,
        window_min=w_min,
        window_max=w_max,
        window_count=len(win),
        window_safe_count=len(safe_vals),
        neighbor_ratio_count=neighbor_count,
        neighbor_ratio_threshold=threshold,
        point_best_episode=point.episode if point is not None else None,
        point_best_reduction=float(point.reduction) if point is not None and point.reduction is not None else None,
        reason=reason,
    )


def select_run_candidate(run_dir: Path, args: argparse.Namespace) -> tuple[Candidate, list[dict[str, Any]]]:
    run_id, seed = run_info_from_dir(run_dir)
    rows = read_run_rows(run_dir, args)
    if not rows:
        raise ValueError(f"No validation rows found in {run_dir}")
    point = point_best(rows)
    if point is None:
        raise ValueError(f"No checkpoint-backed validation row found in {run_dir}")

    scan_rows: list[dict[str, Any]] = []
    strict_candidates: list[Candidate] = []
    relaxed_candidates: list[Candidate] = []

    for row in rows:
        if row.checkpoint_path is None or row.reduction is None:
            continue
        cand_base = build_candidate_for_row(
            row=row,
            rows=rows,
            run_dir=run_dir,
            run_id=run_id,
            seed=seed,
            point=point,
            args=args,
            selection_type="scan",
            selection_rank=-1,
            reason="scan_only",
        )
        current = float(row.reduction)
        strict_ok = (
            row.safe
            and current > float(args.min_current_reduction)
            and cand_base.window_safe_count >= int(args.min_neighbors)
            and cand_base.window_mean is not None
            and cand_base.window_mean > 0.0
            and cand_base.window_min is not None
            and cand_base.window_min > float(args.min_window_reduction)
        )
        relaxed_ok = (
            row.safe
            and current > float(args.min_current_reduction)
            and cand_base.window_safe_count >= int(args.relaxed_min_neighbors)
            and cand_base.window_mean is not None
            and cand_base.window_mean > 0.0
        )
        scan_rows.append(candidate_to_dict(cand_base, include_paths=True) | {
            "strict_ok": strict_ok,
            "relaxed_ok": relaxed_ok,
        })
        if strict_ok:
            strict_candidates.append(
                build_candidate_for_row(
                    row=row,
                    rows=rows,
                    run_dir=run_dir,
                    run_id=run_id,
                    seed=seed,
                    point=point,
                    args=args,
                    selection_type="plateau_strict",
                    selection_rank=2,
                    reason="window_min_positive_and_stable",
                )
            )
        elif relaxed_ok:
            relaxed_candidates.append(
                build_candidate_for_row(
                    row=row,
                    rows=rows,
                    run_dir=run_dir,
                    run_id=run_id,
                    seed=seed,
                    point=point,
                    args=args,
                    selection_type="plateau_relaxed",
                    selection_rank=1,
                    reason="window_mean_positive_relaxed",
                )
            )

    def sort_key(c: Candidate) -> tuple[float, float, float, float]:
        return (
            c.plateau_score,
            c.window_min if c.window_min is not None else -math.inf,
            c.window_mean if c.window_mean is not None else -math.inf,
            float(c.row.reduction) if c.row.reduction is not None else -math.inf,
        )

    if strict_candidates:
        selected = max(strict_candidates, key=sort_key)
    elif relaxed_candidates:
        selected = max(relaxed_candidates, key=sort_key)
    else:
        selected = build_candidate_for_row(
            row=point,
            rows=rows,
            run_dir=run_dir,
            run_id=run_id,
            seed=seed,
            point=point,
            args=args,
            selection_type="point_best_fallback",
            selection_rank=0,
            reason="no_plateau_candidate_found",
        )
    return selected, scan_rows


def candidate_to_dict(c: Candidate, *, include_paths: bool = False) -> dict[str, Any]:
    data: dict[str, Any] = {
        "seed": c.seed,
        "run_id": c.run_id,
        "run_name": f"r{c.run_id}",
        "episode": c.row.episode,
        "current_reduction": c.row.reduction,
        "selection_type": c.selection_type,
        "selection_rank": c.selection_rank,
        "plateau_score": c.plateau_score,
        "window_mean": c.window_mean,
        "window_std": c.window_std,
        "window_min": c.window_min,
        "window_max": c.window_max,
        "window_count": c.window_count,
        "window_safe_count": c.window_safe_count,
        "neighbor_ratio_count": c.neighbor_ratio_count,
        "neighbor_ratio_threshold": c.neighbor_ratio_threshold,
        "point_best_episode": c.point_best_episode,
        "point_best_reduction": c.point_best_reduction,
        "delta_vs_point_best": (
            None
            if c.row.reduction is None or c.point_best_reduction is None
            else float(c.row.reduction) - float(c.point_best_reduction)
        ),
        "reason": c.reason,
    }
    # Preserve useful validation columns when present.
    for key in [
        "hi_deadline_misses_sum",
        "lo_deadline_misses_sum",
        "deadline_misses_sum",
        "mode_change_delta_ratio",
        "no_safe_action_steps_mean",
        "baseline_lc_service_loss_mean",
        "lc_service_loss_mean",
        "baseline_lo_cancellations_mean",
        "lo_cancellations_mean",
        "relative_score",
        "is_pareto_valid",
        "safe_global_best_reason",
        "elite_replay_candidate",
        "elite_replay_reason",
        "elite_replay_buffer_size",
    ]:
        if key in c.row.raw:
            data[key] = c.row.raw.get(key)
    if include_paths:
        data["run_dir"] = str(c.run_dir)
        data["checkpoint_path"] = str(c.row.checkpoint_path) if c.row.checkpoint_path else ""
    return data


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def copy_if_exists(src: Path, dst_dir: Path) -> None:
    if src.exists() and src.is_file():
        shutil.copy2(src, dst_dir / src.name)


def build_candidate_dir(c: Candidate, output_root: Path, args: argparse.Namespace) -> None:
    cand_dir = output_root / "cands" / f"s{c.seed}"
    cand_dir.mkdir(parents=True, exist_ok=True)
    if not args.dry_run:
        if c.row.checkpoint_path is None or not c.row.checkpoint_path.exists():
            raise FileNotFoundError(f"Selected checkpoint does not exist: {c.row.checkpoint_path}")
        shutil.copy2(c.row.checkpoint_path, cand_dir / "model_best.pt")
        copy_if_exists(c.run_dir / "validation_metrics.csv", cand_dir)
        if args.copy_extra_files:
            for name in [
                "config.json",
                "train_metrics.csv",
                "action_histogram.csv",
                "elite_replay_log.csv",
                "best_elite_replay_log.csv",
                "plateau_balanced_log.csv",
                "validation_policy_actions.csv",
            ]:
                copy_if_exists(c.run_dir / name, cand_dir)
    (cand_dir / "source_run_dir.txt").write_text(str(c.run_dir), encoding="utf-8")
    if c.row.checkpoint_path is not None:
        (cand_dir / "source_checkpoint_path.txt").write_text(str(c.row.checkpoint_path), encoding="utf-8")
    (cand_dir / "source_checkpoint_episode.txt").write_text(str(c.row.episode), encoding="utf-8")
    meta = candidate_to_dict(c, include_paths=True)
    meta["dry_run"] = bool(args.dry_run)
    meta["selection_rule"] = {
        "window_episodes": args.window_episodes,
        "min_neighbors": args.min_neighbors,
        "relaxed_min_neighbors": args.relaxed_min_neighbors,
        "std_penalty": args.std_penalty,
        "neighbor_ratio": args.neighbor_ratio,
        "min_window_reduction": args.min_window_reduction,
        "min_current_reduction": args.min_current_reduction,
        "max_mode_delta": args.max_mode_delta,
        "allow_lo_deadline_miss": args.allow_lo_deadline_miss,
        "allow_no_safe_action_steps": args.allow_no_safe_action_steps,
    }
    with (cand_dir / "selection_meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


def main() -> None:
    args = parse_args()
    runs_root: Path = args.runs_root
    output_root: Path = args.output_root
    if not runs_root.exists():
        raise FileNotFoundError(f"runs-root does not exist: {runs_root}")

    run_dirs = [p for p in sorted(runs_root.iterdir()) if p.is_dir() and RUN_DIR_RE.match(p.name)]
    if not run_dirs:
        raise FileNotFoundError(f"No r<run>_s<seed> directories found under {runs_root}")

    scan_all: list[dict[str, Any]] = []
    per_run_candidates: list[Candidate] = []
    warnings: list[str] = []

    for run_dir in run_dirs:
        try:
            selected, scan_rows = select_run_candidate(run_dir, args)
            per_run_candidates.append(selected)
            scan_all.extend(scan_rows)
        except Exception as exc:  # Keep scanning other runs and report failures clearly.
            warnings.append(f"{run_dir}: {type(exc).__name__}: {exc}")

    if not per_run_candidates:
        raise RuntimeError("No run candidates were selected. First warning: " + (warnings[0] if warnings else "none"))

    # Per-seed best-of-runs selection. Prefer strict plateau over relaxed over point fallback,
    # then rank by plateau score and current reduction.
    by_seed: dict[int, list[Candidate]] = {}
    for cand in per_run_candidates:
        by_seed.setdefault(cand.seed, []).append(cand)

    def seed_sort_key(c: Candidate) -> tuple[int, float, float, float, float]:
        return (
            c.selection_rank,
            c.plateau_score,
            c.window_min if c.window_min is not None else -math.inf,
            c.window_mean if c.window_mean is not None else -math.inf,
            float(c.row.reduction) if c.row.reduction is not None else -math.inf,
        )

    per_seed_candidates: list[Candidate] = [max(cands, key=seed_sort_key) for _, cands in sorted(by_seed.items())]

    scan_dir = output_root / "scan"
    write_csv(scan_dir / "plateau_checkpoint_scan_all.csv", scan_all)
    write_csv(scan_dir / "plateau_selected_per_run.csv", [candidate_to_dict(c, include_paths=True) for c in per_run_candidates])
    write_csv(scan_dir / "plateau_selected_per_seed.csv", [candidate_to_dict(c, include_paths=True) for c in per_seed_candidates])

    comparison_rows: list[dict[str, Any]] = []
    for c in per_seed_candidates:
        comparison_rows.append(
            {
                "seed": c.seed,
                "selected_run_id": c.run_id,
                "selected_episode": c.row.episode,
                "selected_type": c.selection_type,
                "selected_current_reduction": c.row.reduction,
                "selected_plateau_score": c.plateau_score,
                "selected_window_mean": c.window_mean,
                "selected_window_std": c.window_std,
                "selected_window_min": c.window_min,
                "selected_neighbor_ratio_count": c.neighbor_ratio_count,
                "point_best_episode_in_selected_run": c.point_best_episode,
                "point_best_reduction_in_selected_run": c.point_best_reduction,
                "delta_vs_point_best_in_selected_run": (
                    None
                    if c.row.reduction is None or c.point_best_reduction is None
                    else float(c.row.reduction) - float(c.point_best_reduction)
                ),
                "source_run_dir": str(c.run_dir),
                "source_checkpoint_path": str(c.row.checkpoint_path) if c.row.checkpoint_path else "",
            }
        )
    write_csv(scan_dir / "plateau_vs_pointbest_summary.csv", comparison_rows)

    for cand in per_seed_candidates:
        build_candidate_dir(cand, output_root, args)

    summary = {
        "runs_root": str(runs_root),
        "output_root": str(output_root),
        "num_run_dirs_found": len(run_dirs),
        "num_run_candidates_selected": len(per_run_candidates),
        "num_seed_candidates_selected": len(per_seed_candidates),
        "selection_type_counts_per_seed": {},
        "warnings": warnings,
        "dry_run": bool(args.dry_run),
        "rule": {
            "window_episodes": args.window_episodes,
            "min_neighbors": args.min_neighbors,
            "relaxed_min_neighbors": args.relaxed_min_neighbors,
            "std_penalty": args.std_penalty,
            "neighbor_ratio": args.neighbor_ratio,
            "min_window_reduction": args.min_window_reduction,
            "min_current_reduction": args.min_current_reduction,
            "max_mode_delta": args.max_mode_delta,
        },
    }
    counts: dict[str, int] = {}
    for cand in per_seed_candidates:
        counts[cand.selection_type] = counts.get(cand.selection_type, 0) + 1
    summary["selection_type_counts_per_seed"] = counts
    with (scan_dir / "plateau_selection_overall.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("Plateau checkpoint selection complete.")
    print(f"  run dirs found:       {len(run_dirs)}")
    print(f"  per-run selected:     {len(per_run_candidates)}")
    print(f"  per-seed selected:    {len(per_seed_candidates)}")
    print(f"  output:               {output_root}")
    print(f"  selection type counts:{counts}")
    if warnings:
        print("Warnings:")
        for warning in warnings[:20]:
            print(f"  - {warning}")
        if len(warnings) > 20:
            print(f"  ... {len(warnings) - 20} more warnings")


if __name__ == "__main__":
    main()
