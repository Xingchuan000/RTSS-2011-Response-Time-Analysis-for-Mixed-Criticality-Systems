"""q-AMC metrics that do not depend on the legacy ``is_degraded`` flag."""

from __future__ import annotations

from dataclasses import dataclass
import json
from collections import Counter, defaultdict

from amc_py.models import Criticality
from amc_py.runtime_models import SimulationResult

from .models import QAmcProfileBundle


@dataclass(frozen=True, slots=True)
class QAmcMetrics:
    paper_quality_sum: int
    paper_quality_per_release: float
    normalized_provided_quality_sum: float
    normalized_quality_qos: float
    zero_service_count: int
    zero_service_ratio: float
    release_target_rank_mean: float
    release_target_normalized_mean: float
    completed_quality_conditional_mean: float
    completed_quality_unconditional_per_release: float
    overrun_stop_count: int
    quality_transition_count: int
    min_quality_exhaustion_count: int
    tasks_ever_degraded: int
    first_degradation_time: int | None
    runtime_level_depth_mean: float
    runtime_level_depth_max: int
    raw_rank_drop_mean: float
    raw_rank_drop_max: int
    release_count_by_raw_rank_json: str
    completed_count_by_raw_rank_json: str
    task_time_at_raw_rank_ratio_json: str
    non_degradable_task_count: int
    trigger_budget_mean_ratio_to_c_lo: float
    trigger_below_design_count: int
    trigger_equal_design_count: int
    trigger_above_design_count: int
    would_overrun_design_count: int
    dqn_budget_update_event_count: int
    dqn_budget_update_task_count: int
    viper_budget_update_event_count: int
    viper_budget_update_task_count: int
    heuristic_budget_update_event_count: int
    heuristic_budget_update_task_count: int
    offline_budget_update_event_count: int
    offline_budget_update_task_count: int
    unspecified_budget_update_event_count: int
    unspecified_budget_update_task_count: int
    profile_fingerprint: str


def compute_qamc_metrics(result: SimulationResult, profile_bundle: QAmcProfileBundle) -> QAmcMetrics:
    jobs = [job for job in result.jobs if job.task.criticality is Criticality.LO and job.qamc_target_runtime_level_at_release is not None]
    releases = len(jobs)
    completed = [job for job in jobs if job.completion_time is not None and not job.dropped and not job.qamc_result_lost_due_to_deadline]
    quality_values = [float(job.qamc_provided_normalized_quality or 0.0) for job in jobs]
    raw_values = [int(job.qamc_provided_raw_rank or 0) for job in jobs]
    target_ranks = [int(job.qamc_target_raw_rank_at_release or 0) for job in jobs]
    target_quality = [float(job.qamc_target_normalized_quality_at_release or 0.0) for job in jobs]
    release_counts = Counter(str(value) for value in target_ranks)
    completed_counts = Counter(str(int(job.qamc_provided_raw_rank or 0)) for job in completed)
    depths: list[int] = []
    raw_drops: list[int] = []
    for task_name, profile in profile_bundle.profiles.items():
        current = profile.initial_runtime_level
        task_changes = [event for event in result.qamc_quality_changes if event.task == task_name]
        if task_changes:
            current = task_changes[-1].new_runtime_level
        depths.append(profile.initial_runtime_level - current)
        raw_drops.append(profile.level(profile.initial_runtime_level).raw_rank - profile.level(current).raw_rank)

    horizon = max(1, result.qamc_evaluation_horizon or result.end_time)
    duration_by_rank_total: defaultdict[str, int] = defaultdict(int)
    for task_name, profile in profile_bundle.profiles.items():
        cursor = 0
        level = profile.initial_runtime_level
        for event in sorted((e for e in result.qamc_quality_changes if e.task == task_name), key=lambda e: e.time):
            duration_by_rank_total[str(profile.level(level).raw_rank)] += max(
                0, min(event.time, horizon) - cursor
            )
            cursor = min(horizon, event.time)
            level = event.new_runtime_level
        duration_by_rank_total[str(profile.level(level).raw_rank)] += max(0, horizon - cursor)
    occupancy_denominator = max(1, len(profile_bundle.profiles) * horizon)
    occupancy = {
        rank: duration / occupancy_denominator
        for rank, duration in sorted(duration_by_rank_total.items())
    }

    trigger_ratios = [job.qamc_trigger_budget_ratio_to_design_c_lo for job in jobs if job.qamc_stopped_by_overrun and job.qamc_trigger_budget_ratio_to_design_c_lo is not None]
    source_counts = Counter(event.source for event in result.budget_update_events)
    source_task_counts = Counter()
    for event in result.budget_update_events:
        source_task_counts[event.source] += len(event.updates)
    return QAmcMetrics(
        paper_quality_sum=sum(raw_values),
        paper_quality_per_release=sum(raw_values) / releases if releases else 0.0,
        normalized_provided_quality_sum=sum(quality_values),
        normalized_quality_qos=sum(quality_values) / releases if releases else 0.0,
        zero_service_count=sum(1 for value in quality_values if value <= 0.0),
        zero_service_ratio=sum(1 for value in quality_values if value <= 0.0) / releases if releases else 0.0,
        release_target_rank_mean=sum(target_ranks) / releases if releases else 0.0,
        release_target_normalized_mean=sum(target_quality) / releases if releases else 0.0,
        completed_quality_conditional_mean=(sum(float(job.qamc_provided_normalized_quality or 0.0) for job in completed) / len(completed) if completed else 0.0),
        completed_quality_unconditional_per_release=sum(quality_values) / releases if releases else 0.0,
        overrun_stop_count=sum(1 for job in jobs if job.qamc_stopped_by_overrun),
        quality_transition_count=len(result.qamc_quality_changes),
        min_quality_exhaustion_count=len(result.qamc_min_quality_exhaustions),
        tasks_ever_degraded=len({event.task for event in result.qamc_quality_changes}),
        first_degradation_time=min((event.time for event in result.qamc_quality_changes), default=None),
        runtime_level_depth_mean=sum(depths) / len(depths) if depths else 0.0,
        runtime_level_depth_max=max(depths, default=0),
        raw_rank_drop_mean=sum(raw_drops) / len(raw_drops) if raw_drops else 0.0,
        raw_rank_drop_max=max(raw_drops, default=0),
        release_count_by_raw_rank_json=json.dumps(dict(sorted(release_counts.items())), sort_keys=True),
        completed_count_by_raw_rank_json=json.dumps(dict(sorted(completed_counts.items())), sort_keys=True),
        task_time_at_raw_rank_ratio_json=json.dumps(occupancy, sort_keys=True),
        non_degradable_task_count=sum(1 for profile in profile_bundle.profiles.values() if not profile.degradable),
        trigger_budget_mean_ratio_to_c_lo=sum(trigger_ratios) / len(trigger_ratios) if trigger_ratios else 0.0,
        trigger_below_design_count=sum(1 for ratio in trigger_ratios if ratio < 1.0),
        trigger_equal_design_count=sum(1 for ratio in trigger_ratios if ratio == 1.0),
        trigger_above_design_count=sum(1 for ratio in trigger_ratios if ratio > 1.0),
        would_overrun_design_count=sum(1 for job in jobs if job.qamc_would_overrun_design_c_lo),
        dqn_budget_update_event_count=source_counts["DQN_ACTION"],
        dqn_budget_update_task_count=source_task_counts["DQN_ACTION"],
        viper_budget_update_event_count=source_counts["VIPER_ACTION"],
        viper_budget_update_task_count=source_task_counts["VIPER_ACTION"],
        heuristic_budget_update_event_count=source_counts["HEURISTIC_ACTION"],
        heuristic_budget_update_task_count=source_task_counts["HEURISTIC_ACTION"],
        offline_budget_update_event_count=source_counts["OFFLINE_PREQUEUED"],
        offline_budget_update_task_count=source_task_counts["OFFLINE_PREQUEUED"],
        unspecified_budget_update_event_count=source_counts["UNSPECIFIED"],
        unspecified_budget_update_task_count=source_task_counts["UNSPECIFIED"],
        profile_fingerprint=profile_bundle.fingerprint,
    )


def qamc_metrics_to_row(metrics: QAmcMetrics, prefix: str = "qamc_") -> dict[str, object]:
    return {f"{prefix}{field}": value for field, value in metrics.__dict__.items()} if hasattr(metrics, "__dict__") else {
        f"{prefix}{field.name}": getattr(metrics, field.name) for field in metrics.__dataclass_fields__.values()
    }


__all__ = ["QAmcMetrics", "compute_qamc_metrics", "qamc_metrics_to_row"]
