from __future__ import annotations

import json
import math

from amc_py.event_runtime import simulate_ordered_taskset_event_driven
from amc_py.metrics import compute_lo_quality_weighted_metrics
from amc_py.models import Criticality, Task
from amc_py.qamc.loss_metrics import compute_qamc_loss_metrics
from amc_py.qamc.metrics_support import compute_qamc_metrics
from amc_py.qamc.profile_spec import QAmcProfileSpec
from amc_py.qamc.profiles import (
    build_qamc_profile_bundle,
    compute_taskset_fingerprint,
)
from amc_py.runtime_models import RuntimeConfig, RuntimeSemantics, SystemMode
from amc_py.runtime_scenarios import make_table_scenario


def _hi_mode_discard_result():
    lo_task = Task("L", 5, 5, 1, 1, Criticality.LO)
    hi_task = Task("H", 20, 20, 2, 10, Criticality.HI)
    tasks = [lo_task, hi_task]
    bundle = build_qamc_profile_bundle(
        tasks,
        taskset_fingerprint=compute_taskset_fingerprint(tasks),
        spec=QAmcProfileSpec(),
    )
    result = simulate_ordered_taskset_event_driven(
        tasks,
        make_table_scenario(
            actual_costs={("L", 0): 1, ("H", 0): 9},
            default_hi="c_lo",
            default_lo="c_lo",
        ),
        RuntimeConfig(
            end_time=9,
            semantics=RuntimeSemantics.Q_AMC,
            record_dropped_lo_releases=True,
        ),
        qamc_profile_bundle=bundle,
    )
    return result, bundle


def _lo_mode_result():
    lo_task = Task("L", 10, 10, 6, 6, Criticality.LO)
    bundle = build_qamc_profile_bundle(
        [lo_task],
        taskset_fingerprint=compute_taskset_fingerprint([lo_task]),
        spec=QAmcProfileSpec(),
    )
    result = simulate_ordered_taskset_event_driven(
        [lo_task],
        make_table_scenario(
            actual_costs={("L", 0): 7, ("L", 1): 2},
            default_hi="c_lo",
            default_lo="c_lo",
        ),
        RuntimeConfig(end_time=20, semantics=RuntimeSemantics.Q_AMC),
        qamc_profile_bundle=bundle,
    )
    return result, bundle


def test_qamc_qos_includes_lo_releases_dropped_in_hi_mode() -> None:
    result, bundle = _hi_mode_discard_result()
    lo_jobs = [
        job for job in result.jobs if job.task.criticality is Criticality.LO
    ]
    hi_mode_dropped_job = lo_jobs[1]

    assert result.mode_switches
    assert hi_mode_dropped_job.released_in_mode is SystemMode.HI
    assert hi_mode_dropped_job.dropped is True
    assert hi_mode_dropped_job.qamc_target_runtime_level_at_release is None

    generic_metrics = compute_lo_quality_weighted_metrics(result)
    qamc_metrics = compute_qamc_metrics(result, bundle)
    assert qamc_metrics.release_count == len(lo_jobs)
    assert math.isclose(
        qamc_metrics.normalized_quality_qos,
        generic_metrics.lo_quality_qos,
        rel_tol=0.0,
        abs_tol=1e-12,
    )
    assert qamc_metrics.zero_service_count == 1


def test_qamc_release_profile_excludes_hi_mode_unsnapshotted_jobs() -> None:
    result, bundle = _hi_mode_discard_result()
    qamc_metrics = compute_qamc_metrics(result, bundle)
    release_counts = json.loads(qamc_metrics.release_count_by_raw_rank_json)

    assert qamc_metrics.release_count == 2
    assert qamc_metrics.managed_release_count == 1
    assert sum(release_counts.values()) == qamc_metrics.managed_release_count
    assert qamc_metrics.managed_release_count < qamc_metrics.release_count


def test_qamc_loss_breakdown_counts_hi_mode_discards() -> None:
    result, _ = _hi_mode_discard_result()
    loss = compute_qamc_loss_metrics(result)
    classified = (
        loss.completed_positive_quality_jobs
        + loss.overrun_stopped_zero_quality_jobs
        + loss.deadline_lost_zero_quality_jobs
        + loss.min_threshold_fallback_zero_quality_jobs
        + loss.hi_mode_discard_zero_quality_jobs
        + loss.other_zero_quality_jobs
    )

    assert loss.hi_mode_discard_zero_quality_jobs == 1
    assert classified == loss.released_lo_jobs == 2


def test_qamc_metrics_unchanged_without_hi_mode_discard() -> None:
    result, bundle = _lo_mode_result()
    qamc_metrics = compute_qamc_metrics(result, bundle)

    assert qamc_metrics.release_count == 2
    assert qamc_metrics.managed_release_count == 2
    assert qamc_metrics.zero_service_count == 1
