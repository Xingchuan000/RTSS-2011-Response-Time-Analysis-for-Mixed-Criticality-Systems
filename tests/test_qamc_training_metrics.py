from __future__ import annotations

import math

from amc_py.event_runtime import simulate_ordered_taskset_event_driven
from amc_py.models import Criticality, Task
from amc_py.qamc.loss_metrics import compute_qamc_loss_metrics
from amc_py.qamc.profile_spec import QAmcProfileSpec
from amc_py.qamc.profiles import (
    build_qamc_profile_bundle,
    compute_taskset_fingerprint,
)
from amc_py.runtime_models import RuntimeConfig, RuntimeSemantics
from amc_py.runtime_scenarios import make_table_scenario
from scripts.train_dqn_amc import LEGACY_DEGRADED_FIELDS, _runtime_metrics_row


def _result_and_bundle():
    task = Task("L", 10, 10, 6, 6, Criticality.LO)
    bundle = build_qamc_profile_bundle(
        [task],
        taskset_fingerprint=compute_taskset_fingerprint([task]),
        spec=QAmcProfileSpec(),
    )
    result = simulate_ordered_taskset_event_driven(
        [task],
        make_table_scenario(
            actual_costs={("L", 0): 7, ("L", 1): 2},
            default_hi="c_lo",
            default_lo="c_lo",
        ),
        RuntimeConfig(end_time=20, semantics=RuntimeSemantics.Q_AMC),
        qamc_profile_bundle=bundle,
    )
    return result, bundle


def test_generic_lo_qos_equals_qamc_normalized_qos() -> None:
    result, bundle = _result_and_bundle()
    row = _runtime_metrics_row(
        result=result,
        semantics=RuntimeSemantics.Q_AMC,
        qamc_profile_bundle=bundle,
    )
    assert math.isclose(
        float(row["lo_quality_qos"]),
        float(row["qamc_normalized_quality_qos"]),
        rel_tol=0.0,
        abs_tol=1e-12,
    )


def test_qamc_loss_breakdown_conserves_releases() -> None:
    result, _ = _result_and_bundle()
    metrics = compute_qamc_loss_metrics(result)
    accounted = (
        metrics.completed_positive_quality_jobs
        + metrics.overrun_stopped_zero_quality_jobs
        + metrics.deadline_lost_zero_quality_jobs
        + metrics.min_threshold_fallback_zero_quality_jobs
        + metrics.hi_mode_discard_zero_quality_jobs
        + metrics.other_zero_quality_jobs
    )
    assert accounted == metrics.released_lo_jobs


def test_overrun_stop_is_zero_service_and_legacy_fields_are_blank() -> None:
    result, bundle = _result_and_bundle()
    row = _runtime_metrics_row(
        result=result,
        semantics=RuntimeSemantics.Q_AMC,
        qamc_profile_bundle=bundle,
    )
    assert int(row["qamc_overrun_stop_count"]) == 1
    assert int(row["qamc_zero_service_count"]) >= 1
    assert row["qamc_legacy_degraded_metrics_applicable"] is False
    assert all(row[name] is None for name in LEGACY_DEGRADED_FIELDS)


def test_profile_fingerprint_is_stable_across_validation_results() -> None:
    first_result, bundle = _result_and_bundle()
    second_result, _ = _result_and_bundle()
    fingerprints = {
        _runtime_metrics_row(
            result=result,
            semantics=RuntimeSemantics.Q_AMC,
            qamc_profile_bundle=bundle,
        )["qamc_profile_fingerprint"]
        for result in (first_result, second_result)
    }
    assert fingerprints == {bundle.fingerprint}
