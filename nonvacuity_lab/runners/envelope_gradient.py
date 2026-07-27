"""Search the first failing integer envelope delta without fixing a seed."""

from __future__ import annotations

from typing import Any, Callable, Mapping


def search_delta_star(
    evaluate_delta: Callable[[int], Mapping[str, Any]],
    *,
    initial_step: int = 1,
    maximum_delta: int = 1024,
) -> dict[str, Any]:
    if initial_step <= 0 or maximum_delta <= 0:
        raise ValueError("gradient step/bound 必须为正整数")
    cache: dict[int, dict[str, Any]] = {}

    def evaluate(delta: int) -> dict[str, Any]:
        if delta not in cache:
            cache[delta] = dict(evaluate_delta(delta))
            cache[delta]["delta"] = delta
        return cache[delta]

    baseline = evaluate(0)
    if baseline.get("result_status") != "DEPLOYED_TREE_PROVED":
        return {
            "schema_version": "envelope_gradient_v1",
            "status": "BASELINE_REGRESSION",
            "delta_star": None,
            "runs": [baseline],
        }
    last_pass = 0
    first_fail = None
    delta = initial_step
    while delta <= maximum_delta:
        row = evaluate(delta)
        if row.get("result_status") == "DEPLOYED_TREE_PROVED":
            last_pass = delta
            delta += initial_step
            continue
        first_fail = delta
        break
    if first_fail is None:
        return {
            "schema_version": "envelope_gradient_v1",
            "status": "NO_FAILURE_WITHIN_BOUND",
            "delta_star": None,
            "runs": [cache[key] for key in sorted(cache)],
        }
    low, high = last_pass + 1, first_fail
    while low < high:
        middle = (low + high) // 2
        row = evaluate(middle)
        if row.get("result_status") == "DEPLOYED_TREE_PROVED":
            low = middle + 1
        else:
            high = middle
    delta_star = low
    slack_values = [
        row.get("slack")
        for _, row in sorted(cache.items())
        if isinstance(row.get("slack"), (int, float))
    ]
    nonincreasing = all(
        right <= left for left, right in zip(slack_values, slack_values[1:])
    )
    return {
        "schema_version": "envelope_gradient_v1",
        "status": "DELTA_STAR_FOUND",
        "delta_star": delta_star,
        "first_failing_obligation": evaluate(delta_star).get("violated_obligation_id"),
        "first_failing_witness": evaluate(delta_star).get("witness"),
        "slack_nonincreasing": nonincreasing,
        "runs": [cache[key] for key in sorted(cache)],
    }
