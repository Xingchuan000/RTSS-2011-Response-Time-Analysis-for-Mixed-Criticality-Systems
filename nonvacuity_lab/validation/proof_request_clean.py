from __future__ import annotations


class IsolationError(ValueError):
    pass


FORBIDDEN_EXPERIMENT_KEYS = {
    "nonvacuity", "nonvacuity_profile", "nonvacuity_params", "mutation_id",
    "mutation_class", "expected_result", "expected_failure", "experiment_status",
}


def assert_experiment_blind_request(value, path="$" ) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            if str(key).lower() in FORBIDDEN_EXPERIMENT_KEYS:
                raise IsolationError(f"forbidden key {path}.{key}")
            assert_experiment_blind_request(child, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            assert_experiment_blind_request(child, f"{path}[{index}]")
