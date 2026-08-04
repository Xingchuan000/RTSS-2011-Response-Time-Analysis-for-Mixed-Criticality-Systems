from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class HoutProfile:
    profile_id: str
    taskset_path: str
    scenario_seeds: tuple[int, ...]
    demand_trace_path: str | None
    horizon: int
    controller_release_times: tuple[int, ...]
    worker_count: int
    runtime_config_path: str
    random_seed: int
    base_command: tuple[str, ...]
    mutated_command: tuple[str, ...]
    summary_relative_path: str = "summary.json"
    events_relative_path: str = "events.jsonl"
    required_metrics: tuple[str, ...] = ()
    required_scenarios: tuple[int, ...] = ()

    def validate(self) -> None:
        if not self.profile_id:
            raise ValueError("profile_id must not be empty")
        if not self.scenario_seeds:
            raise ValueError("scenario_seeds must not be empty")
        if self.horizon <= 0 or self.worker_count <= 0:
            raise ValueError("horizon and worker_count must be positive")
        missing = set(self.required_scenarios) - set(self.scenario_seeds)
        if missing:
            raise ValueError(f"required scenarios missing: {sorted(missing)}")
        if not self.base_command or not self.mutated_command:
            raise ValueError("base_command and mutated_command must not be empty")

    @classmethod
    def from_mapping(cls, value: dict) -> "HoutProfile":
        profile = cls(
            profile_id=str(value["profile_id"]),
            taskset_path=str(value["taskset_path"]),
            scenario_seeds=tuple(int(v) for v in value["scenario_seeds"]),
            demand_trace_path=None if value.get("demand_trace_path") is None else str(value["demand_trace_path"]),
            horizon=int(value["horizon"]),
            controller_release_times=tuple(int(v) for v in value["controller_release_times"]),
            worker_count=int(value["worker_count"]),
            runtime_config_path=str(value["runtime_config_path"]),
            random_seed=int(value["random_seed"]),
            base_command=tuple(str(v) for v in value["base_command"]),
            mutated_command=tuple(str(v) for v in value["mutated_command"]),
            summary_relative_path=str(value.get("summary_relative_path", "summary.json")),
            events_relative_path=str(value.get("events_relative_path", "events.jsonl")),
            required_metrics=tuple(str(v) for v in value.get("required_metrics", ())),
            required_scenarios=tuple(int(v) for v in value.get("required_scenarios", ())),
        )
        profile.validate()
        return profile
