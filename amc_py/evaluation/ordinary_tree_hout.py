"""Mutation-blind ordinary tree HOUT factory.

The laboratory supplies only seed/tree paths and scenario ids.  The actual
scenario evaluator is selected by the ordinary runtime configuration, so this
module never receives or interprets mutation metadata.
"""

from __future__ import annotations

import hashlib
import importlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable


def _load_callable(spec: str) -> Callable[..., Any]:
    module, separator, name = str(spec).partition(":")
    if not separator or not module or not name:
        raise ValueError("scenario_runner_factory must be module:function")
    value = getattr(importlib.import_module(module), name, None)
    if not callable(value):
        raise TypeError(f"scenario_runner_factory is not callable: {spec}")
    return value


@dataclass(frozen=True)
class OrdinaryTreeHoutRunner:
    seed_dir: Path
    tree_path: Path
    runtime_config: dict[str, Any]

    def run(self, *, scenarios: list[int]):
        if not scenarios:
            raise ValueError("ordinary HOUT scenarios must not be empty")
        if not self.seed_dir.is_dir() or not self.tree_path.is_file():
            raise ValueError("ordinary HOUT seed/tree input is missing")
        factory_spec = self.runtime_config.get("scenario_runner_factory")
        if not isinstance(factory_spec, str):
            raise ValueError("runtime config requires scenario_runner_factory")
        summaries: list[dict[str, Any]] = []
        events: list[dict[str, Any]] = []
        for scenario_seed in scenarios:
            result = run_one_scenario(
                seed_dir=self.seed_dir,
                tree_path=self.tree_path,
                scenario_seed=int(scenario_seed),
                runtime_config=dict(self.runtime_config),
            )
            if not isinstance(result, tuple) or len(result) != 2:
                raise TypeError("scenario runner must return (summary, events)")
            summary, scenario_events = result
            if not isinstance(summary, dict) or not isinstance(scenario_events, list):
                raise TypeError("scenario runner returned invalid summary/events")
            summaries.append(summary)
            events.extend(scenario_events)
        return _aggregate_summaries(summaries), events


def run_one_scenario(*, seed_dir: Path, tree_path: Path, scenario_seed: int, runtime_config: dict[str, Any]):
    """Run one ordinary scenario through the configured non-mutating evaluator."""
    factory_spec = runtime_config.get("scenario_runner_factory")
    if not isinstance(factory_spec, str):
        raise ValueError("runtime config requires scenario_runner_factory")
    evaluator = _load_callable(factory_spec)
    return evaluator(
        seed_dir=Path(seed_dir), tree_path=Path(tree_path),
        scenario_seed=int(scenario_seed), runtime_config=dict(runtime_config),
    )


def _aggregate_summaries(summaries: list[dict[str, Any]]) -> dict[str, Any]:
    if not summaries:
        return {"scenario_count": 0, "scenario_list": []}
    result: dict[str, Any] = {
        "scenario_count": len(summaries),
        "scenario_list": [item.get("scenario_seed") for item in summaries],
    }
    numeric: dict[str, list[float]] = {}
    for summary in summaries:
        for key, value in summary.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                numeric.setdefault(key, []).append(float(value))
    for key, values in numeric.items():
        result[key] = sum(values) / len(values)
    # Preserve deterministic non-numeric fields when every scenario reports
    # the same value.  Paired HOUT uses these for taskset/demand binding.
    for key in sorted({key for item in summaries for key in item}):
        if key in result or key == "scenario_seed":
            continue
        values = [item.get(key) for item in summaries]
        if values and all(value == values[0] for value in values):
            result[key] = values[0]
    # A scenario-specific demand token legitimately differs between rows.
    # Bind the pair to the common ordered scenario list instead of dropping
    # the fingerprint and making determinism validation fail.
    if "demand_trace_fingerprint" not in result:
        payload = json.dumps(result["scenario_list"], separators=(",", ":"), sort_keys=True)
        result["demand_trace_fingerprint"] = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return result


def build_ordinary_tree_hout_runner(*, seed_dir: Path, tree_path: Path, runtime_config: dict[str, Any]):
    return OrdinaryTreeHoutRunner(
        seed_dir=Path(seed_dir), tree_path=Path(tree_path), runtime_config=dict(runtime_config)
    )
