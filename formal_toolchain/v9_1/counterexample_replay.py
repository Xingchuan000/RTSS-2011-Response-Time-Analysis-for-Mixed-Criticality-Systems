"""Concrete replay classifier for SAT symbolic windows."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from .safe_prefix_reachability import ReachabilityResult, check_witness_boot_safe_prefix
from .symbolic_state import BoundModel


@dataclass(frozen=True, slots=True)
class ReplayResult:
    status: str
    code: str
    details: Mapping[str, Any]


def replay_concrete_counterexample(
    witness: Mapping[str, Any],
    model: BoundModel,
    *,
    concrete_replayer: Any = None,
    target_task: str,
) -> ReplayResult:
    reachability = check_witness_boot_safe_prefix(witness, model)
    if reachability.status != "PASS":
        return ReplayResult("UNRESOLVED", reachability.code, {"reason": reachability.reason})
    demands = witness.get("exact_periodic_demand_prefix")
    if not isinstance(demands, Mapping):
        return ReplayResult("UNRESOLVED", "SPURIOUS_OR_UNRESOLVED_COUNTEREXAMPLE",
                            {"reason": "concrete periodic demand prefix is missing"})
    if concrete_replayer is None or not callable(getattr(concrete_replayer, "replay", None)):
        return ReplayResult("UNRESOLVED", "SPURIOUS_OR_UNRESOLVED_COUNTEREXAMPLE",
                            {"reason": "independent concrete replay is unavailable"})
    replay = concrete_replayer.replay(
        budgets=dict(witness["z0_budgets"]), demands=dict(demands), target_task=target_task,
    )
    if not isinstance(replay, Mapping):
        return ReplayResult("UNRESOLVED", "SPURIOUS_OR_UNRESOLVED_COUNTEREXAMPLE",
                            {"reason": "replayer returned no machine-readable result"})
    if replay.get("target_first_hi_miss") is not True or replay.get("no_earlier_hi_miss") is not True:
        return ReplayResult("UNRESOLVED", "SPURIOUS_OR_UNRESOLVED_COUNTEREXAMPLE",
                            {"reason": "concrete replay did not reflect the symbolic first-bad theorem",
                             "replay": dict(replay)})
    return ReplayResult("PASS", "CONCRETE_HI_COUNTEREXAMPLE_VERIFIED", dict(replay))


__all__ = ["ReplayResult", "replay_concrete_counterexample"]
