"""Trusted SAT classifier for verifier-regenerated V9.2 windows.

SAT is never equated with deployed unsafety.  We first prove that the specific
SAT z0 is boot-safe-prefix reachable, then require an independent concrete
runtime replayer to reproduce the exact periodic demand prefix and the target's
first HI miss.  Absence or disagreement of the concrete replayer is unresolved,
not unsafe.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from hashlib import sha256
from typing import Any, Mapping

import z3

from .solver_runtime import make_solver

from .formula_solver import canonical_formula_text
from .safe_prefix_invariant import SafePrefixInvariant
from .safe_prefix_reachability import prove_boot_safe_prefix_reachability
from .symbolic_state import BoundModel
from .event_window_encoder import EventWindowEncoding


@dataclass(frozen=True, slots=True)
class DeployedRuntimeCounterexampleReplayer:
    """Independent replay through the actual deployed ``AmcBudgetEnv`` path.

    The replayer does not call any V9.2 transition encoder.  It rebuilds the
    frozen target factory, installs an exact table scenario, evaluates the
    integer CART on production observations, applies the production action mask
    with ranked FirstValid, and advances the concrete event runtime.
    """

    target_recipe: Mapping[str, Any]
    model: BoundModel

    def replay(
        self,
        *,
        demands: Mapping[tuple[str, int], int],
        target_task: str,
        target_deadline_time: int,
        expected_z0_budgets: Mapping[str, int],
        origin_time: int,
    ) -> Mapping[str, Any]:
        from formal_toolchain.adapters.target_factory import build_target
        from formal_toolchain.policy.mask_fallback import select_first_valid
        from formal_toolchain.policy.quantization import replay_quantize
        from amc_py.runtime_scenarios import make_table_scenario
        from amc_py.viper.integer_tree import evaluate_integer_tree

        factory = str(self.target_recipe["factory"])
        kwargs = dict(self.target_recipe.get("kwargs", {}))
        target = build_target(factory, kwargs)
        adapter = target.runtime_adapter
        environment = None if adapter is None else getattr(adapter, "environment", None)
        if environment is None:
            return {"status": "UNRESOLVED", "reason": "deployed runtime environment missing"}
        if self.model.tree is None or self.model.fixed_point_config is None:
            return {"status": "UNRESOLVED", "reason": "deployed tree/fixed-point binding missing"}

        required: set[tuple[str, int]] = set()
        for task in target.ordered_tasks:
            last = target_deadline_time // int(task.period)
            required.update((str(task.name), release_index) for release_index in range(last + 1))
        missing = sorted(required - set(demands))
        if missing:
            return {
                "status": "UNRESOLVED",
                "reason": "exact periodic demand prefix is incomplete",
                "missing": [f"{task}:{index}" for task, index in missing[:32]],
            }

        table = {(str(task), int(index)): int(value) for (task, index), value in demands.items()}
        environment.scenario = make_table_scenario(table, default_hi="c_lo", default_lo="c_lo")
        environment.runtime_config = replace(
            environment.runtime_config,
            end_time=int(target_deadline_time),
            stop_at_first_miss=False,
        )
        observation = environment.reset()
        quant_cfg = dict(self.model.fixed_point_config)
        decision_trace: list[dict[str, Any]] = []
        observed_origin_budget: dict[str, int] | None = None

        while True:
            now = int(observation.time)
            if now == int(origin_time) and environment._engine is not None:
                observed_origin_budget = dict(environment._engine.runtime_budgets.budgets)
            if environment._done:
                break
            mask = tuple(bool(value) for value in environment.formal_valid_action_mask())
            quantized = tuple(replay_quantize(value, quant_cfg)[0] for value in observation.state_vector)
            evaluation = evaluate_integer_tree(self.model.tree, quantized)
            selected = select_first_valid(
                evaluation.action_ranking, mask, action_dim=self.model.action_dim
            )
            if selected is None:
                return {"status": "UNRESOLVED", "reason": "production mask had no valid explicit action"}
            if self.model.noop_id is not None and not mask[int(self.model.noop_id)]:
                return {"status": "UNRESOLVED", "reason": "explicit noop was not valid in concrete runtime"}
            before = (
                dict(environment._engine.runtime_budgets.budgets)
                if environment._engine is not None else {}
            )
            step = environment.step(int(selected))
            decision_trace.append({
                "time": now,
                "leaf_id": int(evaluation.leaf_id),
                "selected_action": int(selected),
                "budget_before": before,
            })
            observation = step.observation
            if step.done:
                break

        if environment._engine is None:
            return {"status": "UNRESOLVED", "reason": "concrete runtime engine disappeared"}
        runtime = environment._engine.finish()
        hi_names = {
            str(task.name) for task in target.ordered_tasks
            if str(getattr(task.criticality, "value", task.criticality)) == "HI"
        }
        hi_misses = sorted(
            (miss for miss in runtime.deadline_misses if str(miss.task) in hi_names),
            key=lambda miss: (int(miss.absolute_deadline), str(miss.task), int(miss.release_index)),
        )
        first = hi_misses[0] if hi_misses else None
        target_match = bool(
            first is not None
            and str(first.task) == str(target_task)
            and int(first.absolute_deadline) == int(target_deadline_time)
        )
        budget_match = (
            None if observed_origin_budget is None
            else observed_origin_budget == {str(k): int(v) for k, v in expected_z0_budgets.items()}
        )
        return {
            "status": "PASS",
            "target_task": str(target_task),
            "miss_time": -1 if first is None else int(first.absolute_deadline),
            "target_first_hi_miss": target_match,
            "no_earlier_hi_miss": bool(first is None or int(first.absolute_deadline) >= int(target_deadline_time)),
            "first_hi_miss_task": None if first is None else str(first.task),
            "first_hi_miss_release_index": None if first is None else int(first.release_index),
            "origin_budget_match": budget_match,
            "decision_count": len(decision_trace),
            "decision_trace_tail": decision_trace[-16:],
        }


@dataclass(frozen=True, slots=True)
class ReplayResult:
    status: str
    code: str
    details: Mapping[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {"status": self.status, "code": self.code, "details": dict(self.details)}


def _eval_int(model: z3.ModelRef, expr: z3.ArithRef) -> int:
    value = model.eval(expr, model_completion=True)
    if not z3.is_int_value(value):
        raise ValueError(f"expected integer model value for {expr}, got {value}")
    return int(value.as_long())


def _window_demands(
    encoding: EventWindowEncoding,
    solver_model: z3.ModelRef,
    model: BoundModel,
) -> dict[tuple[str, int], int]:
    origin = _eval_int(solver_model, encoding.start_state.t)
    result: dict[tuple[str, int], int] = {}
    if getattr(encoding.environment, "lazy_release_demands", False):
        # Explicit Event graphs allocate a fresh demand exactly at each P3
        # occurrence instead of predeclaring a whole-window lookup table.
        # Recover those concrete release demands from the exact post-arrival
        # P4 states stored on the materialized SAT path.
        for step in encoding.event_steps:
            if len(step.phase_states) < 4:
                continue
            p4 = step.phase_states[3]
            for task in model.tasks:
                slot = 0 if task.criticality == "HI" else 1
                job = p4.jobs[(task.name, slot)]
                present = solver_model.eval(job.present, model_completion=True)
                if not z3.is_true(present):
                    continue
                release_time = _eval_int(solver_model, job.release_time)
                if release_time != _eval_int(solver_model, p4.t):
                    continue
                release_index = _eval_int(solver_model, job.release_index)
                result[(task.name, release_index)] = _eval_int(
                    solver_model, job.actual_demand
                )
        return result

    for task in model.tasks:
        for relative_tick in range(encoding.deadline + 1):
            absolute_time = origin + relative_tick
            if absolute_time % task.period != 0:
                continue
            variable = encoding.environment.actual_demands[(task.name, relative_tick)]
            result[(task.name, absolute_time // task.period)] = _eval_int(solver_model, variable)
    return result


def classify_sat_event_window(
    encoding: EventWindowEncoding,
    model: BoundModel,
    invariant: SafePrefixInvariant,
    *,
    concrete_replayer: Any = None,
    timeout_ms: int = 120_000,
    max_boot_ticks: int = 2_000,
) -> ReplayResult:
    """Independently classify one SAT first-bad window.

    The function deliberately re-solves the verifier-generated formula instead
    of consuming the compiler/candidate model.  This keeps the SAT diagnostic
    path under the same trust boundary as the UNSAT safety route.
    """

    formula_text = canonical_formula_text(encoding.formula)
    formula_hash = sha256(formula_text.encode("utf-8")).hexdigest()
    solver = make_solver()
    solver.set(timeout=int(timeout_ms))
    solver.add(encoding.formula)
    result = solver.check()
    if result != z3.sat:
        return ReplayResult(
            "UNRESOLVED", "SPURIOUS_OR_UNRESOLVED_COUNTEREXAMPLE",
            {"reason": f"fresh SAT classifier got {result}", "window_formula_hash": formula_hash},
        )
    sat_model = solver.model()
    reachability = prove_boot_safe_prefix_reachability(
        encoding, sat_model, model, invariant,
        timeout_ms=timeout_ms, max_boot_ticks=max_boot_ticks,
    )
    details: dict[str, Any] = {
        "window_formula_hash": formula_hash,
        "boot_reachability": reachability.as_dict(),
        "target_task": encoding.target_task,
    }
    if reachability.status != "PASS":
        return ReplayResult("UNRESOLVED", "SPURIOUS_OR_UNRESOLVED_COUNTEREXAMPLE", details)

    try:
        origin = _eval_int(sat_model, encoding.start_state.t)
        future = _window_demands(encoding, sat_model, model)
        z0_budgets = {
            task.name: _eval_int(sat_model, encoding.start_state.budgets[task.name])
            for task in model.tasks
        }
    except (KeyError, ValueError) as exc:
        details["reason"] = str(exc)
        return ReplayResult("UNRESOLVED", "SPURIOUS_OR_UNRESOLVED_COUNTEREXAMPLE", details)

    demand_prefix = dict(reachability.demand_prefix or {})
    demand_prefix.update(future)
    details["origin_time"] = origin
    details["target_deadline_time"] = origin + encoding.deadline
    details["z0_budgets"] = z0_budgets
    details["exact_periodic_demand_prefix"] = {
        f"{task}:{release_index}": value
        for (task, release_index), value in sorted(demand_prefix.items())
    }

    replay_method = getattr(concrete_replayer, "replay", None)
    if not callable(replay_method):
        details["reason"] = "independent concrete replay is unavailable"
        return ReplayResult("UNRESOLVED", "SPURIOUS_OR_UNRESOLVED_COUNTEREXAMPLE", details)

    replay = replay_method(
        demands=dict(demand_prefix),
        target_task=encoding.target_task,
        target_deadline_time=origin + encoding.deadline,
        expected_z0_budgets=dict(z0_budgets),
        origin_time=origin,
    )
    if not isinstance(replay, Mapping):
        details["reason"] = "concrete replayer returned no machine-readable result"
        return ReplayResult("UNRESOLVED", "SPURIOUS_OR_UNRESOLVED_COUNTEREXAMPLE", details)
    details["concrete_replay"] = dict(replay)
    if replay.get("no_earlier_hi_miss") is not True:
        details["reason"] = "concrete replay has an earlier HI miss"
        return ReplayResult("UNRESOLVED", "SPURIOUS_OR_UNRESOLVED_COUNTEREXAMPLE", details)
    if replay.get("target_first_hi_miss") is not True:
        details["reason"] = "concrete replay did not reproduce the target first HI miss"
        return ReplayResult("UNRESOLVED", "SPURIOUS_OR_UNRESOLVED_COUNTEREXAMPLE", details)
    if str(replay.get("target_task")) != encoding.target_task:
        details["reason"] = "concrete replay target task mismatch"
        return ReplayResult("UNRESOLVED", "SPURIOUS_OR_UNRESOLVED_COUNTEREXAMPLE", details)
    if int(replay.get("miss_time", -1)) != origin + encoding.deadline:
        details["reason"] = "concrete replay miss time mismatch"
        return ReplayResult("UNRESOLVED", "SPURIOUS_OR_UNRESOLVED_COUNTEREXAMPLE", details)
    return ReplayResult("PASS", "CONCRETE_HI_COUNTEREXAMPLE_VERIFIED", details)


__all__ = ["DeployedRuntimeCounterexampleReplayer", "ReplayResult", "classify_sat_event_window"]
