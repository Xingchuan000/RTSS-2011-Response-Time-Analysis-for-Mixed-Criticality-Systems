"""Build the normative V9.1 machine bindings from the frozen request and current source."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from formal_toolchain.adapters.amc_taskset import export_taskset
from formal_toolchain.adapters.runtime_config import export_formal_target_config
from formal_toolchain.adapters.source_manifest import build_source_manifest
from formal_toolchain.adapters.target_factory import build_target
from formal_toolchain.adapters.tree_artifact import inspect_tree_artifact
from formal_toolchain.core.hashing import sha256_file, sha256_object
from formal_toolchain.policy.tree import validate_tree_and_leaf_partition
from formal_toolchain.policy.tree_io import integer_tree_from_dict
from amc_py.rl.observation import build_default_normalization_bounds
from formal_toolchain.v9_1.constants import (
    CANONICAL_PHASES, KERNEL_STATE_FIELDS, PRIMARY_CLAIM, PROOF_ROUTE, REQUEST_SCHEMA, SCOPE,
)


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _canonical_binding_value(value: Any) -> Any:
    """Convert finite Python floats to canonical decimal strings for proof objects."""

    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("V9.1 bindings cannot contain NaN/Inf")
        return format(value, ".17g")
    if isinstance(value, dict):
        return {str(key): _canonical_binding_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_canonical_binding_value(item) for item in value]
    return value


def load_request(request_path: Path) -> dict[str, Any]:
    data = _read_json(Path(request_path))
    if not isinstance(data, dict):
        raise ValueError("V9.1 proof request must be a JSON object")
    expected = {
        "schema_version": REQUEST_SCHEMA,
        "proof_route": PROOF_ROUTE,
        "scope": SCOPE,
        "primary_claim": PRIMARY_CLAIM,
    }
    for key, value in expected.items():
        if data.get(key) != value:
            raise ValueError(f"V9.1 request {key} mismatch: {data.get(key)!r}")
    forbidden = {"route", "fallback_route", "phase_k_case_map", "protected_prefix", "raw_protected_prefix"}
    if forbidden & set(data):
        raise ValueError("V9.1 request contains legacy route fields")
    return data


def _workspace_root(request_path: Path) -> Path:
    request_path = Path(request_path).resolve()
    if request_path.parent.name != "request":
        raise ValueError("V9.1 proof_request.json must live in <workspace>/request")
    return request_path.parent.parent


def _resolve_input(request_path: Path, relative: str) -> Path:
    root = _workspace_root(request_path)
    path = (root / relative).resolve()
    if root not in path.parents:
        raise ValueError("request input path escapes workspace")
    return path


def _field(config: dict[str, Any], name: str) -> Any:
    row = config["fields"].get(name)
    if not isinstance(row, dict) or "value" not in row:
        raise ValueError(f"effective runtime config missing {name}")
    return row["value"]


def _source_hashes(source_root: Path) -> dict[str, str]:
    files = {
        "event_runtime": "formal_toolchain/semantics/frozen_c_amc_sem_event_runtime.py",
        "runtime_wrapper": "formal_toolchain/semantics/frozen_c_amc_sem_runtime_wrapper.py",
        "observation_runtime": "amc_py/rl/observation.py",
        "observation_formal_schema": "formal_toolchain/semantics/frozen_c_amc_sem_observation.py",
        "feature_state": "amc_py/rl/feature_state.py",
        "feature_config": "amc_py/rl/feature_config.py",
        "rl_environment": "amc_py/rl/env.py",
        "rl_actions": "amc_py/rl/actions.py",
        "rl_action_execution": "amc_py/rl/action_execution.py",
        "rl_safety": "amc_py/rl/safety.py",
        "fixed_point": "amc_py/viper/fixed_point.py",
        "integer_tree": "amc_py/viper/integer_tree.py",
        "runtime_adapter": "formal_toolchain/adapters/amc_real_runtime_adapter.py",
        "action_runtime": "formal_toolchain/semantics/frozen_c_amc_sem_action_runtime.py",
        "budget_runtime": "formal_toolchain/semantics/frozen_c_amc_sem_budget_runtime.py",
    }
    result: dict[str, str] = {}
    for key, relative in files.items():
        path = source_root / relative
        if not path.is_file():
            raise ValueError(f"required V9.1 source binding missing: {relative}")
        result[key] = sha256_file(path)
    return result




def _freeze_normalization_bounds(environment: Any, ordered_tasks: Any) -> dict[str, dict[str, int]]:
    """Freeze the exact per-task normalization intervals used by production v11.

    The current V9.1 numeric proof models these bounds as exact binary64
    integers.  This covers the deployed mc_stratified_dynamic workload and the
    default observation bounds without admitting arbitrary non-integral float
    intervals whose rounding relation is not yet encoded.
    """

    active = getattr(environment, "normalization_bounds", None)
    if active is None:
        active = build_default_normalization_bounds(ordered_tasks)
    rows: dict[str, dict[str, int]] = {}
    for task in ordered_tasks:
        bound = active.get(task.name)
        if bound is None:
            raise ValueError(f"V9_1_P5_NORMALIZATION_BOUND_MISSING:{task.name}")
        lo = float(bound.min_cost)
        hi = float(bound.max_cost)
        if not (math.isfinite(lo) and math.isfinite(hi) and hi > lo):
            raise ValueError(f"V9_1_P5_NORMALIZATION_BOUND_INVALID:{task.name}")
        if not (lo.is_integer() and hi.is_integer()):
            raise ValueError(f"V9_1_P5_NONINTEGRAL_NORMALIZATION_BOUND_UNSUPPORTED:{task.name}")
        if abs(lo) >= 2**53 or abs(hi) >= 2**53:
            raise ValueError(f"V9_1_P5_NORMALIZATION_BOUND_EXCEEDS_EXACT_BINARY64_INTEGER_RANGE:{task.name}")
        rows[str(task.name)] = {"min_cost": int(lo), "max_cost": int(hi)}
    return rows


def _exact_integer_linear_constraints(adapter: Any, task_names: list[str]) -> list[dict[str, Any]]:
    """Freeze the runtime safety checker as exact integer A@B<=rhs rows.

    RuntimeBudgetSafetyChecker constructs every coefficient and rhs from integer
    periods/deadlines/R_LO values.  Refuse any non-integral float here instead
    of silently rounding a deployment guard.
    """

    environment = getattr(adapter, "environment", None)
    if environment is None:
        raise ValueError("V9_1_P5_RUNTIME_ENVIRONMENT_UNBOUND")
    if not bool(getattr(environment, "check_safety", False)):
        return []
    checker = environment._ensure_checker()
    rows = []
    for task_name, constraint_name, rhs, coeff in checker._constraint_rows:
        rhs_f = float(rhs)
        if not rhs_f.is_integer():
            raise ValueError("V9_1_P5_SAFETY_RHS_NOT_EXACT_INTEGER")
        coeff_values = []
        for value in coeff.tolist():
            value_f = float(value)
            if not value_f.is_integer():
                raise ValueError("V9_1_P5_SAFETY_COEFFICIENT_NOT_EXACT_INTEGER")
            coeff_values.append(int(value_f))
        if len(coeff_values) != len(task_names):
            raise ValueError("V9_1_P5_SAFETY_ROW_DIMENSION_MISMATCH")
        if abs(int(rhs_f)) >= 2**53:
            raise ValueError("V9_1_P5_SAFETY_RHS_EXCEEDS_EXACT_BINARY64_INTEGER_RANGE")
        rows.append({
            "task": str(task_name),
            "constraint": str(constraint_name),
            "rhs": int(rhs_f),
            "coefficients": coeff_values,
        })
    if not rows:
        raise ValueError("V9_1_P5_SAFETY_CHECKER_HAS_NO_ROWS")
    return rows


def _environment_domain(
    taskset: dict[str, Any],
    raw_bounds_by_task: dict[str, Any] | None = None,
) -> dict[str, Any]:
    rows_by_name = dict(raw_bounds_by_task or {})
    task_names = {str(row["name"]) for row in taskset["ordered_tasks"]}
    if rows_by_name and set(rows_by_name) != task_names:
        raise ValueError("V9_1_ACTUAL_DEMAND_BOUND_TASKSET_MISMATCH")

    tasks = []
    for row in taskset["ordered_tasks"]:
        name = str(row["name"])
        criticality = str(row["criticality"])
        c_lo = int(row["code_c_lo"])
        c_hi = int(row["code_c_hi"])
        custom = rows_by_name.get(name)
        if custom is None:
            lower = 1
            upper = c_hi if criticality == "HI" else c_lo
            bound_kind = "AMC_TASK_DEMAND_PROFILE"
            support = "TASK_BOUND"
        else:
            lower = int(custom["min"])
            upper = int(custom["max"])
            bound_kind = str(custom.get("kind", "RAW_EXECUTION_COST_INTEGER_ENVELOPE"))
            support = str(custom.get("support", "DECLARED_INTEGER_RANGE"))
        if lower < 1 or upper < lower:
            raise ValueError(f"V9_1_ACTUAL_DEMAND_BOUND_INVALID:{name}")
        if criticality == "HI" and upper > c_hi:
            raise ValueError(f"V9_1_HI_ACTUAL_DEMAND_EXCEEDS_C_HI:{name}")
        actual_demand = {
            "min": lower, "max": upper, "kind": bound_kind, "support": support,
        }
        if criticality == "HI":
            classification = {
                "kind": "DERIVED_FROM_ACTUAL_DEMAND_AT_RELEASE",
                "normal_iff": f"1 <= A <= {c_lo}",
                "abnormal_iff": f"{c_lo} < A <= {c_hi}",
            }
        else:
            classification = {"kind": "LO_ONLY", "value": "LO"}
        tasks.append({
            "task": name,
            "criticality": criticality,
            "period": int(row["period"]),
            "deadline": int(row["deadline"]),
            "release_domain": {
                "kind": "EXACT_PERIODIC_PHASE_ZERO",
                "release_times": "r_k = k*T for every integer k >= 0",
            },
            "actual_demand_domain": actual_demand,
            "classification_domain": classification,
        })
    return {
        "schema_version": "admissible_environment_domain_v9_1",
        "quantifier": "FORALL_ADMISSIBLE_ENVIRONMENTS",
        "release_model": "EXACT_PERIODIC_PHASE_ZERO",
        "demand_choice": "INDEPENDENT_PER_RELEASE_SUBJECT_TO_TASK_BOUNDS",
        "classification_choice": "DETERMINISTIC_FROM_RELEASE_FROZEN_ACTUAL_DEMAND",
        "finite_prefix_extension": "UNIQUE_PERIODIC_RELEASE_SUFFIX_WITH_ARBITRARY_ADMISSIBLE_FUTURE_DEMANDS",
        "tasks": tasks,
    }


def _demand_semantics(runtime_config: dict[str, Any]) -> dict[str, Any]:
    ratio = _field(runtime_config, "c_amc_sem_lo_degradation_ratio")
    return {
        "schema_version": "effective_demand_semantics_v9_1",
        "hi": "E=A; no truncation",
        "lo_primary": "E=min(A, B_rel+1)",
        "lo_degraded": f"E=min(A, C_deg), C_deg=max(1,min(C_LO,round({ratio}*C_LO)))",
        "release_snapshot": "B_rel frozen in P3 before P4/P5",
        "hi_classification": "NORMAL iff 1<=A<=C_LO; ABNORMAL iff C_LO<A<=C_HI; frozen at release",
    }


def build_bindings(request_path: Path, *, source_root: Path) -> dict[str, Any]:
    request_path = Path(request_path).resolve()
    source_root = Path(source_root).resolve()
    request = load_request(request_path)
    source_manifest = build_source_manifest(source_root)
    expected_source = request.get("source_binding", {}).get("source_manifest_semantic_hash")
    if expected_source != source_manifest.get("semantic_hash"):
        raise ValueError("SOURCE_MANIFEST_CHANGED_SINCE_REQUEST_FREEZE")

    tree_dir = _resolve_input(request_path, str(request["tree_artifact_dir"]))
    formal_inputs_dir = _resolve_input(request_path, str(request["formal_inputs_dir"]))
    recipe = request["target_recipe"]
    target = build_target(str(recipe["factory"]), dict(recipe.get("kwargs", {})))
    if int(target.provenance.get("taskset_seed", request["taskset_seed"])) != int(request["taskset_seed"]):
        raise ValueError("TARGET_SEED_IDENTITY_MISMATCH")

    runtime_export = export_formal_target_config(target)
    if runtime_export.get("status") != "PASS":
        raise ValueError("EFFECTIVE_RUNTIME_CONFIG_UNBOUND")
    runtime_config = runtime_export["config"] if "config" in runtime_export else runtime_export
    processor_overhead = _field(runtime_config, "processor_overhead") if "processor_overhead" in runtime_config["fields"] else 0
    if int(processor_overhead) != 0:
        raise ValueError("P0_ZERO_CONTROLLER_OVERHEAD_REQUIRES_PROCESSOR_OVERHEAD_ZERO")
    if str(_field(runtime_config, "observation_mode")) != "v11_full_10d":
        raise ValueError("V9.1 currently binds only observation_mode=v11_full_10d")

    inventory = inspect_tree_artifact(
        tree_dir,
        expected_state_dim=len(target.feature_names),
        expected_action_dim=len(target.action_definitions),
        expected_seed=int(request["taskset_seed"]),
    )
    tree_json = _read_json(tree_dir / "integer_tree.json")
    tree_model = integer_tree_from_dict(tree_json)
    tree_partition = validate_tree_and_leaf_partition(tree_model)
    if tuple(inventory["feature_names"]) != tuple(target.feature_names):
        raise ValueError("TREE_FEATURE_SCHEMA_MISMATCH")
    if list(inventory["action_definitions"]) != [dict(row) for row in target.action_definitions]:
        raise ValueError("TREE_ACTION_SCHEMA_MISMATCH")

    actions = [dict(row) for row in target.action_definitions]
    noop_ids = [int(row.get("action_id", index)) for index, row in enumerate(actions) if row.get("is_noop") is True]
    if len(noop_ids) != 1:
        raise ValueError("V9.1_REQUIRES_EXACTLY_ONE_EXPLICIT_NOOP")
    noop_id = noop_ids[0]
    adapter = target.runtime_adapter
    if adapter is None:
        raise ValueError("V9.1_REQUIRES_RUNTIME_ADAPTER")
    mask_contract = adapter.export_mask_contract()
    if mask_contract.get("selection") != "ranked_first_valid":
        raise ValueError("V9.1_REQUIRES_RANKED_FIRST_VALID")
    if mask_contract.get("shared_with_step") is not True:
        raise ValueError("MASK_AND_STEP_SEMANTICS_NOT_SHARED")
    if mask_contract.get("explicit_noop_always_valid") is not True:
        raise ValueError("EXPLICIT_NOOP_NOT_PROVEN_ALWAYS_VALID")
    if list(mask_contract.get("explicit_noop_action_ids", [])) != [noop_id]:
        raise ValueError("EXPLICIT_NOOP_ID_MISMATCH")
    environment = getattr(adapter, "environment", None)
    if environment is None:
        raise ValueError("V9_1_P5_RUNTIME_ENVIRONMENT_UNBOUND")
    if any(bool(row.get("is_constraint_guided_pair")) or bool(row.get("is_residual_ranked")) for row in actions):
        raise ValueError("V9_1_P5_DYNAMIC_ACTION_SPACE_UNSUPPORTED")
    if bool(getattr(environment, "enable_residual_safety_fallback", False)):
        raise ValueError("V9_1_P5_RESIDUAL_SAFETY_FALLBACK_MUST_BE_DISABLED")
    if str(environment.budget_rounding_mode) != "ceil_floor":
        raise ValueError("V9_1_P5_ONLY_CEIL_FLOOR_ROUNDING_IS_BOUND")
    normalization_bounds = _freeze_normalization_bounds(environment, target.ordered_tasks)

    taskset = export_taskset(target.ordered_tasks, target.provenance.get("budget_by_task"))
    safety_rows = _exact_integer_linear_constraints(
        adapter, [str(task.name) for task in target.ordered_tasks]
    )
    upper_by_name = {str(row["name"]): int(row["action_hard_upper"]) for row in taskset["ordered_tasks"]}
    task_names = [str(task.name) for task in target.ordered_tasks]
    for row in safety_rows:
        lhs_abs_bound = sum(
            abs(int(coeff)) * upper_by_name[name]
            for coeff, name in zip(row["coefficients"], task_names)
        )
        if lhs_abs_bound >= 2**53:
            raise ValueError("V9_1_P5_SAFETY_LHS_EXCEEDS_EXACT_BINARY64_INTEGER_RANGE")
    env_domain = _environment_domain(
        taskset, target.provenance.get("actual_demand_bounds_by_task")
    )
    demand_semantics = _demand_semantics(runtime_config)
    source_hashes = _source_hashes(source_root)
    agent_period = int(_field(runtime_config, "agent_period"))
    if agent_period <= 0:
        raise ValueError("agent_period must be positive")

    event_binding = {
        "schema_version": "p0_event_order_binding_v9_1",
        "canonical_phase_schema": [{"phase": index, "name": name} for index, name in enumerate(CANONICAL_PHASES)],
        "zero_time_order": "strictly increasing p",
        "time_advance": "only P7 advances t by exactly one integer service quantum and resets p=0",
        "release_snapshot_boundary": "P3 freezes B_rel,A,class,release_entry_mode before P4/P5",
        "mode_switch_boundary": "P4 LO->HI iff P3 release batch contains abnormal HI",
        "controller_boundary": "P5 after P4 and before final dispatch",
        "controller_trigger_predicate": f"CtrlEnabled(z) iff p=5 and t mod {agent_period}=0",
        "deadline_observe_boundary": "P2 observe-only; incomplete HI job is never removed by deadline observation",
        "dispatch": "fixed priority; smaller priority_index wins; equal priority preserves active_jobs insertion order",
        "work_conserving_preemptive": True,
        "source_hashes": {k: source_hashes[k] for k in ("event_runtime", "runtime_wrapper")},
    }
    closure = {
        "schema_version": "same_timestamp_closure_rank_v9_1",
        "rank": "mu(z)=7-p",
        "domain": "p in {0,1,2,3,4,5,6,7}",
        "zero_time_step_property": "p' = p+1, therefore mu(z') = mu(z)-1",
        "only_time_step": "P7",
        "status": "PROVED_BY_FINITE_PHASE_ORDER",
    }
    history_domain_by_task = {
        row["task"]: {
            "cost_min": 0,
            "cost_max": max(
                int(next(task_row["code_c_lo"] for task_row in taskset["ordered_tasks"]
                         if str(task_row["name"]) == str(row["task"]))),
                int(row["actual_demand_domain"]["max"]),
            ),
            "semantics": "OVERAPPROX_OF_INITIAL_C_LO_AND_RUNTIME_EXECUTION_SAMPLES",
        }
        for row in env_domain["tasks"]
    }

    numeric = {
        "schema_version": "numeric_observation_binding_v9_1",
        "feature_names": list(target.feature_names),
        "observation_mode": "v11_full_10d",
        "raw_feature_equations_binding": "EXECUTABLE_SOURCE_BOUND",
        "feature_history_update_binding": "EXECUTABLE_SOURCE_BOUND",
        "history_domain_by_task": history_domain_by_task,
        "history_domain_hash": sha256_object(history_domain_by_task),
        "normalization_binding": "FROZEN_EXACT_PER_TASK_INTERVAL",
        "normalization_bounds": normalization_bounds,
        "normalization_bounds_hash": sha256_object(normalization_bounds),
        "clipping": {"input_min": inventory["fixed_point_config"]["input_min"],
                     "input_max": inventory["fixed_point_config"]["input_max"]},
        "quantization": dict(inventory["fixed_point_config"]),
        "rounding": inventory["fixed_point_config"]["rounding_mode"],
        "feature_config": {
            "ema_alpha": float(environment.feature_config.ema_alpha),
            "overrun_ema_alpha": float(environment.feature_config.overrun_ema_alpha),
            "history_k": int(environment.feature_config.history_k),
            "event_window": int(environment.feature_config.event_window),
            "max_cost_weight": float(environment.feature_config.max_cost_weight),
            "risk_max_scale": float(environment.feature_config.risk_max_scale),
            "include_safety_margin": bool(environment.feature_config.include_safety_margin),
        },
        "numeric_width_overflow": "Python integer after binary64 observation; fixed-point output explicitly bounded",
        "float_semantics": "CPython IEEE-754 binary64 for executable observation before Decimal(str(value)) quantization",
        "comparison_semantics": "integer tree left iff q <= threshold_int; right iff q > threshold_int",
        "source_hashes": {k: source_hashes[k] for k in (
            "observation_runtime", "observation_formal_schema", "feature_state", "feature_config",
            "rl_environment", "rl_safety", "fixed_point", "integer_tree",
        )},
        "fixed_point_config_hash": inventory["fixed_point_config_hash"],
        "tree_partition": tree_partition,
    }
    action_binding = {
        "schema_version": "policy_action_binding_v9_1",
        "action_alphabet": actions,
        "noop_id": noop_id,
        "noop_semantics": "identity budget update and ordinary ranked candidate",
        "selection": "FirstValid(ranking,Mask(z)); total because explicit noop is always valid",
        "implicit_noop": False,
        "fallback": "NONE",
        "mask_contract": {k: mask_contract.get(k) for k in (
            "action_ids", "shared_with_step", "explicit_noop", "explicit_noop_action_ids",
            "explicit_noop_always_valid", "check_safety", "candidate_reject_helper", "candidate_evaluator",
            "selection", "rounding_mode", "min_budget_delta",
        )},
        "execution_config": {
            "rounding_mode": str(environment.budget_rounding_mode),
            "min_budget_delta": int(environment.min_budget_delta),
            "forbid_decreasing_hi_budgets": bool(environment.forbid_decreasing_hi_budgets),
            "enable_deploy_cap_mask": bool(environment.enable_deploy_cap_mask),
            "deploy_cap_mask_ratio": float(environment.deploy_cap_mask_ratio),
            "deploy_cap_mask_criticality": str(environment.deploy_cap_mask_criticality),
            "check_safety": bool(environment.check_safety),
            "safety_constraints": safety_rows,
        },
        "source_hashes": {k: source_hashes[k] for k in (
            "runtime_adapter", "action_runtime", "budget_runtime", "rl_environment",
            "rl_actions", "rl_action_execution", "rl_safety",
        )},
    }
    kernel_schema = {
        "schema_version": "policy_timing_kernel_state_v9_1",
        "state_tuple": list(KERNEL_STATE_FIELDS),
        "phase_domain": list(range(8)),
        "mode_domain": ["LO", "HI"],
        "eta": "per-task release eligibility age eta_i in [0,T_i], eligible iff eta_i=T_i; exact-periodic binding constrains releases to k*T",
        "jobs": [
            "job_key", "task_id", "release_time", "absolute_deadline", "priority", "tie_break",
            "criticality", "release_entry_mode", "classification", "B_rel", "A", "E",
            "executed_service", "removed", "ready",
        ],
        "frontier": "effective event frontier F",
        "hi_miss_ledger": "sticky M; NoPriorHIMiss iff M=0",
        "policy_history": "chi contains all history required by numeric observation",
    }

    seed_task_binding = {
        "schema_version": "seed_task_binding_v9_1",
        "seed_id": int(request["taskset_seed"]),
        "taskset_hash": taskset["fingerprint"],
        "task_model_kind": "EXACT_PERIODIC",
        "priority_assignment_hash": sha256_object(taskset["priority_order"]),
        "tie_break_hash": sha256_object({"rule": event_binding["dispatch"], "source": source_hashes["event_runtime"]}),
        "runtime_config_hash": sha256_object(runtime_config),
    }
    degraded_cost_by_task = {
        str(row["name"]): max(
            1,
            min(
                int(row["code_c_lo"]),
                int(round(float(_field(runtime_config, "c_amc_sem_lo_degradation_ratio")) * int(row["code_c_lo"]))),
            ),
        )
        for row in taskset["ordered_tasks"]
        if str(row["criticality"]) == "LO"
    }
    environment_binding = {
        "schema_version": "environment_binding_v9_1",
        "domain": env_domain,
        "degraded_cost_by_task": degraded_cost_by_task,
        "release_model_hash": sha256_object({"release_model": env_domain["release_model"], "tasks": env_domain["tasks"]}),
        "initial_release_state_hash": sha256_object({row["task"]: 0 for row in env_domain["tasks"]}),
        "release_domain_hash": sha256_object([row["release_domain"] for row in env_domain["tasks"]]),
        "classification_domain_hash": sha256_object([row["classification_domain"] for row in env_domain["tasks"]]),
        "demand_domain_hash": sha256_object([row["actual_demand_domain"] for row in env_domain["tasks"]]),
        "raw_actual_demand_bounds_by_task": {
            row["task"]: dict(row["actual_demand_domain"]) for row in env_domain["tasks"]
        },
        "demand_bound_semantics_hash": sha256_object(demand_semantics),
        "demand_semantics": demand_semantics,
    }

    bindings = {
        "schema_version": "v9_1_binding_bundle_v1",
        "proof_route": PROOF_ROUTE,
        "scope": SCOPE,
        "seed_task_binding": seed_task_binding,
        "environment_binding": environment_binding,
        "p0_event_order_binding": event_binding,
        "numeric_observation_binding": numeric,
        "policy_action_binding": action_binding,
        "kernel_state_schema": kernel_schema,
        "finite_same_timestamp_closure": closure,
        "taskset": taskset,
        "tree_identity": {
            "tree_hash": sha256_file(tree_dir / "integer_tree.json"),
            "tree_threshold_encoding_hash": sha256_object({
                "nodes": tree_json["nodes"], "comparison": numeric["comparison_semantics"]}),
            "artifact_inventory_hash": sha256_object(inventory["files"]),
            "integer_tree": tree_json,
        },
        "source_manifest_semantic_hash": source_manifest["semantic_hash"],
        "formal_inputs_dir_hash": sha256_object(sorted(p.name for p in formal_inputs_dir.iterdir())),
    }
    bindings = _canonical_binding_value(bindings)
    bindings["binding_root_hash"] = sha256_object(bindings)
    return bindings
