"""Build the normative V9.1 machine bindings from the frozen request and current source."""

from __future__ import annotations

import json
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
from formal_toolchain.v9_1.constants import (
    CANONICAL_PHASES, KERNEL_STATE_FIELDS, PRIMARY_CLAIM, PROOF_ROUTE, REQUEST_SCHEMA, SCOPE,
)


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


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


def _environment_domain(taskset: dict[str, Any]) -> dict[str, Any]:
    tasks = []
    for row in taskset["ordered_tasks"]:
        criticality = str(row["criticality"])
        c_lo = int(row["code_c_lo"])
        c_hi = int(row["code_c_hi"])
        if criticality == "HI":
            actual_demand = {"min": 1, "max": c_hi}
            classification = {
                "kind": "DERIVED_FROM_ACTUAL_DEMAND_AT_RELEASE",
                "normal_iff": f"1 <= A <= {c_lo}",
                "abnormal_iff": f"{c_lo} < A <= {c_hi}",
            }
        else:
            actual_demand = {"min": 1, "max": c_lo}
            classification = {"kind": "LO_ONLY", "value": "LO"}
        tasks.append({
            "task": row["name"],
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

    taskset = export_taskset(target.ordered_tasks, target.provenance.get("budget_by_task"))
    env_domain = _environment_domain(taskset)
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
    numeric = {
        "schema_version": "numeric_observation_binding_v9_1",
        "feature_names": list(target.feature_names),
        "observation_mode": "v11_full_10d",
        "raw_feature_equations_binding": "EXECUTABLE_SOURCE_BOUND",
        "feature_history_update_binding": "EXECUTABLE_SOURCE_BOUND",
        "normalization_binding": "EXECUTABLE_SOURCE_BOUND",
        "clipping": {"input_min": inventory["fixed_point_config"]["input_min"],
                     "input_max": inventory["fixed_point_config"]["input_max"]},
        "quantization": dict(inventory["fixed_point_config"]),
        "rounding": inventory["fixed_point_config"]["rounding_mode"],
        "numeric_width_overflow": "Python integer after binary64 observation; fixed-point output explicitly bounded",
        "float_semantics": "CPython IEEE-754 binary64 for executable observation before Decimal(str(value)) quantization",
        "comparison_semantics": "integer tree left iff q <= threshold_int; right iff q > threshold_int",
        "source_hashes": {k: source_hashes[k] for k in ("observation_runtime", "observation_formal_schema", "feature_state", "fixed_point", "integer_tree")},
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
        "source_hashes": {k: source_hashes[k] for k in ("runtime_adapter", "action_runtime", "budget_runtime")},
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
    environment_binding = {
        "schema_version": "environment_binding_v9_1",
        "domain": env_domain,
        "release_model_hash": sha256_object({"release_model": env_domain["release_model"], "tasks": env_domain["tasks"]}),
        "initial_release_state_hash": sha256_object({row["task"]: 0 for row in env_domain["tasks"]}),
        "release_domain_hash": sha256_object([row["release_domain"] for row in env_domain["tasks"]]),
        "classification_domain_hash": sha256_object([row["classification_domain"] for row in env_domain["tasks"]]),
        "demand_domain_hash": sha256_object([row["actual_demand_domain"] for row in env_domain["tasks"]]),
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
        },
        "source_manifest_semantic_hash": source_manifest["semantic_hash"],
        "formal_inputs_dir_hash": sha256_object(sorted(p.name for p in formal_inputs_dir.iterdir())),
    }
    bindings["binding_root_hash"] = sha256_object(bindings)
    return bindings
