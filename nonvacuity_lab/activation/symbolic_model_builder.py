from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json

from ..canonical import file_hash
from .action_semantics_encoder import encode_candidate_budget, parse_ratio
from .model_schema import IntDomain, SymbolicAction, SymbolicTaskBudget
from .tree_guard_encoder import encode_feature_domains, encode_leaf_guard, find_leaf_path


@dataclass
class BuiltSymbolicProblem:
    solver: object
    feature_vars: dict
    budget_vars: dict
    formulas: dict
    metadata: dict


def _read(value):
    if isinstance(value, (str, Path)):
        return json.loads(Path(value).read_text(encoding="utf-8"))
    return value


def _path(binding, name):
    if isinstance(binding, dict):
        return binding.get(name)
    return getattr(binding, name, None)


def _load_budgets(binding):
    taskset = _read(_path(binding, "taskset_path") or _path(binding, "taskset"))
    envelope = _read(_path(binding, "certified_envelope_path") or _path(binding, "certified_envelope"))
    floors = _read(_path(binding, "budget_floor_path") or _path(binding, "budget_floors"))
    result = []
    for task in taskset.get("tasks", []):
        task_id = str(task["task_id"])
        reference = int(task["reference_budget"])
        floor = int(floors[task_id] if isinstance(floors, dict) and task_id in floors else task.get("minimum_budget", 0))
        upper = int(envelope[task_id] if isinstance(envelope, dict) and task_id in envelope else task.get("certified_upper_bound", reference))
        result.append(SymbolicTaskBudget(task_id, str(task["criticality"]), IntDomain(f"budget__{task_id}", floor, upper), floor, upper, reference))
    return tuple(result)


def build_symbolic_problem(resolved_target, binding, formula_kind):
    import z3
    target = dict(resolved_target)
    tree_path = Path(target["tree_path"])
    if target.get("tree_sha256") and file_hash(tree_path) != target["tree_sha256"]:
        raise ValueError("tree hash mismatch")
    tree = _read(tree_path)
    feature_schema = _read(_path(binding, "feature_schema_path") or _path(binding, "feature_schema"))
    budgets = _load_budgets(binding)
    feature_vars = {int(item["index"]): z3.Int(f"q_{item['index']}") for item in feature_schema.get("features", [])}
    budget_vars = {item.task_id: z3.Int(f"budget__{item.task_id}") for item in budgets}
    path = find_leaf_path(tree, int(target["leaf_id"]))
    guard = encode_leaf_guard(path, feature_vars)
    domains = encode_feature_domains(feature_schema, feature_vars)
    invariant_terms = []
    for item in budgets:
        var = budget_vars[item.task_id]
        invariant_terms.extend((var >= item.minimum_budget, var <= item.certified_upper_bound))
        if item.criticality == "HI":
            invariant_terms.append(var >= item.reference_budget)
    invariant = z3.And(*invariant_terms) if invariant_terms else z3.BoolVal(True)
    action_defs = _read(_path(binding, "action_definitions_path") or _path(binding, "action_definitions"))
    def raw_for(action_id):
        value = action_defs.get(str(int(action_id)), action_defs.get(int(action_id))) if isinstance(action_defs, dict) else None
        if value is None and isinstance(action_defs, dict):
            value = next((a for a in action_defs.get("actions", []) if int(a["action_id"]) == int(action_id)), None)
        return value
    raw_action = raw_for(target["action_id"])
    if not isinstance(raw_action, dict):
        raise ValueError("action definition missing")
    action = SymbolicAction(int(target["action_id"]), raw_action.get("task_id"), str(raw_action.get("operation", "noop")), parse_ratio(raw_action["ratio"]) if raw_action.get("ratio") is not None else None, int(raw_action.get("minimum_increment", 0)), str(raw_action.get("rounding_mode", "ceil")))
    task = next((item for item in budgets if item.task_id == action.task_id), None)
    if task is None:
        raise ValueError("action task missing")
    current = budget_vars[task.task_id]
    candidate = encode_candidate_budget(action, current)
    legal = z3.And(candidate >= task.minimum_budget, candidate <= task.certified_upper_bound, *( [candidate >= task.reference_budget] if task.criticality == "HI" else []))
    post_terms = [candidate >= task.minimum_budget, candidate <= task.certified_upper_bound]
    if task.criticality == "HI":
        post_terms.append(candidate >= task.reference_budget)
    post = z3.And(*post_terms)
    solver = z3.Solver()
    solver.add(domains, invariant)
    if formula_kind == "A_MASK_REJECT":
        solver.add(guard, z3.Not(legal))
    elif formula_kind == "B_RAW_TOP1_BREAKS_INVARIANT":
        solver.add(guard, z3.Not(legal), z3.Not(post))
    elif formula_kind == "B3_ALL_INVALID":
        ranking = [int(item) for item in target.get("original_ranking", target.get("ranking", (target["action_id"],)))]
        legal_terms = []
        for action_id in ranking:
            candidate_action = raw_for(action_id)
            if not isinstance(candidate_action, dict):
                raise ValueError(f"action definition missing: {action_id}")
            candidate_obj = SymbolicAction(action_id, candidate_action.get("task_id"), str(candidate_action.get("operation", "noop")), parse_ratio(candidate_action["ratio"]) if candidate_action.get("ratio") is not None else None, int(candidate_action.get("minimum_increment", 0)), str(candidate_action.get("rounding_mode", "ceil")))
            candidate_task = next((item for item in budgets if item.task_id == candidate_obj.task_id), None)
            if candidate_task is None:
                continue
            candidate_var = budget_vars[candidate_task.task_id]
            candidate_value = encode_candidate_budget(candidate_obj, candidate_var)
            legal_terms.append(z3.And(candidate_value >= candidate_task.minimum_budget, candidate_value <= candidate_task.certified_upper_bound, *([candidate_value >= candidate_task.reference_budget] if candidate_task.criticality == "HI" else [])))
        if not legal_terms:
            raise ValueError("B3 requires ranked action definitions")
        solver.add(guard, z3.And(*(z3.Not(item) for item in legal_terms)))
    elif formula_kind == "B4_GUARD_NECESSITY":
        disabled = target.get("disabled_guard_constraint")
        if not isinstance(disabled, dict):
            raise ValueError("B4 requires disabled_guard_constraint")
        without_guard = _encode_simple_constraint(z3, disabled, feature_vars, budget_vars)
        solver.add(guard, invariant, z3.Not(legal), without_guard, z3.Not(post))
    else:
        raise ValueError(f"unknown formula kind {formula_kind}")
    return BuiltSymbolicProblem(solver, feature_vars, budget_vars, {"leaf_guard": guard, "feature_domains": domains, "pre_invariant": invariant, "action_legal": legal, "post_invariant": post}, {"tree_sha256": file_hash(tree_path), "leaf_id": int(target["leaf_id"]), "action_id": int(target["action_id"]), "formula_kind": formula_kind})


def _encode_simple_constraint(z3, constraint, feature_vars, budget_vars):
    name = str(constraint["variable"])
    variable = feature_vars.get(int(name)) if name.isdigit() else budget_vars.get(name)
    if variable is None:
        raise ValueError(f"constraint variable missing: {name}")
    op, value = str(constraint["op"]), int(constraint["value"])
    return {"<=": variable <= value, "<": variable < value, ">=": variable >= value, ">": variable > value, "==": variable == value, "!=": variable != value}[op]
