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


def _optional_read(value, default):
    return default if value in (None, "") else _read(value)


def _load_budgets(binding):
    taskset = _read(_path(binding, "taskset_path") or _path(binding, "taskset"))
    envelope = _optional_read(
        _path(binding, "certified_envelope_path") or _path(binding, "certified_envelope"), {}
    )
    floors = _optional_read(
        _path(binding, "budget_floor_path") or _path(binding, "budget_floors"), {}
    )
    raw_tasks = taskset.get("tasks", taskset.get("ordered_tasks", []))
    result = []
    for task in raw_tasks:
        task_id = str(task.get("task_id", task.get("name")))
        reference = int(task.get("reference_budget", task.get("initial_runtime_budget", task.get("code_c_lo", 0))))
        floor = int(
            floors[task_id] if isinstance(floors, dict) and task_id in floors
            else task.get("minimum_budget", task.get("budget_floor", reference))
        )
        upper = int(
            envelope[task_id] if isinstance(envelope, dict) and task_id in envelope
            else task.get("certified_upper_bound", task.get("action_hard_upper", task.get("code_c_hi", reference)))
        )
        result.append(SymbolicTaskBudget(
            task_id, str(task["criticality"]), IntDomain(f"budget__{task_id}", floor, upper),
            floor, upper, reference,
        ))
    if not result:
        raise ValueError("taskset does not expose symbolic task budgets")
    return tuple(result)


def _normalize_feature_schema(raw, tree, binding):
    if isinstance(raw, dict) and isinstance(raw.get("features"), list):
        return raw
    names = []
    if isinstance(raw, dict):
        names = list(raw.get("feature_names", ()))
    elif isinstance(raw, list):
        names = list(raw)
    if not names:
        names = list(tree.get("feature_names", ()))
    scale = int(_path(binding, "feature_scale") or 1000)
    return {"features": [
        {"index": index, "name": str(name), "integer_lower": 0, "integer_upper": scale}
        for index, name in enumerate(names)
    ]}


def _normalize_action(raw_action, binding):
    if bool(raw_action.get("is_residual_ranked")) or bool(raw_action.get("is_constraint_guided_pair")):
        action_kind = raw_action.get("residual_action_type", "constraint_guided_pair")
        raise ValueError(f"UNSUPPORTED_DYNAMIC_ACTION_SLOT:{action_kind}")
    operation = raw_action.get("operation", raw_action.get("direction"))
    task_id = raw_action.get("task_id", raw_action.get("target_task"))
    ratio = raw_action.get("ratio")
    if operation is None:
        if raw_action.get("increase_task") is not None:
            operation = "increase"
            task_id = raw_action.get("increase_task")
            ratio = raw_action.get("increase_ratio", ratio)
        elif raw_action.get("decrease_tasks"):
            operation = "decrease"
            task_id = list(raw_action.get("decrease_tasks"))[0]
            ratio = raw_action.get("decrease_ratio", ratio)
        elif bool(raw_action.get("is_noop")) or raw_action.get("residual_action_type") == "noop":
            operation = "noop"
        else:
            raise ValueError("UNSUPPORTED_ACTION_DEFINITION_WITHOUT_STATIC_TASK")
    operation = str(operation)
    if ratio is None:
        ratio = _path(binding, "default_action_ratio") or "1/50"
    if bool(raw_action.get("is_noop")) or operation == "noop":
        ratio = None
        operation = "noop"
    rounding = raw_action.get("rounding_mode")
    if rounding is None:
        rounding = "ceil" if operation == "increase" else "floor"
    return SymbolicAction(
        int(raw_action["action_id"]), task_id, operation,
        parse_ratio(ratio) if ratio is not None else None,
        int(raw_action.get("minimum_increment", 1 if operation != "noop" else 0)),
        str(rounding),
    )


def build_symbolic_problem(resolved_target, binding, formula_kind):
    import z3
    target = dict(resolved_target)
    tree_path = Path(target["tree_path"])
    if target.get("tree_sha256") and file_hash(tree_path) != target["tree_sha256"]:
        raise ValueError("tree hash mismatch")
    tree = _read(tree_path)
    feature_schema_raw = _read(_path(binding, "feature_schema_path") or _path(binding, "feature_schema"))
    feature_schema = _normalize_feature_schema(feature_schema_raw, tree, binding)
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
    action = _normalize_action(raw_action, binding)
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
    elif formula_kind == "B2_NO_FIRST_VALID_DIFFERENCE":
        ranking = [
            int(item)
            for item in target.get(
                "original_ranking",
                target.get("ranking", (target["action_id"],)),
            )
        ]
        if not ranking or ranking[0] != int(target["action_id"]):
            raise ValueError("B2 requires the resolved action to be raw top-1")
        lower_legal_terms = []
        for action_id in ranking[1:]:
            candidate_action = raw_for(action_id)
            if not isinstance(candidate_action, dict):
                raise ValueError(f"action definition missing: {action_id}")
            candidate_obj = _normalize_action(candidate_action, binding)
            if candidate_obj.operation == "noop":
                lower_legal_terms.append(z3.BoolVal(True))
                continue
            candidate_task = next(
                (item for item in budgets if item.task_id == candidate_obj.task_id),
                None,
            )
            if candidate_task is None:
                raise ValueError(f"action task missing: {action_id}")
            candidate_var = budget_vars[candidate_task.task_id]
            candidate_value = encode_candidate_budget(candidate_obj, candidate_var)
            lower_legal_terms.append(
                z3.And(
                    candidate_value >= candidate_task.minimum_budget,
                    candidate_value <= candidate_task.certified_upper_bound,
                    *(
                        [candidate_value >= candidate_task.reference_budget]
                        if candidate_task.criticality == "HI"
                        else []
                    ),
                )
            )
        if not lower_legal_terms:
            raise ValueError("B2 requires at least one lower-ranked action")
        solver.add(guard, z3.Not(legal), z3.Or(*lower_legal_terms))
    elif formula_kind == "B_RAW_TOP1_BREAKS_INVARIANT":
        solver.add(guard, z3.Not(legal), z3.Not(post))
    elif formula_kind == "B3_ALL_INVALID":
        ranking = [int(item) for item in target.get("original_ranking", target.get("ranking", (target["action_id"],)))]
        legal_terms = []
        for action_id in ranking:
            candidate_action = raw_for(action_id)
            if not isinstance(candidate_action, dict):
                raise ValueError(f"action definition missing: {action_id}")
            candidate_obj = _normalize_action(candidate_action, binding)
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
        disabled = target.get("disabled_guard_constraint", {"constant": True})
        if not isinstance(disabled, dict):
            raise ValueError("B4 requires disabled_guard_constraint")
        without_guard = _encode_simple_constraint(z3, disabled, feature_vars, budget_vars)
        solver.add(guard, invariant, z3.Not(legal), without_guard, z3.Not(post))
    elif formula_kind == "C2_ROUNDING_DIFFERENCE":
        nearest_action = SymbolicAction(
            action.action_id, action.task_id, action.operation, action.ratio,
            action.minimum_increment, "nearest",
        )
        nearest_candidate = encode_candidate_budget(nearest_action, current)
        solver.add(guard, invariant, legal, candidate != nearest_candidate)
    else:
        raise ValueError(f"unknown formula kind {formula_kind}")
    return BuiltSymbolicProblem(solver, feature_vars, budget_vars, {"leaf_guard": guard, "feature_domains": domains, "pre_invariant": invariant, "action_legal": legal, "post_invariant": post}, {"tree_sha256": file_hash(tree_path), "leaf_id": int(target["leaf_id"]), "action_id": int(target["action_id"]), "formula_kind": formula_kind})


def _encode_simple_constraint(z3, constraint, feature_vars, budget_vars):
    if "constant" in constraint:
        return z3.BoolVal(bool(constraint["constant"]))
    name = str(constraint["variable"])
    variable = feature_vars.get(int(name)) if name.isdigit() else budget_vars.get(name)
    if variable is None:
        raise ValueError(f"constraint variable missing: {name}")
    op, value = str(constraint["op"]), int(constraint["value"])
    return {"<=": variable <= value, "<": variable < value, ">=": variable >= value, ">": variable > value, "==": variable == value, "!=": variable != value}[op]
