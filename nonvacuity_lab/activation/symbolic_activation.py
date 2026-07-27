"""Finite-domain/Z3 activation solver for B1/B3/B4 witnesses."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from ..schema import ActivationStatus
from .policy_witness import evaluate_policy_witness
from .schema import ActivationResult


def solve_symbolic_activation(
    *,
    mutation_id: str,
    rule: Mapping[str, Any],
    output_path: Path | None = None,
) -> ActivationResult:
    symbolic_model = rule.get("symbolic_model")
    if isinstance(symbolic_model, Mapping):
        solved = _solve_z3_model(
            mutation_id=mutation_id,
            activation_kind=str(rule.get("activation_kind", "b1")).lower(),
            symbolic_model=symbolic_model,
            output_path=output_path,
        )
        if solved.status in {
            ActivationStatus.ACTIVATED,
            ActivationStatus.ACTIVATION_SETUP_INVALID,
        }:
            return solved
        if not rule.get("candidate_witnesses"):
            return solved
    candidates = rule.get("candidate_witnesses", ())
    if not isinstance(candidates, list):
        return ActivationResult(
            mutation_id=mutation_id,
            status=ActivationStatus.ACTIVATION_SETUP_INVALID,
            details={"reason": "candidate_witnesses 必须为 array"},
        )
    activation_kind = str(rule.get("activation_kind", "b1")).lower()
    key = {"b1": "activated_b1", "b3": "activated_b3", "b4": "activated_b4"}.get(
        activation_kind
    )
    if key is None:
        return ActivationResult(
            mutation_id=mutation_id,
            status=ActivationStatus.ACTIVATION_SETUP_INVALID,
            details={"reason": f"不支持的 activation_kind: {activation_kind}"},
        )
    evaluated = []
    try:
        for candidate in candidates:
            result = evaluate_policy_witness(candidate)
            evaluated.append(result)
            if result[key]:
                if output_path is not None:
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    output_path.write_text(
                        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
                        encoding="utf-8",
                    )
                return ActivationResult(
                    mutation_id=mutation_id,
                    status=ActivationStatus.ACTIVATED,
                    evidence_modes=("SYMBOLIC",),
                    leaf_id=result.get("leaf_id"),
                    action_id=result.get("action_id"),
                    all_invalid_count=int(bool(result.get("all_invalid"))),
                    guard_satisfiable=bool(result.get("leaf_guard_satisfied")),
                    illegal_action_witness=str(output_path) if output_path else None,
                    post_invariant_violation=bool(
                        result.get("mutated_post_invariant_violation")
                    ),
                    details={
                        "activation_kind": activation_kind,
                        "candidate_count": len(candidates),
                        "witness": result,
                    },
                )
    except (KeyError, TypeError, ValueError) as exc:
        return ActivationResult(
            mutation_id=mutation_id,
            status=ActivationStatus.ACTIVATION_SETUP_INVALID,
            evidence_modes=("SYMBOLIC",),
            details={"reason": str(exc)},
        )
    return ActivationResult(
        mutation_id=mutation_id,
        status=ActivationStatus.NOT_ACTIVATED,
        evidence_modes=("SYMBOLIC",),
        guard_satisfiable=any(
            bool(item.get("leaf_guard_satisfied")) for item in evaluated
        ),
        details={
            "activation_kind": activation_kind,
            "candidate_count": len(candidates),
            "evaluated": evaluated,
        },
    )


def _solve_z3_model(
    *,
    mutation_id: str,
    activation_kind: str,
    symbolic_model: Mapping[str, Any],
    output_path: Path | None,
) -> ActivationResult:
    try:
        import z3
    except ImportError:
        return ActivationResult(
            mutation_id=mutation_id,
            status=ActivationStatus.ACTIVATION_INCONCLUSIVE,
            evidence_modes=("SYMBOLIC",),
            details={"reason": "z3 unavailable"},
        )
    try:
        variables_raw = symbolic_model["variables"]
        lower = symbolic_model["lower"]
        upper = symbolic_model["upper"]
        ranking = [int(item) for item in symbolic_model["ranking"]]
        actions = symbolic_model["actions"]
        if (
            not isinstance(variables_raw, Mapping)
            or not isinstance(lower, Mapping)
            or not isinstance(upper, Mapping)
            or not isinstance(actions, Mapping)
            or not ranking
        ):
            raise ValueError("symbolic_model variables/lower/upper/actions/ranking 非法")
        variables = {str(name): z3.Int(str(name)) for name in variables_raw}
        solver = z3.Solver()
        for name, spec in variables_raw.items():
            if not isinstance(spec, Mapping):
                raise ValueError(f"variable spec 非法: {name}")
            if "min" in spec:
                solver.add(variables[str(name)] >= int(spec["min"]))
            if "max" in spec:
                solver.add(variables[str(name)] <= int(spec["max"]))
        for name, value in lower.items():
            solver.add(variables[str(name)] >= int(value))
        for name, value in upper.items():
            solver.add(variables[str(name)] <= int(value))
        for constraint in symbolic_model.get("leaf_guard", ()):
            solver.add(_z3_constraint(z3, variables, constraint))

        legal: dict[int, Any] = {}
        post_invariant: dict[int, Any] = {}
        for action_id in ranking:
            action = actions.get(str(action_id), actions.get(action_id))
            if not isinstance(action, Mapping):
                raise ValueError(f"symbolic action 缺失: {action_id}")
            delta = action.get("delta", {})
            if not isinstance(delta, Mapping):
                raise ValueError(f"symbolic action delta 非法: {action_id}")
            post_terms = []
            for name, variable in variables.items():
                post = variable + int(delta.get(name, 0))
                if name in lower:
                    post_terms.append(post >= int(lower[name]))
                if name in upper:
                    post_terms.append(post <= int(upper[name]))
            post_invariant[action_id] = z3.And(*post_terms) if post_terms else z3.BoolVal(True)
            guard_terms = [
                _z3_constraint(z3, variables, guard)
                for guard in action.get("guards", ())
            ]
            legal[action_id] = z3.And(
                post_invariant[action_id],
                *(guard_terms or [z3.BoolVal(True)]),
            )
        top1 = ranking[0]
        if activation_kind == "b1":
            solver.add(z3.Not(legal[top1]))
            solver.add(z3.Or(*(legal[action] for action in ranking[1:])))
            solver.add(z3.Not(post_invariant[top1]))
        elif activation_kind == "b3":
            solver.add(z3.And(*(z3.Not(legal[action]) for action in ranking)))
        elif activation_kind == "b4":
            disabled_guard = str(symbolic_model["disabled_guard"])
            top_action = actions.get(str(top1), actions.get(top1))
            target_guards = [
                guard
                for guard in top_action.get("guards", ())
                if str(guard.get("name")) == disabled_guard
            ]
            other_guards = [
                guard
                for guard in top_action.get("guards", ())
                if str(guard.get("name")) != disabled_guard
            ]
            if len(target_guards) != 1:
                raise ValueError("B4 symbolic model 必须有且仅有一个目标 guard")
            solver.add(z3.Not(_z3_constraint(z3, variables, target_guards[0])))
            solver.add(post_invariant[top1])
            solver.add(
                z3.And(
                    *(
                        [_z3_constraint(z3, variables, guard) for guard in other_guards]
                        or [z3.BoolVal(True)]
                    )
                )
            )
        else:
            raise ValueError(f"不支持的 activation_kind: {activation_kind}")
        if solver.check() != z3.sat:
            return ActivationResult(
                mutation_id=mutation_id,
                status=ActivationStatus.NOT_ACTIVATED,
                evidence_modes=("SYMBOLIC",),
                guard_satisfiable=False,
                details={"activation_kind": activation_kind, "solver": "Z3", "result": "UNSAT"},
            )
        model = solver.model()
        state = {
            name: model.eval(variable, model_completion=True).as_long()
            for name, variable in variables.items()
        }
        witness = {
            "leaf_id": symbolic_model.get("leaf_id"),
            "state": state,
            "lower": dict(lower),
            "upper": dict(upper),
            "leaf_guard": list(symbolic_model.get("leaf_guard", ())),
            "ranking": ranking,
            "actions": dict(actions),
        }
        if activation_kind == "b4":
            witness["disabled_guard"] = symbolic_model["disabled_guard"]
        evaluated = evaluate_policy_witness(witness)
        if output_path is not None:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(
                json.dumps(evaluated, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
        return ActivationResult(
            mutation_id=mutation_id,
            status=ActivationStatus.ACTIVATED,
            evidence_modes=("SYMBOLIC",),
            leaf_id=evaluated.get("leaf_id"),
            action_id=evaluated.get("action_id"),
            all_invalid_count=int(bool(evaluated.get("all_invalid"))),
            guard_satisfiable=True,
            illegal_action_witness=str(output_path) if output_path else None,
            post_invariant_violation=bool(
                evaluated.get("mutated_post_invariant_violation")
            ),
            details={
                "activation_kind": activation_kind,
                "solver": "Z3",
                "result": "SAT",
                "witness": evaluated,
            },
        )
    except (KeyError, TypeError, ValueError, z3.Z3Exception) as exc:
        return ActivationResult(
            mutation_id=mutation_id,
            status=ActivationStatus.ACTIVATION_SETUP_INVALID,
            evidence_modes=("SYMBOLIC",),
            details={"reason": str(exc), "solver": "Z3"},
        )


def _z3_constraint(z3: Any, variables: Mapping[str, Any], raw: Mapping[str, Any]):
    field = str(raw["field"])
    expression = variables[field]
    target = int(raw["value"])
    op = str(raw["op"])
    comparisons = {
        "<=": expression <= target,
        "<": expression < target,
        ">=": expression >= target,
        ">": expression > target,
        "==": expression == target,
        "!=": expression != target,
    }
    if op not in comparisons:
        raise ValueError(f"不支持的 symbolic constraint op: {op}")
    return comparisons[op]
