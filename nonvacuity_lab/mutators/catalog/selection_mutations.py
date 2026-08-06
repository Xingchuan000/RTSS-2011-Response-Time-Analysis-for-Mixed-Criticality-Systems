"""Current-source coherent bindings for B1/B2/B3/B5 selection mutations."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ..python_binding import bind_symbol
from ...canonical import python_symbol_hash


@dataclass(frozen=True)
class SelectionPatch:
    role: str
    target_file: str
    target_symbol: str
    before_ast_hash: str
    before_snippet: str
    after_snippet: str
    occurrence: int = 1

    def to_dict(self) -> dict:
        return self.__dict__.copy()


def _bound_patch(
    source_root: Path,
    *,
    role: str,
    relative: str,
    target: str,
    before: str,
    after: str,
    occurrence: int = 1,
) -> SelectionPatch:
    path = source_root / relative
    source = path.read_text(encoding="utf-8")
    bound = bind_symbol(source, target)
    actual = bound.source.count(before)
    if actual != occurrence:
        raise ValueError(
            f"SELECTION_PATCH_BINDING_NOT_UNIQUE:{relative}:{target}:"
            f"expected={occurrence}:actual={actual}"
        )
    return SelectionPatch(
        role=role,
        target_file=relative,
        target_symbol=target,
        before_ast_hash=python_symbol_hash(source, target),
        before_snippet=before,
        after_snippet=after,
        occurrence=occurrence,
    )


def _policy_patch(source_root: Path, *, semantics: str) -> SelectionPatch:
    relative = "amc_py/viper/tree_policy.py"
    target = "IntegerTreeBudgetPolicy.select_action_id"
    path = source_root / relative
    source = path.read_text(encoding="utf-8")
    bound = bind_symbol(source, target)
    marker = "        if valid_action_mask is None:\n"
    start = bound.source.index(marker)
    before = bound.source[start:]
    if semantics == "raw_top1":
        after = """        if valid_action_mask is None:
            if trace is not None:
                base.update(trace)
            return raw_top1, base
        if len(valid_action_mask) != len(self.action_definitions):
            raise ValueError("valid_action_mask 长度必须与 action_dim 一致")
        raw_invalid = not bool(valid_action_mask[raw_top1])
        base.update({
            "tree_raw_top1_invalid": raw_invalid,
            "tree_fallback_used": False,
            "tree_no_valid_action": False,
            "tree_selected_action_id": raw_top1,
            "tree_selected_rank": 0,
        })
        if trace is not None:
            base.update(trace)
        return raw_top1, base
"""
    elif semantics == "top1_valid_else_noop":
        after = """        if valid_action_mask is None:
            if trace is not None:
                base.update(trace)
            return raw_top1, base
        if len(valid_action_mask) != len(self.action_definitions):
            raise ValueError("valid_action_mask 长度必须与 action_dim 一致")
        raw_invalid = not bool(valid_action_mask[raw_top1])
        base["tree_raw_top1_invalid"] = raw_invalid
        if raw_invalid:
            base.update({
                "tree_fallback_used": False,
                "tree_top1_invalid_noop": True,
                "tree_no_valid_action": False,
                "tree_selected_action_id": None,
                "tree_selected_rank": None,
            })
            if trace is not None:
                base.update(trace)
            return None, base
        base["tree_top1_invalid_noop"] = False
        if trace is not None:
            base.update(trace)
        return raw_top1, base
"""
    elif semantics == "all_invalid_force_top1":
        after = """        if valid_action_mask is None:
            if trace is not None:
                base.update(trace)
            return raw_top1, base
        if len(valid_action_mask) != len(self.action_definitions):
            raise ValueError("valid_action_mask 长度必须与 action_dim 一致")
        raw_invalid = not bool(valid_action_mask[raw_top1])
        base["tree_raw_top1_invalid"] = raw_invalid
        for rank, candidate in enumerate(ranking):
            if bool(valid_action_mask[candidate]):
                base.update({
                    "tree_fallback_used": bool(raw_invalid and rank > 0),
                    "tree_selected_action_id": candidate,
                    "tree_selected_rank": rank,
                })
                if trace is not None:
                    base.update(trace)
                return candidate, base
        base.update({
            "tree_no_valid_action": True,
            "tree_selected_action_id": raw_top1,
            "tree_selected_rank": 0,
        })
        if trace is not None:
            base.update(trace)
        return raw_top1, base
"""
    else:
        raise ValueError(f"UNSUPPORTED_SELECTION_SEMANTICS:{semantics}")
    return SelectionPatch(
        role="DEPLOYED_SELECTION",
        target_file=relative,
        target_symbol=target,
        before_ast_hash=python_symbol_hash(source, target),
        before_snippet=before,
        after_snippet=after,
    )


def _deployed_apply_patch(source_root: Path, *, semantics: str) -> SelectionPatch:
    if semantics != "unchecked_apply":
        raise ValueError(f"UNSUPPORTED_APPLY_SEMANTICS:{semantics}")
    before = '''                evaluation = self.evaluate_budget_candidate(
                    action=action,
                    budget_before=budget_before,
                    hi_pressure_threshold=float(
                        self._reward_mode_config.reward_parameters.get("hi_mode_pressure_threshold", 0.8)
                    ),
                    lo_pressure_threshold=float(
                        self._reward_mode_config.reward_parameters.get("lo_pressure_threshold", 0.8)
                    ),
                )
'''
    after = '''                evaluation = self.evaluate_budget_candidate_unchecked(
                    action=action, budget_before=budget_before
                )
'''
    return _bound_patch(
        source_root,
        role="DEPLOYED_APPLY",
        relative="amc_py/rl/env.py",
        target="AmcBudgetEnv.step",
        before=before,
        after=after,
    )


def _formal_selection_patch(source_root: Path, *, semantics: str) -> SelectionPatch:
    return _bound_patch(
        source_root,
        role="FROZEN_SELECTION",
        relative="formal_toolchain/adapters/s185_target.py",
        target="build_target",
        before='selection_semantics="ranked_first_valid"',
        after=f'selection_semantics="{semantics}"',
    )


def _formal_apply_patches(source_root: Path, *, semantics: str) -> tuple[SelectionPatch, SelectionPatch]:
    if semantics != "unchecked_apply":
        raise ValueError(f"UNSUPPORTED_APPLY_SEMANTICS:{semantics}")
    behavior = _bound_patch(
        source_root,
        role="FROZEN_APPLY",
        relative="formal_toolchain/adapters/amc_real_runtime_adapter.py",
        target="AMCRealRuntimeAdapter.apply_action",
        before='''        diagnosis = environment.diagnose_candidate_budget_update(new_budgets=after)
        if not diagnosis.accepted:
            raise RuntimeError("REAL_RUNTIME_ACTION_REJECTED_BY_ENVIRONMENT")
        return after
''',
        after='''        return after
''',
    )
    contract = _bound_patch(
        source_root,
        role="FROZEN_APPLY",
        relative="formal_toolchain/adapters/amc_real_runtime_adapter.py",
        target="AMCRealRuntimeAdapter.export_mask_contract",
        before='"step_guard_semantics": "checked"',
        after='"step_guard_semantics": "unchecked_apply"',
    )
    return behavior, contract


def build_selection_catalog(source_root: Path) -> dict[str, tuple[dict, ...]]:
    """Build complete deployed/formal source overlays for B1/B2/B3/B5."""

    raw_top1 = (
        _policy_patch(source_root, semantics="raw_top1").to_dict(),
        _deployed_apply_patch(source_root, semantics="unchecked_apply").to_dict(),
        _formal_selection_patch(source_root, semantics="raw_top1").to_dict(),
        *[item.to_dict() for item in _formal_apply_patches(source_root, semantics="unchecked_apply")],
    )
    return {
        "B1": raw_top1,
        "B2": (
            _policy_patch(source_root, semantics="top1_valid_else_noop").to_dict(),
            _formal_selection_patch(source_root, semantics="top1_valid_else_noop").to_dict(),
        ),
        "B3": (
            _policy_patch(source_root, semantics="all_invalid_force_top1").to_dict(),
            _deployed_apply_patch(source_root, semantics="unchecked_apply").to_dict(),
            _formal_selection_patch(source_root, semantics="all_invalid_force_top1").to_dict(),
            *[item.to_dict() for item in _formal_apply_patches(source_root, semantics="unchecked_apply")],
        ),
    }
