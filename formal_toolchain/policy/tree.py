"""Phase G02：整数树结构和叶区间证书。"""

from __future__ import annotations

from typing import Any

from amc_py.viper.integer_tree import IntegerTreeModel, evaluate_integer_tree


def validate_tree_and_leaf_partition(model: IntegerTreeModel, *, state_min: int = 0,
                                     state_max: int = 1_000_000) -> dict[str, Any]:
    """复用生产模型的结构校验，并以独立遍历确认每个叶均可达。"""
    # 这里不调用生产模块的私有 validator，避免 verifier 与 production
    # validator 同源；结构字段由本函数独立检查。
    node_ids = {node.node_id for node in model.nodes}
    leaf_ids = {leaf.node_id for leaf in model.leaves}
    if model.root_node_id not in node_ids | leaf_ids or len(node_ids) != len(model.nodes) or len(leaf_ids) != len(model.leaves):
        raise ValueError("integer tree root 或 node/leaf ID 非法")
    if len(model.feature_names) != model.state_dim or model.action_dim <= 0:
        raise ValueError("integer tree dimension 非法")
    if any(leaf.raw_action_id != leaf.action_ranking[0] or
           tuple(sorted(leaf.action_ranking)) != tuple(range(model.action_dim)) for leaf in model.leaves):
        raise ValueError("leaf ranking 不是完整 action permutation")
    for node in model.nodes:
        if not (0 <= node.feature_index < model.state_dim) or node.left_child == node.right_child:
            raise ValueError("tree node feature/branch 非法")
        if node.left_child not in node_ids | leaf_ids or node.right_child not in node_ids | leaf_ids:
            raise ValueError("tree child 不存在")
    reachable: set[int] = set()
    active: set[int] = set()
    nodes = {node.node_id: node for node in model.nodes}
    def visit(node_id: int) -> None:
        if node_id in active:
            raise ValueError("tree graph 存在环")
        if node_id in reachable:
            return
        active.add(node_id)
        if node_id in nodes:
            visit(nodes[node_id].left_child); visit(nodes[node_id].right_child)
        active.remove(node_id); reachable.add(node_id)
    visit(model.root_node_id)
    if reachable != node_ids | leaf_ids:
        raise ValueError("tree 存在不可达 node/leaf")
    if state_min > state_max:
        raise ValueError("状态整数域非法")
    try:
        import z3
    except ImportError:
        # synthetic fixture 的 fallback 是对 axis-aligned conjunction 的精确
        # 区间求解，不是随机抽样或边界猜测；真实复杂树仍由 Z3 backend 处理。
        return _validate_with_exact_intervals(model, state_min=state_min, state_max=state_max)
    leaves = {leaf.node_id for leaf in model.leaves}
    variables = [z3.Int(f"q_{i}") for i in range(model.state_dim)]
    bounds = [z3.And(value >= state_min, value <= state_max) for value in variables]
    guards: dict[int, list[Any]] = {}
    nodes = {node.node_id: node for node in model.nodes}

    def walk(node_id: int, conditions: list[Any]) -> None:
        if node_id in leaves:
            guards[node_id] = list(conditions)
            return
        node = nodes[node_id]
        feature = variables[node.feature_index]
        walk(node.left_child, conditions + [feature <= node.threshold_int])
        walk(node.right_child, conditions + [feature > node.threshold_int])

    walk(model.root_node_id, [])
    if set(guards) != leaves:
        raise ValueError("leaf guard 生成未覆盖全部 leaf")
    for leaf_id, guard in guards.items():
        solver = z3.Solver(); solver.add(*(bounds + guard))
        if solver.check() != z3.sat:
            raise ValueError(f"leaf {leaf_id} guard 不可满足")
    leaf_ids = sorted(guards)
    for index, left_id in enumerate(leaf_ids):
        for right_id in leaf_ids[index + 1:]:
            solver = z3.Solver(); solver.add(*(bounds + guards[left_id] + guards[right_id]))
            if solver.check() != z3.unsat:
                raise ValueError(f"leaf guard overlap: {left_id}/{right_id}")
    coverage_solver = z3.Solver()
    coverage_solver.add(*bounds)
    coverage_solver.add(z3.Not(z3.Or(*(z3.And(*guard) for guard in guards.values()))))
    if coverage_solver.check() != z3.unsat:
        raise ValueError("leaf guards 存在 uncovered region")
    return {"status": "PASS", "schema_version": "tree_leaf_partition_v1",
            "node_count": len(model.nodes), "leaf_count": len(model.leaves),
            "state_domain": [state_min, state_max], "coverage": "z3_exhaustive",
            "reachable_leaf_ids": sorted(leaves)}


def _validate_with_exact_intervals(model: IntegerTreeModel, *, state_min: int, state_max: int) -> dict[str, Any]:
    """无 Z3 时对树路径 conjunction 做精确整数区间判定。"""
    leaves = {leaf.node_id for leaf in model.leaves}; nodes = {node.node_id: node for node in model.nodes}
    guards: dict[int, list[tuple[int, str, int]]] = {}
    def walk(node_id: int, path: list[tuple[int, str, int]]) -> None:
        if node_id in leaves:
            guards[node_id] = list(path); return
        node = nodes[node_id]
        walk(node.left_child, path + [(node.feature_index, "<=", node.threshold_int)])
        walk(node.right_child, path + [(node.feature_index, ">", node.threshold_int)])
    walk(model.root_node_id, [])
    def interval(path: list[tuple[int, str, int]]) -> dict[int, tuple[int, int]] | None:
        result: dict[int, list[int]] = {}
        for feature, operator, threshold in path:
            current = result.setdefault(feature, [state_min, state_max])
            if operator == "<=": current[1] = min(current[1], threshold)
            else: current[0] = max(current[0], threshold + 1)
            if current[0] > current[1]: return None
        return {key: (value[0], value[1]) for key, value in result.items()}
    intervals = {leaf: interval(path) for leaf, path in guards.items()}
    if any(value is None for value in intervals.values()):
        raise ValueError("leaf guard 不可满足")
    # 两个树路径的 conjunction 只有在每个共同 feature 的区间均相交时才可满足。
    leaf_ids = sorted(intervals)
    for index, left in enumerate(leaf_ids):
        for right in leaf_ids[index + 1:]:
            shared = set(intervals[left]) & set(intervals[right])
            if all(max(intervals[left][feature][0], intervals[right][feature][0]) <=
                   min(intervals[left][feature][1], intervals[right][feature][1]) for feature in shared):
                raise ValueError(f"leaf guard overlap: {left}/{right}")
    return {"status": "PASS", "schema_version": "tree_leaf_partition_v1",
            "node_count": len(model.nodes), "leaf_count": len(model.leaves),
            "state_domain": [state_min, state_max], "coverage": "exact_interval_fallback",
            "reachable_leaf_ids": sorted(leaves)}
