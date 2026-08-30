"""Exact integer CART encoding specialized as leaf-path guards."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import z3


@dataclass(frozen=True, slots=True)
class TreeLeafCase:
    leaf_id: int
    ranking: tuple[int, ...]
    guard: z3.BoolRef


@dataclass(frozen=True, slots=True)
class TreeEncoding:
    leaf_id: z3.ArithRef
    leaf_cases: tuple[TreeLeafCase, ...]
    constraints: tuple[z3.BoolRef, ...]


def _indexes(tree: Any):
    return ({node.node_id: node for node in tree.nodes},
            {leaf.node_id: leaf for leaf in tree.leaves})


def encode_tree_leaf_and_ranking(
    q: Sequence[z3.ArithRef], tree: Any, *, prefix: str = "tree"
) -> TreeEncoding:
    """Compile CART as disjoint root-to-leaf path guards.

    The previous encoder materialized every ranking position as a nested ITE
    over the whole tree.  FirstValid then compared those symbolic action IDs
    against every mask entry, multiplying the tree ITEs by the action alphabet.
    Here the tree is represented once as its finite disjoint leaf paths.  Each
    leaf carries its already-concrete ranking, exactly matching deployed CART
    evaluation while keeping ranking arithmetic out of SMT.
    """

    nodes, leaves = _indexes(tree)
    if len(q) != int(tree.state_dim):
        raise ValueError("TREE_STATE_DIMENSION_MISMATCH")

    cases: list[TreeLeafCase] = []

    def visit(node_id: int, guard: z3.BoolRef) -> None:
        if node_id in leaves:
            leaf = leaves[node_id]
            ranking = tuple(int(value) for value in leaf.action_ranking)
            if len(ranking) != int(tree.action_dim):
                raise ValueError("TREE_ACTION_RANKING_DIMENSION_MISMATCH")
            cases.append(TreeLeafCase(int(leaf.node_id), ranking, guard))
            return
        node = nodes[node_id]
        predicate = q[int(node.feature_index)] <= int(node.threshold_int)
        visit(int(node.left_child), z3.And(guard, predicate))
        visit(int(node.right_child), z3.And(guard, z3.Not(predicate)))

    visit(int(tree.root_node_id), z3.BoolVal(True))
    if not cases:
        raise ValueError("TREE_HAS_NO_LEAVES")

    leaf_id = z3.Int(f"{prefix}.leaf_id")
    constraints: list[z3.BoolRef] = [
        z3.Or(*(case.guard for case in cases)),
    ]
    constraints.extend(
        z3.Implies(case.guard, leaf_id == case.leaf_id)
        for case in cases
    )
    return TreeEncoding(leaf_id, tuple(cases), tuple(constraints))


__all__ = ["TreeEncoding", "TreeLeafCase", "encode_tree_leaf_and_ranking"]
