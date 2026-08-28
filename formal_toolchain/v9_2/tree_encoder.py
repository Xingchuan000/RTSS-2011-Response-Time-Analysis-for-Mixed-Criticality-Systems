"""Integer CART encoding; no concrete leaf is fixed into the SMT formula."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import z3


@dataclass(frozen=True, slots=True)
class TreeEncoding:
    leaf_id: z3.ArithRef
    ranking: tuple[z3.ArithRef, ...]
    constraints: tuple[z3.BoolRef, ...]


def _indexes(tree: Any):
    return ({node.node_id: node for node in tree.nodes},
            {leaf.node_id: leaf for leaf in tree.leaves})


def encode_tree_leaf_and_ranking(
    q: Sequence[z3.ArithRef], tree: Any, *, prefix: str = "tree"
) -> TreeEncoding:
    nodes, leaves = _indexes(tree)
    if len(q) != int(tree.state_dim):
        raise ValueError("TREE_STATE_DIMENSION_MISMATCH")
    constraints: list[z3.BoolRef] = []

    def visit(node_id: int) -> tuple[z3.ArithRef, tuple[z3.ArithRef, ...]]:
        if node_id in leaves:
            leaf = leaves[node_id]
            return z3.IntVal(leaf.node_id), tuple(z3.IntVal(value) for value in leaf.action_ranking)
        node = nodes[node_id]
        predicate = q[node.feature_index] <= node.threshold_int
        left_leaf, left_rank = visit(node.left_child)
        right_leaf, right_rank = visit(node.right_child)
        return z3.If(predicate, left_leaf, right_leaf), tuple(
            z3.If(predicate, left_rank[index], right_rank[index])
            for index in range(int(tree.action_dim))
        )

    leaf_id, ranking = visit(int(tree.root_node_id))
    return TreeEncoding(leaf_id, ranking, tuple(constraints))


__all__ = ["TreeEncoding", "encode_tree_leaf_and_ranking"]
