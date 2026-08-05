from __future__ import annotations

from .model_schema import PathPredicate


def _index(tree):
    result = {}
    nodes = tree.get("nodes")
    if isinstance(nodes, list):
        result.update({int(node.get("node_id", node.get("id"))): node for node in nodes})
    elif isinstance(nodes, dict):
        result.update({int(k): v for k, v in nodes.items()})
    leaves = tree.get("leaves")
    if isinstance(leaves, list):
        result.update({int(node.get("node_id", node.get("leaf_id", node.get("id")))): node for node in leaves})
    return result


def _is_leaf(node):
    return (
        "leaf_id" in node
        or "action_ranking" in node
        or "actions" in node
        or ("feature_index" not in node and "left_child" not in node and "left" not in node)
    )


def find_leaf_path(tree: dict, leaf_id: int) -> list[PathPredicate]:
    """Return the root-to-leaf guard for both legacy and integer_tree_v1 artifacts."""
    nodes = _index(tree)
    if not nodes:
        raise ValueError("integer tree does not expose internal nodes")
    root = int(tree.get("root_node_id", tree.get("root_id", tree.get("root", 0))))
    comparator = str(tree.get("comparator", "<="))

    def visit(node_id, path):
        node = nodes[node_id]
        if _is_leaf(node):
            actual_leaf = int(node.get("leaf_id", node.get("node_id", node.get("id"))))
            return path if actual_leaf == int(leaf_id) else None
        feature = int(node["feature_index"])
        threshold = int(node.get("threshold_int", node.get("threshold")))
        left = int(node.get("left_child", node.get("left")))
        right = int(node.get("right_child", node.get("right")))
        left_cmp = str(node.get("left_comparator", comparator))
        right_cmp = str(node.get("right_comparator", ">" if left_cmp == "<=" else ">="))
        found = visit(left, path + [PathPredicate(feature, left_cmp, threshold, node_id)])
        return found if found is not None else visit(
            right, path + [PathPredicate(feature, right_cmp, threshold, node_id)]
        )

    if root not in nodes:
        raise ValueError(f"root node not found: {root}")
    path = visit(root, [])
    if path is None:
        raise ValueError(f"leaf {leaf_id} not found")
    return path


def encode_leaf_guard(path, feature_vars):
    import z3
    terms = []
    for predicate in path:
        variable = feature_vars[predicate.feature_index]
        if predicate.comparator in ("<=", "le"):
            terms.append(variable <= predicate.threshold)
        elif predicate.comparator in ("<",):
            terms.append(variable < predicate.threshold)
        elif predicate.comparator in (">", "gt"):
            terms.append(variable > predicate.threshold)
        elif predicate.comparator in (">=", "ge"):
            terms.append(variable >= predicate.threshold)
        else:
            raise ValueError(f"unsupported comparator {predicate.comparator}")
    return z3.And(*terms) if terms else z3.BoolVal(True)


def encode_feature_domains(feature_schema, feature_vars):
    import z3
    terms = []
    for feature in feature_schema.get("features", []):
        variable = feature_vars[int(feature["index"])]
        terms.extend((
            variable >= int(feature.get("integer_lower", 0)),
            variable <= int(feature.get("integer_upper", 1000)),
        ))
    return z3.And(*terms) if terms else z3.BoolVal(True)
