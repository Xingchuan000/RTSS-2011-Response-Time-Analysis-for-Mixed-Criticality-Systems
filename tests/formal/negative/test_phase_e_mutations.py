"""Phase E 规定的语义 mutation 必须改变 IR 或被拒绝。"""

from formal_toolchain.binding.python_ast_ir import function_to_ir


def test_deadline_priority_mutation_changes_comparison_ir():
    first = function_to_ir("def f(x, d):\n    if x > d:\n        return True\n    return False\n", "f")
    mutated = function_to_ir("def f(x, d):\n    if x >= d:\n        return True\n    return False\n", "f")
    assert first != mutated


def test_unsupported_dynamic_and_unbounded_mutations_fail_closed():
    for source in (
        "def f(x):\n    return eval(x)\n",
        "def f(x):\n    while x:\n        x -= 1\n    return x\n",
        "def f(x):\n    __import__('os')\n    return x\n",
    ):
        result = function_to_ir(source, "f")
        assert result["status"] == "UNRESOLVED"
        assert result["failure"]["code"] == "UNSUPPORTED_AST_NODE"


def test_all_invalid_fallback_mutation_is_observable():
    first = function_to_ir("def f(mask):\n    if mask:\n        return None\n    return None\n", "f")
    mutated = function_to_ir("def f(mask):\n    if mask:\n        return None\n    return 0\n", "f")
    assert first != mutated
