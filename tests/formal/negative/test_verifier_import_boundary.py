"""R06：verifier 源码不能依赖 compiler 或旧 raw-evidence 总入口。"""

import ast
from pathlib import Path


def test_verifier_does_not_import_compiler_or_formal_checks() -> None:
    root = Path(__file__).parents[3] / "formal_toolchain" / "verifier"
    forbidden = {"formal_toolchain.compiler", "formal_toolchain.core.formal_checks"}
    imports: set[str] = set()
    for path in root.glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imports.add(node.module)
    assert not any(module == item or module.startswith(item + ".")
                   for module in imports for item in forbidden)
