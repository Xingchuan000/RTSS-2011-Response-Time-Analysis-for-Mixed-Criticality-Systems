"""seed workspace 的轻量构造函数。"""

from pathlib import Path


def workspace_path(seed_dir: Path, output_dir: Path | None = None) -> Path:
    return output_dir or (seed_dir / ".formal_workspace")
