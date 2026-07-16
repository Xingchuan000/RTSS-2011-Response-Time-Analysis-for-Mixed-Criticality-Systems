"""candidate summary 的窄写入接口；不生成 verified claim。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping


def write_candidate_report(summary: Mapping[str, Any], output: Path) -> None:
    """把 candidate 状态写成展示文件，明确保留 CANDIDATE。"""

    Path(output).write_text(
        "# Candidate proof report\n\n" + json.dumps(dict(summary), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
