#!/usr/bin/env python3
"""运行并记录 Phase A 要求的现有回归基线。

runner 不把不存在的测试路径当作成功，也不把输出写入 proof bundle；它只生成
开发阶段的 baseline JSON，方便检查新增 formal_toolchain 是否导致旧测试退化。
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import re
from datetime import datetime, timezone
from pathlib import Path


DEFAULT_TESTS = [
    "tests/test_event_models.py",
    "tests/test_event_runtime_c_amc_sem.py",
    "tests/test_runtime_mode_recovery.py",
    "tests/test_viper_fixed_point.py",
    "tests/test_viper_integer_tree.py",
    "tests/test_viper_tree_policy.py",
    "tests/test_rl_action_mask.py",
]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("baseline_result.json"))
    parser.add_argument("tests", nargs="*", default=DEFAULT_TESTS)
    args = parser.parse_args(argv)
    missing = [item for item in args.tests if not Path(item).is_file()]
    result = {"timestamp_utc": datetime.now(timezone.utc).isoformat(),
              "python": sys.version, "platform": platform.platform(),
              "tests": args.tests}
    if missing:
        result.update(status="FAIL", missing_tests=missing)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return 2
    completed = subprocess.run([sys.executable, "-m", "pytest", "-q", *args.tests], text=True,
                               capture_output=True)
    print(completed.stdout, end="")
    print(completed.stderr, end="", file=sys.stderr)
    summary = re.search(r"(?:(?P<failed>\d+) failed, )?(?P<passed>\d+) passed(?:, (?P<skipped>\d+) skipped)?", completed.stdout)
    passed = int(summary.group("passed")) if summary else 0
    failed = int(summary.group("failed") or 0) if summary else 0
    skipped = int(summary.group("skipped") or 0) if summary else 0
    result.update(status="PASS" if completed.returncode == 0 and passed == 38 else "FAIL",
                  returncode=completed.returncode, passed=passed, failed=failed,
                  skipped=skipped, collected=passed + failed + skipped)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    # 计划要求固定 38 项；数量漂移即使 pytest 本身返回 0，也必须让 runner 失败。
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
