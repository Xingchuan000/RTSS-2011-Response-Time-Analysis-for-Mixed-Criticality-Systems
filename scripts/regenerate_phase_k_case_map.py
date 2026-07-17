"""从当前源码重新生成 Phase K transition path map。

脚本不读取旧 map 中任何 guard/effect/hash 作为新值；旧文件只提供输出位置
和人工声明的 path ID 集合，所有语义字段均由当前 AST/CFG 现场计算。
"""

from __future__ import annotations

import json
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from formal_toolchain.adapters.source_manifest import build_source_manifest
from formal_toolchain.bridge.p0_case_manifest import p0_case_manifest_hash
from formal_toolchain.bridge.runtime_branch_map import PATH_SPECS, _path_row, build_normal_runtime_path_coverage
from formal_toolchain.core.hashing import sha256_object


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path,
                        default=ROOT / "tests/formal/fixtures/synthetic_p0/phase_k_case_map.json")
    args = parser.parse_args(argv)
    source_hash = build_source_manifest(ROOT)["semantic_hash"]
    paths = {spec[0]: _path_row(ROOT, spec) for spec in PATH_SPECS}
    coverage = build_normal_runtime_path_coverage(ROOT)
    if coverage.get("status") != "PASS":
        raise SystemExit(f"branch map coverage incomplete: {coverage}")
    result = {
        "schema_version": "phase_k_transition_path_map_v2_cfg_ir",
        "source_hash": source_hash,
        "paths": paths,
        "coverage": coverage,
        "case_manifest_hash": p0_case_manifest_hash(),
        "path_map_hash": sha256_object({"paths": paths, "coverage": coverage["artifact_hash"]}),
    }
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
