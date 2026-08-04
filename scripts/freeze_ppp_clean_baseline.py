from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

# Keep the helper executable both as ``python scripts/...`` and from an
# installed package checkout.  The baseline freezer is a repository tool, so
# resolving the repository root here is intentional and does not alter proof
# semantics.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from formal_toolchain.workflow.prove_seed import prove_seed
from tests.formal.regression.ppp_baseline_helpers import extract_ppp_baseline


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed-dir", required=True, type=Path)
    parser.add_argument("--tree-variant", default="best_overall")
    parser.add_argument("--code-root", type=Path, default=Path.cwd())
    parser.add_argument("--work-dir", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)
    work_dir = args.work_dir.resolve()
    if work_dir.exists():
        shutil.rmtree(work_dir)
    code, result = prove_seed(
        seed_dir=args.seed_dir,
        tree_variant=args.tree_variant,
        code_root=args.code_root,
        out=work_dir,
        overwrite=False,
        refresh_phase_k_map=True,
        proof_route="protected_prefix",
    )
    request_path = work_dir / "request" / "proof_request.json"
    if not request_path.is_file():
        raise RuntimeError(
            "PPP baseline freeze did not produce a request workspace: "
            f"exit_code={code} result={json.dumps(result, ensure_ascii=False, sort_keys=True)}"
        )
    snapshot = extract_ppp_baseline(work_dir)
    snapshot["workflow_exit_code"] = code
    snapshot["workflow_result_status"] = result.get("result_status")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(snapshot, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
