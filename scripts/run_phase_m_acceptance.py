"""Phase M synthetic_p0 全链验收。

脚本只接受仓库内 canonical fixture；它不扫描真实 seed、HOUT 或用户主目录。
正向结果必须同时满足单命令和手动四命令 root 一致，且报告明确把真实 seed
标记为 DEFERRED。
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FIXTURE_ROOT = ROOT / "tests" / "formal" / "fixtures" / "synthetic_p0"


def _run(module: str, args: list[str]) -> subprocess.CompletedProcess[str]:
    """所有四命令都以新 Python 进程运行，避免共享 compiler/verifier 状态。"""

    return subprocess.run([sys.executable, "-m", module, *args], cwd=ROOT,
                          capture_output=True, text=True, check=False)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Phase M synthetic_p0 acceptance")
    parser.add_argument("--fixture", default="synthetic_p0")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.fixture != "synthetic_p0" or FIXTURE_ROOT.resolve() != (ROOT / "tests/formal/fixtures/synthetic_p0").resolve():
        print(json.dumps({"phase_result": "PHASE_M_REJECTED", "failure_code": "SYNTHETIC_FIXTURE_REQUIRED"}, ensure_ascii=False))
        return 1

    out = Path(args.out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    single = subprocess.run([
        sys.executable, "-m", "formal_toolchain.cli.prove_seed",
        "--seed-dir", str(FIXTURE_ROOT), "--tree-variant", "best_overall",
        "--code-root", str(ROOT), "--out", str(out), "--overwrite", "--json",
    ], cwd=ROOT, capture_output=True, text=True, check=False)
    if single.returncode != 0:
        print(single.stdout)
        return single.returncode or 1
    single_result = json.loads(single.stdout.strip().splitlines()[-1])
    request = out / "request" / "proof_request.json"
    manual_candidate = out.parent / "manual_synthetic_p0_candidate"
    manual_verified = out.parent / "manual_synthetic_p0_verified"
    inspect = _run("formal_toolchain.cli.inspect_target", ["--request", str(request)])
    compile_run = _run("formal_toolchain.cli.compile_seed", ["--request", str(request), "--out", str(manual_candidate)])
    verify_run = _run("formal_toolchain.cli.verify_bundle", ["--request", str(request), "--bundle", str(manual_candidate), "--out", str(manual_verified)]) if compile_run.returncode == 0 else None
    report_run = _run("formal_toolchain.cli.render_report", ["--verified", str(manual_verified), "--out", str(manual_verified / "human_readable_report.md")]) if verify_run and verify_run.returncode == 0 else None
    if inspect.returncode != 0 or compile_run.returncode != 0 or verify_run is None or verify_run.returncode != 0 or report_run is None or report_run.returncode != 0:
        print(json.dumps({"phase_result": "PHASE_M_REJECTED", "failure_code": "MANUAL_FOUR_COMMAND_FAILED"}, ensure_ascii=False))
        return 1
    single_summary = json.loads((out / "verified" / "proof_summary.json").read_text(encoding="utf-8"))
    manual_summary = json.loads((manual_verified / "proof_summary.json").read_text(encoding="utf-8"))
    roots_equal = single_summary.get("outer_bundle_root") == manual_summary.get("outer_bundle_root")
    contexts_equal = single_summary.get("certificate_context_hash") == manual_summary.get("certificate_context_hash")

    # 同一 fixture 换树只验证 context invalidation；不把 best_overall 证书复制到
    # alternate bundle，故这里使用独立输出目录和独立 fresh verifier。
    alternate_out = out.parent / "synthetic_p0_best_balanced"
    alternate = subprocess.run([
        sys.executable, "-m", "formal_toolchain.cli.prove_seed",
        "--seed-dir", str(FIXTURE_ROOT), "--tree-variant", "best_balanced",
        "--code-root", str(ROOT), "--out", str(alternate_out), "--overwrite", "--json",
    ], cwd=ROOT, capture_output=True, text=True, check=False)
    alternate_result = json.loads(alternate.stdout.strip().splitlines()[-1]) if alternate.stdout.strip() else {}
    alt_summary_path = alternate_out / "verified" / "proof_summary.json"
    alternate_summary = json.loads(alt_summary_path.read_text(encoding="utf-8")) if alt_summary_path.is_file() else {}
    context_invalidated = (
        alternate_summary.get("certificate_context_hash") != single_summary.get("certificate_context_hash")
        or alternate_summary.get("outer_bundle_root") != single_summary.get("outer_bundle_root")
    )

    # 在临时副本中篡改 tree 文件并保持旧 manifest，preflight 必须在 compiler
    # 启动前拒绝；这个负向检查不改变仓库 fixture。
    with tempfile.TemporaryDirectory(prefix="phase_m_no_real_seed_") as temp:
        tampered = Path(temp) / "synthetic_p0"
        shutil.copytree(FIXTURE_ROOT, tampered)
        tree_path = tampered / "best_overall" / "integer_tree.json"
        tree = json.loads(tree_path.read_text(encoding="utf-8"))
        tree["nodes"][0]["threshold_int"] += 1
        tree_path.write_text(json.dumps(tree, separators=(",", ":")), encoding="utf-8")
        tampered_out = Path(temp) / "tampered_out"
        tampered_run = subprocess.run([
            sys.executable, "-m", "formal_toolchain.cli.prove_seed",
            "--seed-dir", str(tampered), "--tree-variant", "best_overall",
            "--code-root", str(ROOT), "--out", str(tampered_out), "--json",
        ], cwd=ROOT, capture_output=True, text=True, check=False)
        tamper_rejected = tampered_run.returncode != 0

    result = {
        "phase": "M",
        "phase_result": "PHASE_M_ACCEPTED" if (
            single_result.get("result_status") == "DEPLOYED_TREE_PROVED"
            and single_result.get("fixture_claim_result") == "DEPLOYED_TREE_PROVED"
            and single_result.get("real_seed_evaluation") == "DEFERRED"
            and single_result.get("exit_code") == 0
            and single_summary.get("fixture_id") == "synthetic_p0"
            and single_summary.get("fixture_kind") == "SYNTHETIC_P0"
            and roots_equal and contexts_equal and context_invalidated and tamper_rejected
        ) else "PHASE_M_REJECTED",
        "fixture": "synthetic_p0",
        "fixture_claim_result": single_result.get("result_status"),
        "real_seed_evaluation": "DEFERRED",
        "single_command_root": single_summary.get("outer_bundle_root"),
        "manual_four_command_root": manual_summary.get("outer_bundle_root"),
        "single_command_root_equals_manual_root": roots_equal,
        "certificate_context_hash_equal": contexts_equal,
        "alternate_tree_result": alternate_result.get("result_status"),
        "alternate_tree_context_invalidated": context_invalidated,
        "tamper_rejected": tamper_rejected,
    }
    (out / "phase_m_result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))
    return 0 if result["phase_result"] == "PHASE_M_ACCEPTED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
