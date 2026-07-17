"""R00：关键证据缺失时不得误报最终 PROVED。"""

from pathlib import Path
import shutil

from formal_toolchain.workflow.prove_seed import prove_seed


def test_missing_bridge_cannot_prove(tmp_path: Path) -> None:
    # 复制一个明确缺少 Phase K case map 的 seed，真实测试的是“bridge 缺失”，
    # 而不是让完整 synthetic fixture 反向证明该负向断言。
    source = Path(__file__).parents[1] / "fixtures" / "synthetic_p0"
    seed = tmp_path / "synthetic_without_phase_k"
    shutil.copytree(source, seed, ignore=shutil.ignore_patterns("phase_k_case_map.json"))
    result_code, result = prove_seed(
        seed_dir=seed, tree_variant="best_overall", code_root=Path(__file__).parents[3],
        out=tmp_path / "bundle", overwrite=True)
    assert result_code != 0
    assert result["result_status"] in {"UNRESOLVED", "PROOF_BUNDLE_INVALID"}
    assert result["result_status"] != "DEPLOYED_TREE_PROVED"
