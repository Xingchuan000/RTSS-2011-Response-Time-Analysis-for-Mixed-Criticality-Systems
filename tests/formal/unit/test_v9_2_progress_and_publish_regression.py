from pathlib import Path

import formal_toolchain.workflow.prove_seed as prove_seed_module


ROOT = Path(__file__).parents[3]


def test_incremental_progress_merges_dynamic_fields_before_keyword_expansion():
    verifier = (ROOT / "formal_toolchain/v9_2/verifier.py").read_text(encoding="utf-8")
    assert "payload = dict(progress_base)" in verifier
    assert "payload.update(details)" in verifier
    assert '"probe_timeout_ms": int(timeout_ms)' in verifier
    assert "**payload" in verifier
    assert "solver_strategy=SOLVER_STRATEGY,\n                **details" not in verifier


def test_publish_staging_falls_back_to_copy_when_windows_rename_is_denied(tmp_path, monkeypatch):
    staging = tmp_path / ".proof.staging"
    out = tmp_path / "proof"
    staging.mkdir()
    (staging / "proof_result.json").write_text('{"result_status":"PASS"}\n', encoding="utf-8")

    original_rename = Path.rename

    def deny_staging_rename(self, target):
        if self == staging:
            raise PermissionError(5, "Access is denied")
        return original_rename(self, target)

    monkeypatch.setattr(Path, "rename", deny_staging_rename)
    mode = prove_seed_module._publish_staging(staging, out)

    assert mode == "COPIED_AFTER_RENAME_DENIED"
    assert staging.is_dir()
    assert (out / "proof_result.json").is_file()
