from pathlib import Path
import sys

from nonvacuity_lab.analysis.expectations import classify_experiment
from nonvacuity_lab.runners.integrity_reuse import _read_integrity_result
from nonvacuity_lab.runners.campaign import _campaign_summary
from nonvacuity_lab.schema import ExpectedResult
from nonvacuity_lab.subprocess_runner import run_command


def test_subprocess_timeout_is_captured(tmp_path: Path):
    receipt = run_command(
        [sys.executable, "-c", "import time; time.sleep(2)"],
        cwd=tmp_path,
        log_dir=tmp_path / "logs",
        timeout_seconds=1,
    )
    assert receipt["timed_out"] is True
    assert receipt["execution_status"] == "TIMED_OUT"
    assert receipt["returncode"] is None


def test_integrity_missing_summary_is_not_a_kill(tmp_path: Path):
    stdout = tmp_path / "stdout.log"
    stderr = tmp_path / "stderr.log"
    stdout.write_text("", encoding="utf-8")
    stderr.write_text("", encoding="utf-8")
    proof = _read_integrity_result(
        tmp_path / "verify",
        {
            "stdout": str(stdout),
            "stderr": str(stderr),
            "returncode": 1,
            "timed_out": False,
            "traceback_marker": False,
        },
    )
    classified = classify_experiment(
        expected=ExpectedResult(),
        proof_result=proof,
        activation_result=None,
        integrity=True,
    )
    assert proof["result_status"] == "VERIFIER_OUTPUT_MISSING"
    assert classified["status"] == "VERIFIER_OUTPUT_MISSING"
    assert classified["status"] != "INTEGRITY_REJECTION_EXPECTED"


def test_integrity_timeout_is_not_a_kill(tmp_path: Path):
    stdout = tmp_path / "stdout.log"
    stderr = tmp_path / "stderr.log"
    stdout.write_text("", encoding="utf-8")
    stderr.write_text("", encoding="utf-8")
    proof = _read_integrity_result(
        tmp_path / "verify",
        {
            "stdout": str(stdout),
            "stderr": str(stderr),
            "returncode": None,
            "timed_out": True,
            "traceback_marker": False,
        },
    )
    classified = classify_experiment(
        expected=ExpectedResult(),
        proof_result=proof,
        activation_result=None,
        integrity=True,
    )
    assert proof["result_status"] == "VERIFIER_TIMEOUT"
    assert classified["status"] == "VERIFIER_TIMEOUT"


def test_tool_failures_are_excluded_from_kill_rate():
    summary = _campaign_summary(
        [
            {"status": "FAIL_EXPECTED"},
            {"status": "VERIFIER_TIMEOUT"},
            {"status": "VERIFIER_OUTPUT_MISSING"},
            {"status": "TOOL_EXECUTION_FAILED"},
        ],
        fail_on_not_activated=True,
    )
    assert summary["kill_rate_numerator"] == 1
    assert summary["kill_rate_denominator"] == 1
