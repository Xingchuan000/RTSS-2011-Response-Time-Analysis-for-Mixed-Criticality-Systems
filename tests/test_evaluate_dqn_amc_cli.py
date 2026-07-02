"""正式 DQN 评估 CLI 测试。"""

from __future__ import annotations

import csv
import os
import sys
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    """读取 CSV 全部行，供串并行结果对比复用。"""

    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def test_evaluate_fieldnames_include_degradation_metrics() -> None:
    """正式评估 CSV 必须稳定包含论文 degraded-service 指标列。"""

    from scripts.evaluate_dqn_amc import _eval_summary_fieldnames

    fields = set(_eval_summary_fieldnames())
    required = {
        "hdm",
        "jne",
        "ldm",
        "nid",
        "tid",
        "total_time",
        "jne_plus_ldm",
        "dqn_runtime_semantics",
        "c_amc_sem_xf",
        "lo_job_losses_total",
        "lo_budget_cancellations",
        "lo_release_dropped_in_degraded_mode",
        "lo_active_dropped_on_mode_switch",
        "jne_residual_not_in_cancellations",
        "active_drop_share_of_jne",
    }
    assert required.issubset(fields)
    assert "lo_equiv_jne_rate" in fields
    assert "lo_quality_qos" in fields
    assert "lo_degraded_released" in fields
    assert "lo_full_quality_ratio" in fields
    assert "tid_ratio" in fields
    assert "tree_selected_action_count" in fields
    assert "tree_selected_action_match_teacher_count" in fields
    assert "tree_selected_action_match_teacher_rate" in fields
    assert "tree_raw_action_match_teacher_rate" in fields


def test_formal_evaluate_runtime_configs_disable_trace_and_record_dropped_lo_releases() -> None:
    """正式 HOUT 评估使用的 runtime 配置应统一关闭 trace 并记录 dropped LO release。"""

    from scripts.evaluate_dqn_amc import _baseline_runtime_config, _formal_agent_runtime_config
    from amc_py.runtime_models import RuntimeSemantics

    for semantics in (
        RuntimeSemantics.AMC_PLUS,
        RuntimeSemantics.AMC_RA,
        RuntimeSemantics.AMC_RH,
    ):
        baseline_cfg = _baseline_runtime_config(end_time=100, semantics=semantics)
        assert baseline_cfg.capture_trace is False
        assert baseline_cfg.capture_debug_events is False
        assert baseline_cfg.record_dropped_lo_releases is True

    c_amc_sem_cfg = _baseline_runtime_config(
        end_time=100,
        semantics=RuntimeSemantics.C_AMC_SEM,
        c_amc_sem_xf=0.5,
    )
    assert c_amc_sem_cfg.capture_trace is False
    assert c_amc_sem_cfg.capture_debug_events is False
    assert c_amc_sem_cfg.record_dropped_lo_releases is True
    assert c_amc_sem_cfg.drop_lo_jobs_on_hi_switch is False
    assert c_amc_sem_cfg.c_amc_sem_lo_degradation_ratio == 0.5
    assert c_amc_sem_cfg.c_amc_sem_primary_on_switch_time is True

    agent_cfg = _formal_agent_runtime_config(
        end_time=100,
        semantics=RuntimeSemantics.C_AMC_SEM,
        c_amc_sem_xf=0.75,
    )
    assert agent_cfg.capture_trace is False
    assert agent_cfg.capture_debug_events is False
    assert agent_cfg.record_dropped_lo_releases is True
    assert agent_cfg.drop_lo_jobs_on_hi_switch is False
    assert agent_cfg.c_amc_sem_lo_degradation_ratio == 0.75
    assert agent_cfg.c_amc_sem_primary_on_switch_time is True

    agent_rh_cfg = _formal_agent_runtime_config(
        end_time=100,
        semantics=RuntimeSemantics.AMC_RH,
        c_amc_sem_xf=0.75,
    )
    assert agent_rh_cfg.drop_lo_jobs_on_hi_switch is True
    assert agent_rh_cfg.c_amc_sem_primary_on_switch_time is False
    assert agent_rh_cfg.c_amc_sem_lo_degradation_ratio == 0.75


def test_evaluate_dqn_amc_cli_runs_after_training(tmp_path: Path) -> None:
    """训练后的正式模型应可被评估 CLI 加载并输出汇总。"""

    output_dir = tmp_path / "dqn_amc"
    model_path = output_dir / "model_final.pt"
    eval_path = output_dir / "eval_summary.csv"
    unified_summary_path = output_dir / "eval_summary_unified_summary.csv"
    env = {**os.environ, "KMP_DUPLICATE_LIB_OK": "TRUE"}

    subprocess.run(
        [
            sys.executable,
            "scripts/train_dqn_amc.py",
            "--episodes",
            "2",
            "--end-time",
            "50",
            "--seed",
            "0",
            "--output-dir",
            str(output_dir),
        ],
        check=True,
        cwd=PROJECT_ROOT,
        env=env,
    )
    subprocess.run(
        [
            sys.executable,
            "scripts/evaluate_dqn_amc.py",
            "--model",
            str(model_path),
            "--seeds",
            "0,1",
            "--end-time",
            "50",
            "--baselines",
            "amc_plus_baseline,amc_ra_baseline,amc_rh_baseline,c_amc_sem_baseline,dqn_agent",
            "--c-amc-sem-xf",
            "0.5",
            "--output",
            str(eval_path),
        ],
        check=True,
        cwd=PROJECT_ROOT,
        env=env,
    )

    assert eval_path.exists()
    assert unified_summary_path.exists()
    with eval_path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    with unified_summary_path.open("r", encoding="utf-8", newline="") as f:
        unified_rows = list(csv.DictReader(f))

    assert rows
    assert unified_rows
    methods = {row["method"] for row in rows}
    assert "dqn_agent" in methods
    assert "amc_plus_baseline" in methods
    assert "c_amc_sem_baseline" in methods
    assert all(row["c_amc_sem_xf"] == "0.5" for row in rows)
    expected_summary_fields = {
        "row_type",
        "method",
        "reference_method",
        "mode_changes_mean",
        "lo_cancellations_mean",
        "lo_job_losses_total_mean",
        "lo_active_dropped_on_mode_switch_mean",
        "delta_lc_service_loss",
        "relative_lc_loss_reduction",
        "accepted_action_count_mean",
        "rejected_action_count_mean",
        "noop_action_count_mean",
        "noop_action_rate_mean",
        "noop_q_rank_mean",
        "noop_q_sample_count",
        "masked_action_count_mean",
        "valid_action_count_mean",
    }
    assert expected_summary_fields.issubset(set(unified_rows[0].keys()))
    assert "lo_equiv_jne_rate_mean" in unified_rows[0]
    assert "lo_quality_qos_mean" in unified_rows[0]
    assert "lo_degraded_released_mean" in unified_rows[0]
    assert "tid_ratio_mean" in unified_rows[0]
    assert "delta_lo_equiv_jne_rate" in unified_rows[0]
    assert "lo_zero_service_ratio" in rows[0]
    assert "lo_budget_cancellations" in rows[0]
    assert "lo_active_dropped_on_mode_switch" in rows[0]
    assert "lo_release_dropped_in_degraded_mode" in rows[0]
    row_types = {row["row_type"] for row in unified_rows}
    assert row_types >= {"method_summary", "dqn_vs_reference"}
    unified_methods = {row["method"] for row in unified_rows}
    assert "c_amc_sem_baseline" in unified_methods
    assert any(
        row["row_type"] == "dqn_vs_reference"
        and row["reference_method"] == "c_amc_sem_baseline"
        for row in unified_rows
    )
    assert "noop_q_rank_mean" in rows[0]
    baseline_row = next(row for row in rows if row["method"] == "amc_plus_baseline")
    assert baseline_row["noop_q_rank_mean"] == ""
    assert baseline_row["dqn_runtime_semantics"] == "AMC_PLUS"
    assert "budget_floor_ratio" in rows[0]
    assert "masked_budget_floor_violation_count" in rows[0]
    assert "masked_budget_floor_violation_rate" in rows[0]
    assert "masked_deploy_cap_increase_count" in rows[0]
    assert "masked_deploy_cap_increase_rate" in rows[0]


def test_evaluate_cli_supports_amc_ra_rh_baselines(tmp_path: Path) -> None:
    """评估 CLI 显式启用 AMC-RA/AMC-RH baseline 后应输出对应方法行与退化指标列。"""

    output_dir = tmp_path / "dqn_amc_ra_rh_eval"
    model_path = output_dir / "model_final.pt"
    eval_path = output_dir / "eval_ra_rh.csv"
    env = {**os.environ, "KMP_DUPLICATE_LIB_OK": "TRUE"}

    subprocess.run(
        [
            sys.executable,
            "scripts/train_dqn_amc.py",
            "--episodes",
            "1",
            "--end-time",
            "50",
            "--seed",
            "0",
            "--output-dir",
            str(output_dir),
        ],
        check=True,
        cwd=PROJECT_ROOT,
        env=env,
    )

    subprocess.run(
        [
            sys.executable,
            "scripts/evaluate_dqn_amc.py",
            "--model",
            str(model_path),
            "--seeds",
            "0",
            "--end-time",
            "100",
            "--baselines",
            "amc_plus_baseline,amc_ra_baseline,amc_rh_baseline,dqn_agent",
            "--output",
            str(eval_path),
        ],
        check=True,
        cwd=PROJECT_ROOT,
        env=env,
    )

    rows = _read_csv_rows(eval_path)
    methods = {row["method"] for row in rows}
    assert "amc_plus_baseline" in methods
    assert "amc_ra_baseline" in methods
    assert "amc_rh_baseline" in methods
    assert "dqn_agent" in methods

    required = {"hdm", "jne", "ldm", "nid", "tid", "total_time", "jne_plus_ldm"}
    for row in rows:
        assert required.issubset(row.keys())
        assert int(row["jne_plus_ldm"]) == int(row["jne"]) + int(row["ldm"])


def test_evaluate_cli_supports_dqn_on_rh_runtime_semantics(tmp_path: Path) -> None:
    """显式指定 AMC_RH 时，dqn_agent 与 wrapper baselines 应写出 AMC_RH 语义。"""

    output_dir = tmp_path / "dqn_on_rh_eval"
    model_path = output_dir / "model_final.pt"
    eval_path = output_dir / "eval_dqn_on_rh.csv"
    env = {**os.environ, "KMP_DUPLICATE_LIB_OK": "TRUE"}

    subprocess.run(
        [
            sys.executable,
            "scripts/train_dqn_amc.py",
            "--episodes",
            "1",
            "--end-time",
            "40",
            "--seed",
            "0",
            "--output-dir",
            str(output_dir),
        ],
        check=True,
        cwd=PROJECT_ROOT,
        env=env,
    )

    subprocess.run(
        [
            sys.executable,
            "scripts/evaluate_dqn_amc.py",
            "--model",
            str(model_path),
            "--seeds",
            "0",
            "--end-time",
            "40",
            "--dqn-runtime-semantics",
            "AMC_RH",
            "--baselines",
            "amc_plus_baseline,amc_ra_baseline,amc_rh_baseline,noop_agent,dqn_agent",
            "--output",
            str(eval_path),
        ],
        check=True,
        cwd=PROJECT_ROOT,
        env=env,
    )

    rows = _read_csv_rows(eval_path)
    by_method = {row["method"]: row for row in rows}
    assert by_method["dqn_agent"]["dqn_runtime_semantics"] == "AMC_RH"
    assert by_method["noop_agent"]["dqn_runtime_semantics"] == "AMC_RH"


def test_evaluate_cli_short_hout_smoke_outputs_ra_rh_and_dqn_methods(tmp_path: Path) -> None:
    """短时域 smoke 应能同时输出 AMC+、RA、RH 与 DQN 的正式评估结果。"""

    output_dir = tmp_path / "dqn_amc_hout_smoke"
    model_path = output_dir / "model_final.pt"
    eval_path = output_dir / "eval_hout_smoke.csv"
    env = {**os.environ, "KMP_DUPLICATE_LIB_OK": "TRUE"}

    subprocess.run(
        [
            sys.executable,
            "scripts/train_dqn_amc.py",
            "--episodes",
            "1",
            "--end-time",
            "40",
            "--seed",
            "0",
            "--output-dir",
            str(output_dir),
        ],
        check=True,
        cwd=PROJECT_ROOT,
        env=env,
    )

    subprocess.run(
        [
            sys.executable,
            "scripts/evaluate_dqn_amc.py",
            "--model",
            str(model_path),
            "--seeds",
            "0",
            "--end-time",
            "40",
            "--evaluation-workers",
            "1",
            "--baselines",
            "amc_plus_baseline,amc_ra_baseline,amc_rh_baseline,dqn_agent",
            "--output",
            str(eval_path),
        ],
        check=True,
        cwd=PROJECT_ROOT,
        env=env,
    )

    rows = _read_csv_rows(eval_path)
    methods = {row["method"] for row in rows}
    assert methods >= {
        "amc_plus_baseline",
        "amc_ra_baseline",
        "amc_rh_baseline",
        "dqn_agent",
    }
    for row in rows:
        assert "hdm" in row
        assert "jne" in row
        assert "ldm" in row
        assert "nid" in row
        assert "tid" in row
        assert "jne_plus_ldm" in row


def test_evaluate_cli_supports_dqn_on_c_amc_sem_runtime_semantics(tmp_path: Path) -> None:
    """HOUT CLI 应支持使用 C-AMC-sem runtime 评估 DQN agent。"""

    output_dir = tmp_path / "dqn_on_c_amc_sem_eval"
    model_path = output_dir / "model_final.pt"
    eval_path = output_dir / "eval_dqn_on_c_amc_sem.csv"
    env = {**os.environ, "KMP_DUPLICATE_LIB_OK": "TRUE"}

    subprocess.run(
        [
            sys.executable,
            "scripts/train_dqn_amc.py",
            "--episodes",
            "1",
            "--end-time",
            "40",
            "--agent-period",
            "10",
            "--seed",
            "0",
            "--dqn-runtime-semantics",
            "C_AMC_SEM",
            "--validation-baseline-semantics",
            "C_AMC_SEM",
            "--c-amc-sem-xf",
            "0.5",
            "--dqn-device",
            "cpu",
            "--output-dir",
            str(output_dir),
        ],
        check=True,
        cwd=PROJECT_ROOT,
        env=env,
    )

    subprocess.run(
        [
            sys.executable,
            "scripts/evaluate_dqn_amc.py",
            "--model",
            str(model_path),
            "--seeds",
            "0",
            "--end-time",
            "40",
            "--agent-period",
            "10",
            "--dqn-runtime-semantics",
            "C_AMC_SEM",
            "--c-amc-sem-xf",
            "0.5",
            "--baselines",
            "c_amc_sem_baseline,noop_agent,dqn_agent",
            "--output",
            str(eval_path),
        ],
        check=True,
        cwd=PROJECT_ROOT,
        env=env,
    )

    rows = _read_csv_rows(eval_path)
    by_method = {row["method"]: row for row in rows}

    assert "c_amc_sem_baseline" in by_method
    assert "noop_agent" in by_method
    assert "dqn_agent" in by_method
    assert by_method["dqn_agent"]["dqn_runtime_semantics"] == "C_AMC_SEM"
    assert by_method["noop_agent"]["dqn_runtime_semantics"] == "C_AMC_SEM"
    assert all(row["c_amc_sem_xf"] == "0.5" for row in rows)

    for required_col in (
        "hdm",
        "jne",
        "ldm",
        "nid",
        "tid",
        "jne_plus_ldm",
        "lo_degraded_released",
        "lo_quality_qos",
        "lo_equiv_jne_rate",
        "lo_full_quality_ratio",
    ):
        assert required_col in rows[0]


def test_evaluate_cli_rejects_legacy_reward_mode(tmp_path: Path) -> None:
    """评估 CLI 不应再接受旧 reward mode。"""

    output_dir = tmp_path / "dqn_amc_legacy_reward_eval"
    env = {**os.environ, "KMP_DUPLICATE_LIB_OK": "TRUE"}
    subprocess.run(
        [
            sys.executable,
            "scripts/train_dqn_amc.py",
            "--episodes",
            "1",
            "--end-time",
            "20",
            "--seed",
            "0",
            "--output-dir",
            str(output_dir),
        ],
        check=True,
        cwd=PROJECT_ROOT,
        env=env,
    )
    result = subprocess.run(
        [
            sys.executable,
            "scripts/evaluate_dqn_amc.py",
            "--model",
            str(output_dir / "model_final.pt"),
            "--seeds",
            "0",
            "--reward-mode",
            "event_delta_no_job_start",
            "--output",
            str(output_dir / "eval.csv"),
        ],
        check=False,
        cwd=PROJECT_ROOT,
        env=env,
    )
    assert result.returncode != 0


def test_evaluate_cli_rejects_invalid_budget_floor_ratio(tmp_path: Path) -> None:
    """评估 CLI 应拒绝超出 [0,1] 的 budget floor 参数。"""

    output_dir = tmp_path / "dqn_amc_invalid_floor_eval"
    env = {**os.environ, "KMP_DUPLICATE_LIB_OK": "TRUE"}
    subprocess.run(
        [
            sys.executable,
            "scripts/train_dqn_amc.py",
            "--episodes",
            "1",
            "--end-time",
            "20",
            "--seed",
            "0",
            "--output-dir",
            str(output_dir),
        ],
        check=True,
        cwd=PROJECT_ROOT,
        env=env,
    )
    result = subprocess.run(
        [
            sys.executable,
            "scripts/evaluate_dqn_amc.py",
            "--model",
            str(output_dir / "model_final.pt"),
            "--seeds",
            "0",
            "--budget-floor-ratio",
            "-0.1",
            "--output",
            str(output_dir / "eval.csv"),
        ],
        check=False,
        cwd=PROJECT_ROOT,
        env=env,
    )
    assert result.returncode != 0


def test_evaluate_cli_rejects_unknown_baseline(tmp_path: Path) -> None:
    """评估 CLI 仍应拒绝计划外的未知 baseline 名称。"""

    output_dir = tmp_path / "dqn_amc_unknown_baseline_eval"
    env = {**os.environ, "KMP_DUPLICATE_LIB_OK": "TRUE"}
    subprocess.run(
        [
            sys.executable,
            "scripts/train_dqn_amc.py",
            "--episodes",
            "1",
            "--end-time",
            "20",
            "--seed",
            "0",
            "--output-dir",
            str(output_dir),
        ],
        check=True,
        cwd=PROJECT_ROOT,
        env=env,
    )
    result = subprocess.run(
        [
            sys.executable,
            "scripts/evaluate_dqn_amc.py",
            "--model",
            str(output_dir / "model_final.pt"),
            "--seeds",
            "0",
            "--baselines",
            "amc_ra_baseline,unknown_baseline",
            "--output",
            str(output_dir / "eval.csv"),
        ],
        check=False,
        cwd=PROJECT_ROOT,
        env=env,
    )
    assert result.returncode != 0


def test_evaluate_cli_rejects_invalid_deploy_cap_mask_ratio(tmp_path: Path) -> None:
    """评估 CLI 应拒绝小于等于 1.0 的 deploy cap ratio。"""

    output_dir = tmp_path / "dqn_amc_invalid_deploy_cap_eval"
    env = {**os.environ, "KMP_DUPLICATE_LIB_OK": "TRUE"}
    subprocess.run(
        [
            sys.executable,
            "scripts/train_dqn_amc.py",
            "--episodes",
            "1",
            "--end-time",
            "20",
            "--seed",
            "0",
            "--output-dir",
            str(output_dir),
        ],
        check=True,
        cwd=PROJECT_ROOT,
        env=env,
    )
    result = subprocess.run(
        [
            sys.executable,
            "scripts/evaluate_dqn_amc.py",
            "--model",
            str(output_dir / "model_final.pt"),
            "--seeds",
            "0",
            "--enable-deploy-cap-mask",
            "--deploy-cap-mask-ratio",
            "1.0",
            "--output",
            str(output_dir / "eval.csv"),
        ],
        check=False,
        cwd=PROJECT_ROOT,
        env=env,
    )
    assert result.returncode != 0


def test_evaluate_cli_supports_parallel_seed_workers(tmp_path: Path) -> None:
    """评估 CLI 开启多进程按 seed 并行后应正常产出结果。"""

    output_dir = tmp_path / "dqn_amc_parallel_eval"
    env = {**os.environ, "KMP_DUPLICATE_LIB_OK": "TRUE"}
    subprocess.run(
        [
            sys.executable,
            "scripts/train_dqn_amc.py",
            "--episodes",
            "2",
            "--end-time",
            "50",
            "--seed",
            "0",
            "--output-dir",
            str(output_dir),
        ],
        check=True,
        cwd=PROJECT_ROOT,
        env=env,
    )
    eval_path = output_dir / "eval_parallel.csv"
    subprocess.run(
        [
            sys.executable,
            "scripts/evaluate_dqn_amc.py",
            "--model",
            str(output_dir / "model_final.pt"),
            "--seeds",
            "0,1",
            "--end-time",
            "50",
            "--evaluation-workers",
            "2",
            "--baselines",
            "amc_plus_baseline,amc_ra_baseline,amc_rh_baseline,dqn_agent",
            "--output",
            str(eval_path),
        ],
        check=True,
        cwd=PROJECT_ROOT,
        env=env,
    )

    rows = _read_csv_rows(eval_path)
    assert rows
    assert {row["method"] for row in rows} >= {
        "dqn_agent",
        "amc_plus_baseline",
        "amc_ra_baseline",
        "amc_rh_baseline",
    }


def test_evaluate_cli_parallel_and_serial_outputs_match(tmp_path: Path) -> None:
    """固定模型与 seeds 时，串行/并行评估输出应保持一致。"""

    output_dir = tmp_path / "dqn_amc_eval_consistency"
    env = {**os.environ, "KMP_DUPLICATE_LIB_OK": "TRUE"}
    subprocess.run(
        [
            sys.executable,
            "scripts/train_dqn_amc.py",
            "--episodes",
            "2",
            "--end-time",
            "50",
            "--seed",
            "0",
            "--output-dir",
            str(output_dir),
        ],
        check=True,
        cwd=PROJECT_ROOT,
        env=env,
    )

    serial_eval_path = output_dir / "eval_serial.csv"
    parallel_eval_path = output_dir / "eval_parallel.csv"
    base_cmd = [
        sys.executable,
        "scripts/evaluate_dqn_amc.py",
        "--model",
        str(output_dir / "model_final.pt"),
        "--seeds",
        "0,1",
        "--end-time",
        "50",
        "--baselines",
        "amc_plus_baseline,amc_ra_baseline,amc_rh_baseline,dqn_agent",
    ]
    subprocess.run(
        base_cmd + ["--evaluation-workers", "1", "--output", str(serial_eval_path)],
        check=True,
        cwd=PROJECT_ROOT,
        env=env,
    )
    subprocess.run(
        base_cmd + ["--evaluation-workers", "2", "--output", str(parallel_eval_path)],
        check=True,
        cwd=PROJECT_ROOT,
        env=env,
    )

    assert _read_csv_rows(serial_eval_path) == _read_csv_rows(parallel_eval_path)
    assert _read_csv_rows(
        serial_eval_path.with_name(f"{serial_eval_path.stem}_unified_summary.csv")
    ) == _read_csv_rows(
        parallel_eval_path.with_name(f"{parallel_eval_path.stem}_unified_summary.csv")
    )


def test_evaluate_cli_rejects_invalid_evaluation_workers(tmp_path: Path) -> None:
    """评估 worker 数小于 1 时，评估 CLI 应显式报错。"""

    output_dir = tmp_path / "dqn_amc_invalid_eval_workers"
    env = {**os.environ, "KMP_DUPLICATE_LIB_OK": "TRUE"}
    subprocess.run(
        [
            sys.executable,
            "scripts/train_dqn_amc.py",
            "--episodes",
            "1",
            "--end-time",
            "20",
            "--seed",
            "0",
            "--output-dir",
            str(output_dir),
        ],
        check=True,
        cwd=PROJECT_ROOT,
        env=env,
    )
    result = subprocess.run(
        [
            sys.executable,
            "scripts/evaluate_dqn_amc.py",
            "--model",
            str(output_dir / "model_final.pt"),
            "--seeds",
            "0",
            "--evaluation-workers",
            "0",
            "--output",
            str(output_dir / "eval.csv"),
        ],
        check=False,
        cwd=PROJECT_ROOT,
        env=env,
    )
    assert result.returncode != 0


def test_evaluate_cli_constraint_guided_pair_smoke_and_dim_mismatch(tmp_path: Path) -> None:
    """constraint_guided_pair 模型可评估，且动作维度不匹配时应报错。"""

    env = {**os.environ, "KMP_DUPLICATE_LIB_OK": "TRUE"}
    cg_train_dir = tmp_path / "cg_train"
    single_train_dir = tmp_path / "single_train"
    cg_eval_path = cg_train_dir / "eval_cg.csv"

    subprocess.run(
        [
            sys.executable,
            "scripts/train_dqn_amc.py",
            "--workload",
            "small",
            "--episodes",
            "1",
            "--end-time",
            "60",
            "--agent-period",
            "20",
            "--observation-mode",
            "v11_full_10d",
            "--action-space",
            "constraint_guided_pair",
            "--constraint-guided-pair-top-k-risk",
            "3",
            "--constraint-guided-pair-top-k-decrease",
            "5",
            "--include-explicit-noop",
            "--output-dir",
            str(cg_train_dir),
        ],
        check=True,
        cwd=PROJECT_ROOT,
        env=env,
    )
    subprocess.run(
        [
            sys.executable,
            "scripts/evaluate_dqn_amc.py",
            "--model",
            str(cg_train_dir / "model_final.pt"),
            "--seeds",
            "0",
            "--end-time",
            "60",
            "--agent-period",
            "20",
            "--observation-mode",
            "v11_full_10d",
            "--action-space",
            "constraint_guided_pair",
            "--constraint-guided-pair-top-k-risk",
            "3",
            "--constraint-guided-pair-top-k-decrease",
            "5",
            "--baselines",
            "dqn_agent",
            "--include-explicit-noop",
            "--output",
            str(cg_eval_path),
        ],
        check=True,
        cwd=PROJECT_ROOT,
        env=env,
    )
    assert cg_eval_path.exists()
    rows = _read_csv_rows(cg_eval_path)
    assert rows
    dqn_rows = [row for row in rows if row.get("method") == "dqn_agent"]
    assert dqn_rows
    # bundled transfer 口径：constraint_guided_pair(alias) + explicit noop 下动作维度应为 4。
    assert all(int(row["action_count"]) == 4 for row in dqn_rows)
    assert all(row["action_space_type"] == "constraint_guided_transfer" for row in dqn_rows)

    subprocess.run(
        [
            sys.executable,
            "scripts/train_dqn_amc.py",
            "--workload",
            "small",
            "--episodes",
            "1",
            "--end-time",
            "60",
            "--agent-period",
            "20",
            "--observation-mode",
            "v11_full_10d",
            "--action-space",
            "single",
            "--include-explicit-noop",
            "--output-dir",
            str(single_train_dir),
        ],
        check=True,
        cwd=PROJECT_ROOT,
        env=env,
    )
    mismatch = subprocess.run(
        [
            sys.executable,
            "scripts/evaluate_dqn_amc.py",
            "--model",
            str(single_train_dir / "model_final.pt"),
            "--seeds",
            "0",
            "--end-time",
            "60",
            "--agent-period",
            "20",
            "--observation-mode",
            "v11_full_10d",
            "--action-space",
            "constraint_guided_pair",
            "--constraint-guided-pair-top-k-risk",
            "3",
            "--constraint-guided-pair-top-k-decrease",
            "5",
            "--baselines",
            "dqn_agent",
            "--include-explicit-noop",
            "--output",
            str(single_train_dir / "eval_mismatch.csv"),
        ],
        check=False,
        cwd=PROJECT_ROOT,
        env=env,
    )
    assert mismatch.returncode != 0
