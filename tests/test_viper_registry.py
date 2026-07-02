"""VIPER teacher registry 测试。"""

from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path

import torch

from amc_py.dqn import DqnBudgetAgent, DqnConfig

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _make_agent_model(path: Path) -> None:
    agent = DqnBudgetAgent(
        observation_dim=2,
        action_dim=2,
        config=DqnConfig(
            gamma=0.9,
            learning_rate=1e-3,
            replay_capacity=8,
            min_replay_size=1,
            batch_size=1,
            target_update_freq=1,
            epsilon_start=0.0,
            epsilon_end=0.0,
            epsilon_decay_steps=1,
            hidden_layers=(4,),
            seed=0,
        ),
    )
    with torch.no_grad():
        for param in agent.policy_network.parameters():
            param.zero_()
        for param in agent.target_network.parameters():
            param.zero_()
    agent.save(path)


def test_build_viper_teacher_registry_supports_run_dir_template(tmp_path: Path) -> None:
    teacher_root = tmp_path / "tr"
    run_dir = teacher_root / "r0_s185"
    run_dir.mkdir(parents=True)
    _make_agent_model(run_dir / "model_final.pt")
    output_csv = tmp_path / "teacher_registry.csv"
    subprocess.run(
        [
            sys.executable,
            "scripts/build_viper_teacher_registry.py",
            "--teacher-root",
            str(teacher_root),
            "--seeds",
            "185",
            "--fallback-checkpoint-name",
            "model_final.pt",
            "--output",
            str(output_csv),
        ],
        check=True,
        cwd=PROJECT_ROOT,
    )
    with output_csv.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["taskset_seed"] == "185"
    assert rows[0]["train_output_dir"].endswith("tr/r0_s185")
