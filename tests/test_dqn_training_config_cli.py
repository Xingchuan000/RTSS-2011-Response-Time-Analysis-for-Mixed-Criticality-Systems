"""DQN 训练配置 CLI 测试。"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_hidden_layers_and_replay_related_params_are_applied(tmp_path: Path) -> None:
    """CLI 传入的训练规模参数应写入配置并可正常启动训练。"""

    output_dir = tmp_path / "dqn_cfg"
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
            "--batch-size",
            "64",
            "--replay-capacity",
            "10000",
            "--min-replay-size",
            "500",
            "--hidden-layers",
            "128,128",
            "--target-update-frequency",
            "100",
            "--output-dir",
            str(output_dir),
        ],
        check=True,
        cwd=PROJECT_ROOT,
        env={**os.environ, "KMP_DUPLICATE_LIB_OK": "TRUE"},
    )

    config_path = output_dir / "config.json"
    with config_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    dqn_cfg = payload["dqn_config"]
    assert dqn_cfg["batch_size"] == 64
    assert dqn_cfg["replay_capacity"] == 10000
    assert dqn_cfg["min_replay_size"] == 500
    assert dqn_cfg["target_update_freq"] == 100
    assert tuple(dqn_cfg["hidden_layers"]) == (128, 128)
