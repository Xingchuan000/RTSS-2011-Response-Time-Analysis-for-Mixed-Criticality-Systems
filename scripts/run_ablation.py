"""阶段 0：一键运行 DQN ablation（训练 + 评估）。"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def _load_json(path: Path) -> dict:
    """读取 JSON 配置文件。"""

    if not path.exists():
        raise FileNotFoundError(f"配置文件不存在: {path}")
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError("ablation 配置必须是 JSON object")
    return payload


def _append_cli_arg(command: list[str], key: str, value: object) -> None:
    """把配置键值转换为命令行参数。

    设计约束：
    1. 不做“猜测式兜底”，只接受显式配置；
    2. 布尔值严格映射为 `--key` / `--no-key`；
    3. 其它类型统一转字符串。
    """

    cli_key = f"--{key.replace('_', '-')}"
    if isinstance(value, bool):
        if value:
            command.append(cli_key)
        else:
            command.append(f"--no-{key.replace('_', '-')}")
        return
    command.extend([cli_key, str(value)])


def _build_command(script_path: Path, args_payload: dict[str, object]) -> list[str]:
    """按配置构造子命令。"""

    if not isinstance(args_payload, dict):
        raise ValueError("脚本参数段必须是 JSON object")
    command = [sys.executable, str(script_path)]
    for key, value in args_payload.items():
        _append_cli_arg(command, key, value)
    return command


def build_parser() -> argparse.ArgumentParser:
    """构建命令行参数解析器。"""

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/ablation/current_pair_no_effective_noop.json"),
    )
    parser.add_argument("--experiment-name", type=str, required=True)
    parser.add_argument("--ablation-name", type=str, required=True)
    parser.add_argument("--output-root", type=Path, default=Path("outputs/ablations"))
    return parser


def main() -> None:
    """执行 ablation：先训练，再评估。"""

    args = build_parser().parse_args()
    payload = _load_json(args.config)
    if "train" not in payload or "eval" not in payload:
        raise ValueError("ablation 配置必须同时包含 train 与 eval 两个字段")
    if not isinstance(payload["train"], dict) or not isinstance(payload["eval"], dict):
        raise ValueError("train/eval 字段都必须是 JSON object")

    run_root = args.output_root / args.experiment_name / args.ablation_name
    run_root.mkdir(parents=True, exist_ok=True)
    train_dir = run_root / "train"
    eval_path = run_root / "eval_summary.csv"

    train_args = dict(payload["train"])
    eval_args = dict(payload["eval"])
    train_args["output_dir"] = str(train_dir)
    eval_args["model"] = str(train_dir / "model_final.pt")
    eval_args["output"] = str(eval_path)

    train_cmd = _build_command(Path("scripts/train_dqn_amc.py"), train_args)
    eval_cmd = _build_command(Path("scripts/evaluate_dqn_amc.py"), eval_args)

    # 固定在仓库根目录执行，避免调用方从其它 cwd 触发时找不到脚本/相对路径。
    subprocess.run(train_cmd, check=True, cwd=Path(__file__).resolve().parents[1])
    subprocess.run(eval_cmd, check=True, cwd=Path(__file__).resolve().parents[1])

    # 统一保存本次运行的命令与配置，确保阶段 0 复现实验可追溯。
    manifest = {
        "experiment_name": args.experiment_name,
        "ablation_name": args.ablation_name,
        "config_path": str(args.config),
        "train_command": train_cmd,
        "eval_command": eval_cmd,
        "train_dir": str(train_dir),
        "eval_summary_csv": str(eval_path),
        "eval_unified_summary_csv": str(eval_path.with_name(f"{eval_path.stem}_unified_summary.csv")),
    }
    with (run_root / "run_manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
