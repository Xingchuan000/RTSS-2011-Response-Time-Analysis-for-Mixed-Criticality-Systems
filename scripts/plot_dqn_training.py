"""DQN 训练过程诊断绘图脚本。"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

# 在受限环境中显式设置 matplotlib 缓存目录，避免权限相关告警。
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

matplotlib.use("Agg")


def build_parser() -> argparse.ArgumentParser:
    """构建命令行参数解析器。"""

    parser = argparse.ArgumentParser()
    parser.add_argument("--train-log", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def _plot_episode_reward(df: pd.DataFrame, output_path: Path) -> None:
    """绘制每个 episode 的累计 reward 曲线。"""

    episode_df = df.groupby("episode", as_index=False)["episode_reward"].max()
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(episode_df["episode"], episode_df["episode_reward"], marker="o")
    ax.set_title("Episode Reward")
    ax.set_xlabel("episode")
    ax.set_ylabel("episode_reward")
    ax.grid(True, alpha=0.3)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_loss(df: pd.DataFrame, output_path: Path) -> None:
    """绘制优化 loss 曲线。"""

    loss_df = df[df["loss"].notna()].copy()
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(loss_df["step"], loss_df["loss"], marker="o", linewidth=1.0)
    ax.set_title("Training Loss")
    ax.set_xlabel("step")
    ax.set_ylabel("loss")
    ax.grid(True, alpha=0.3)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_epsilon(df: pd.DataFrame, output_path: Path) -> None:
    """绘制 epsilon 衰减曲线。"""

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(df["step"], df["epsilon"], linewidth=1.2)
    ax.set_title("Epsilon")
    ax.set_xlabel("step")
    ax.set_ylabel("epsilon")
    ax.grid(True, alpha=0.3)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_action_counts(df: pd.DataFrame, output_path: Path) -> None:
    """绘制每个 episode 的 accepted/rejected/noop 数量。"""

    action_df = pd.DataFrame(
        {
            "episode": df["episode"],
            "accepted": df["accepted"].fillna(False).astype(bool).astype(int),
            "rejected": df["rejected"].fillna(False).astype(bool).astype(int),
            "noop": df["noop_due_to_no_valid_action"].fillna(False).astype(bool).astype(int),
        }
    )
    grouped = action_df.groupby("episode", as_index=False)[["accepted", "rejected", "noop"]].sum()

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(grouped["episode"], grouped["accepted"], marker="o", label="accepted")
    ax.plot(grouped["episode"], grouped["rejected"], marker="o", label="rejected")
    ax.plot(grouped["episode"], grouped["noop"], marker="o", label="noop")
    ax.set_title("Action Counts")
    ax.set_xlabel("episode")
    ax.set_ylabel("count")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    """读取训练日志并输出诊断图。"""

    args = build_parser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.train_log)
    df["loss"] = pd.to_numeric(df["loss"], errors="coerce")
    df["epsilon"] = pd.to_numeric(df["epsilon"], errors="coerce")
    df["episode_reward"] = pd.to_numeric(df["episode_reward"], errors="coerce")
    df["step"] = pd.to_numeric(df["step"], errors="coerce")

    _plot_episode_reward(df, args.output_dir / "episode_reward.png")
    _plot_loss(df, args.output_dir / "loss.png")
    _plot_epsilon(df, args.output_dir / "epsilon.png")
    _plot_action_counts(df, args.output_dir / "action_counts.png")


if __name__ == "__main__":
    main()
