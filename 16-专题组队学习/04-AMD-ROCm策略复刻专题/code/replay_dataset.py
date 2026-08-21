#!/usr/bin/env python
"""从 LeRobot 数据集回放一个 episode 的 agent/wrist 视角视频。

示例：
  python code/replay_dataset.py --list
  python code/replay_dataset.py --episode 0
  python code/replay_dataset.py --episode 3 --view wrist --fps 20

数据集、模型和输出不随轻量分支提交。默认数据目录是
DATA_ROOT/omy_pnp_language，也可以通过 DATASET_ROOT 显式指定。
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import numpy as np


TOPIC_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = Path(os.environ.get("DATA_ROOT", TOPIC_ROOT / "data")).expanduser()
DATASET_ROOT = Path(
    os.environ.get("DATASET_ROOT", DATA_ROOT / "omy_pnp_language")
).expanduser()
OUTPUT_ROOT = Path(os.environ.get("OUTPUT_ROOT", TOPIC_ROOT / "outputs")).expanduser()
DATASET_REPO_ID = os.environ.get("DATASET_REPO_ID", "datawhale_eai_pnp_language")


def episode_rows() -> list[dict]:
    path = DATASET_ROOT / "meta" / "episodes.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"找不到 episodes.jsonl：{path}")
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def list_episodes() -> None:
    rows = episode_rows()
    print(f"{'ep':>3} {'task':<40} {'frames':>6} {'时长s':>6}")
    print("-" * 60)
    for row in rows:
        task = str(row.get("tasks", [""])[0])
        length = int(row.get("length", 0))
        print(
            f"{int(row.get('episode_index', 0)):>3} "
            f"{task:<40} {length:>6} {length / 20:>6.1f}"
        )


def to_uint8(image) -> np.ndarray:
    if hasattr(image, "detach"):
        image = image.detach().cpu().numpy()
    image = np.asarray(image)

    if image.ndim == 3 and image.shape[0] in (1, 3) and image.shape[-1] not in (1, 3):
        image = np.moveaxis(image, 0, -1)
    if image.ndim == 2:
        image = np.repeat(image[..., None], 3, axis=-1)
    if image.ndim != 3 or image.shape[-1] not in (1, 3):
        raise ValueError(f"无法识别图像形状：{image.shape}")
    if image.shape[-1] == 1:
        image = np.repeat(image, 3, axis=-1)

    if np.issubdtype(image.dtype, np.floating):
        scale = 255.0 if float(np.nanmax(image)) <= 1.5 else 1.0
        image = image * scale
    return np.clip(image, 0, 255).astype(np.uint8)


def choose_feature(dataset, view: str) -> str:
    available = set()
    meta = getattr(dataset, "meta", None)
    features = getattr(meta, "features", {}) if meta is not None else {}
    if isinstance(features, dict):
        available = set(features)

    candidates = (
        ("observation.image", "observation.agent_image", "agent_image")
        if view == "agent"
        else ("observation.wrist_image", "wrist_image")
    )
    if available:
        for candidate in candidates:
            if candidate in available:
                return candidate
        raise KeyError(
            f"数据集没有 {view} 视角特征；当前可用特征：{sorted(available)}"
        )
    return candidates[0]


def replay(
    episode_index: int,
    view: str,
    output_path: Path,
    fps: int,
    max_frames: int,
) -> None:
    from video_io import write_video
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

    if max_frames <= 0:
        raise ValueError("max_frames 必须大于 0。")
    rows = episode_rows()
    indices = [int(row.get("episode_index", -1)) for row in rows]
    if episode_index not in indices:
        raise IndexError(
            f"episode {episode_index} 不存在，可用编号为：{indices}"
        )

    dataset = LeRobotDataset(DATASET_REPO_ID, root=str(DATASET_ROOT))
    feature = choose_feature(dataset, view)
    ep_start = int(dataset.episode_data_index["from"][episode_index].item())
    ep_end = int(dataset.episode_data_index["to"][episode_index].item())
    frame_count = ep_end - ep_start
    step = max(1, math.ceil(frame_count / max_frames))

    frames = [
        to_uint8(dataset[index][feature])
        for index in range(ep_start, ep_end, step)
    ]
    if not frames:
        raise RuntimeError(f"episode {episode_index} 没有可回放的帧。")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    codec = write_video(output_path, frames, fps)
    print(
       f"完成：episode={episode_index}, view={view}, "
       f"feature={feature}, codec={codec}, frames={len(frames)} -> {output_path}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--episode", type=int, default=0)
    parser.add_argument(
        "-v", "--view", choices=["agent", "wrist"], default="agent"
    )
    parser.add_argument("-o", "--output", type=Path, default=None)
    parser.add_argument("--dataset-root", type=Path, default=None)
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--max-frames", type=int, default=200)
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args()

    global DATASET_ROOT
    if args.dataset_root is not None:
        DATASET_ROOT = args.dataset_root.expanduser()

    if args.list:
        list_episodes()
        return

    output = args.output
    if output is None:
        output = OUTPUT_ROOT / f"replay_ep{args.episode}_{args.view}.mp4"
    replay(args.episode, args.view, output, args.fps, args.max_frames)


if __name__ == "__main__":
    main()
