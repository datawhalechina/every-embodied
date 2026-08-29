#!/usr/bin/env python3
"""Convert successful DISCOVERSE AIRBOT episodes to LeRobotDataset v2."""

from __future__ import annotations

import argparse
import inspect
import json
import shutil
from pathlib import Path

import cv2
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--repo-id", default="local/discoverse-place-block")
    parser.add_argument("--task", default="Place the green block in the pink bowl")
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument(
        "--action-bias",
        type=int,
        default=-1,
        help="Use act[t+1] for DISCOVERSE post-action observations.",
    )
    parser.add_argument("--max-episodes", type=int)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def dataset_features(image_size: int) -> dict[str, dict[str, object]]:
    image_shape = (image_size, image_size, 3)
    return {
        "observation.image": {
            "dtype": "image",
            "shape": image_shape,
            "names": ["height", "width", "channels"],
        },
        "observation.wrist_image": {
            "dtype": "image",
            "shape": image_shape,
            "names": ["height", "width", "channels"],
        },
        "observation.state": {
            "dtype": "float32",
            "shape": (7,),
            "names": ["state"],
        },
        "action": {
            "dtype": "float32",
            "shape": (7,),
            "names": ["action"],
        },
    }


def read_rgb_video(path: Path, image_size: int) -> list[np.ndarray]:
    capture = cv2.VideoCapture(str(path))
    frames: list[np.ndarray] = []
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(
                frame, (image_size, image_size), interpolation=cv2.INTER_AREA
            )
            frames.append(np.ascontiguousarray(frame))
    finally:
        capture.release()
    if not frames:
        raise ValueError(f"No video frames decoded from {path}")
    return frames


def discover_episode_dirs(input_root: Path) -> list[Path]:
    episodes = [
        path
        for path in input_root.iterdir()
        if path.is_dir() and (path / "obs_action.json").is_file()
    ]
    return sorted(episodes, key=lambda path: int(path.name))


def load_dataset_class():
    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
    except ImportError:
        from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
    return LeRobotDataset


def main() -> int:
    args = parse_args()
    if args.overwrite and args.output_root.exists():
        shutil.rmtree(args.output_root)
    if args.output_root.exists():
        raise SystemExit(f"Output already exists; use --overwrite: {args.output_root}")

    dataset_class = load_dataset_class()
    dataset = dataset_class.create(
        repo_id=args.repo_id,
        root=args.output_root,
        robot_type="airbot_play",
        fps=args.fps,
        features=dataset_features(args.image_size),
        image_writer_threads=8,
        image_writer_processes=0,
    )
    add_frame_accepts_task = "task" in inspect.signature(dataset.add_frame).parameters

    episode_dirs = discover_episode_dirs(args.input_root)
    if args.max_episodes is not None:
        episode_dirs = episode_dirs[: args.max_episodes]
    if not episode_dirs:
        raise ValueError(f"No DISCOVERSE episodes found under {args.input_root}")

    manifest = []
    for output_index, episode_dir in enumerate(episode_dirs):
        record = json.loads((episode_dir / "obs_action.json").read_text())
        states = np.asarray(record["obs"]["jq"], dtype=np.float32)
        raw_actions = np.asarray(record["act"], dtype=np.float32)
        if len(raw_actions) == 0:
            raise ValueError(f"Episode {episode_dir.name} has no actions")
        action_indices = np.clip(
            np.arange(len(raw_actions)) - args.action_bias,
            0,
            len(raw_actions) - 1,
        )
        actions = raw_actions[action_indices]
        camera0 = read_rgb_video(episode_dir / "cam_0.mp4", args.image_size)
        camera1 = read_rgb_video(episode_dir / "cam_1.mp4", args.image_size)

        lengths = {
            "state": len(states),
            "action": len(actions),
            "camera0": len(camera0),
            "camera1": len(camera1),
        }
        if len(set(lengths.values())) != 1:
            raise ValueError(f"Episode {episode_dir.name} is not aligned: {lengths}")
        if states.shape[1:] != (7,) or actions.shape[1:] != (7,):
            raise ValueError(
                f"Episode {episode_dir.name} expected 7-D state/action, got "
                f"{states.shape} and {actions.shape}"
            )
        if not np.isfinite(states).all() or not np.isfinite(actions).all():
            raise ValueError(f"Episode {episode_dir.name} contains NaN or Inf")

        for frame_index in range(len(actions)):
            frame = {
                "observation.image": camera0[frame_index],
                "observation.wrist_image": camera1[frame_index],
                "observation.state": states[frame_index],
                "action": actions[frame_index],
            }
            if add_frame_accepts_task:
                dataset.add_frame(frame, task=args.task)
            else:
                frame["task"] = args.task
                dataset.add_frame(frame)
        dataset.save_episode()
        manifest.append(
            {
                "output_episode": output_index,
                "source_episode": episode_dir.name,
                "frames": len(actions),
                "first_action_index": int(action_indices[0]),
                "last_action_index": int(action_indices[-1]),
            }
        )
        print(
            f"converted {output_index + 1}/{len(episode_dirs)} "
            f"source={episode_dir.name} frames={len(actions)}",
            flush=True,
        )

    manifest_path = args.output_root / "discoverse_conversion_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "source_root": str(args.input_root.resolve()),
                "episodes": len(manifest),
                "frames": sum(item["frames"] for item in manifest),
                "task": args.task,
                "action_bias": args.action_bias,
                "items": manifest,
            },
            indent=2,
        )
    )
    print(f"conversion complete: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
