#!/usr/bin/env python3
"""Append an approximate tcp_to_plate vector to a phase-state LeRobot dataset.

This diagnostic transform is meant for staged Pi0 finisher experiments.  The
source dataset already uses 19-D state:

    joint6 + timestamp + phase_index_norm + phase_onehot11

The destination dataset appends three values:

    tcp_xyz_approx - plate_xyz

For frame 0 the TCP approximation comes from the collector summary
``prefix_end_debug.tcp_pos`` when available.  For later frames it uses the
previous recorded eef_abs target.  This is intentionally a smoke-test feature:
the closed-loop evaluator should use the real MuJoCo ``tcp_link`` position.
"""

from __future__ import annotations

import argparse
import copy
import json
import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np


AUTO_FEATURES = {"timestamp", "frame_index", "episode_index", "index", "task_index"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src-repo-id", required=True)
    parser.add_argument("--src-root", type=Path, required=True)
    parser.add_argument("--dst-repo-id", required=True)
    parser.add_argument("--dst-root", type=Path, required=True)
    parser.add_argument("--state-key", default="observation.state")
    parser.add_argument("--action-key", default="action")
    parser.add_argument("--robot-type", default="omy")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--image-writer-processes", type=int, default=5)
    parser.add_argument("--image-writer-threads", type=int, default=10)
    return parser.parse_args()


def data_features(features: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in features.items() if key not in AUTO_FEATURES}


def normalize_feature(value: Any, spec: dict[str, Any]) -> np.ndarray:
    arr = value.detach().cpu().numpy() if hasattr(value, "detach") else np.asarray(value)
    if spec.get("dtype") == "image":
        if arr.ndim == 3 and arr.shape[0] in (1, 3) and arr.shape[-1] not in (1, 3):
            arr = np.transpose(arr, (1, 2, 0))
        if arr.dtype != np.uint8:
            arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
        return arr
    return arr.astype(np.float32 if str(spec.get("dtype", "")).startswith("float") else arr.dtype)


def episode_range(dataset: Any, episode_index: int) -> tuple[int, int]:
    start = int(dataset.episode_data_index["from"][episode_index].item())
    end = int(dataset.episode_data_index["to"][episode_index].item())
    return start, end


def make_features(src_features: dict[str, Any], state_key: str) -> tuple[dict[str, Any], int]:
    features = {key: copy.deepcopy(value) for key, value in data_features(src_features).items()}
    state_spec = features[state_key]
    old_shape = tuple(state_spec.get("shape", ()))
    if len(old_shape) != 1:
        raise ValueError(f"{state_key} must be 1-D, got {old_shape}")
    old_dim = int(old_shape[0])
    new_dim = old_dim + 3
    state_spec["shape"] = (new_dim,)
    state_spec["names"] = ["state_phase_tcpplate" for _ in range(new_dim)]
    return features, old_dim


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def load_collection_manifest(src_root: Path) -> list[dict[str, Any]]:
    manifest = src_root / "collection_manifest.jsonl"
    if not manifest.exists():
        raise FileNotFoundError(f"Missing collection manifest: {manifest}")
    rows = read_jsonl(manifest)
    for row in rows:
        row["_resolve_root"] = str(src_root)
    return rows


def load_manifest_rows(src_root: Path) -> list[dict[str, Any]]:
    direct = src_root / "collection_manifest.jsonl"
    if direct.exists():
        return load_collection_manifest(src_root)

    state_phase_summary = src_root / "state_phase_summary.json"
    state_phase_manifest = src_root / "state_phase_manifest.jsonl"
    if not state_phase_summary.exists() or not state_phase_manifest.exists():
        raise FileNotFoundError(
            f"Missing collection manifest and state-phase source metadata under {src_root}"
        )

    summary = json.loads(state_phase_summary.read_text(encoding="utf-8"))
    raw_root = Path(summary["src_root"])
    if not raw_root.is_absolute():
        raw_root = src_root.parent / raw_root
    raw_rows = load_collection_manifest(raw_root)
    state_rows = read_jsonl(state_phase_manifest)
    rows = []
    for state_row in state_rows:
        source_episode = int(state_row["source_episode_index"])
        row = dict(raw_rows[source_episode])
        row["_state_phase_episode_index"] = int(state_row["new_episode_index"])
        rows.append(row)
    return rows


def resolve_npz(path_text: str, src_root: Path) -> Path:
    path = Path(path_text)
    candidates = [path, src_root / path, src_root.parent / path]
    resolved = next((candidate for candidate in candidates if candidate.exists()), None)
    if resolved is None:
        raise FileNotFoundError(f"Could not resolve actions_npz: {path_text}")
    return resolved


def load_summary_by_npz(npz_path: Path) -> dict[str, Any] | None:
    summary = npz_path.parent / "pi0_dagger_tail_summary.jsonl"
    if not summary.exists():
        return None
    for line in summary.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if Path(row.get("actions_npz", "")).name == npz_path.name:
            return row
    return None


def plate_from_obj_init(item: dict[str, Any]) -> np.ndarray:
    obj_init = np.asarray(item["obj_init"], dtype=np.float32).reshape(-1)
    if obj_init.shape[0] < 9:
        raise ValueError(f"Expected obj_init dim >= 9, got {obj_init.shape}")
    return obj_init[-3:].astype(np.float32)


def task_for_item(item: dict[str, Any], fallback: str) -> str:
    task = item.get("task")
    return task if isinstance(task, str) and task else fallback


def main() -> int:
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

    args = parse_args()
    if args.dst_root.exists():
        if not args.overwrite:
            raise SystemExit(f"Output root exists, use --overwrite: {args.dst_root}")
        shutil.rmtree(args.dst_root)

    src = LeRobotDataset(args.src_repo_id, root=args.src_root)
    manifest_rows = load_manifest_rows(args.src_root)
    if len(manifest_rows) != src.num_episodes:
        raise ValueError(f"Manifest rows {len(manifest_rows)} != dataset episodes {src.num_episodes}")
    features, old_state_dim = make_features(src.features, args.state_key)
    dst = LeRobotDataset.create(
        repo_id=args.dst_repo_id,
        root=args.dst_root,
        robot_type=args.robot_type,
        fps=int(src.fps),
        features=features,
        image_writer_threads=args.image_writer_threads,
        image_writer_processes=args.image_writer_processes,
    )

    out_manifest = args.dst_root / "state_phase_tcpplate_manifest.jsonl"
    fallback_task = "Pick up the object and place it on the plate."
    total_frames = 0
    for episode_index, row in enumerate(manifest_rows):
        start, end = episode_range(src, episode_index)
        resolve_root = Path(row.get("_resolve_root", str(args.src_root)))
        npz_path = resolve_npz(row["actions_npz"], resolve_root)
        npz = np.load(npz_path, allow_pickle=True)
        oracle_actions = np.asarray(npz["oracle"], dtype=np.float32)
        if oracle_actions.shape[0] != end - start:
            raise ValueError(
                f"Episode {episode_index}: oracle actions {oracle_actions.shape[0]} != frames {end - start}"
            )
        summary_row = load_summary_by_npz(npz_path)
        start_tcp = None
        if summary_row is not None:
            prefix_debug = summary_row.get("prefix_end_debug") or {}
            if "tcp_pos" in prefix_debug:
                start_tcp = np.asarray(prefix_debug["tcp_pos"], dtype=np.float32).reshape(3)
        if start_tcp is None:
            start_tcp = oracle_actions[0, :3].astype(np.float32)

        first = src[start]
        task = task_for_item(first, fallback_task)
        plate = plate_from_obj_init(first)
        tcp_to_plate_first = None
        tcp_to_plate_last = None
        for local_idx, idx in enumerate(range(start, end)):
            item = src[idx]
            frame: dict[str, Any] = {}
            for key, spec in features.items():
                if key == args.state_key:
                    state = np.asarray(item[key], dtype=np.float32).reshape(-1)
                    if state.shape[0] != old_state_dim:
                        raise ValueError(f"Expected state dim {old_state_dim}, got {state.shape[0]}")
                    tcp = start_tcp if local_idx == 0 else oracle_actions[local_idx - 1, :3].astype(np.float32)
                    tcp_to_plate = (tcp - plate).astype(np.float32)
                    if tcp_to_plate_first is None:
                        tcp_to_plate_first = tcp_to_plate.copy()
                    tcp_to_plate_last = tcp_to_plate.copy()
                    frame[key] = np.concatenate([state, tcp_to_plate]).astype(np.float32)
                else:
                    frame[key] = normalize_feature(item[key], spec)
            dst.add_frame(frame, task=task)
        dst.save_episode()
        out_row = {
            "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "episode_index": episode_index,
            "source_actions_npz": str(npz_path),
            "frames": end - start,
            "task": task,
            "old_state_dim": old_state_dim,
            "new_state_dim": old_state_dim + 3,
            "tcp_to_plate_first": None if tcp_to_plate_first is None else np.round(tcp_to_plate_first, 5).tolist(),
            "tcp_to_plate_last": None if tcp_to_plate_last is None else np.round(tcp_to_plate_last, 5).tolist(),
        }
        with out_manifest.open("a", encoding="utf-8") as f:
            f.write(json.dumps(out_row, ensure_ascii=False) + "\n")
        print(json.dumps(out_row, ensure_ascii=False), flush=True)
        total_frames += end - start

    summary = {
        "src_repo_id": args.src_repo_id,
        "src_root": str(args.src_root),
        "dst_repo_id": args.dst_repo_id,
        "dst_root": str(args.dst_root),
        "episodes": int(src.num_episodes),
        "frames": int(total_frames),
        "old_state_dim": int(old_state_dim),
        "new_state_dim": int(old_state_dim + 3),
        "feature": "append tcp_to_plate = tcp_xyz_approx - plate_xyz",
        "features": features,
    }
    (args.dst_root / "state_phase_tcpplate_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print("SUMMARY " + json.dumps(summary, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
