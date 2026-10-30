#!/usr/bin/env python3
"""Per-episode physical replay audit for the randomized PnP LeRobot dataset."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd


os.environ.setdefault("SUCCESS_REQUIRE_EE_HEIGHT", "0")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", default="datawhale_eai_pnp")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--hz", type=float, default=20.0)
    parser.add_argument("--settle-actions", type=int, default=20)
    parser.add_argument("--episodes", default="all", help="all or comma-separated episode indexes")
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--summary-json", type=Path, required=True)
    parser.add_argument("--physical-min-lift", type=float, default=0.06)
    parser.add_argument("--physical-min-lift-steps", type=int, default=3)
    parser.add_argument("--physical-final-upright-cos", type=float, default=0.85)
    return parser.parse_args()


def load_manifest(root: Path) -> dict[int, dict]:
    path = root / "collection_manifest.jsonl"
    if not path.exists():
        return {}
    rows = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        rows[int(row["episode_in_run"])] = row
    return rows


def parse_episode_indexes(spec: str, total: int) -> list[int]:
    if spec.strip().lower() == "all":
        return list(range(total))
    indexes = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        indexes.append(int(part))
    return indexes


def load_episode_lengths(root: Path) -> list[int]:
    path = root / "meta" / "episodes.jsonl"
    lengths: list[int] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        lengths.append(int(row["length"]))
    return lengths


def episode_path(root: Path, episode_index: int) -> Path:
    info = json.loads((root / "meta" / "info.json").read_text(encoding="utf-8"))
    chunk_size = int(info.get("chunks_size", 1000))
    episode_chunk = episode_index // chunk_size
    rel = info.get(
        "data_path",
        "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
    ).format(episode_chunk=episode_chunk, episode_index=episode_index)
    return root / rel


def load_episode_actions(root: Path, episode_index: int) -> np.ndarray:
    df = pd.read_parquet(episode_path(root, episode_index), columns=["action"])
    return np.stack([np.asarray(action, dtype=np.float32) for action in df["action"].tolist()])


def replay_episode(actions: np.ndarray, env, episode_index: int, seed: int | None, args: argparse.Namespace) -> dict:
    from eval_policy_success import (
        get_act_physical_debug,
        init_act_physical_tracker,
        update_act_physical_tracker,
    )

    length = int(actions.shape[0])
    total_actions = length + max(args.settle_actions, 0)
    env.reset(seed=seed)

    tracker = init_act_physical_tracker(env)
    action_steps = 0
    sim_steps = 0
    success_ever = False
    first_success_step = None
    physical_success_ever = False
    first_physical_success_step = None
    last_debug = {}
    last_action = actions[0]

    start_time = time.time()
    while action_steps < total_actions and env.env.is_viewer_alive():
        env.step_env()
        sim_steps += 1
        if not env.env.loop_every(HZ=args.hz):
            continue

        if action_steps < length:
            action = actions[action_steps]
            last_action = action
            phase = "dataset"
        else:
            action = last_action
            phase = "settle"

        env.step(action)
        update_act_physical_tracker(env, tracker, args)
        if hasattr(env, "get_success_debug"):
            last_debug = env.get_success_debug()
        else:
            last_debug = {"success": bool(env.check_success())}
        last_debug = get_act_physical_debug(env, tracker, args, last_debug)

        if env.check_success():
            success_ever = True
            if first_success_step is None:
                first_success_step = action_steps
        if last_debug["physical_success"]:
            physical_success_ever = True
            if first_physical_success_step is None:
                first_physical_success_step = action_steps

        action_steps += 1

    if hasattr(env, "get_success_debug"):
        last_debug = env.get_success_debug()
    else:
        last_debug = {"success": bool(env.check_success())}
    last_debug = get_act_physical_debug(env, tracker, args, last_debug)
    physical_success = bool(physical_success_ever or last_debug["physical_success"])
    success = bool(success_ever or env.check_success())
    last_debug.update(
        {
            "legacy_success_ever": success,
            "first_legacy_success_step": first_success_step,
            "physical_success_ever": physical_success,
            "first_physical_success_step": first_physical_success_step,
        }
    )
    return {
        "episode_index": episode_index,
        "seed": seed,
        "dataset_length": length,
        "settle_actions": max(args.settle_actions, 0),
        "action_steps": action_steps,
        "sim_steps": sim_steps,
        "elapsed_s": round(time.time() - start_time, 3),
        "last_phase": phase if action_steps else None,
        "success": success,
        "physical_success": physical_success,
        "debug": last_debug,
    }


def main() -> int:
    from mujoco_env.y_env import SimpleEnv

    args = parse_args()
    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    if args.output_jsonl.exists():
        args.output_jsonl.unlink()

    episode_lengths = load_episode_lengths(args.root)
    manifest = load_manifest(args.root)
    indexes = parse_episode_indexes(args.episodes, len(episode_lengths))
    env = SimpleEnv("./asset/example_scene_y.xml", action_type="joint_angle")

    rows = []
    try:
        for episode_index in indexes:
            manifest_row = manifest.get(episode_index, {})
            seed = manifest_row.get("seed")
            actions = load_episode_actions(args.root, episode_index)
            row = replay_episode(actions, env, episode_index, seed, args)
            row["manifest"] = {
                "seed": seed,
                "frames": manifest_row.get("frames"),
                "obj_init": manifest_row.get("obj_init"),
            }
            rows.append(row)
            with args.output_jsonl.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
            print(json.dumps(row, ensure_ascii=False), flush=True)
    finally:
        try:
            env.env.close_viewer()
        except Exception:
            pass

    summary = {
        "repo_id": args.repo_id,
        "root": str(args.root),
        "episodes": len(rows),
        "legacy_success_count": sum(1 for row in rows if row["success"]),
        "physical_success_count": sum(1 for row in rows if row["physical_success"]),
        "legacy_success_rate": sum(1 for row in rows if row["success"]) / max(len(rows), 1),
        "physical_success_rate": sum(1 for row in rows if row["physical_success"]) / max(len(rows), 1),
        "episode_indexes": indexes,
        "physical_min_lift": args.physical_min_lift,
        "physical_min_lift_steps": args.physical_min_lift_steps,
        "physical_final_upright_cos": args.physical_final_upright_cos,
    }
    args.summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("SUMMARY " + json.dumps(summary, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
