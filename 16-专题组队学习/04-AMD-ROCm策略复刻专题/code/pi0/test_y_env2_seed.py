#!/usr/bin/env python3
"""Verify that SimpleEnv2 reset seeds are distinct and repeatable."""

from __future__ import annotations

from mujoco_env.y_env2 import SimpleEnv2


def main() -> int:
    env = SimpleEnv2("./asset/example_scene_y2.xml", action_type="joint_angle")
    positions = []
    try:
        for seed in (0, 1, 2, 1):
            env.env.reset(step=False)
            env.reset(seed=seed)
            position = env.get_obj_pose()[1].round(6).tolist()
            positions.append(position)
            print(seed, position)
    finally:
        env.env.close_viewer()
    if positions[0] == positions[1] or positions[1] == positions[2]:
        raise SystemExit("Different seeds produced identical blue-mug positions")
    if positions[1] != positions[3]:
        raise SystemExit("Repeated seed was not deterministic")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
