"""Same-frame FK-residual validation for collected LeRobot episodes.

Follow-up to issue #105 (https://github.com/datawhalechina/every-embodied/issues/105):
the collection bug recorded the *current* joint state as the action label
(`PnPEnv.step()` return value) instead of the *commanded* joint targets
(`PnPEnv.get_commanded_joint_action()`), which made trained policies freeze
in place. The label fix is merged; this script is the standing data check
agreed in the issue thread (maintainer + reporter, 2026-10).

How it works
------------
For every frame *t* of an episode where the observed joint state actually
moved since frame ``t-1`` (a "motion frame", per the agreed consecutive-frame
criterion), we run forward kinematics twice on the same MuJoCo model:

* ``FK(observation.state[t])``  -- where the arm actually was at frame ``t``;
* ``FK(action[t, :n_arm])``     -- where the recorded label claims to be.

The residual is the Euclidean distance between the two end-effector
positions. On a correctly labeled episode the residual on a motion frame is
roughly the per-frame end-effector travel (a few millimetres at 20 Hz for
this task). On a mislabeled episode the action is a copy of the state, so
the residual is exactly zero on every frame -- a fingerprint that slow
motion cannot fake.

Result
------
The script reports the fraction of motion frames whose residual is below
``--threshold`` (default ``1e-6``). A ratio near 1.0 means the recorded
action is a copy of the observed state (the issue #105 bug); a ratio near
0.0 is consistent with correct labeling.

Threshold caveat: ``1e-6`` and the resulting ratios are results observed
for *this dataset and robot* (``demo_data_example`` / OMY). They are not
general thresholds; re-check joint order, frames, and units before applying
the check to another robot (xArm6/7, Nova5), as agreed in the issue thread.

Usage
-----
    MUJOCO_GL=disable python check_fk_residual.py \
        --data demo_data_example --robot robotis_omy/omy.xml

Requires: mujoco, numpy, pandas/pyarrow (for the LeRobot parquet frames).
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

if __name__ == "__main__" and os.environ.get("MUJOCO_GL", "") == "":
    os.environ["MUJOCO_GL"] = "disable"  # headless-safe default

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from mujoco_env.y_env import SimpleEnv  # noqa: E402
from test_collection_actions import make_env, robot_parser  # noqa: E402

# Motion-frame criterion agreed in the issue thread: a frame counts as
# moving when the largest per-joint change of observation.state since the
# previous frame exceeds this value (rad). Stationary frames are skipped
# because their residual is trivially ~0 on both healthy and broken data.
DEFAULT_MOTION_THRESHOLD = 1e-8
# Residual threshold agreed in the issue thread. Dataset/robot-specific
# result, NOT a general threshold -- see module docstring.
DEFAULT_RESIDUAL_THRESHOLD = 1e-6
# A ratio above this on motion frames indicates action == state (the bug).
BROKEN_RATIO_THRESHOLD = 0.5


def load_episode_arrays(dataset_dir):
    """Load (observation.state, action) arrays for every episode parquet.

    Returns a list of (states, actions) tuples, one per episode file,
    ordered by chunk/episode path name.
    """
    import pandas as pd

    dataset_dir = Path(dataset_dir)
    info_path = dataset_dir / "meta" / "info.json"
    if info_path.exists():
        with open(info_path) as f:
            info = json.load(f)
        fps = info.get("fps")
    else:
        fps = None

    parquets = sorted(dataset_dir.glob("data/chunk-*/episode_*.parquet"))
    if not parquets:
        raise FileNotFoundError(f"no episode parquet found under {dataset_dir}/data/chunk-*")

    episodes = []
    for path in parquets:
        df = pd.read_parquet(
            path,
            columns=["observation.state", "action"],
        )
        states = np.stack(df["observation.state"].to_numpy()).astype(np.float64)
        actions = np.stack(df["action"].to_numpy()).astype(np.float64)
        episodes.append((states, actions))
    return episodes, fps


class FkResidualChecker:
    """Same-frame FK-residual check against a real robot model."""

    def __init__(self, robot_xml="robotis_omy/omy.xml", n_arm=6):
        env = make_env(n_arm=n_arm)
        env.env = robot_parser(robot_xml)
        env._configure_robot_bindings()
        self.env = env
        self.joint_names = list(env.joint_names)
        self.ee_body_name = env.ee_body_name
        if len(self.joint_names) != n_arm:
            raise ValueError(
                f"{robot_xml}: expected {n_arm} arm joints, got {len(self.joint_names)}: {self.joint_names}"
            )

    def fk_position(self, q_arm):
        """End-effector position (metres) for arm joint angles ``q_arm``."""
        # set_qpos_joints already runs mj_forward; forward(q=None) is a no-op re-forward.
        self.env.env.set_qpos_joints(joint_names=self.joint_names, qpos=np.asarray(q_arm, dtype=np.float64))
        self.env.env.forward(q=None, increase_tick=False)
        p, _ = self.env.env.get_pR_body(body_name=self.ee_body_name)
        return np.asarray(p, dtype=np.float64)

    def episode_residuals(self, states, actions, motion_threshold=DEFAULT_MOTION_THRESHOLD):
        """Same-frame FK residuals for motion frames only.

        Returns (residuals, motion_indices). ``states`` is (T, n_arm),
        ``actions`` is (T, n_arm + 1) with the gripper scalar in last place.
        """
        n_arm = len(self.joint_names)
        if states.shape[0] != actions.shape[0]:
            raise ValueError(f"frame count mismatch: state {states.shape[0]} vs action {actions.shape[0]}")
        if actions.shape[1] < n_arm:
            raise ValueError(f"action dim {actions.shape[1]} < arm joints {n_arm}; check joint order/units")

        deltas = np.abs(np.diff(states[:, :n_arm], axis=0)).max(axis=1) if states.shape[0] > 1 else np.array([])
        motion_indices = np.flatnonzero(deltas > motion_threshold) + 1  # frames t >= 1

        residuals = np.zeros(len(motion_indices), dtype=np.float64)
        state_positions = {}
        action_positions = {}
        for k, t in enumerate(motion_indices):
            st = tuple(states[t, :n_arm])
            ac = tuple(actions[t, :n_arm])
            if st not in state_positions:
                state_positions[st] = self.fk_position(st)
            if ac not in action_positions:
                action_positions[ac] = self.fk_position(ac)
            residuals[k] = float(np.linalg.norm(state_positions[st] - action_positions[ac]))
        return residuals, motion_indices


def check_dataset(
    dataset_dir,
    robot_xml="robotis_omy/omy.xml",
    residual_threshold=DEFAULT_RESIDUAL_THRESHOLD,
    motion_threshold=DEFAULT_MOTION_THRESHOLD,
):
    checker = FkResidualChecker(robot_xml=robot_xml)
    episodes, fps = load_episode_arrays(dataset_dir)

    all_residuals = []
    n_motion = 0
    for states, actions in episodes:
        residuals, motion_idx = checker.episode_residuals(states, actions, motion_threshold)
        all_residuals.extend(residuals.tolist())
        n_motion += len(motion_idx)

    all_residuals = np.asarray(all_residuals, dtype=np.float64)
    if n_motion == 0:
        raise ValueError("no motion frames found; nothing to check")

    n_near_zero = int(np.sum(all_residuals < residual_threshold))
    ratio = n_near_zero / n_motion
    median_residual = float(np.median(all_residuals))
    broken = ratio >= BROKEN_RATIO_THRESHOLD

    report = {
        "dataset": str(dataset_dir),
        "robot": robot_xml,
        "ee_body": checker.ee_body_name,
        "fps": fps,
        "motion_frames": n_motion,
        "residual_threshold": residual_threshold,
        "near_zero_motion_frames": n_near_zero,
        "near_zero_ratio": ratio,
        "median_residual_m": median_residual,
        "action_is_copy_of_state": bool(broken),
        "verdict": "BROKEN (action appears to be a copy of observation.state)" if broken
        else "HEALTHY (action differs from state on motion frames)",
    }
    return report


def format_report(report):
    lines = [
        "Same-frame FK residual check (issue #105 follow-up)",
        f"  dataset:                {report['dataset']}",
        f"  robot:                  {report['robot']} (ee: {report['ee_body']})",
        f"  fps:                    {report['fps']}",
        f"  motion frames:          {report['motion_frames']}",
        f"  residual threshold:     {report['residual_threshold']:g} (dataset/robot-specific)",
        f"  near-zero frames:       {report['near_zero_motion_frames']}/{report['motion_frames']}",
        f"  near-zero ratio:        {report['near_zero_ratio']:.6f}",
        f"  median residual:        {report['median_residual_m']*1000:.3f} mm",
        f"  verdict:                {report['verdict']}",
    ]
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", default=str(ROOT / "demo_data_example"), help="LeRobot dataset dir")
    parser.add_argument("--robot", default="robotis_omy/omy.xml", help="robot XML relative to asset/")
    parser.add_argument("--n-arm", type=int, default=6, help="number of arm joints")
    parser.add_argument("--threshold", type=float, default=DEFAULT_RESIDUAL_THRESHOLD)
    parser.add_argument("--motion-threshold", type=float, default=DEFAULT_MOTION_THRESHOLD)
    parser.add_argument("--budget", type=float, default=120.0, help="wall-clock budget in seconds")
    args = parser.parse_args(argv)

    deadline = time.monotonic() + args.budget
    report = check_dataset(
        args.data,
        robot_xml=args.robot,
        residual_threshold=args.threshold,
        motion_threshold=args.motion_threshold,
    )
    if time.monotonic() > deadline:
        print("TIME BUDGET EXCEEDED", file=sys.stderr)
        return 2
    print(format_report(report))
    # Exit code 1 flags a mislabeled dataset; 0 means the check itself passed.
    return 1 if report["action_is_copy_of_state"] else 0


if __name__ == "__main__":
    sys.exit(main())
