"""Capture native R1 Pro state/images and safely replay actual G0.5 arm chunks."""

import argparse
import json
from pathlib import Path

import numpy as np

from galaxea_piano import build_task
from physical_piano import position_targets
from g05_piano_probe import sha256


def load_prediction(path, report_path, call_index):
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report["backend"] not in {"G0.5_Qwen35_autoregressive", "G0.5_Qwen35_policy"} or report["embodiment"] != "galaxea_r1pro":
        raise ValueError("Expected identified native G0.5 R1 Pro inference")
    call = report["calls"][call_index]
    if path.name != call["actions_file"] or sha256(path) != call["actions_sha256"]:
        raise ValueError("Model action identity mismatch")
    allowed = {"left_control", "right_control", "left_gripper", "right_gripper", "lower_body"}
    absent = set(call["absent_action_groups"])
    if not absent <= allowed:
        raise ValueError("Unknown absent action group")
    with np.load(path, allow_pickle=False) as archive:
        actions = {k: archive[k] for k in archive.files}
    for side in ("left", "right"):
        for part, dim in (("arm", 7), ("gripper", 1)):
            key = side + "_" + part
            value = actions[key]
            if value.ndim == 3 and value.shape[0] == 1:
                value = value[0]
            if value.ndim != 2 or value.shape[1] != dim or not 1 <= len(value) <= 64 or not np.isfinite(value).all():
                raise ValueError("Bad native action shape: " + key)
            actions[key] = value
    if len({len(v) for v in actions.values()}) != 1:
        raise ValueError("Inconsistent action horizons")
    return actions, absent, report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--actions", type=Path)
    parser.add_argument("--inference-report", type=Path)
    parser.add_argument("--call-index", type=int, default=0)
    args = parser.parse_args()
    if args.actions is not None and args.inference_report is None:
        parser.error("--actions requires --inference-report to preserve absent action masks")
    args.out.mkdir(parents=True, exist_ok=False)
    import imageio.v2 as imageio
    import mujoco
    from dm_control import mjcf
    from mujoco_utils import composer_utils

    task = build_task(args.model, native_grippers=True)
    env = composer_utils.Environment(task=task, strip_singleton_obs_buffer_dim=True, recompile_physics=False)
    writer = None
    try:
        env.reset()
        physics, model = env.physics, env.physics.model.ptr
        for name, value in task.robot.seed.items():
            physics.bind(task.robot.mjcf_model.find("joint", name)).qpos = value
        physics.forward()
        initial = physics.data.qpos.copy()
        initial_vel = physics.data.qvel.copy()
        control = position_targets(model, physics.data.ptr)
        mjcf.export_with_assets(task.root_entity.mjcf_model, str(args.out / "scene"), out_file_name="scene.xml")
        observation = {}
        for name, size in (("head_rgb", (640, 360)), ("left_wrist_rgb", (640, 480)), ("right_wrist_rgb", (640, 480))):
            camera = task.robot.mjcf_model.find("camera", name).full_identifier
            pixels = physics.render(width=size[0], height=size[1], camera_id=camera)
            imageio.imwrite(args.out / (name + ".png"), pixels)
            observation[name] = pixels.transpose(2, 0, 1)
        for side in ("left", "right"):
            observation[side + "_arm"] = np.asarray([physics.bind(j).qpos[0] for j in task.robot.arm_joints[side]], dtype=np.float32)
            observation[side + "_gripper"] = np.array([float(physics.bind(task.robot.mjcf_model.find("joint", side + "_gripper_finger_joint1")).qpos[0]) * 2000], dtype=np.float32)
        np.savez_compressed(args.out / "observation.npz", **observation)
        imageio.imwrite(args.out / "whole.png", physics.render(width=960, height=720, camera_id="whole_robot"))
        if args.actions is None:
            return
        actions, absent_groups, inference_report = load_prediction(args.actions, args.inference_report, args.call_index)
        horizon = min(len(v) for v in actions.values())
        if sha256(args.out / "observation.npz") != inference_report["observation_sha256"]:
            raise ValueError("Replay observation differs from the model input; recapture and infer")
        writer = imageio.get_writer(str(args.out / "probe.mp4"), fps=25, codec="libx264", macro_block_size=1)
        previous = np.zeros(88, dtype=bool)
        onsets, controls, states, safety = [], [], [], []
        baseline = observation
        group_map = {"left_control": "left_arm", "right_control": "right_arm", "left_gripper": "left_gripper", "right_gripper": "right_gripper"}
        absent = {group_map.get(k, k) for k in absent_groups}
        # Bounded absolute joint targets: 0.35 rad from initial state, 1 rad/s slew.
        for step in range(150):
            row = min(int(step * 0.02 * 15), horizon - 1)
            clipped = 0
            for side in ("left", "right"):
                requested = baseline[side + "_arm"] if side + "_arm" in absent else actions[side + "_arm"][row]
                target = np.clip(requested, baseline[side + "_arm"] - 0.35, baseline[side + "_arm"] + 0.35)
                clipped += int(np.count_nonzero(target != requested))
                for joint, value in zip(task.robot.arm_joints[side], target):
                    actuator = task.robot.mjcf_model.find("actuator", joint.name + "_act")
                    index = int(physics.bind(actuator).element_id)
                    safe = np.clip(value, control[index] - 0.02, control[index] + 0.02)
                    control[index] = np.clip(safe, *model.actuator_ctrlrange[index])
                opening = baseline[side + "_gripper"][0] if side + "_gripper" in absent else float(actions[side + "_gripper"][row, 0])
                for i, sign in ((1, 1), (2, -1)):
                    actuator = task.robot.mjcf_model.find("actuator", f"{side}_gripper_finger_joint{i}_act")
                    index = int(physics.bind(actuator).element_id)
                    target = sign * np.clip(opening, 0, 100) / 2000
                    control[index] = np.clip(target, control[index] - 0.001, control[index] + 0.001)
            env.step(control)
            if not np.isfinite(physics.data.qpos).all():
                raise ValueError("Non-finite simulation state")
            active = task.piano.activation.copy()
            onsets.extend({"midi": int(k + 21), "time_seconds": float(physics.data.time)} for k in np.flatnonzero(active & ~previous))
            previous = active
            controls.append(control.copy()); states.append(physics.data.qpos.copy()); safety.append(clipped)
            if step % 2 == 0:
                pixels = physics.render(width=960, height=720, camera_id="whole_robot")
                writer.append_data(pixels)
                if step in (0, 74, 148):
                    imageio.imwrite(args.out / f"frame-{step:04d}.png", pixels)
        writer.close(); writer = None
        np.savez_compressed(args.out / "trajectory.npz", controls=controls, qpos=states, initial_qpos=initial, initial_qvel=initial_vel)
        report = {"native_g05_actions_executed": True, "source_actions": str(args.actions), "actions_sha256": sha256(args.actions), "checkpoint_sha256": inference_report["checkpoint_sha256"], "gpt_plan_sha256": inference_report["gpt_plan_sha256"], "call_index": args.call_index, "absent_groups_held": sorted(absent), "duration_seconds": 3, "horizon": horizon, "new_inferences_during_replay": 0, "closed_loop": False, "arm_max_offset_radians": 0.35, "arm_slew_radians_per_second": 1, "clipped_arm_coordinates_total": sum(safety), "measured_piano_onsets": onsets, "song_success_verified": False, "base": "fixed_chassis", "grippers": "native_parallel", "finger_joint_teleport_during_execution": False}
        (args.out / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(report, indent=2))
    finally:
        if writer is not None:
            writer.close()
        env.close()


if __name__ == "__main__":
    main()
