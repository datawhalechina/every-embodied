"""Replay the exported actuator sequence with native MuJoCo, not qpos animation."""

import argparse
import json
import time
from pathlib import Path


def key_activation(qpos, ranges, tolerance):
    import numpy as np

    if not np.isfinite(qpos).all():
        raise ValueError("Non-finite key state")
    # MuJoCo joint limits are soft. Clip only the measured readout, as RoboPianist does.
    state = np.clip(qpos, ranges[:, 0], ranges[:, 1])
    return np.abs(state - ranges[:, 1]) <= tolerance


def compare_parts(onsets, expected):
    """Compare disjoint pitch parts without imposing ordering on simultaneous hands."""
    pitches = {side: set(part) for side, part in expected.items()}
    if pitches["left"] & pitches["right"]:
        raise ValueError("Pitch-based comparison requires disjoint left/right registers")
    actual = {side: [event["midi"] for event in onsets if event["midi"] in register] for side, register in pitches.items()}
    unexpected = [event for event in onsets if event["midi"] not in pitches["left"] | pitches["right"]]
    return actual, unexpected, actual == expected and not unexpected


def replay(run, viewer=False):
    import mujoco
    import numpy as np

    manifest = json.loads((run / "scene_manifest.json").read_text(encoding="utf-8"))
    if mujoco.__version__ != manifest["mujoco_version"]:
        raise ValueError("Use the recorded MuJoCo version before comparing a replay")
    scene = (run / manifest["scene"]).resolve()
    if not scene.is_relative_to(run.resolve()):
        raise ValueError("Scene must be inside the exported run")
    # MuJoCo 3.1.6 on Windows cannot reliably open CJK paths; use its asset VFS.
    assets = {p.name: p.read_bytes() for p in scene.parent.iterdir() if p.is_file() and p.suffix.lower() in (".obj", ".stl", ".msh", ".png", ".jpg", ".jpeg", ".xml") and p != scene}
    model = mujoco.MjModel.from_xml_string(scene.read_text(encoding="utf-8"), assets=assets)
    data = mujoco.MjData(model)
    trajectory = np.load(run / "trajectory.npz", allow_pickle=False)
    controls = trajectory["controls"]
    if controls.ndim != 2 or controls.shape[1] != model.nu or not np.isfinite(controls).all():
        raise ValueError("Invalid actuator trajectory")
    if (controls < model.actuator_ctrlrange[:, 0]).any() or (controls > model.actuator_ctrlrange[:, 1]).any():
        raise ValueError("Actuator targets exceed the recorded limits")
    if any((mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i) or "").startswith("piano/") for i in range(model.nu)):
        raise ValueError("Piano actuators are not permitted in the contact replay")
    ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in manifest["key_joints"]]
    if len(ids) != 88 or min(ids) < 0:
        raise ValueError("The exported 88-key joint contract is incomplete")
    addresses = model.jnt_qposadr[ids]
    key_geoms = []
    for joint in ids:
        candidates = np.flatnonzero(model.geom_bodyid == model.jnt_bodyid[joint])
        if len(candidates) != 1:
            raise ValueError("Expected one physical geom per piano key")
        key_geoms.append(int(candidates[0]))
    original_colors = model.geom_rgba[key_geoms].copy()
    if not np.isfinite(trajectory["initial_qpos"]).all() or not np.isfinite(trajectory["initial_qvel"]).all():
        raise ValueError("Non-finite initial state")
    if abs(manifest["physics_steps_per_control"] * model.opt.timestep - manifest["control_timestep_seconds"]) > 1e-9:
        raise ValueError("Recorded control and physics clocks are inconsistent")
    # Initial-state loading is separate from execution; qpos is never assigned in the loop.
    data.qpos[:] = trajectory["initial_qpos"]
    data.qvel[:] = trajectory["initial_qvel"]
    mujoco.mj_forward(model, data)
    previous = np.zeros(len(ids), dtype=bool)
    onsets = []
    handle = None
    try:
        if viewer:
            import mujoco.viewer

            handle = mujoco.viewer.launch_passive(model, data)
            camera = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, manifest.get("camera", "piano/back"))
            handle.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
            handle.cam.fixedcamid = camera
        for control in controls:
            if handle is not None and not handle.is_running():
                raise RuntimeError("Viewer closed before the replay completed")
            start = time.perf_counter()
            data.ctrl[:] = control
            for _ in range(manifest["physics_steps_per_control"]):
                mujoco.mj_step(model, data)
            if not np.isfinite(data.qpos).all():
                raise ValueError("Non-finite physical state")
            active = key_activation(data.qpos[addresses], model.jnt_range[ids], manifest["activation_tolerance_radians"])
            model.geom_rgba[key_geoms] = np.where(active[:, None], [0.2, 0.8, 0.2, 1.0], original_colors)
            onsets.extend({"midi": int(key + manifest["minimum_midi"]), "time_seconds": round(float(data.time), 4)} for key in np.flatnonzero(active & ~previous))
            previous = active
            if handle is not None:
                handle.sync()
                time.sleep(max(0, manifest["control_timestep_seconds"] - (time.perf_counter() - start)))
        source = json.loads((run / "report.json").read_text(encoding="utf-8"))
        if "expected_parts" in source:
            actual, unexpected, matches = compare_parts(onsets, source["expected_parts"])
        else:
            actual, unexpected = None, []
            matches = [n["midi"] for n in onsets] == source["expected_onsets"]
        report = {"mode": "native_actuator_replay_no_model_calls", "measured_onsets": onsets, "measured_parts": actual, "unexpected_onsets": unexpected, "onset_sequence_matches": matches, "model_calls": 0, "key_state_teleported": False, "audio": "See demo.mp4 for the measured audio from the source run; native replay is silent."}
        (run / "native_replay_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        return report
    finally:
        if handle is not None:
            handle.close()
        trajectory.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--viewer", action="store_true")
    args = parser.parse_args()
    report = replay(args.run, args.viewer)
    print(json.dumps(report, indent=2))
    if not report["onset_sequence_matches"]:
        raise SystemExit("Replay did not reproduce the expected onsets")


if __name__ == "__main__":
    main()
