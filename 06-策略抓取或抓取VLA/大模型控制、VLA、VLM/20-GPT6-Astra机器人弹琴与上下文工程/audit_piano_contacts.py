"""Audit all physical substeps of an exported actuator replay, including contact ownership."""

import argparse
import json
from pathlib import Path

from piano_context import write_json
from replay_windows import compare_parts, key_activation


def audit(run):
    import mujoco
    import numpy as np

    run = Path(run).resolve()
    manifest = json.loads((run / "scene_manifest.json").read_text(encoding="utf-8"))
    source = json.loads((run / "report.json").read_text(encoding="utf-8"))
    if mujoco.__version__ != manifest["mujoco_version"]:
        raise ValueError("Use the recorded MuJoCo version")
    scene = (run / manifest["scene"]).resolve()
    if not scene.is_relative_to(run):
        raise ValueError("Scene must be inside the run")
    assets = {path.name: path.read_bytes() for path in scene.parent.iterdir() if path.is_file() and path != scene}
    model = mujoco.MjModel.from_xml_string(scene.read_text(encoding="utf-8"), assets=assets)
    data = mujoco.MjData(model)
    if any((mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, index) or "").startswith("piano/") for index in range(model.nu)):
        raise ValueError("Piano actuators are not allowed")
    ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in manifest["key_joints"]]
    if len(ids) != 88 or min(ids) < 0:
        raise ValueError("Expected the 88-key joint contract")
    key_geom = {}
    for index, joint in enumerate(ids):
        geoms = np.flatnonzero(model.geom_bodyid == model.jnt_bodyid[joint])
        if len(geoms) != 1:
            raise ValueError("Expected one contact geometry per key")
        key_geom[int(geoms[0])] = index + manifest["minimum_midi"]
    hands = {side: {index for index in range(model.ngeom) if "/" + prefix + "shadow_hand/" in (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, index) or "")} for side, prefix in (("left", "lh_"), ("right", "rh_"))}
    if not all(hands.values()):
        raise ValueError("The exported scene has no identifiable pair of Shadow Hands")
    owners = {geom: side for side, geoms in hands.items() for geom in geoms}
    period = manifest["control_timestep_seconds"]
    substeps = manifest["physics_steps_per_control"]
    if abs(substeps * model.opt.timestep - period) > 1e-9:
        raise ValueError("Inconsistent physics clock")
    previous = np.zeros(88, dtype=bool)
    recent, onsets, interhand_steps = {}, [], 0
    with np.load(run / "trajectory.npz", allow_pickle=False) as trace:
        controls = trace["controls"]
        if controls.ndim != 2 or controls.shape[1] != model.nu or not np.isfinite(controls).all():
            raise ValueError("Invalid controls")
        if (controls < model.actuator_ctrlrange[:, 0]).any() or (controls > model.actuator_ctrlrange[:, 1]).any():
            raise ValueError("Control exceeds actuator bounds")
        if not np.isfinite(trace["initial_qpos"]).all() or not np.isfinite(trace["initial_qvel"]).all():
            raise ValueError("Non-finite initial state")
        data.qpos[:] = trace["initial_qpos"]
        data.qvel[:] = trace["initial_qvel"]
        mujoco.mj_forward(model, data)
        for control in controls:
            data.ctrl[:] = control
            for _ in range(substeps):
                mujoco.mj_step(model, data)
                if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():
                    raise ValueError("Non-finite physical state")
                now = float(data.time)
                touching = {index + manifest["minimum_midi"]: set() for index in range(88)}
                between_hands = False
                for contact in data.contact:
                    a, b = int(contact.geom1), int(contact.geom2)
                    if a in owners and b in owners and owners[a] != owners[b]:
                        between_hands = True
                    for key, hand in ((a, b), (b, a)):
                        if key in key_geom and hand in owners:
                            pitch, side = key_geom[key], owners[hand]
                            touching[pitch].add(side)
                            recent[pitch, side] = now
                interhand_steps += int(between_hands)
                active = key_activation(data.qpos[model.jnt_qposadr[ids]], model.jnt_range[ids], manifest["activation_tolerance_radians"])
                for index in np.flatnonzero(active & ~previous):
                    pitch = int(index + manifest["minimum_midi"])
                    onsets.append({
                        "midi": pitch, "time_seconds": round(now, 6),
                        "contact_hands_at_onset": sorted(touching[pitch]),
                        "contact_hands_in_previous_control_period": sorted(side for side in hands if now - recent.get((pitch, side), -1) <= period + 1e-9),
                    })
                previous = active
    actual, unexpected, matches = compare_parts(onsets, source["expected_parts"])
    expected_hand = {pitch: side for side, pitches in source["expected_parts"].items() for pitch in pitches}
    attribution = all(event["contact_hands_in_previous_control_period"] == [expected_hand.get(event["midi"])] for event in onsets)
    report = {
        "mode": "physical_substep_actuator_replay_audit", "sample_hz": 1.0 / model.opt.timestep,
        "reference_plan_sha256": source["plan_sha256"], "measured_onsets": onsets,
        "measured_parts": actual, "unexpected_onsets": unexpected, "parts_match": matches,
        "onset_contact_hand_matches": attribution,
        "contact_attribution_history_seconds": period,
        "interhand_contact_physics_steps": interhand_steps,
        "joint_state_teleported_during_execution": False, "model_calls": 0,
    }
    write_json(run / "contact-audit.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    report = audit(args.run)
    print(json.dumps(report, indent=2))
    if not report["parts_match"] or not report["onset_contact_hand_matches"] or report["interhand_contact_physics_steps"]:
        parser.exit(1, "Physical note/contact audit failed\n")


if __name__ == "__main__":
    main()
