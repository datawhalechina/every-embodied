"""Reproduce every physical substep and identify which finger sounded each note."""

import argparse
import hashlib
import json
from pathlib import Path

from piano_context import write_json
from polyphonic_score import KeyCycles, evaluate, performance_passed


def audit(run):
    import mujoco
    import numpy as np

    run = Path(run).resolve()
    manifest = json.loads((run / "scene_manifest.json").read_text(encoding="utf-8"))
    source = json.loads((run / "report.json").read_text(encoding="utf-8"))
    plan_bytes = (run / "source.plan.json").read_bytes()
    if hashlib.sha256(plan_bytes).hexdigest() != source["plan_sha256"]:
        raise ValueError("Reference score changed after recording")
    score = json.loads(plan_bytes)
    if mujoco.__version__ != manifest["mujoco_version"]:
        raise ValueError("Use the recorded MuJoCo version")
    scene = (run / manifest["scene"]).resolve()
    if not scene.is_relative_to(run):
        raise ValueError("Scene must be inside the run")
    assets = {p.name: p.read_bytes() for p in scene.parent.iterdir() if p.is_file() and p != scene}
    model = mujoco.MjModel.from_xml_string(scene.read_text(encoding="utf-8"), assets=assets)
    data = mujoco.MjData(model)
    if any((mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i) or "").startswith("piano/") for i in range(model.nu)):
        raise ValueError("Piano actuators are not allowed")
    joints = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in manifest["key_joints"]]
    if len(joints) != 88 or min(joints) < 0:
        raise ValueError("Expected 88 keys")
    key_geoms = {int(g): k + 21 for k, joint in enumerate(joints) for g in np.flatnonzero(model.geom_bodyid == model.jnt_bodyid[joint])}
    codes = {"th": "thumb", "ff": "index", "mf": "middle", "rf": "ring", "lf": "little"}
    owners, hands = {}, {}
    for geom in range(model.ngeom):
        body = int(model.geom_bodyid[geom])
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body) or ""
        for side, prefix in (("left", "lh_"), ("right", "rh_")):
            if "/" + prefix + "shadow_hand/" in name:
                hands[geom] = side
                for code, finger in codes.items():
                    if name.split("/")[-1].startswith(prefix + code):
                        owners[geom] = side + "/" + finger
    recent, events = {}, []
    sensor = KeyCycles()
    between_hands = 0
    period, substeps = manifest["control_timestep_seconds"], manifest["physics_steps_per_control"]
    if abs(substeps * model.opt.timestep - period) > 1e-9:
        raise ValueError("Inconsistent physics clock")
    with np.load(run / "trajectory.npz", allow_pickle=False) as trace:
        controls = trace["controls"]
        offsets = trace["piano_offsets"] if "piano_offsets" in trace else None
        mover = None
        if offsets is not None:
            from piano_relocation import PianoTranslation
            if offsets.shape != (len(controls), 3) or not np.isfinite(offsets).all():
                raise ValueError("Invalid recorded piano offsets")
            mover = PianoTranslation(model)
        elif manifest.get("piano_translation"):
            raise ValueError("Missing piano translation trace")
        if controls.ndim != 2 or controls.shape[1] != model.nu or not np.isfinite(controls).all():
            raise ValueError("Invalid controls")
        if (controls < model.actuator_ctrlrange[:, 0]).any() or (controls > model.actuator_ctrlrange[:, 1]).any():
            raise ValueError("Out-of-range controls")
        data.qpos[:] = trace["initial_qpos"]
        data.qvel[:] = trace["initial_qvel"]
        mujoco.mj_forward(model, data)
        for step, control in enumerate(controls):
            if mover is not None:
                mover.apply(offsets[step])
            data.ctrl[:] = control
            for _ in range(substeps):
                mujoco.mj_step(model, data)
                if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():
                    raise ValueError("Non-finite replay state")
                now = float(data.time)
                touching, hand_contact = {}, False
                for contact in data.contact:
                    a, b = int(contact.geom1), int(contact.geom2)
                    hand_contact |= a in hands and b in hands and hands[a] != hands[b]
                    for key, finger in ((a, b), (b, a)):
                        if key in key_geoms and finger in owners:
                            pitch, owner = key_geoms[key], owners[finger]
                            touching.setdefault(pitch, set()).add(owner)
                            recent[pitch, owner] = now
                between_hands += int(hand_contact)
                ranges = model.jnt_range[joints]
                travel = (data.qpos[model.jnt_qposadr[joints]] - ranges[:, 0]) / (ranges[:, 1] - ranges[:, 0])
                for index, active in sensor.update(travel):
                    pitch = int(index + 21)
                    events.append({"type": "NoteOn" if active else "NoteOff", "midi": pitch,
                        "time_seconds": now, "contact_fingers": sorted(touching.get(pitch, set())),
                        "recent_contact_fingers": sorted(owner for (key, owner), when in recent.items() if key == pitch and now - when <= period + 1e-9)})
        final_error = float(np.max(np.abs(data.qpos-trace["qpos"][-1])))
        if mover is not None and final_error > 1e-6:
            raise ValueError(f"Relocation replay diverged: {final_error}")
    result = evaluate(score, events, song_mode=source.get("song_mode", False))
    result["replay_final_qpos_error"] = final_error
    result["piano_translation_replayed"] = mover is not None
    ownership = []
    for note in result["matched"]:
        event = next(e for e in events if e["type"] == "NoteOn" and e["midi"] == note["midi"] and abs(e["time_seconds"] - note["actual_on_seconds"]) < 1e-9)
        ownership.append({"midi": note["midi"], "time_seconds": event["time_seconds"],
            "expected_finger": note["hand"] + "/" + note["finger"], "measured_fingers": event["recent_contact_fingers"],
            "passed": event["recent_contact_fingers"] == [note["hand"] + "/" + note["finger"]]})
    result.update({"mode": "physical_substep_polyphonic_audit", "sample_hz": 1 / model.opt.timestep,
        "finger_matches": all(record["passed"] for record in ownership), "finger_ownership": ownership,
        "events": events, "interhand_contact_steps": between_hands, "contact_history_seconds": period,
        "key_sensor": {"press_travel_fraction": 0.90, "release_travel_fraction": 0.50, "uses_reference_score": False},
        "state_teleported_during_execution": False, "model_calls_during_replay": 0})
    write_json(run / "polyphonic-audit.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.run)
    print(json.dumps({k: result[k] for k in ("note_count_match", "timing_match", "chords_match", "finger_matches", "interhand_contact_steps")}, indent=2))
    if not performance_passed(result):
        parser.exit(1, "Polyphonic contact audit failed\n")
