"""Paired, simulation-only arm recovery tests. No finger policy or song playback.

Reuse an exported R1/Shadow scene. G0.5 sees fresh images at each decision;
all controllers share the same measured start, target, limits and safety stop.
"""

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from g05_piano_probe import sha256
from physical_piano import position_targets
from pianomime_galaxea import reduced_arm_problem, unique_suffix


MODES = ("ik_only", "g05_direct", "g05_assisted")
CASES = {
    "small": np.array([.04, -.04, .02, 0, 0, 0, 0, -.04, .04, -.02, 0, 0, 0, 0]),
    "medium": np.array([.08, -.08, .04, -.03, 0, 0, 0, -.08, .08, -.04, -.03, 0, 0, 0]),
    "asymmetric": np.array([.02, -.02, 0, 0, 0, 0, 0, -.10, .06, -.05, -.03, 0, 0, 0]),
    "aligned": np.zeros(14),
}


def piano_shift(value):
    shift = np.asarray(value, dtype=float)
    if (shift.shape != (3,) or not np.isfinite(shift).all()
            or shift[2] != 0 or np.linalg.norm(shift[:2]) > .08):
        raise ValueError("Piano translation must be horizontal and at most 8 cm")
    return shift


def hover_offset(value):
    if not np.isfinite(value) or not .02 <= value <= .12:
        raise ValueError("Hover height offset must be within [0.02, 0.12] meters")
    return np.array([0., 0., value])


def bounded_command(requested, previous, anchor, limits, dt=.01, speed=.6, radius=.35):
    arrays = [np.asarray(v, dtype=float) for v in (requested, previous, anchor, limits)]
    requested, previous, anchor, limits = arrays
    if (requested.ndim != 1 or previous.shape != requested.shape or anchor.shape != requested.shape
            or limits.shape != (len(requested), 2) or not all(np.isfinite(x).all() for x in arrays)
            or np.any(limits[:, 0] >= limits[:, 1])
            or not all(np.isfinite(x) and x > 0 for x in (dt, speed, radius))):
        raise ValueError("Invalid bounded controller inputs")
    low = np.maximum.reduce([limits[:, 0], anchor - radius, previous - speed * dt])
    high = np.minimum.reduce([limits[:, 1], anchor + radius, previous + speed * dt])
    if np.any(low > high):
        raise ValueError("Infeasible safety bounds")
    return np.clip(requested, low, high)


def sustained_success(history, window=20, position=.005, rotation=.05):
    if len(history) < window:
        return False
    return all(np.isfinite(row["position_m"]) and np.isfinite(row["rotation_rad"])
               and row["position_m"] <= position and row["rotation_rad"] <= rotation
               for row in history[-window:])


def summarize(rows):
    comparisons = []
    for case in sorted({r["case"] for r in rows}):
        by_mode = {r["mode"]: r for r in rows if r["case"] == case}
        base, assisted = by_mode.get("ik_only"), by_mode.get("g05_assisted")
        if not base or not assisted:
            continue
        if base["initial_qpos_sha256"] != assisted["initial_qpos_sha256"]:
            raise ValueError("Paired starts differ")
        comparable = (not base["safety_stop"] and not assisted["safety_stop"]
                      and abs(base["simulated_seconds"] - assisted["simulated_seconds"]) < .001)
        comparisons.append({"case": case,
            "equal_horizon_without_safety_stop": comparable,
            "assisted_minus_baseline_integrated_position_error": assisted["integrated_position_error"] - base["integrated_position_error"] if comparable else None,
            "baseline_success": base["success"], "assisted_success": assisted["success"],
            "assisted_has_lower_error": comparable and assisted["integrated_position_error"] < base["integrated_position_error"]})
    return comparisons


def run(args):
    import imageio.v2 as imageio
    import mink
    import mujoco
    import qpsolvers
    from dm_control import mujoco as dm_mujoco
    from g05_piano_client import PianoPolicy

    if (not np.isfinite(args.duration) or not .5 <= args.duration <= 10
            or not np.isfinite(args.model_weight) or not 0 < args.model_weight <= 10):
        raise ValueError("Invalid experiment duration or model weight")
    if any(mode != "ik_only" for mode in args.modes) and not all(
            (args.g05_python, args.g05_repo, args.g05_models, args.g05_checkpoint)):
        raise ValueError("G0.5 modes require all four model paths")
    shift = piano_shift(args.piano_shift)
    args.out.mkdir(parents=True, exist_ok=False)
    scene = args.reference / "scene/scene.xml"
    p = dm_mujoco.Physics.from_xml_path(str(scene))
    model = p.model.ptr
    if not np.isclose(model.opt.timestep, .002):
        raise ValueError("Expected the 500 Hz source scene")
    with np.load(args.reference / "trajectory.npz", allow_pickle=False) as data:
        initial = data["initial_qpos"].copy()
    if initial.shape != (model.nq,) or not np.isfinite(initial).all():
        raise ValueError("Invalid source initial state")
    names = lambda kind, count: [mujoco.mj_id2name(model, kind, i) for i in range(count)]
    joint_names = names(mujoco.mjtObj.mjOBJ_JOINT, model.njnt)
    actuator_names = names(mujoco.mjtObj.mjOBJ_ACTUATOR, model.nu)
    body_names = names(mujoco.mjtObj.mjOBJ_BODY, model.nbody)
    site_names = names(mujoco.mjtObj.mjOBJ_SITE, model.nsite)
    arm_j = [unique_suffix(joint_names, f"{side}_arm_joint{i}") for side in ("left", "right") for i in range(1, 8)]
    qids, vids = model.jnt_qposadr[arm_j], model.jnt_dofadr[arm_j]
    aids = [unique_suffix(actuator_names, f"{side}_arm_joint{i}_act") for side in ("left", "right") for i in range(1, 8)]
    limits = model.jnt_range[arm_j].copy()
    mounts = [unique_suffix(body_names, prefix + "forearm") for prefix in ("lh_", "rh_")]
    tip_ids = [[site_names.index(f"galaxea_r1pro/{prefix}shadow_hand/{finger}distal_site") for finger in ("th", "ff")]
               for prefix in ("lh_", "rh_")]
    piano_j = [i for i, name in enumerate(joint_names) if name and name.startswith("piano/")]
    key_q = model.jnt_qposadr[piano_j]
    robot_geoms = np.array([(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i) or "").startswith("galaxea_r1pro/")
                            for i in range(model.ngeom)])
    config = mink.Configuration(model)
    config.update(initial)
    frames = [mink.FrameTask(body_names[i], "body", position_cost=100, orientation_cost=10, lm_damping=.001) for i in mounts]
    goals = []
    lift = hover_offset(args.hover_height)
    for frame, idx in zip(frames, mounts):
        goal = mink.SE3.from_rotation_and_translation(mink.SO3.from_matrix(config.data.xmat[idx].reshape(3, 3)),
            config.data.xpos[idx] + lift)
        goals.append(goal)
        frame.set_target(goal)
    weights = np.zeros(model.nv)
    weights[vids] = args.model_weight
    model_prior = mink.PostureTask(model, cost=weights)

    def solve(qpos, tasks, bound=None):
        config.update(qpos)
        problem = mink.build_ik(config, tasks, .01, damping=1e-5, limits=[])
        P, q, lo, hi = reduced_arm_problem(problem, vids, qids, config.q, limits)
        if bound is not None:
            lo, hi = np.maximum(lo, -bound), np.minimum(hi, bound)
        dq = qpsolvers.solve_qp(P, q, lb=lo, ub=hi, solver="daqp")
        if dq is None or not np.isfinite(dq).all():
            raise RuntimeError("Arm IK failed")
        result = qpos.copy()
        result[qids] += dq
        return result

    hover = initial.copy()
    for _ in range(100):
        hover = solve(hover, frames, .02)

    # Keep the robot at the original hover pose; relocate the scene and its goals.
    # This is a static relocation test, not a dynamic pushed-piano simulation.
    if np.any(shift):
        piano_body = body_names.index("piano/")
        model.body_pos[piano_body] += shift
        for i in range(model.ngeom):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i) or ""
            if name.startswith("piano_leg_"):
                if model.geom_bodyid[i] != 0:
                    raise ValueError("Expected piano legs attached to the world")
                model.geom_pos[i] += shift
        goals = [mink.SE3.from_rotation_and_translation(goal.rotation(), goal.translation() + shift)
                 for goal in goals]
        for frame, goal in zip(frames, goals):
            frame.set_target(goal)

    def errors():
        position, rotation = [], []
        for idx, goal in zip(mounts, goals):
            position.append(float(np.linalg.norm(p.data.xpos[idx] - goal.translation())))
            rotation.append(float(np.arccos(np.clip((np.trace(goal.rotation().as_matrix().T @ p.data.xmat[idx].reshape(3, 3)) - 1) / 2, -1, 1))))
        return {"position_m": max(position), "rotation_rad": max(rotation)}

    def penetrating_contacts():
        return {tuple(sorted((int(c.geom1), int(c.geom2)))): float(c.dist)
                for c in p.data.contact[:p.data.ncon] if c.dist < -.001 and (robot_geoms[c.geom1] or robot_geoms[c.geom2])}

    def observe():
        obs = {}
        for name, size in (("head_rgb", (640, 360)), ("left_wrist_rgb", (640, 480)), ("right_wrist_rgb", (640, 480))):
            obs[name] = p.render(width=size[0], height=size[1], camera_id="galaxea_r1pro/" + name).transpose(2, 0, 1)
        for i, side in enumerate(("left", "right")):
            obs[side + "_arm"] = p.data.qpos[qids[i*7:i*7+7]].astype(np.float32).copy()
            a, b = tip_ids[i]
            aperture = np.linalg.norm(p.data.site_xpos[a] - p.data.site_xpos[b])
            obs[side + "_gripper"] = np.array([np.clip(aperture * 1000, 0, 100)], dtype=np.float32)
        return obs

    manifest = {"source_scene_sha256": sha256(scene), "source_trajectory_sha256": sha256(args.reference / "trajectory.npz"),
        "modes": args.modes, "cases": args.cases, "duration_seconds": args.duration,
        "model_posture_weight": args.model_weight, "goal_hover_height_offset_m": args.hover_height, "g05_config_mode": args.g05_config_mode,
        "piano_shift_world_m": shift.tolist(),
        "relocation": "static scene translation before rollout; original robot hover retained",
        "control_hz": 100, "physics_hz": 500, "model_query_interval_simulated_seconds": .4,
        "inference_pauses_simulation": True, "seed_schedule": "1000 + case_index * 100 + query_index",
        "success": "both mounts within 5 mm and 0.05 rad for last 0.2 seconds, no safety stop",
        "speed_limit_rad_s": .6, "anchor_radius_rad": .35, "actual_speed_is_measured_not_guaranteed": True,
        "goal_source": "known simulator geometry; exact target xyz and wxyz provided in model task instruction as well as to IK",
        "limitations": ["simulation only", "fixed base", "ideal gravity compensation", "Shadow gripper-state proxy",
            "no finger actions from G0.5", "no song evaluated", "no dynamic piano movement or visual pose estimation evaluated",
            "collision stop is not a certified safety controller", "Astra not invoked in this benchmark"]}
    (args.out / "protocol.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    rows, policy = [], None
    try:
        if any(mode != "ik_only" for mode in args.modes):
            policy = PianoPolicy(args, None, None)
        for case in args.cases:
            start = hover.copy()
            start[qids] = np.clip(start[qids] + CASES[case], limits[:, 0], limits[:, 1])
            for mode in args.modes:
                p.reset()
                p.data.qpos[:] = start
                p.data.qvel[:] = 0
                p.forward()
                baseline_contacts = penetrating_contacts()
                control = position_targets(model, p.data.ptr)
                anchor = start[qids].copy()
                initial_error = errors()
                trial = args.out / f"{case}-{mode}"
                trial.mkdir()
                imageio.imwrite(trial / "initial.png", p.render(width=960, height=720, camera_id="whole_robot"))
                writer = imageio.get_writer(str(trial / "recovery.mp4"), fps=20, codec="libx264") if args.video else None
                history, controls, states, calls, intended, times = [], [], [], [], [], []
                chunk, reason, max_speed, prior_distance = {}, None, 0., 0.
                began = time.monotonic()
                try:
                    for tick in range(round(args.duration * 100)):
                        if mode != "ik_only" and tick % 40 == 0:
                            target_text = "; ".join(f"{side} xyz={np.round(goal.translation(), 3).tolist()} wxyz={np.round(goal.rotation().wxyz, 3).tolist()}"
                                for side, goal in zip(("left", "right"), goals))
                            chunk = policy.infer_observation(float(p.data.time), observe(),
                                "Move both hand mounts to these piano hover targets, in world meters and wxyz quaternions: " + target_text + ". Do not press keys; keep fingers still.",
                                "Recover a comfortable two-handed piano preparation posture. The fingers are held still by another controller; output the arm actions only.",
                                chunk=True, seed=1000 + list(CASES).index(case) * 100 + tick // 40)
                            calls.append(policy.calls[-1]["call_index"])
                        measured = p.data.qpos.copy()
                        proposal = measured.copy()
                        row = int((tick % 40) * .01 * 15)
                        for i, side in enumerate(("left", "right")):
                            if side in chunk:
                                proposal[qids[i*7:i*7+7]] = chunk[side][min(row, len(chunk[side])-1)]
                        tasks = list(frames)
                        if mode == "g05_assisted":
                            model_prior.set_target(proposal)
                            tasks.append(model_prior)
                        target = proposal[qids] if mode == "g05_direct" else solve(measured, tasks, .02)[qids]
                        command = bounded_command(target, control[aids], anchor, limits)
                        if mode == "g05_assisted":
                            counterfactual = bounded_command(solve(measured, frames, .02)[qids], control[aids], anchor, limits)
                            prior_distance += float(np.linalg.norm(command - counterfactual)) * .01
                        control[aids] = command
                        # All non-arm controls remain identical to their initial hold values.
                        for _ in range(5):
                            p.set_control(control)
                            p.step()
                            if not np.isfinite(p.data.qpos).all():
                                raise RuntimeError("Non-finite physics")
                            max_speed = max(max_speed, float(np.max(np.abs(p.data.qvel[vids]))))
                            contacts = penetrating_contacts()
                            if any(depth < baseline_contacts.get(pair, 0.) - .001 for pair, depth in contacts.items()):
                                reason = "new_or_deeper_robot_contact"
                            if len(key_q) and np.any(p.data.qpos[key_q] > .8 * model.jnt_range[piano_j, 1]):
                                reason = "unintended_key_press"
                            if reason:
                                break
                        current = {"time_seconds": float(p.data.time), **errors()}
                        history.append(current)
                        controls.append(control.copy())
                        states.append(p.data.qpos.copy())
                        intended.append(proposal[qids].copy())
                        times.append(float(p.data.time))
                        if writer and tick % 5 == 0:
                            writer.append_data(p.render(width=960, height=720, camera_id="whole_robot"))
                        if reason:
                            break
                    success = reason is None and sustained_success(history)
                    elapsed = time.monotonic() - began
                    dt = np.diff(np.r_[0, times])
                    report = {"case": case, "mode": mode, "initial_qpos_sha256": hashlib.sha256(start.tobytes()).hexdigest(),
                        "piano_shift_world_m": shift.tolist(),
                        "initial_error": initial_error, "final_error": history[-1], "success": success,
                        "safety_stop": reason, "model_calls": calls, "wall_seconds": elapsed,
                        "simulated_seconds": times[-1], "max_actual_arm_speed_rad_s": max_speed,
                        "first_sustained_success_seconds": next((history[i-1]["time_seconds"] for i in range(20, len(history)+1) if sustained_success(history[:i])), None),
                        "initial_penetrating_contacts": len(baseline_contacts),
                        "integrated_position_error": float(sum(r["position_m"] * t for r, t in zip(history, dt))),
                        "mean_position_error_m": float(np.mean([r["position_m"] for r in history])),
                        "max_actual_arm_change_rad": float(np.max(np.abs(np.asarray(states)[:, qids] - start[qids]))),
                        "model_induced_command_distance_rad_seconds": prior_distance,
                        "qpos_teleported_during_execution": False, "song_success_claimed": False}
                    (trial / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
                    (trial / "errors.json").write_text(json.dumps(history), encoding="utf-8")
                    np.savez_compressed(trial / "trajectory.npz", initial_qpos=start, controls=controls, qpos=states,
                        proposed_arm_targets=intended, times=times)
                    imageio.imwrite(trial / "final.png", p.render(width=960, height=720, camera_id="whole_robot"))
                    rows.append(report)
                    (args.out / "results.json").write_text(json.dumps({"trials": rows, "paired": summarize(rows)}, indent=2), encoding="utf-8")
                    print(json.dumps({k: report[k] for k in ("case", "mode", "success", "final_error", "safety_stop", "model_induced_command_distance_rad_seconds")}), flush=True)
                finally:
                    if writer:
                        writer.close()
    finally:
        if policy:
            policy.close()
        p.free()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--modes", nargs="+", choices=MODES, default=list(MODES))
    parser.add_argument("--cases", nargs="+", choices=tuple(CASES), default=["small", "medium", "asymmetric"])
    parser.add_argument("--piano-shift", type=float, nargs=3, default=[0., 0., 0.],
                        metavar=("X", "Y", "Z"), help="Static horizontal piano translation, world meters, <=8 cm")
    parser.add_argument("--duration", type=float, default=4.)
    parser.add_argument("--model-weight", type=float, default=2.)
    parser.add_argument("--hover-height", type=float, default=.08)
    parser.add_argument("--g05-config-mode", choices=("legacy-ar", "checkpoint"), default="checkpoint")
    parser.add_argument("--video", action="store_true")
    for name in ("python", "repo", "models", "checkpoint"):
        parser.add_argument("--g05-" + name, type=Path)
    run(parser.parse_args())
