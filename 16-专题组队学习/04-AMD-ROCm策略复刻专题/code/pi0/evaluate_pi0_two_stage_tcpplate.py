#!/usr/bin/env python3
"""Run the staged Pi0 evaluator with 22-D target-relative state support.

This is a thin diagnostic wrapper around ``evaluate_pi0_two_stage_eef_abs.py``.
The base evaluator already implements oracle/policy prefix, finisher handoff,
dataset-schedule phase playback, and physical-success logging.  This wrapper
adds one state layout used by the Pi0 finisher experiments:

    joint6 + timestamp + phase_index_norm + phase_onehot11 + tcp_to_plate3

The closed-loop value of ``tcp_to_plate`` is computed from MuJoCo every control
tick as ``tcp_link - body_obj_plate_11``.  Dataset schedules still provide only
the timestamp/phase portion; current joint state and current tcp_to_plate come
from the live simulator.

For prefix-policy ablations, ``--tcpplate-prefix-target-state`` keeps the same
22-D layout but appends ``tcp_link - env.obj_target`` while the prefix policy is
running.  Use it together with a 22-D prefix dataset built by
``build_lerobot_state_phase_tcptarget.py``.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np


def pop_wrapper_args(argv: list[str]) -> tuple[list[str], argparse.Namespace]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--tcpplate-base-evaluator",
        type=Path,
        default=Path(__file__).with_name("evaluate_pi0_two_stage_eef_abs.py"),
        help="Path to the original staged Pi0 evaluator.",
    )
    parser.add_argument(
        "--tcpplate-schedule-start-phase",
        default=None,
        help="Optional phase name where dataset_schedule playback should start.",
    )
    parser.add_argument(
        "--tcpplate-force-schedule-episode",
        type=int,
        default=None,
        help="Use one finisher dataset episode as the phase schedule for every seed.",
    )
    parser.add_argument(
        "--tcpplate-prefix-target-state",
        action="store_true",
        help=(
            "For 22-D non-dataset-schedule policy state, append live "
            "tcp_link - env.obj_target instead of tcp_link - plate. This is "
            "intended for policy-prefix experiments; scheduled finishers keep "
            "tcp_to_plate."
        ),
    )
    parser.add_argument(
        "--tcpplate-gripper-head-path",
        type=Path,
        default=None,
        help=(
            "Optional learned logistic gripper head (.npz). When provided together "
            "with --finisher-scripted-gripper, the head replaces the phase rule."
        ),
    )
    parser.add_argument(
        "--tcpplate-gripper-head-threshold",
        type=float,
        default=0.5,
        help="Closed-gripper probability threshold for --tcpplate-gripper-head-path.",
    )
    parser.add_argument(
        "--tcpplate-adaptive-move-preplace",
        action="store_true",
        help=(
            "For dataset_schedule, hold the last move_preplace row until live "
            "TCP-to-plate XY is close enough, then continue the schedule tail."
        ),
    )
    parser.add_argument(
        "--tcpplate-adaptive-move-preplace-xy-threshold",
        type=float,
        default=0.07,
        help="Live TCP-to-plate XY threshold in meters for leaving move_preplace.",
    )
    parser.add_argument(
        "--tcpplate-adaptive-move-preplace-min-steps",
        type=int,
        default=20,
        help="Minimum finisher steps before adaptive move_preplace transition.",
    )
    parser.add_argument(
        "--tcpplate-adaptive-move-preplace-max-steps",
        type=int,
        default=180,
        help="Safety fallback: leave move_preplace after this many finisher steps.",
    )
    parser.add_argument(
        "--tcpplate-transition-head-path",
        type=Path,
        default=None,
        help=(
            "Optional learned logistic transition head (.npz). When set, it "
            "decides when the held move_preplace row may enter the lower tail."
        ),
    )
    parser.add_argument(
        "--tcpplate-transition-head-threshold",
        type=float,
        default=0.5,
        help="Ready-to-lower probability threshold for --tcpplate-transition-head-path.",
    )
    parser.add_argument(
        "--tcpplate-contact-primitive",
        choices=["off", "clamp_xy", "guided_contact_lift"],
        default="off",
        help=(
            "Optional prefix-stage contact scaffold. clamp_xy only prevents "
            "descending below the target grasp floor; guided_contact_lift "
            "executes a target-relative grasp/close/lift primitive after the "
            "configured start step."
        ),
    )
    parser.add_argument(
        "--tcpplate-contact-start-step",
        type=int,
        default=40,
        help="Earliest prefix step at which guided_contact_lift may take over.",
    )
    parser.add_argument(
        "--tcpplate-contact-trigger-xy",
        type=float,
        default=0.08,
        help="TCP-to-target XY threshold that may trigger contact handling.",
    )
    parser.add_argument(
        "--tcpplate-contact-xy-tol",
        type=float,
        default=0.025,
        help="XY tolerance for target-relative contact primitive phase changes.",
    )
    parser.add_argument(
        "--tcpplate-contact-z-tol",
        type=float,
        default=0.012,
        help="Z tolerance for target-relative contact primitive phase changes.",
    )
    parser.add_argument(
        "--tcpplate-contact-close-hold",
        type=int,
        default=25,
        help="Number of control actions to hold a closed gripper before lifting.",
    )
    parser.add_argument(
        "--tcpplate-contact-lift-hold",
        type=int,
        default=70,
        help="Safety fallback number of lift actions before switching to hold_lift.",
    )
    parser.add_argument(
        "--tcpplate-contact-pregrasp-hold",
        type=int,
        default=999,
        help="Safety fallback number of pregrasp actions before descending in strict transition-head tests.",
    )
    parser.add_argument(
        "--tcpplate-contact-descend-hold",
        type=int,
        default=999,
        help="Safety fallback number of descend actions before closing in strict transition-head tests.",
    )
    parser.add_argument(
        "--tcpplate-contact-target-z-offset",
        type=float,
        default=None,
        help="Target-relative TCP Z floor for closing; defaults to --grasp-offset-z.",
    )
    parser.add_argument(
        "--tcpplate-contact-head-path",
        type=Path,
        default=None,
        help=(
            "Optional learned softmax phase head (.npz) for guided_contact_lift. "
            "When set, the head replaces the hand-written phase transitions while "
            "the target-relative primitive still generates safe motion targets."
        ),
    )
    parser.add_argument(
        "--tcpplate-contact-head-min-confidence",
        type=float,
        default=0.0,
        help="Ignore contact phase head predictions below this confidence.",
    )
    parser.add_argument(
        "--tcpplate-contact-transition-head-path",
        type=Path,
        default=None,
        help=(
            "Optional learned logistic transition head (.npz) for guided_contact_lift. "
            "It predicts contact phase transition readiness, replacing fixed "
            "contact counters when confident."
        ),
    )
    parser.add_argument(
        "--tcpplate-contact-transition-head-threshold",
        type=float,
        default=0.5,
        help="Ready probability threshold for --tcpplate-contact-transition-head-path.",
    )
    parser.add_argument(
        "--tcpplate-contact-transition-head-strict",
        action="store_true",
        help=(
            "When the contact transition head contains a task for the current "
            "phase transition, do not use the hand-written geometric transition "
            "except for the configured safety timeout."
        ),
    )
    parser.add_argument(
        "--tcpplate-contact-descend-floor-guard",
        action="store_true",
        help="Do not allow a learned descend_to_close transition until TCP is near the grasp floor.",
    )
    parser.add_argument(
        "--tcpplate-contact-descend-floor-guard-tol",
        type=float,
        default=0.012,
        help="Z tolerance in meters for --tcpplate-contact-descend-floor-guard.",
    )
    known, remaining = parser.parse_known_args(argv[1:])
    return [argv[0], *remaining], known


def load_module(path: Path):
    spec = importlib.util.spec_from_file_location("evaluate_pi0_two_stage_eef_abs_base", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import evaluator from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def tcp_to_plate(env) -> np.ndarray:
    tcp = np.asarray(env.env.get_p_body("tcp_link")[:3], dtype=np.float32)
    plate = np.asarray(env.env.get_p_body("body_obj_plate_11")[:3], dtype=np.float32)
    return (tcp - plate).astype(np.float32)


def tcp_to_target(env) -> np.ndarray:
    target_body = getattr(env, "obj_target", None)
    if not target_body:
        raise ValueError("Cannot compute tcp_to_target: env.obj_target is not set")
    tcp = np.asarray(env.env.get_p_body("tcp_link")[:3], dtype=np.float32)
    target = np.asarray(env.env.get_p_body(target_body)[:3], dtype=np.float32)
    return (tcp - target).astype(np.float32)


def live_tcp(env) -> np.ndarray:
    return np.asarray(env.env.get_p_body("tcp_link")[:3], dtype=np.float32)


def live_target(env) -> np.ndarray:
    target_body = getattr(env, "obj_target", None)
    if not target_body:
        raise ValueError("Cannot read target pose: env.obj_target is not set")
    return np.asarray(env.env.get_p_body(target_body)[:3], dtype=np.float32)


def contact_targets(env, args, contact_config: dict) -> dict[str, np.ndarray]:
    target = live_target(env)
    z_offset = contact_config.get("target_z_offset")
    if z_offset is None:
        z_offset = float(getattr(args, "grasp_offset_z", 0.0))
    grasp = target + np.asarray(
        [
            float(getattr(args, "grasp_offset_x", 0.0)),
            float(getattr(args, "grasp_offset_y", 0.0)),
            float(z_offset),
        ],
        dtype=np.float32,
    )
    pregrasp = grasp.copy()
    pregrasp[2] = target[2] + float(getattr(args, "pregrasp_z", 0.135))
    lift = grasp.copy()
    lift[2] = target[2] + float(getattr(args, "lift_z", 0.145))
    return {
        "target": target,
        "grasp": grasp,
        "pregrasp": pregrasp,
        "lift": lift,
        "floor_z": np.asarray([grasp[2]], dtype=np.float32),
    }


def clipped_xyz_delta(current: np.ndarray, target: np.ndarray, max_step: float) -> np.ndarray:
    return np.clip(
        np.asarray(target, dtype=np.float32).reshape(3) - np.asarray(current, dtype=np.float32).reshape(3),
        -float(max_step),
        float(max_step),
    ).astype(np.float32)


CONTACT_PHASES = ["pregrasp", "descend", "close", "lift", "hold_lift"]
CONTACT_PHASE_ORDER = {name: idx for idx, name in enumerate(CONTACT_PHASES)}


DEFAULT_CONTACT_FEATURE_NAMES = [
    "tcp_to_target_x",
    "tcp_to_target_y",
    "tcp_to_target_z",
    "tcp_to_target_xy",
    "abs_tcp_to_target_z",
    "local_step_norm",
]


def contact_feature_map(env, stage_step: int, step_scale: float) -> dict[str, float]:
    rel = tcp_to_target(env)
    xy = float(np.linalg.norm(rel[:2]))
    return {
        "tcp_to_target_x": float(rel[0]),
        "tcp_to_target_y": float(rel[1]),
        "tcp_to_target_z": float(rel[2]),
        "tcp_to_target_xy": xy,
        "abs_tcp_to_target_z": abs(float(rel[2])),
        "local_step_norm": float(stage_step) / max(float(step_scale), 1.0),
    }


def contact_features(env, stage_step: int, step_scale: float, feature_names: list[str]) -> np.ndarray:
    values = contact_feature_map(env, stage_step, step_scale)
    return np.asarray([values[name] for name in feature_names], dtype=np.float32)


def load_contact_phase_head(head_path: Path | None, min_confidence: float):
    if head_path is None:
        return None
    data = np.load(head_path, allow_pickle=False)
    mean = np.asarray(data["state_mean"], dtype=np.float32).reshape(-1)
    std = np.asarray(data["state_std"], dtype=np.float32).reshape(-1)
    weight = np.asarray(data["weight"], dtype=np.float32)
    bias = np.asarray(data["bias"], dtype=np.float32).reshape(-1)
    if "feature_names" in data.files:
        feature_names = [str(name) for name in np.asarray(data["feature_names"]).reshape(-1).tolist()]
    else:
        feature_names = list(DEFAULT_CONTACT_FEATURE_NAMES)
    if "class_names" in data.files:
        class_names = [str(name) for name in np.asarray(data["class_names"]).reshape(-1).tolist()]
    else:
        class_names = list(CONTACT_PHASES)
    step_scale = float(np.asarray(data["step_scale"]).reshape(())) if "step_scale" in data.files else 180.0
    if weight.ndim != 2:
        raise ValueError(f"Contact head weight must be 2-D, got {weight.shape}")
    if mean.shape != std.shape or mean.shape[0] != weight.shape[0]:
        raise ValueError(
            f"Bad contact head feature shapes: mean={mean.shape}, std={std.shape}, weight={weight.shape}"
        )
    if bias.shape[0] != weight.shape[1] or len(class_names) != weight.shape[1]:
        raise ValueError(
            f"Bad contact head class shapes: bias={bias.shape}, weight={weight.shape}, classes={len(class_names)}"
        )
    if len(feature_names) != mean.shape[0]:
        raise ValueError(
            f"Contact feature count {len(feature_names)} does not match head dim {mean.shape[0]}"
        )
    std = np.where(std < 1e-6, 1.0, std).astype(np.float32)

    def predict_phase(env, stage_step: int) -> dict:
        x = contact_features(env, int(stage_step), step_scale, feature_names)
        logits = np.dot((x - mean) / std, weight) + bias
        logits = logits.astype(np.float64)
        logits -= float(np.max(logits))
        probs = np.exp(logits)
        probs /= max(float(np.sum(probs)), 1e-12)
        index = int(np.argmax(probs))
        return {
            "phase": class_names[index],
            "confidence": float(probs[index]),
            "probs": {class_names[i]: float(probs[i]) for i in range(len(class_names))},
            "features": {feature_names[i]: float(x[i]) for i in range(len(feature_names))},
        }

    return {
        "path": str(head_path),
        "min_confidence": float(min_confidence),
        "feature_names": feature_names,
        "class_names": class_names,
        "step_scale": float(step_scale),
        "predict_phase": predict_phase,
    }


DEFAULT_CONTACT_TRANSITION_FEATURE_NAMES = [
    "tcp_to_target_xy",
    "abs_tcp_to_target_z",
    "local_step_norm",
    "phase_elapsed_norm",
    "pregrasp_count_norm",
    "descend_count_norm",
    "close_count_norm",
    "lift_count_norm",
]


def contact_transition_feature_map(
    env,
    stage_step: int,
    contact_state: dict,
    step_scale: float,
    count_scale: float,
    task_name: str | None = None,
) -> dict[str, float]:
    values = contact_feature_map(env, int(stage_step), float(step_scale))
    rel = np.asarray(
        [
            values["tcp_to_target_x"],
            values["tcp_to_target_y"],
            values["tcp_to_target_z"],
        ],
        dtype=np.float32,
    )
    grasp_offset_x = float(contact_state.get("grasp_offset_x", 0.006))
    grasp_offset_y = float(contact_state.get("grasp_offset_y", 0.060))
    target_z_offset = float(contact_state.get("target_z_offset", 0.0))
    pregrasp_z = float(contact_state.get("pregrasp_z", 0.135))
    grasp_xy = float(
        np.linalg.norm(rel[:2] - np.asarray([grasp_offset_x, grasp_offset_y], dtype=np.float32))
    )
    pregrasp_z_err = float(rel[2]) - pregrasp_z
    floor_z = float(rel[2]) - target_z_offset
    values.update(
        {
            "tcp_to_grasp_xy": grasp_xy,
            "tcp_to_pregrasp_z": pregrasp_z_err,
            "abs_tcp_to_pregrasp_z": abs(pregrasp_z_err),
            "tcp_to_floor_z": floor_z,
            "abs_tcp_to_floor_z": abs(floor_z),
        }
    )
    raw_pregrasp_count = int(contact_state.get("pregrasp_count", 0))
    raw_descend_count = int(contact_state.get("descend_count", 0))
    raw_close_count = int(contact_state.get("close_count", 0))
    raw_lift_count = int(contact_state.get("lift_count", 0))
    if task_name == "pregrasp_to_descend":
        pregrasp_count = raw_pregrasp_count
        descend_count = 0
        close_count = 0
        lift_count = 0
        phase_elapsed = raw_pregrasp_count
    elif task_name == "descend_to_close":
        pregrasp_count = 0
        descend_count = raw_descend_count
        close_count = 0
        lift_count = 0
        phase_elapsed = raw_descend_count
    elif task_name == "close_to_lift":
        pregrasp_count = 0
        descend_count = 0
        close_count = raw_close_count
        lift_count = 0
        phase_elapsed = raw_close_count
    elif task_name == "lift_to_hold":
        pregrasp_count = 0
        descend_count = 0
        close_count = 0
        lift_count = raw_lift_count
        phase_elapsed = raw_lift_count
    else:
        phase = str(contact_state.get("phase", "idle"))
        pregrasp_count = raw_pregrasp_count
        descend_count = raw_descend_count
        close_count = raw_close_count
        lift_count = raw_lift_count
        phase_elapsed_by_phase = {
            "pregrasp": raw_pregrasp_count,
            "descend": raw_descend_count,
            "close": raw_close_count,
            "lift": raw_lift_count,
        }
        phase_elapsed = phase_elapsed_by_phase.get(phase, 0)
    scale = max(float(count_scale), 1.0)
    values.update(
        {
            "phase_elapsed_norm": float(phase_elapsed) / scale,
            "pregrasp_count_norm": float(pregrasp_count) / scale,
            "descend_count_norm": float(descend_count) / scale,
            "close_count_norm": float(close_count) / scale,
            "lift_count_norm": float(lift_count) / scale,
        }
    )
    return values


def contact_transition_features(
    env,
    stage_step: int,
    contact_state: dict,
    step_scale: float,
    count_scale: float,
    feature_names: list[str],
    task_name: str | None = None,
) -> np.ndarray:
    values = contact_transition_feature_map(
        env,
        int(stage_step),
        contact_state,
        step_scale,
        count_scale,
        task_name=task_name,
    )
    return np.asarray([values[name] for name in feature_names], dtype=np.float32)


def load_contact_transition_head(head_path: Path | None, threshold: float):
    if head_path is None:
        return None
    data = np.load(head_path, allow_pickle=False)
    mean = np.asarray(data["state_mean"], dtype=np.float32).reshape(-1)
    std = np.asarray(data["state_std"], dtype=np.float32).reshape(-1)
    weight = np.asarray(data["weight"], dtype=np.float32)
    bias = np.asarray(data["bias"], dtype=np.float32).reshape(-1)
    if "feature_names" in data.files:
        feature_names = [str(name) for name in np.asarray(data["feature_names"]).reshape(-1).tolist()]
    else:
        feature_names = list(DEFAULT_CONTACT_TRANSITION_FEATURE_NAMES)
    if "task_names" in data.files:
        task_names = [str(name) for name in np.asarray(data["task_names"]).reshape(-1).tolist()]
    else:
        task_names = ["close_to_lift", "lift_to_hold"]
    step_scale = float(np.asarray(data["step_scale"]).reshape(())) if "step_scale" in data.files else 180.0
    count_scale = float(np.asarray(data["count_scale"]).reshape(())) if "count_scale" in data.files else 64.0
    if weight.ndim == 1:
        weight = weight.reshape(1, -1)
    if weight.ndim != 2:
        raise ValueError(f"Contact transition weight must be 2-D, got {weight.shape}")
    if mean.shape != std.shape or mean.shape[0] != weight.shape[1]:
        raise ValueError(
            f"Bad contact transition feature shapes: mean={mean.shape}, std={std.shape}, weight={weight.shape}"
        )
    if bias.shape[0] != weight.shape[0] or len(task_names) != weight.shape[0]:
        raise ValueError(
            f"Bad contact transition task shapes: bias={bias.shape}, weight={weight.shape}, tasks={len(task_names)}"
        )
    if len(feature_names) != mean.shape[0]:
        raise ValueError(
            f"Contact transition feature count {len(feature_names)} does not match head dim {mean.shape[0]}"
        )
    task_to_index = {name: idx for idx, name in enumerate(task_names)}
    std = np.where(std < 1e-6, 1.0, std).astype(np.float32)

    def predict_ready_probability(env, stage_step: int, contact_state: dict, task_name: str) -> dict:
        if task_name not in task_to_index:
            raise ValueError(f"Unknown contact transition task {task_name!r}; choices={task_names}")
        x = contact_transition_features(
            env,
            int(stage_step),
            contact_state,
            float(step_scale),
            float(count_scale),
            feature_names,
            task_name=task_name,
        )
        task_index = int(task_to_index[task_name])
        logit = float(np.dot((x - mean) / std, weight[task_index]) + bias[task_index])
        prob = float(1.0 / (1.0 + np.exp(-np.clip(logit, -60.0, 60.0))))
        return {
            "task": task_name,
            "probability": prob,
            "features": {feature_names[i]: float(x[i]) for i in range(len(feature_names))},
        }

    return {
        "path": str(head_path),
        "threshold": float(threshold),
        "feature_names": feature_names,
        "task_names": task_names,
        "step_scale": float(step_scale),
        "count_scale": float(count_scale),
        "predict_ready_probability": predict_ready_probability,
    }


DEFAULT_TRANSITION_FEATURE_NAMES = [
    "tcp_to_plate_x",
    "tcp_to_plate_y",
    "tcp_to_plate_z",
    "tcp_to_plate_xy",
    "abs_tcp_to_plate_z",
    "local_step_norm",
]


def transition_feature_map(rel: np.ndarray, action_step: int, step_scale: float) -> dict[str, float]:
    rel = np.asarray(rel, dtype=np.float32).reshape(-1)
    if rel.shape[0] < 3:
        raise ValueError(f"Expected tcp_to_plate vector with 3 values, got {rel.shape}")
    xy = float(np.linalg.norm(rel[:2]))
    return {
        "tcp_to_plate_x": float(rel[0]),
        "tcp_to_plate_y": float(rel[1]),
        "tcp_to_plate_z": float(rel[2]),
        "tcp_to_plate_xy": xy,
        "abs_tcp_to_plate_z": abs(float(rel[2])),
        "local_step_norm": float(action_step) / max(float(step_scale), 1.0),
    }


def transition_features(
    rel: np.ndarray,
    action_step: int,
    step_scale: float,
    feature_names: list[str],
) -> np.ndarray:
    values = transition_feature_map(rel, action_step, step_scale)
    return np.asarray([values[name] for name in feature_names], dtype=np.float32)


def load_transition_head(head_path: Path | None, threshold: float):
    if head_path is None:
        return None
    data = np.load(head_path, allow_pickle=False)
    mean = np.asarray(data["state_mean"], dtype=np.float32).reshape(-1)
    std = np.asarray(data["state_std"], dtype=np.float32).reshape(-1)
    weight = np.asarray(data["weight"], dtype=np.float32).reshape(-1)
    bias = float(np.asarray(data["bias"]).reshape(()))
    step_scale = float(np.asarray(data["step_scale"]).reshape(())) if "step_scale" in data.files else 180.0
    if "feature_names" in data.files:
        feature_names = [str(name) for name in np.asarray(data["feature_names"]).reshape(-1).tolist()]
    else:
        feature_names = list(DEFAULT_TRANSITION_FEATURE_NAMES)
    if mean.shape != std.shape or mean.shape != weight.shape:
        raise ValueError(
            f"Bad transition head shapes: mean={mean.shape}, std={std.shape}, weight={weight.shape}"
        )
    if len(feature_names) != mean.shape[0]:
        raise ValueError(
            f"Transition feature count {len(feature_names)} does not match head dim {mean.shape[0]}"
        )
    std = np.where(std < 1e-6, 1.0, std).astype(np.float32)

    def predict_ready_probability(rel: np.ndarray, action_step: int) -> float:
        x = transition_features(rel, int(action_step), step_scale, feature_names)
        if x.shape[0] != mean.shape[0]:
            raise ValueError(f"Transition feature dim {x.shape[0]} does not match head dim {mean.shape[0]}")
        logit = float(np.dot((x - mean) / std, weight) + bias)
        return float(1.0 / (1.0 + np.exp(-np.clip(logit, -60.0, 60.0))))

    return {
        "path": str(head_path),
        "threshold": float(threshold),
        "step_scale": float(step_scale),
        "feature_names": feature_names,
        "predict_ready_probability": predict_ready_probability,
    }


def install_tcpplate_patch(
    module,
    adaptive_config=None,
    prefix_target_state: bool = False,
    contact_config: dict | None = None,
):
    from audit_smolvla_physical import (
        expected_state_dim,
        phase_feature,
        phase_index_from_state,
        set_current_phase,
    )

    original_observation_state = module.observation_state_for_policy
    adaptive_state = {
        "schedule_id": None,
        "release_step": None,
        "lower_start": None,
    }
    contact_state = {
        "active": False,
        "phase": "idle",
        "pregrasp_count": 0,
        "descend_count": 0,
        "close_count": 0,
        "lift_count": 0,
    }

    def reset_contact_state() -> None:
        contact_state.update(
            {
                "active": False,
                "phase": "idle",
                "pregrasp_count": 0,
                "descend_count": 0,
                "close_count": 0,
                "lift_count": 0,
            }
        )

    def apply_contact_primitive(env, env_action: np.ndarray, info: dict, stage_step: int, args) -> tuple[np.ndarray, dict]:
        if not contact_config or str(contact_config.get("mode", "off")) == "off":
            return env_action, info
        if int(stage_step) == 0:
            reset_contact_state()

        mode = str(contact_config.get("mode", "off"))
        current = live_tcp(env)
        targets = contact_targets(env, args, contact_config)
        grasp = targets["grasp"]
        pregrasp = targets["pregrasp"]
        lift = targets["lift"]
        floor_z = float(targets["floor_z"][0])
        contact_state["grasp_offset_x"] = float(getattr(args, "grasp_offset_x", 0.006))
        contact_state["grasp_offset_y"] = float(getattr(args, "grasp_offset_y", 0.060))
        contact_state["target_z_offset"] = float(
            contact_config.get("target_z_offset")
            if contact_config.get("target_z_offset") is not None
            else getattr(args, "grasp_offset_z", 0.0)
        )
        contact_state["pregrasp_z"] = float(getattr(args, "pregrasp_z", 0.135))
        xy = float(np.linalg.norm((current - grasp)[:2]))
        z_err = float(current[2] - floor_z)
        predicted_next_z = float(current[2] + float(env_action[2]))
        max_step = float(contact_config.get("max_step") or getattr(args, "eef_abs_max_step", 0.004))
        xy_tol = float(contact_config.get("xy_tol", 0.025))
        z_tol = float(contact_config.get("z_tol", 0.012))

        primitive_info = {
            "mode": mode,
            "phase": contact_state["phase"],
            "active": bool(contact_state["active"]),
            "stage_step": int(stage_step),
            "tcp_to_grasp_xy": round(xy, 6),
            "tcp_to_floor_z": round(z_err, 6),
            "floor_z": round(floor_z, 6),
        }

        if mode == "clamp_xy":
            if predicted_next_z < floor_z:
                env_action = np.asarray(env_action, dtype=np.float32).copy()
                env_action[2] = max(float(env_action[2]), floor_z - float(current[2]))
                primitive_info["clamped_z"] = True
            if xy <= float(contact_config.get("trigger_xy", 0.08)) and current[2] <= floor_z + z_tol:
                env_action[6] = 1.0
                primitive_info["close_on_floor"] = True
            info["tcpplate_contact_primitive"] = primitive_info
            info["env_action"] = module.round_list(env_action, 5)
            return env_action, info

        if mode != "guided_contact_lift":
            raise ValueError(f"Unknown contact primitive mode: {mode}")

        should_start = int(stage_step) >= int(contact_config.get("start_step", 40))
        should_start = should_start or xy <= float(contact_config.get("trigger_xy", 0.08))
        if should_start and not contact_state["active"]:
            contact_state.update(
                {
                    "active": True,
                    "phase": "pregrasp",
                    "pregrasp_count": 0,
                    "descend_count": 0,
                    "close_count": 0,
                    "lift_count": 0,
                }
            )

        if not contact_state["active"]:
            info["tcpplate_contact_primitive"] = primitive_info
            return env_action, info

        contact_head = contact_config.get("contact_head")
        contact_transition_head = contact_config.get("contact_transition_head")
        head_prediction = None
        transition_prediction = None
        transition_reason = None
        transition_blocked_reason = None
        if contact_head is not None:
            head_prediction = contact_head["predict_phase"](env, int(stage_step))
            predicted_phase = str(head_prediction["phase"])
            confidence = float(head_prediction["confidence"])
            if predicted_phase not in CONTACT_PHASE_ORDER:
                raise ValueError(f"Unknown contact head phase {predicted_phase!r}")
            if confidence >= float(contact_head.get("min_confidence", 0.0)):
                current_order = CONTACT_PHASE_ORDER.get(str(contact_state["phase"]), 0)
                predicted_order = CONTACT_PHASE_ORDER[predicted_phase]
                if predicted_order > current_order:
                    contact_state["phase"] = predicted_phase
                    if predicted_phase == "descend":
                        contact_state["descend_count"] = 0
                    if predicted_phase == "close":
                        contact_state["close_count"] = 0
                    if predicted_phase in ("lift", "hold_lift"):
                        contact_state["lift_count"] = max(int(contact_state["lift_count"]), 0)

        phase = str(contact_state["phase"])
        env_action = np.asarray(env_action, dtype=np.float32).copy()
        env_action[:3] = 0.0
        env_action[6] = 0.0
        target_name = None

        if phase == "pregrasp":
            target_name = "pregrasp"
            env_action[:3] = clipped_xyz_delta(current, pregrasp, max_step)
            contact_state["pregrasp_count"] = int(contact_state["pregrasp_count"]) + 1
            task_name = "pregrasp_to_descend"
            head_has_task = (
                contact_transition_head is not None
                and task_name in set(contact_transition_head.get("task_names", []))
            )
            ready_to_descend = False
            if head_has_task:
                transition_prediction = contact_transition_head["predict_ready_probability"](
                    env,
                    int(stage_step),
                    contact_state,
                    task_name,
                )
                ready_to_descend = float(transition_prediction["probability"]) >= float(
                    contact_transition_head["threshold"]
                )
            geometry_ready = xy <= max(xy_tol, 0.04) and abs(float(current[2] - pregrasp[2])) <= max(z_tol, 0.02)
            pregrasp_timeout = int(contact_state["pregrasp_count"]) >= int(contact_config.get("pregrasp_hold", 999))
            allow_geometry_transition = (not bool(contact_config.get("transition_head_strict"))) or not head_has_task
            if ready_to_descend or (allow_geometry_transition and geometry_ready) or pregrasp_timeout:
                contact_state["phase"] = "descend"
                contact_state["descend_count"] = 0
                if ready_to_descend:
                    transition_reason = "contact_transition_head"
                elif geometry_ready:
                    transition_reason = "pregrasp_geometry"
                else:
                    transition_reason = "pregrasp_hold"
        elif phase == "descend":
            target_name = "grasp_floor"
            env_action[:3] = clipped_xyz_delta(current, grasp, max_step)
            if float(current[2] + float(env_action[2])) < floor_z:
                env_action[2] = max(float(env_action[2]), floor_z - float(current[2]))
            contact_state["descend_count"] = int(contact_state["descend_count"]) + 1
            task_name = "descend_to_close"
            head_has_task = (
                contact_transition_head is not None
                and task_name in set(contact_transition_head.get("task_names", []))
            )
            ready_to_close = False
            if head_has_task:
                transition_prediction = contact_transition_head["predict_ready_probability"](
                    env,
                    int(stage_step),
                    contact_state,
                    task_name,
                )
                ready_to_close = float(transition_prediction["probability"]) >= float(
                    contact_transition_head["threshold"]
                )
            floor_guard_ready = current[2] <= floor_z + float(contact_config.get("descend_floor_guard_tol", z_tol))
            if ready_to_close and bool(contact_config.get("descend_floor_guard")) and not floor_guard_ready:
                ready_to_close = False
                transition_blocked_reason = "descend_floor_guard"
            geometry_ready = xy <= xy_tol and current[2] <= floor_z + z_tol
            descend_timeout = int(contact_state["descend_count"]) >= int(contact_config.get("descend_hold", 999))
            allow_geometry_transition = (not bool(contact_config.get("transition_head_strict"))) or not head_has_task
            if ready_to_close or (allow_geometry_transition and geometry_ready) or descend_timeout:
                contact_state["phase"] = "close"
                contact_state["close_count"] = 0
                if ready_to_close:
                    transition_reason = "contact_transition_head"
                elif geometry_ready:
                    transition_reason = "descend_geometry"
                else:
                    transition_reason = "descend_hold"
        elif phase == "close":
            target_name = "grasp_floor"
            env_action[:3] = clipped_xyz_delta(current, grasp, max_step)
            if float(current[2] + float(env_action[2])) < floor_z:
                env_action[2] = max(float(env_action[2]), floor_z - float(current[2]))
            env_action[6] = 1.0
            contact_state["close_count"] = int(contact_state["close_count"]) + 1
            ready_to_lift = False
            if contact_transition_head is not None:
                transition_prediction = contact_transition_head["predict_ready_probability"](
                    env,
                    int(stage_step),
                    contact_state,
                    "close_to_lift",
                )
                ready_to_lift = float(transition_prediction["probability"]) >= float(
                    contact_transition_head["threshold"]
                )
            close_timeout = int(contact_state["close_count"]) >= int(contact_config.get("close_hold", 25))
            if ready_to_lift or close_timeout:
                contact_state["phase"] = "lift"
                contact_state["lift_count"] = 0
                transition_reason = "contact_transition_head" if ready_to_lift else "close_hold"
        elif phase == "lift":
            target_name = "lift"
            env_action[:3] = clipped_xyz_delta(current, lift, max_step)
            env_action[6] = 1.0
            contact_state["lift_count"] = int(contact_state["lift_count"]) + 1
            ready_to_hold = False
            if contact_transition_head is not None:
                transition_prediction = contact_transition_head["predict_ready_probability"](
                    env,
                    int(stage_step),
                    contact_state,
                    "lift_to_hold",
                )
                ready_to_hold = float(transition_prediction["probability"]) >= float(
                    contact_transition_head["threshold"]
                )
            lift_reached = abs(float(current[2] - lift[2])) <= max(z_tol, 0.02)
            lift_timeout = int(contact_state["lift_count"]) >= int(contact_config.get("lift_hold", 70))
            if ready_to_hold or lift_reached or lift_timeout:
                contact_state["phase"] = "hold_lift"
                if ready_to_hold:
                    transition_reason = "contact_transition_head"
                elif lift_reached:
                    transition_reason = "lift_z_reached"
                else:
                    transition_reason = "lift_hold"
        elif phase == "hold_lift":
            target_name = "hold_lift"
            env_action[:3] = 0.0
            env_action[6] = 1.0
        else:
            raise ValueError(f"Unknown contact primitive phase: {phase}")

        primitive_info.update(
            {
                "active": bool(contact_state["active"]),
                "phase": str(contact_state["phase"]),
                "target": target_name,
                "pregrasp_count": int(contact_state["pregrasp_count"]),
                "descend_count": int(contact_state["descend_count"]),
                "close_count": int(contact_state["close_count"]),
                "lift_count": int(contact_state["lift_count"]),
                "overrode_action": True,
            }
        )
        if head_prediction is not None:
            primitive_info["contact_head_path"] = str(contact_head["path"])
            primitive_info["contact_head_phase"] = str(head_prediction["phase"])
            primitive_info["contact_head_confidence"] = round(float(head_prediction["confidence"]), 6)
        if transition_prediction is not None:
            primitive_info["contact_transition_head_path"] = str(contact_transition_head["path"])
            primitive_info["contact_transition_head_task"] = str(transition_prediction["task"])
            primitive_info["contact_transition_head_prob"] = round(float(transition_prediction["probability"]), 6)
            primitive_info["contact_transition_head_threshold"] = float(contact_transition_head["threshold"])
        if transition_reason is not None:
            primitive_info["contact_transition_reason"] = str(transition_reason)
        if transition_blocked_reason is not None:
            primitive_info["contact_transition_blocked_reason"] = str(transition_blocked_reason)
        info["tcpplate_contact_primitive"] = primitive_info
        info["env_action"] = module.round_list(env_action, 5)
        return env_action, info

    def select_scheduled_row(env, action_step: int, scheduled_states: np.ndarray) -> np.ndarray:
        if not adaptive_config or not adaptive_config.get("enabled"):
            return np.asarray(
                scheduled_states[min(int(action_step), len(scheduled_states) - 1)],
                dtype=np.float32,
            ).reshape(-1)

        schedule_id = id(scheduled_states)
        if int(action_step) == 0 or adaptive_state.get("schedule_id") != schedule_id:
            lower_index = int(module.PHASES.index("lower_to_plate"))
            lower_start = None
            for idx, row in enumerate(scheduled_states):
                if int(phase_index_from_state(row)) >= lower_index:
                    lower_start = int(idx)
                    break
            if lower_start is None:
                lower_start = len(scheduled_states) - 1
            adaptive_state.update(
                {
                    "schedule_id": schedule_id,
                    "release_step": None,
                    "lower_start": lower_start,
                }
            )

        lower_start = int(adaptive_state["lower_start"])
        release_step = adaptive_state.get("release_step")
        if release_step is None:
            rel = tcp_to_plate(env)
            xy = float(np.linalg.norm(rel[:2]))
            transition_head = adaptive_config.get("transition_head")
            can_release = int(action_step) >= int(adaptive_config["min_steps"])
            if transition_head is not None:
                prob = float(transition_head["predict_ready_probability"](rel, int(action_step)))
                close_enough = prob >= float(transition_head["threshold"])
                adaptive_state["last_transition_head_prob"] = prob
            else:
                close_enough = xy <= float(adaptive_config["xy_threshold"])
            timeout = int(action_step) >= int(adaptive_config["max_steps"])
            if can_release and (close_enough or timeout):
                adaptive_state["release_step"] = int(action_step)
                adaptive_state["release_reason"] = "transition_head" if close_enough else "max_steps"
                release_step = int(action_step)

        if release_step is None:
            row_index = min(int(action_step), max(lower_start - 1, 0))
        else:
            row_index = min(lower_start + int(action_step) - int(release_step), len(scheduled_states) - 1)
        return np.asarray(scheduled_states[row_index], dtype=np.float32).reshape(-1)

    def observation_state_for_policy(
        env,
        policy,
        action_step: int,
        hz: float,
        args,
        phase_tracker,
        scheduled_states: np.ndarray | None = None,
        stage: str | None = None,
    ) -> np.ndarray:
        dim = expected_state_dim(policy)
        if dim != 22:
            return original_observation_state(
                env,
                policy,
                action_step,
                hz,
                args,
                phase_tracker,
                scheduled_states=scheduled_states,
            )

        if str(args.pi0_phase_state) == "dataset_schedule" and scheduled_states is not None:
            if len(scheduled_states) == 0:
                raise ValueError("--pi0-phase-state=dataset_schedule received an empty schedule")
            row = select_scheduled_row(env, int(action_step), scheduled_states)
            if row.shape[0] != 22:
                raise ValueError(f"Scheduled state dim {row.shape[0]} does not match expected dim 22")
            base_state = np.asarray(env.get_joint_state()[:6], dtype=np.float32).reshape(-1)
            rel = tcp_to_plate(env)
            set_current_phase(args, phase_index_from_state(row))
            return np.concatenate([base_state, row[6:19], rel]).astype(np.float32)

        base_state = np.asarray(env.get_joint_state()[:6], dtype=np.float32).reshape(-1)
        timestamp = np.asarray([float(action_step) / float(hz)], dtype=np.float32)
        if str(args.pi0_phase_state) == "zeros":
            phase_index = 0
        else:
            if phase_tracker is None:
                raise ValueError("22-D phase-conditioned state requires a phase tracker")
            phase_index = int(phase_tracker.observe(env))
        set_current_phase(args, phase_index)
        use_target_state = bool(prefix_target_state and str(stage) == "prefix")
        rel = tcp_to_target(env) if use_target_state else tcp_to_plate(env)
        return np.concatenate([base_state, timestamp, phase_feature(phase_index), rel]).astype(np.float32)

    module.observation_state_for_policy = observation_state_for_policy

    original_select_env_action = module.select_env_action

    def select_env_action(
        env,
        policy,
        stage_step: int,
        args,
        helper_args,
        phase_tracker,
        scheduled_states: np.ndarray | None = None,
        scripted_gripper: bool = False,
        stage: str | None = None,
    ):
        env_action, info = original_select_env_action(
            env,
            policy,
            stage_step,
            args,
            helper_args,
            phase_tracker,
            scheduled_states=scheduled_states,
            scripted_gripper=scripted_gripper,
            stage=stage,
        )
        if adaptive_config and adaptive_config.get("enabled"):
            info["tcpplate_adaptive_move_preplace"] = True
            info["tcpplate_adaptive_release_step"] = adaptive_state.get("release_step")
            info["tcpplate_adaptive_release_reason"] = adaptive_state.get("release_reason")
            if "last_transition_head_prob" in adaptive_state:
                info["tcpplate_transition_head_prob"] = round(
                    float(adaptive_state["last_transition_head_prob"]),
                    6,
                )
        if stage == "prefix":
            env_action, info = apply_contact_primitive(env, env_action, info, int(stage_step), args)
        return env_action, info

    module.select_env_action = select_env_action


def install_schedule_patch(module, start_phase: str | None, force_episode: int | None):
    if not start_phase and force_episode is None:
        return
    if start_phase and start_phase not in module.PHASES:
        raise ValueError(f"Unknown phase {start_phase!r}; choices are {module.PHASES}")

    from audit_smolvla_physical import phase_index_from_state

    target_index = int(module.PHASES.index(start_phase)) if start_phase else None
    original_loader = module.load_finisher_phase_schedules

    def load_finisher_phase_schedules(args, seeds: list[int]) -> dict[int, np.ndarray]:
        original_spec = getattr(args, "finisher_phase_schedule_episodes", None)
        if force_episode is not None:
            args.finisher_phase_schedule_episodes = ",".join(str(int(force_episode)) for _ in seeds)
        try:
            schedules = original_loader(args, seeds)
        finally:
            args.finisher_phase_schedule_episodes = original_spec
        shifted: dict[int, np.ndarray] = {}
        for seed, states in schedules.items():
            start = 0
            if target_index is not None:
                for idx, row in enumerate(states):
                    if int(phase_index_from_state(row)) == target_index:
                        start = idx
                        break
            shifted[int(seed)] = states[start:]
        return shifted

    module.load_finisher_phase_schedules = load_finisher_phase_schedules


def install_gripper_head_patch(module, head_path: Path | None, threshold: float):
    if head_path is None:
        return
    data = np.load(head_path, allow_pickle=False)
    mean = np.asarray(data["state_mean"], dtype=np.float32).reshape(-1)
    std = np.asarray(data["state_std"], dtype=np.float32).reshape(-1)
    weight = np.asarray(data["weight"], dtype=np.float32).reshape(-1)
    bias = float(np.asarray(data["bias"]).reshape(()))
    if "feature_indices" in data.files:
        feature_indices = np.asarray(data["feature_indices"], dtype=np.int64).reshape(-1)
    else:
        feature_indices = np.arange(mean.shape[0], dtype=np.int64)
    if mean.shape != std.shape or mean.shape != weight.shape:
        raise ValueError(
            f"Bad gripper head shapes: mean={mean.shape}, std={std.shape}, weight={weight.shape}"
        )
    if feature_indices.shape[0] != mean.shape[0]:
        raise ValueError(
            f"Bad gripper head feature count: indices={feature_indices.shape}, mean={mean.shape}"
        )
    std = np.where(std < 1e-6, 1.0, std).astype(np.float32)
    threshold = float(threshold)

    original_select_env_action = module.select_env_action

    def predict_closed_probability(state: np.ndarray) -> float:
        state = np.asarray(state, dtype=np.float32).reshape(-1)
        if int(feature_indices.max(initial=-1)) >= state.shape[0]:
            raise ValueError(
                f"Gripper head feature index {int(feature_indices.max())} outside state dim {state.shape[0]}"
            )
        x = (state[feature_indices] - mean) / std
        logit = float(np.dot(x, weight) + bias)
        return float(1.0 / (1.0 + np.exp(-np.clip(logit, -60.0, 60.0))))

    def select_env_action(
        env,
        policy,
        stage_step: int,
        args,
        helper_args,
        phase_tracker,
        scheduled_states: np.ndarray | None = None,
        scripted_gripper: bool = False,
        stage: str | None = None,
    ):
        env_action, info = original_select_env_action(
            env,
            policy,
            stage_step,
            args,
            helper_args,
            phase_tracker,
            scheduled_states=scheduled_states,
            scripted_gripper=scripted_gripper,
            stage=stage,
        )
        if scripted_gripper:
            prob = predict_closed_probability(np.asarray(info["state"], dtype=np.float32))
            env_action[6] = 1.0 if prob >= threshold else 0.0
            info["env_action"] = module.round_list(env_action, 5)
            info["tcpplate_gripper_head_path"] = str(head_path)
            info["tcpplate_gripper_head_closed_prob"] = round(prob, 6)
            info["tcpplate_gripper_head_threshold"] = threshold
            info["tcpplate_gripper_head_closed"] = bool(prob >= threshold)
            info["scripted_gripper"] = False
            info["scripted_gripper_replaced_by_learned_head"] = True
        return env_action, info

    module.select_env_action = select_env_action


def main() -> int:
    forwarded_argv, wrapper_args = pop_wrapper_args(sys.argv)
    module = load_module(wrapper_args.tcpplate_base_evaluator)
    transition_head = load_transition_head(
        wrapper_args.tcpplate_transition_head_path,
        wrapper_args.tcpplate_transition_head_threshold,
    )
    contact_head = load_contact_phase_head(
        wrapper_args.tcpplate_contact_head_path,
        wrapper_args.tcpplate_contact_head_min_confidence,
    )
    contact_transition_head = load_contact_transition_head(
        wrapper_args.tcpplate_contact_transition_head_path,
        wrapper_args.tcpplate_contact_transition_head_threshold,
    )
    install_tcpplate_patch(
        module,
        {
            "enabled": bool(wrapper_args.tcpplate_adaptive_move_preplace or transition_head is not None),
            "xy_threshold": float(wrapper_args.tcpplate_adaptive_move_preplace_xy_threshold),
            "min_steps": int(wrapper_args.tcpplate_adaptive_move_preplace_min_steps),
            "max_steps": int(wrapper_args.tcpplate_adaptive_move_preplace_max_steps),
            "transition_head": transition_head,
        },
        prefix_target_state=bool(wrapper_args.tcpplate_prefix_target_state),
        contact_config={
            "mode": str(wrapper_args.tcpplate_contact_primitive),
            "start_step": int(wrapper_args.tcpplate_contact_start_step),
            "trigger_xy": float(wrapper_args.tcpplate_contact_trigger_xy),
            "xy_tol": float(wrapper_args.tcpplate_contact_xy_tol),
            "z_tol": float(wrapper_args.tcpplate_contact_z_tol),
            "pregrasp_hold": int(wrapper_args.tcpplate_contact_pregrasp_hold),
            "descend_hold": int(wrapper_args.tcpplate_contact_descend_hold),
            "close_hold": int(wrapper_args.tcpplate_contact_close_hold),
            "lift_hold": int(wrapper_args.tcpplate_contact_lift_hold),
            "target_z_offset": wrapper_args.tcpplate_contact_target_z_offset,
            "contact_head": contact_head,
            "contact_transition_head": contact_transition_head,
            "transition_head_strict": bool(wrapper_args.tcpplate_contact_transition_head_strict),
            "descend_floor_guard": bool(wrapper_args.tcpplate_contact_descend_floor_guard),
            "descend_floor_guard_tol": float(wrapper_args.tcpplate_contact_descend_floor_guard_tol),
        },
    )
    install_schedule_patch(
        module,
        wrapper_args.tcpplate_schedule_start_phase,
        wrapper_args.tcpplate_force_schedule_episode,
    )
    install_gripper_head_patch(
        module,
        wrapper_args.tcpplate_gripper_head_path,
        wrapper_args.tcpplate_gripper_head_threshold,
    )
    sys.argv = forwarded_argv
    return int(module.main())


if __name__ == "__main__":
    raise SystemExit(main())
