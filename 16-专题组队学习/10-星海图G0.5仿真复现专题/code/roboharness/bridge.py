"""Convert postprocessed BEHAVIOR policy actions to RoboHarness joint tools.

The caller owns model normalization and BehaviorPerKeyTransform.backward.
Commands here contain physical joint targets and model gripper values in
[-1, 1]. They are never accepted as a flattened ActionCodec vector. The
workspace's joint_control tool owns validation, physics and action receipts.
"""

from dataclasses import dataclass
import math


SEMANTICS = "behavior_r1pro_postprocessed_v1"
PART_WIDTHS = {"left_arm": 7, "left_gripper": 1, "right_arm": 7,
               "right_gripper": 1, "lower_body": 7}


def vector(values, size, name):
    if not isinstance(values, (list, tuple)) or len(values) != size:
        raise ValueError(f"{name} requires {size} values")
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) or
           not math.isfinite(v) for v in values):
        raise ValueError(f"{name} requires finite numeric values")
    return [float(v) for v in values]


@dataclass(frozen=True)
class Calibration:
    """Physical parameters whose live qualification is separately recorded."""

    max_joint_delta_rad: float = 0.10
    close_effort_n: float = -10.0
    open_effort_n: float = 10.0
    close_threshold: float = -0.5
    open_threshold: float = 0.5
    tool_roll_tolerance_rad: float = 0.05

    def __post_init__(self):
        values = tuple(self.__dict__.values())
        if any(isinstance(v, bool) or not math.isfinite(v) for v in values):
            raise ValueError("calibration values must be finite numbers")
        if not 0 < self.max_joint_delta_rad <= 0.5:
            raise ValueError("invalid single-command joint bound")
        if not -20 <= self.close_effort_n < 0 < self.open_effort_n <= 20:
            raise ValueError("gripper effort convention is negative-close")
        if not -1 <= self.close_threshold < self.open_threshold <= 1:
            raise ValueError("invalid gripper hysteresis")
        if not 0 <= self.tool_roll_tolerance_rad <= 0.1:
            raise ValueError("invalid tool-roll tolerance")


def state_from_proprio(proprio, calibration=Calibration()):
    """Read allowed 65-value proprioception into pre-transform BEHAVIOR parts.

    J8 is unrepresented by the seven-joint policy. Its near-zero posture is
    required, and the output never silently drops an arbitrary tool rotation.
    Gripper positions remain two measured metre values; the official model
    preprocessor performs its documented width collapse and normalization.
    """
    q = vector(proprio, 65, "proprioception")
    if any(abs(q[i]) > calibration.tool_roll_tolerance_rad for i in (10, 37)):
        raise ValueError("unrepresented J8 must be restored before G0.5 inference")
    return {"left_arm": q[3:10], "left_gripper": q[26:28],
            "right_arm": q[30:37], "right_gripper": q[53:55],
            "trunk_qpos": q[57:61], "base_qvel": q[0:3]}


def validate_proposal(proposal, episode_id, sequence):
    """Bind a candidate to the exact current episode and observation."""
    if proposal.get("semantics") != SEMANTICS:
        raise ValueError("candidate must contain postprocessed BEHAVIOR actions")
    if proposal.get("episode_id") != episode_id:
        raise ValueError("candidate belongs to another episode")
    if isinstance(proposal.get("sequence"), bool) or proposal.get("sequence") != sequence:
        raise ValueError("candidate uses a stale observation")
    if not isinstance(proposal.get("proposal_id"), str) or not proposal["proposal_id"]:
        raise ValueError("proposal_id is required")
    parts = proposal.get("action")
    if not isinstance(parts, dict) or set(parts) != set(PART_WIDTHS):
        raise ValueError("candidate action parts do not match the BEHAVIOR contract")
    return {k: vector(parts[k], n, k) for k, n in PART_WIDTHS.items()}


def joint_tool_arguments(proposal, proprio, episode_id, sequence,
                         calibration=Calibration()):
    """Construct one atomic joint_control call without stepping the simulator.

    The first bridge version holds the base and torso. Navigation and whole-
    body changes remain with the original geometric tools. Candidate lower-
    body targets must agree with that scope. Large arm deltas are rejected,
    not clipped. Every accepted candidate executes exactly one physics step.
    """
    q = vector(proprio, 65, "proprioception")
    state_from_proprio(q, calibration)
    parts = validate_proposal(proposal, episode_id, sequence)
    lb = parts["lower_body"]  # Official backward transform: base(3), torso(4).
    if any(abs(v) > 1e-8 for v in lb[:3]):
        raise ValueError("base motion belongs to geometric navigation in this phase")
    if any(abs(target - current) > 1e-5 for target, current in zip(lb[3:], q[57:61])):
        raise ValueError("torso motion is outside this manipulation-only bridge")
    result = {"joints": {}, "steps": 1}
    for i in range(3):
        result["joints"][f"torso_joint{i+1}"] = lb[3 + i]
    for side, start in (("left", 3), ("right", 30)):
        targets = parts[f"{side}_arm"]
        for i, target in enumerate(targets):
            if abs(target - q[start+i]) > calibration.max_joint_delta_rad:
                raise ValueError(f"{side} arm target exceeds single-command bound")
            result["joints"][f"{side}_arm_joint{i+1}"] = target
        grip = parts[f"{side}_gripper"][0]
        if not -1 <= grip <= 1:
            raise ValueError("postprocessed gripper must be in [-1, 1]")
        if grip <= calibration.close_threshold:
            result[f"{side}_finger_effort"] = calibration.close_effort_n
        elif grip >= calibration.open_threshold:
            result[f"{side}_finger_effort"] = calibration.open_effort_n
        # Omitted effort keeps the workspace's last commanded force.
    return result


def choose_executor(*, tracking_valid, candidate_valid, error_before_m,
                    error_after_m, precise_alignment=False):
    """Choose the next controller from measured tracking residuals.

    Invalid observations require reacquisition. Missing residuals require a
    geometric check. A 1 mm progress threshold is an initial development
    parameter and must be frozen before live comparison.
    """
    if not tracking_valid:
        return "reobserve"
    if precise_alignment or not candidate_valid:
        return "geometry"
    residuals = (error_before_m, error_after_m)
    if any(v is None or isinstance(v, bool) or not isinstance(v, (int, float)) or
           not math.isfinite(v) or v < 0 for v in residuals):
        return "geometry"
    return "g05" if error_before_m - error_after_m >= 0.001 else "geometry"
