#!/usr/bin/env python3
"""Evaluate a staged Pi0 EEF-abs policy on MuJoCo PnP.

The first policy controls a fixed prefix from reset.  The second policy then
starts with its own action/phase clock and tries to finish from that on-policy
state.  This is a diagnostic staged-policy evaluator, not a raw single-policy
Pi0 evaluator.
"""

from __future__ import annotations

import argparse
import gc
import inspect
import json
import os
import random
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from audit_smolvla_physical import (
    PHASES,
    PHASE_SCRIPTED_CLOSED,
    ScriptedPhaseTracker,
    action_for_environment,
    close_env,
    configure_env_action_space,
    init_tracker,
    make_pi0_policy_for_dataset,
    observation_state_for_policy,
    physical_debug,
    round_list,
    update_tracker,
)
from eval_policy_success import to_tensor_image


DEFAULT_TIMED_PHASE_DWELL = {
    "move_pregrasp": 40,
    "move_grasp": 50,
    "lift_mug": 60,
    "move_preplace": 130,
    "lower_to_plate": 40,
    "retreat": 40,
}


def parse_seed_list(spec: str | None) -> list[int]:
    if spec is None or not str(spec).strip():
        return []
    return [int(part.strip()) for part in str(spec).split(",") if part.strip()]


def parse_phase_dwell_spec(spec: str | None) -> dict[str, int]:
    values = dict(DEFAULT_TIMED_PHASE_DWELL)
    if not spec:
        return values
    for part in str(spec).split(","):
        part = part.strip()
        if not part:
            continue
        if ":" not in part:
            raise ValueError(f"Invalid dwell spec entry {part!r}; expected phase:steps")
        phase, steps = part.split(":", 1)
        values[phase.strip()] = int(steps.strip())
    return values


class TimedPhaseTracker(ScriptedPhaseTracker):
    """Scripted phase tracker with max dwell fallbacks for move phases."""

    def __init__(self, args: argparse.Namespace):
        super().__init__(args)
        self.dwell_limits = parse_phase_dwell_spec(getattr(args, "timed_phase_dwell_spec", None))
        self.phase_observations = 0
        self.last_phase_index = int(self.phase_index)

    def observe(self, env) -> int:
        before = int(self.phase_index)
        phase_index = int(super().observe(env))
        if int(self.phase_index) != before or int(self.phase_index) != self.last_phase_index:
            self.phase_observations = 0
            self.last_phase_index = int(self.phase_index)
            return int(self.phase_index)

        if self.phase_index < len(self.phases):
            phase, kind, _ = self.phases[self.phase_index]
            if kind == "move":
                self.phase_observations += 1
                limit = int(self.dwell_limits.get(phase, 0))
                if limit > 0 and self.phase_observations >= limit:
                    self.phase_index += 1
                    if self.phase_index < len(self.phases):
                        self.hold_remaining = self.phases[self.phase_index][2]
                    self.phase_observations = 0
                    self.last_phase_index = int(self.phase_index)
                    return int(self.phase_index)
        return phase_index


def make_phase_tracker(phase_state: str, helper_args: argparse.Namespace):
    if str(phase_state) not in ("auto", "dynamic_oracle", "dynamic_timed"):
        return None
    if str(phase_state) == "dynamic_timed":
        return TimedPhaseTracker(helper_args)
    return ScriptedPhaseTracker(helper_args)


def set_tracker_start_phase(tracker, phase_name: str | None) -> None:
    if tracker is None or not phase_name:
        return
    for index, (name, _kind, hold_steps) in enumerate(tracker.phases):
        if name == phase_name:
            tracker.phase_index = int(index)
            tracker.last_phase_index = int(index)
            tracker.phase_observations = 0
            tracker.hold_remaining = int(hold_steps)
            return
    raise ValueError(f"Unknown tracker phase {phase_name!r}")


def policy_args(args: argparse.Namespace, phase_state: str) -> argparse.Namespace:
    return SimpleNamespace(
        policy_type="pi0",
        pi0_action_mode="eef_abs",
        eef_abs_max_step=float(args.eef_abs_max_step),
        pi0_phase_state=str(phase_state),
        hz=float(args.hz),
        physical_min_lift=float(args.physical_min_lift),
        physical_min_lift_steps=int(args.physical_min_lift_steps),
        physical_final_upright_cos=float(args.physical_final_upright_cos),
        place_step_scale=float(args.place_step_scale),
        pos_tol=float(args.pos_tol),
        initial_hold=int(args.initial_hold),
        close_hold=int(args.close_hold),
        pre_release_hold=int(args.pre_release_hold),
        open_hold=int(args.open_hold),
        final_hold=int(args.final_hold),
        grasp_offset_x=float(args.grasp_offset_x),
        grasp_offset_y=float(args.grasp_offset_y),
        grasp_offset_z=float(args.grasp_offset_z),
        pregrasp_z=float(args.pregrasp_z),
        lift_z=float(args.lift_z),
        release_offset_x=float(args.release_offset_x),
        release_offset_y=float(args.release_offset_y),
        release_offset_z=float(args.release_offset_z),
        retreat_z=float(args.retreat_z),
        timed_phase_dwell_spec=str(args.timed_phase_dwell_spec),
    )


def select_env_action(
    env,
    policy,
    stage_step: int,
    args: argparse.Namespace,
    helper_args: argparse.Namespace,
    phase_tracker: ScriptedPhaseTracker,
    scheduled_states: np.ndarray | None = None,
    scripted_gripper: bool = False,
    stage: str | None = None,
) -> tuple[np.ndarray, dict]:
    state_kwargs = {
        "scheduled_states": scheduled_states,
    }
    if "stage" in inspect.signature(observation_state_for_policy).parameters:
        state_kwargs["stage"] = stage
    state = observation_state_for_policy(
        env,
        policy,
        stage_step,
        args.hz,
        helper_args,
        phase_tracker,
        **state_kwargs,
    )
    image, wrist_image = env.grab_image()
    batch = {
        "observation.state": torch.tensor(np.asarray([state]), dtype=torch.float32, device=args.device),
        "observation.image": to_tensor_image(image).unsqueeze(0).to(args.device),
        "observation.wrist_image": to_tensor_image(wrist_image).unsqueeze(0).to(args.device),
        "task": [env.instruction],
    }
    with torch.no_grad():
        action = policy.select_action(batch)[0, :7].detach().cpu().numpy().astype(np.float32)
    env_action, bridge_info = action_for_environment(action, env, helper_args)
    scripted_phase_name = None
    if scripted_gripper:
        scripted_phase_name = phase_name_from_state(state)
        env_action[6] = 1.0 if scripted_phase_name in PHASE_SCRIPTED_CLOSED else 0.0
    info = {
        "policy_action": round_list(action, 5),
        "env_action": round_list(env_action, 5),
        "bridge_info": bridge_info,
        "state": round_list(state, 5),
        "scripted_gripper": bool(scripted_gripper),
        "scripted_gripper_phase": scripted_phase_name,
    }
    return env_action, info


def phase_name_from_state(state: np.ndarray) -> str:
    state = np.asarray(state, dtype=np.float32).reshape(-1)
    if state.shape[0] >= 8 + len(PHASES):
        onehot = state[8 : 8 + len(PHASES)]
        if onehot.shape[0] == len(PHASES) and float(np.max(onehot)) >= 0.5:
            return PHASES[int(np.argmax(onehot))]
        index = int(round(float(state[7]) * max(len(PHASES) - 1, 1)))
        return PHASES[int(np.clip(index, 0, len(PHASES) - 1))]
    return PHASES[0]


def select_oracle_prefix_action(
    env,
    stage_step: int,
    args: argparse.Namespace,
    phase_tracker: ScriptedPhaseTracker,
) -> tuple[np.ndarray, dict]:
    try:
        from collect_pi0_dagger_tail import target_positions
    except Exception as exc:
        raise RuntimeError("Oracle prefix mode requires collect_pi0_dagger_tail.py on PYTHONPATH") from exc

    phase_index = int(phase_tracker.observe(env))
    phase_name = PHASES[int(np.clip(phase_index, 0, len(PHASES) - 1))]
    points = target_positions(env, args)
    target_key = {
        "move_pregrasp": "pregrasp",
        "move_grasp": "grasp",
        "lift_mug": "lift",
        "move_preplace": "preplace",
        "lower_to_plate": "release",
        "retreat": "retreat",
    }.get(phase_name)

    env_action = np.zeros(7, dtype=np.float32)
    if target_key is not None:
        current = np.asarray(env.env.get_p_body("tcp_link")[:3], dtype=np.float32)
        target = np.asarray(points[target_key], dtype=np.float32)
        step_scale = float(args.place_step_scale) if phase_name == "lower_to_plate" else 1.0
        max_step = float(args.move_step) * step_scale
        env_action[:3] = np.clip(target - current, -max_step, max_step).astype(np.float32)
    env_action[6] = 1.0 if phase_name in PHASE_SCRIPTED_CLOSED else 0.0
    info = {
        "source": "scripted_oracle_prefix",
        "stage_step": int(stage_step),
        "phase_index": int(phase_index),
        "phase_name": phase_name,
        "target_key": target_key,
        "env_action": round_list(env_action, 5),
    }
    return env_action, info


def run_episode(
    env,
    seed: int,
    prefix_policy,
    finisher_policy,
    args: argparse.Namespace,
    finisher_scheduled_states: np.ndarray | None,
) -> dict:
    prefix_helper_args = policy_args(args, args.prefix_phase_state)
    finisher_helper_args = policy_args(args, args.finisher_phase_state)
    if bool(getattr(args, "hard_reset_sim_data", False)):
        inner_env = getattr(env, "env", None)
        if inner_env is not None and hasattr(inner_env, "reset"):
            inner_env.reset(step=False)
    env.reset(seed=int(seed))
    if args.instruction:
        env.set_instruction(args.instruction)
    policy_seed = int(seed) + int(args.policy_seed_offset)
    random.seed(policy_seed)
    np.random.seed(policy_seed)
    torch.manual_seed(policy_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(policy_seed)
    configure_env_action_space(env, prefix_helper_args)
    if prefix_policy is not None:
        prefix_policy.reset()
    finisher_policy.reset()
    prefix_tracker = (
        ScriptedPhaseTracker(prefix_helper_args)
        if str(args.prefix_source) == "oracle"
        else make_phase_tracker(str(args.prefix_phase_state), prefix_helper_args)
    )
    finisher_tracker = make_phase_tracker(str(args.finisher_phase_state), finisher_helper_args)
    set_tracker_start_phase(finisher_tracker, args.finisher_start_phase)
    tracker = init_tracker(env)

    action_steps = 0
    sim_steps = 0
    prefix_steps_done = 0
    finisher_steps_done = 0
    first_legacy_success_step = None
    first_physical_success_step = None
    physical_success_ever = False
    legacy_success_ever = False
    last_debug = physical_debug(env, tracker, prefix_helper_args)
    start = time.time()

    while action_steps < int(args.max_action_steps) and env.env.is_viewer_alive():
        env.step_env()
        sim_steps += 1
        if not env.env.loop_every(HZ=args.hz):
            continue

        active_helper_args = prefix_helper_args if action_steps < int(args.prefix_steps) else finisher_helper_args
        update_tracker(env, tracker, active_helper_args)
        last_debug = physical_debug(env, tracker, active_helper_args)
        if last_debug["success"] and first_legacy_success_step is None:
            first_legacy_success_step = action_steps
        if last_debug["physical_success"] and first_physical_success_step is None:
            first_physical_success_step = action_steps
        legacy_success_ever = bool(legacy_success_ever or last_debug["success"])
        physical_success_ever = bool(physical_success_ever or last_debug["physical_success"])
        if last_debug["physical_success"]:
            break

        if action_steps < int(args.prefix_steps):
            stage = "prefix"
            stage_step = prefix_steps_done
            phase_tracker = prefix_tracker
            helper_args = prefix_helper_args
            scheduled_states = None
            prefix_steps_done += 1
        else:
            stage = "finisher"
            stage_step = finisher_steps_done
            policy = finisher_policy
            phase_tracker = finisher_tracker
            helper_args = finisher_helper_args
            scheduled_states = finisher_scheduled_states
            finisher_steps_done += 1

        if stage == "prefix" and str(args.prefix_source) == "oracle":
            if phase_tracker is None:
                raise ValueError("Oracle prefix mode requires a phase tracker")
            env_action, action_info = select_oracle_prefix_action(env, stage_step, args, phase_tracker)
        else:
            policy = prefix_policy if stage == "prefix" else finisher_policy
            if policy is None:
                raise ValueError("Policy prefix mode requires --prefix-policy-path")
            env_action, action_info = select_env_action(
                env,
                policy,
                stage_step,
                args,
                helper_args,
                phase_tracker,
                scheduled_states=scheduled_states,
                scripted_gripper=(stage == "finisher" and bool(args.finisher_scripted_gripper)),
                stage=stage,
            )
        env.step(env_action)
        action_steps += 1

        update_tracker(env, tracker, helper_args)
        last_debug = physical_debug(env, tracker, helper_args)
        if last_debug["success"] and first_legacy_success_step is None:
            first_legacy_success_step = action_steps
        if last_debug["physical_success"] and first_physical_success_step is None:
            first_physical_success_step = action_steps
        legacy_success_ever = bool(legacy_success_ever or last_debug["success"])
        physical_success_ever = bool(physical_success_ever or last_debug["physical_success"])

        should_log_transition = (
            (stage == "prefix" and stage_step >= max(int(args.prefix_steps) - 3, 0))
            or (stage == "finisher" and stage_step < 5)
        )
        if args.log_steps_jsonl and (
            action_steps <= 10 or action_steps % int(args.log_every) == 0 or should_log_transition
        ):
            row = {
                "event": "step",
                "seed": int(seed),
                "global_action_step": int(action_steps),
                "stage": stage,
                "stage_step": int(stage_step),
                "instruction": env.instruction,
                "action_info": action_info,
                "debug": last_debug,
            }
            with args.log_steps_jsonl.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

        if args.render:
            env.render()
        if last_debug["physical_success"]:
            break

    last_debug = physical_debug(env, tracker, finisher_helper_args)
    legacy_success_ever = bool(legacy_success_ever or last_debug["success"])
    physical_success_ever = bool(physical_success_ever or last_debug["physical_success"])
    last_debug.update(
        {
            "legacy_success_ever": legacy_success_ever,
            "first_legacy_success_step": first_legacy_success_step,
            "physical_success_ever": physical_success_ever,
            "first_physical_success_step": first_physical_success_step,
        }
    )
    return {
        "policy": "pi0_two_stage_eef_abs",
        "seed": int(seed),
        "policy_seed": int(policy_seed),
        "success": legacy_success_ever,
        "physical_success": physical_success_ever,
        "action_steps": int(action_steps),
        "prefix_steps_done": int(prefix_steps_done),
        "finisher_steps_done": int(finisher_steps_done),
        "sim_steps": int(sim_steps),
        "elapsed_s": round(time.time() - start, 3),
        "instruction": getattr(env, "instruction", None),
        "debug": last_debug,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefix-source", choices=["policy", "oracle"], default="policy")
    parser.add_argument("--prefix-policy-path", type=Path, default=None)
    parser.add_argument("--prefix-dataset-repo-id", default=None)
    parser.add_argument("--prefix-dataset-root", type=Path, default=None)
    parser.add_argument("--finisher-policy-path", type=Path, required=True)
    parser.add_argument("--finisher-dataset-repo-id", required=True)
    parser.add_argument("--finisher-dataset-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--episodes", type=int, default=2)
    parser.add_argument("--seed-start", type=int, default=1000)
    parser.add_argument("--seeds", default=None)
    parser.add_argument("--policy-seed-offset", type=int, default=0)
    parser.add_argument("--instruction", default=None)
    parser.add_argument("--hz", type=float, default=20.0)
    parser.add_argument("--prefix-steps", type=int, default=120)
    parser.add_argument("--max-action-steps", type=int, default=500)
    parser.add_argument("--eef-abs-max-step", type=float, default=0.004)
    parser.add_argument("--move-step", type=float, default=0.004)
    parser.add_argument("--pi0-phase-state", choices=["auto", "dynamic_oracle", "dynamic_timed", "zeros"], default="dynamic_oracle")
    parser.add_argument("--prefix-phase-state", choices=["auto", "dynamic_oracle", "dynamic_timed", "zeros"], default=None)
    parser.add_argument(
        "--finisher-phase-state",
        choices=["auto", "dynamic_oracle", "dynamic_timed", "zeros", "dataset_schedule"],
        default="dynamic_oracle",
    )
    parser.add_argument(
        "--finisher-start-phase",
        choices=PHASES,
        default=None,
        help=(
            "Optional start phase for dynamic finisher trackers. This is useful "
            "when comparing dynamic_timed/dynamic_oracle finishers against a "
            "dataset_schedule that was previously shifted to move_preplace."
        ),
    )
    parser.add_argument(
        "--timed-phase-dwell-spec",
        default=",".join(f"{phase}:{steps}" for phase, steps in DEFAULT_TIMED_PHASE_DWELL.items()),
        help="Comma-separated phase:max_steps fallback used by --*-phase-state=dynamic_timed.",
    )
    parser.add_argument(
        "--finisher-phase-schedule-episodes",
        default=None,
        help="Comma-separated finisher dataset episode ids aligned with --seeds; defaults to 0,1,2,...",
    )
    parser.add_argument(
        "--finisher-scripted-gripper",
        action="store_true",
        help="Diagnostic: keep Pi0 EEF/arm action but set finisher gripper open/closed from the current phase.",
    )
    parser.add_argument("--physical-min-lift", type=float, default=0.03)
    parser.add_argument("--physical-min-lift-steps", type=int, default=3)
    parser.add_argument("--physical-final-upright-cos", type=float, default=0.7)
    parser.add_argument("--render", action="store_true")
    parser.add_argument(
        "--fresh-env-per-episode",
        action="store_true",
        help=(
            "Create a new MuJoCo SimpleEnv2 for each seed. This avoids cross-episode "
            "qvel/ctrl/free-joint residue when the environment reset is not a full "
            "sim-data reset."
        ),
    )
    parser.add_argument(
        "--hard-reset-sim-data",
        action="store_true",
        help=(
            "Call the underlying MuJoCo parser reset before each env.reset(seed), "
            "clearing qvel/ctrl/free-joint residue while reusing the same viewer/env."
        ),
    )
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--log-steps-jsonl", type=Path, default=None)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--summary-json", type=Path, required=True)
    parser.add_argument("--place-step-scale", type=float, default=0.65)
    parser.add_argument("--pos-tol", type=float, default=0.008)
    parser.add_argument("--initial-hold", type=int, default=0)
    parser.add_argument("--close-hold", type=int, default=64)
    parser.add_argument("--pre-release-hold", type=int, default=12)
    parser.add_argument("--open-hold", type=int, default=70)
    parser.add_argument("--final-hold", type=int, default=35)
    parser.add_argument("--grasp-offset-x", type=float, default=0.006)
    parser.add_argument("--grasp-offset-y", type=float, default=0.060)
    parser.add_argument("--grasp-offset-z", type=float, default=0.000)
    parser.add_argument("--pregrasp-z", type=float, default=0.135)
    parser.add_argument("--lift-z", type=float, default=0.145)
    parser.add_argument("--release-offset-x", type=float, default=0.008)
    parser.add_argument("--release-offset-y", type=float, default=0.026)
    parser.add_argument("--release-offset-z", type=float, default=0.095)
    parser.add_argument("--retreat-z", type=float, default=0.145)
    return parser.parse_args()


def parse_episode_list(spec: str | None, seeds: list[int]) -> list[int]:
    if spec is None or not str(spec).strip():
        return list(range(len(seeds)))
    episode_ids = [int(part.strip()) for part in str(spec).split(",") if part.strip()]
    if len(episode_ids) != len(seeds):
        raise ValueError(
            f"--finisher-phase-schedule-episodes has {len(episode_ids)} entries, "
            f"but {len(seeds)} seeds were requested"
        )
    return episode_ids


def load_finisher_phase_schedules(args: argparse.Namespace, seeds: list[int]) -> dict[int, np.ndarray]:
    if str(args.finisher_phase_state) != "dataset_schedule":
        return {}
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

    dataset = LeRobotDataset(args.finisher_dataset_repo_id, root=args.finisher_dataset_root)
    episode_ids = parse_episode_list(args.finisher_phase_schedule_episodes, seeds)
    state_column = dataset.hf_dataset["observation.state"]
    schedules: dict[int, np.ndarray] = {}
    for seed, episode_id in zip(seeds, episode_ids):
        if episode_id < 0 or episode_id >= dataset.num_episodes:
            raise ValueError(f"Episode {episode_id} is outside dataset range 0..{dataset.num_episodes - 1}")
        start = int(dataset.episode_data_index["from"][episode_id].item())
        end = int(dataset.episode_data_index["to"][episode_id].item())
        states = [np.asarray(state_column[idx], dtype=np.float32).reshape(-1) for idx in range(start, end)]
        if not states:
            raise ValueError(f"Episode {episode_id} has no frames")
        schedules[int(seed)] = np.stack(states).astype(np.float32)
    return schedules


def main() -> int:
    args = parse_args()
    if args.prefix_phase_state is None:
        args.prefix_phase_state = args.pi0_phase_state
    os.environ.setdefault("HF_HOME", str(Path.home() / ".cache" / "huggingface"))
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    if args.output_jsonl.exists():
        args.output_jsonl.unlink()
    if args.log_steps_jsonl:
        args.log_steps_jsonl.parent.mkdir(parents=True, exist_ok=True)
        if args.log_steps_jsonl.exists():
            args.log_steps_jsonl.unlink()

    seeds = parse_seed_list(args.seeds)
    if not seeds:
        seeds = list(range(int(args.seed_start), int(args.seed_start) + int(args.episodes)))
    finisher_phase_schedules = load_finisher_phase_schedules(args, seeds)

    prefix_policy = make_pi0_policy_for_dataset(
        args.device,
        args.prefix_policy_path,
        args.prefix_dataset_repo_id,
        args.prefix_dataset_root,
    ) if str(args.prefix_source) == "policy" else None
    if str(args.prefix_source) == "policy" and (
        args.prefix_policy_path is None or args.prefix_dataset_repo_id is None or args.prefix_dataset_root is None
    ):
        raise ValueError("Policy prefix mode requires --prefix-policy-path, --prefix-dataset-repo-id, and --prefix-dataset-root")
    finisher_policy = make_pi0_policy_for_dataset(
        args.device,
        args.finisher_policy_path,
        args.finisher_dataset_repo_id,
        args.finisher_dataset_root,
    )

    results = []
    env = None
    try:
        for seed in seeds:
            from mujoco_env.y_env2 import SimpleEnv2

            if env is None or bool(args.fresh_env_per_episode):
                if env is not None:
                    close_env(env)
                env = SimpleEnv2("./asset/example_scene_y2.xml", action_type="joint_angle")
            row = run_episode(
                env,
                seed,
                prefix_policy,
                finisher_policy,
                args,
                finisher_scheduled_states=finisher_phase_schedules.get(int(seed)),
            )
            results.append(row)
            with args.output_jsonl.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
            print(json.dumps(row, ensure_ascii=False), flush=True)
    finally:
        if env is not None:
            close_env(env)
        del prefix_policy
        del finisher_policy
        gc.collect()

    success_count = sum(1 for row in results if row["success"])
    physical_success_count = sum(1 for row in results if row["physical_success"])
    summary = {
        "policy": "pi0_two_stage_eef_abs",
        "episodes": len(results),
        "success_count": success_count,
        "physical_success_count": physical_success_count,
        "success_rate": success_count / max(len(results), 1),
        "physical_success_rate": physical_success_count / max(len(results), 1),
        "seeds": seeds,
        "prefix_steps": int(args.prefix_steps),
        "prefix_source": str(args.prefix_source),
        "max_action_steps": int(args.max_action_steps),
        "prefix_policy_path": str(args.prefix_policy_path) if args.prefix_policy_path is not None else None,
        "finisher_policy_path": str(args.finisher_policy_path),
        "prefix_dataset_repo_id": str(args.prefix_dataset_repo_id) if args.prefix_dataset_repo_id is not None else None,
        "finisher_dataset_repo_id": str(args.finisher_dataset_repo_id),
        "prefix_phase_state": str(args.prefix_phase_state),
        "finisher_phase_state": str(args.finisher_phase_state),
        "finisher_scripted_gripper": bool(args.finisher_scripted_gripper),
        "physical_min_lift": float(args.physical_min_lift),
        "physical_min_lift_steps": int(args.physical_min_lift_steps),
        "physical_final_upright_cos": float(args.physical_final_upright_cos),
    }
    args.summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print("SUMMARY " + json.dumps(summary, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
