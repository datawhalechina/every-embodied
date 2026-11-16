"""Evaluate PI 0.5 policies on DexJoCo simulation environments."""

import multiprocessing as mp
import os
import random
import signal
import time
from collections import deque
from dataclasses import dataclass
from multiprocessing.synchronize import Event as MpEvent
from pathlib import Path
from queue import Empty
from typing import Literal

import imageio
import numpy as np
import yaml
from openpi_client import websocket_client_policy
from scipy.spatial.transform import Rotation as R

from .dexjoco_openpi_env import DexJoCoOpenPIEnv


@dataclass
class Observation:
    obs: dict
    timestamp: int


@dataclass
class Action:
    action: np.ndarray
    timestamp: int


ActionChunk = Action


def get_latest(q: mp.Queue):
    """Return the newest queued item and discard older buffered items."""
    latest = None
    try:
        while True:
            latest = q.get_nowait()
    except Empty:
        pass
    return latest


def _set_seed(seed: int):
    np.random.seed(seed)
    # torch.manual_seed(seed)
    random.seed(seed)
    # torch.cuda.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False


def _interp_rotvec_geodesic(
    rotvec0: np.ndarray, rotvec1: np.ndarray, t: float
) -> np.ndarray:
    """Interpolate rotation vectors on SO(3) instead of component-wise lerp."""
    if t <= 0.0:
        return rotvec0.copy()
    if t >= 1.0:
        return rotvec1.copy()

    r0 = R.from_rotvec(rotvec0)
    r1 = R.from_rotvec(rotvec1)
    relative_rotvec = (r0.inv() * r1).as_rotvec()
    return (r0 * R.from_rotvec(relative_rotvec * t)).as_rotvec()


def _interp_single_arm_action(
    old_action: np.ndarray, new_action: np.ndarray, t: float
) -> np.ndarray:
    """Interpolate single-arm action [xyz, rotvec, hand]."""
    interp_action = (1.0 - t) * old_action + t * new_action
    rotvec_slice = slice(3, 6)
    interp_action[rotvec_slice] = _interp_rotvec_geodesic(
        old_action[rotvec_slice], new_action[rotvec_slice], t
    ).astype(interp_action.dtype, copy=False)
    return interp_action


def _interp_dual_arm_action(
    old_action: np.ndarray, new_action: np.ndarray, t: float
) -> np.ndarray:
    """Interpolate dual-arm action [r_xyz, r_rotvec, r_hand, l_xyz, l_rotvec, l_hand]."""
    interp_action = (1.0 - t) * old_action + t * new_action
    right_rotvec_slice = slice(3, 6)
    left_rotvec_slice = slice(25, 28)
    interp_action[right_rotvec_slice] = _interp_rotvec_geodesic(
        old_action[right_rotvec_slice], new_action[right_rotvec_slice], t
    ).astype(interp_action.dtype, copy=False)
    interp_action[left_rotvec_slice] = _interp_rotvec_geodesic(
        old_action[left_rotvec_slice], new_action[left_rotvec_slice], t
    ).astype(interp_action.dtype, copy=False)
    return interp_action


def inference_process(
    obs_queue: mp.Queue,
    action_queue: mp.Queue,
    stop_event: MpEvent,
    port: int,
    inferencing_event: MpEvent,
    seed: int,
    host: str,
):
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    _set_seed(seed)

    # Inference worker: receive observations and query the OpenPI policy server.
    client = websocket_client_policy.WebsocketClientPolicy(host=host, port=port)

    while not stop_event.is_set():
        obs: Observation | None = get_latest(obs_queue)
        if obs is None:
            stop_event.wait(0.01)
            continue

        result = client.infer(obs.obs)
        action_chunk = result["actions"]

        action_queue.put(ActionChunk(action=action_chunk, timestamp=obs.timestamp))
        inferencing_event.clear()


def receive_actions(
    action_queue: mp.Queue,
    actions_buffer: deque,
    now_timestamp: int,
    dual_arm: bool,
):
    """Receive action chunks and merge them into a timestamped action buffer.

    now_timestamp has not been executed yet.
    """
    interp_action_fn = (
        _interp_dual_arm_action if dual_arm else _interp_single_arm_action
    )

    # Drop expired actions that are older than the current timestamp.
    while actions_buffer and actions_buffer[0].timestamp < now_timestamp:
        actions_buffer.popleft()

    while True:
        try:
            action_chunk: ActionChunk = action_queue.get_nowait()

            # Chunk timestamp comes from observation, so it should not exceed now_timestamp.
            assert action_chunk.timestamp <= now_timestamp

            # All timestamp ranges below use half-open intervals: [start, end).
            action_chunk_timestamp_range = (
                now_timestamp,
                action_chunk.timestamp + action_chunk.action.shape[0],
            )
            if action_chunk_timestamp_range[1] <= now_timestamp:
                continue

            action = action_chunk.action[
                (action_chunk_timestamp_range[0] - action_chunk.timestamp) : (
                    action_chunk_timestamp_range[1] - action_chunk.timestamp
                )
            ]

            if actions_buffer:
                buffer_timestamp_range = (
                    actions_buffer[0].timestamp,
                    actions_buffer[-1].timestamp + 1,
                )
                assert buffer_timestamp_range[1] - buffer_timestamp_range[0] == len(
                    actions_buffer
                ), "Buffer timestamps must be continuous"
            else:
                buffer_timestamp_range = (now_timestamp, now_timestamp)

            # Blend overlapping actions already in buffer.
            overlap_range = (
                max(action_chunk_timestamp_range[0], buffer_timestamp_range[0]),
                min(action_chunk_timestamp_range[1], buffer_timestamp_range[1]),
            )
            overlap_len = overlap_range[1] - overlap_range[0]
            for ts in range(overlap_range[0], overlap_range[1]):
                buffer_idx = ts - buffer_timestamp_range[0]
                action_idx = ts - action_chunk_timestamp_range[0]

                # Keep interpolation away from 0/1 endpoints for smoother transitions.
                interp_t = (ts - overlap_range[0] + 1) / (overlap_len + 1)

                interp_action = interp_action_fn(
                    actions_buffer[buffer_idx].action,
                    action[action_idx],
                    interp_t,
                )
                actions_buffer[buffer_idx] = Action(action=interp_action, timestamp=ts)

            # Append non-overlapping tail actions.
            non_overlap_timestamp_range = (
                buffer_timestamp_range[1],
                action_chunk_timestamp_range[1],
            )
            for ts in range(
                non_overlap_timestamp_range[0], non_overlap_timestamp_range[1]
            ):
                action_idx = ts - action_chunk_timestamp_range[0]
                actions_buffer.append(Action(action=action[action_idx], timestamp=ts))
        except Empty:
            break


def _append_video_frames(video_writers: dict, raw_images: dict):
    for cam_name, writer in video_writers.items():
        writer.append_data(raw_images[cam_name])


def main(
    config: Path,
    seed: int = 0,
    rand_full: bool = False,
    randomize_dynamics: bool = False,
    port: int = 8000,
    host: str = "0.0.0.0",
    output: Path | None = None,
    render_mode: Literal["rgb_array", "human"] = "rgb_array",
    replan_ratio: float = 0.8,
    episodes: int = 50,
    pad_state_dim46: bool = False,
    record_pressed_digits: bool | None = None,
):
    if render_mode == "rgb_array":
        os.environ.setdefault("MUJOCO_GL", "egl")
    else:
        os.environ.setdefault("MUJOCO_GL", "glfw")
    _set_seed(seed)

    # Load evaluation configuration.
    with open(config, "r") as f:
        cfg = yaml.safe_load(f)

    env_name = cfg["env_name"]
    camera_mapping = cfg["camera_mapping"]
    robot_type = cfg["robot_type"]
    dual_arm = robot_type == "dual_arm"
    prompt = cfg["prompt"]
    action_horizon = 30  # the policy trained on

    # Record password input only for iPad tasks unless explicitly configured.
    if record_pressed_digits is None:
        record_pressed_digits = env_name == "bimanual_unlock_ipad"

    # Write episode videos under a temporary name before assigning the result suffix.
    if output is None:
        output_dir = (
            Path("outputs")
            / f"{env_name}{'_rand_full' if rand_full else ''}_seed{seed}"
        )
    else:
        output_dir = output
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create the DexJoCo environment wrapper used by the OpenPI policy.
    env = DexJoCoOpenPIEnv(
        env_name=env_name,
        camera_mapping=camera_mapping,
        seed=seed,
        rand_full=rand_full,
        randomize_dynamics=randomize_dynamics,
        dual_arm=dual_arm,
        prompt=prompt,
        render_mode=render_mode,
        pad_state_dim46=pad_state_dim46,
        password=cfg.get("password", None),  # Pass password from config if available
    )
    env.start()

    # Queues connect the control loop with the asynchronous inference worker.
    obs_queue = mp.Queue()
    action_queue = mp.Queue()
    stop_event = mp.Event()
    inferencing_event = mp.Event()

    inference_proc = mp.Process(
        target=inference_process,
        args=(obs_queue, action_queue, stop_event, port, inferencing_event, seed, host),
    )
    video_writers = None

    try:
        inference_proc.start()
        num_success = 0

        for ep in range(episodes):
            print(f"Episode {ep + 1}/{episodes}")

            # Setup video writers in a temporary episode directory.
            video_dir = output_dir / f"episode_{ep:02d}_temp"
            video_dir.mkdir(parents=True, exist_ok=True)
            video_writers = {
                cam_name: imageio.get_writer(video_dir / f"{cam_name}.mp4", fps=30)
                for cam_name in camera_mapping.values()
            }

            env.reset()

            timestamp = 0
            actions_buffer = deque()

            if env_name == "click_mouse":
                # Align with dataset.
                for _ in range(30):
                    # fmt: off
                    env.step(
                        action=np.array([
                            -4.4294e-01, 1.3729e-06, 1.5170e00,
                            -3.14156462e00, -6.91584035e-05, -1.40317984e-03,
                            0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                            0, 0, 0.263, 0, 0, 0
                        ])
                    )
                    # fmt: on

            # Send the first observation and mark inference as active before enqueueing it.
            inferencing_event.set()
            obs_queue.put(Observation(env.get_obs(), timestamp))

            # Save the reset frame.
            raw_images = env.get_raw_images()
            _append_video_frames(video_writers, raw_images)

            in_stay_state = (
                False  # Track whether the previous step already used stay().
            )
            password = []

            # Episode loop.
            while True:
                receive_actions(action_queue, actions_buffer, timestamp, dual_arm)

                # Execute the scheduled action for this timestamp, or hold the pose.
                if actions_buffer:
                    assert actions_buffer[0].timestamp == timestamp, (
                        "Buffer head timestamp must match current timestamp"
                    )
                    action = actions_buffer.popleft().action
                    pressed_digits = env.step(action)
                    in_stay_state = False
                else:
                    pressed_digits = env.stay(continue_stay=in_stay_state)
                    in_stay_state = True

                if record_pressed_digits and pressed_digits:
                    password.append(pressed_digits)

                timestamp += 1

                raw_images = env.get_raw_images()
                _append_video_frames(video_writers, raw_images)

                # Request a replan when the buffered action horizon is below threshold
                # and no observation, inference request, or action chunk is pending.
                should_send_obs = (
                    len(actions_buffer) < replan_ratio * action_horizon
                    and obs_queue.empty()  # no observation waiting for inferencing
                    and not inferencing_event.is_set()  # no inferencing observations
                    and action_queue.empty()  # no actions to append to the buffer
                )

                if should_send_obs:
                    inferencing_event.set()
                    obs_queue.put(Observation(env.get_obs(), timestamp))
                    # inferencing_event is cleared in the inference process after inference finishes.

                # Stop after the environment reports terminal state.
                if env.is_done:
                    if env.is_success:
                        num_success += 1
                        print("Success!")
                    else:
                        print("Failed")
                    break

            for writer in video_writers.values():
                writer.close()
            video_writers = None

            # Rename the temporary episode directory with the final result label.
            result_suffix = "success" if env.is_success else "failure"
            if record_pressed_digits:
                if password:
                    password_suffix = "_".join(
                        "".join(str(digit) for digit in digits) for digits in password
                    )
                else:
                    password_suffix = "no_password_input"
                final_video_dir = (
                    output_dir / f"episode_{ep:02d}_{result_suffix}_{password_suffix}"
                )
            else:
                final_video_dir = output_dir / f"episode_{ep:02d}_{result_suffix}"
            video_dir.rename(final_video_dir)

            # Drain in-flight work before starting the next episode.
            while True:
                try:
                    obs_queue.get_nowait()
                except Empty:
                    break
            while inferencing_event.is_set():
                time.sleep(0.1)
            while not action_queue.empty():
                action_queue.get()

        print(
            f"\nSuccess rate: {num_success}/{episodes} ({100 * num_success / episodes:.1f}%)"
        )
        (output_dir / f"success_rate_{num_success}_{episodes}.txt").touch()

    finally:
        # Shut down worker and release multiprocessing resources.
        stop_event.set()
        inference_proc.join(timeout=2)
        if inference_proc.is_alive():
            inference_proc.terminate()
            inference_proc.join(timeout=2)
        obs_queue.cancel_join_thread()
        obs_queue.close()
        action_queue.cancel_join_thread()
        action_queue.close()
        env.close()
        if video_writers is not None:
            for writer in video_writers.values():
                writer.close()
