"""Build causal BEHAVIOR policy inputs from allowed, measured observations.

Callers provide native control-step indices and RGB arrays after the official
camera-role mapping. Startup history repeats the earliest measured frame with
padding flags. No simulator truth or unobserved robot state is synthesized.
"""
import bisect
import math

import numpy as np

from bridge import state_from_proprio,vector

CAMERAS = ('head_rgb', 'left_wrist_rgb', 'right_wrist_rgb')


def integer(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(name + ' must be a nonnegative integer')
    return value


def validate(snapshot,*,policy_state=True):
    """Validate one episode-bound observation without importing a simulator."""
    if not isinstance(snapshot.get('episode_id'), str) or not snapshot['episode_id']:
        raise ValueError('episode_id is required')
    integer(snapshot['sequence'], 'sequence')
    integer(snapshot['control_step'], 'control_step')
    measured=vector(snapshot['proprio'],65,'measured proprioception')
    state = state_from_proprio(measured) if policy_state else None
    if set(snapshot['images']) != set(CAMERAS):
        raise ValueError('all three measured RGB cameras are required')
    images = {}
    for camera in CAMERAS:
        array = np.asarray(snapshot['images'][camera])
        if array.dtype != np.uint8 or array.ndim != 3 or array.shape[-1] != 3 or min(array.shape[:2]) < 1:
            raise ValueError(camera + ' requires a nonempty uint8 HWC RGB array')
        images[camera] = array.copy()
    return state, images


def build_input(history, task, *, control_hz=30, stride_seconds=1.0,
                history_count=6, action_horizon=32, max_history_lag_steps=1):
    """Select six causal frames and produce the official processor dictionary.

    Sampling uses simulated control steps, never wall time spent awaiting a
    model. Missing interior history is rejected. Only startup before the first
    recorded step can be padded. Returned binding identifies the latest input.
    This function performs no model inference and does not execute actions.
    """
    import torch
    if not history or not isinstance(task, str) or not task.strip():
        raise ValueError('measured history and a task instruction are required')
    if isinstance(control_hz, bool) or not isinstance(control_hz, (int, float)) or not math.isfinite(control_hz) or control_hz <= 0:
        raise ValueError('control_hz must be positive and finite')
    if isinstance(stride_seconds, bool) or not isinstance(stride_seconds, (int, float)) or not math.isfinite(stride_seconds) or stride_seconds <= 0:
        raise ValueError('stride_seconds must be positive and finite')
    if integer(history_count, 'history_count') < 1 or integer(action_horizon, 'action_horizon') < 1:
        raise ValueError('positive history and action horizons are required')
    integer(max_history_lag_steps, 'max_history_lag_steps')
    stride = round(control_hz * stride_seconds)
    if stride < 1 or abs(stride - control_hz * stride_seconds) > 1e-6:
        raise ValueError('history stride must contain an integral number of native steps')
    episode = history[-1]['episode_id']
    steps, sequences, validated = [], [], []
    for snapshot in history:
        if snapshot['episode_id'] != episode:
            raise ValueError('history crosses an episode reset')
        validated.append(validate(snapshot))
        steps.append(snapshot['control_step'])
        sequences.append(snapshot['sequence'])
    if any(b <= a for a, b in zip(steps, steps[1:])) or any(b <= a for a, b in zip(sequences, sequences[1:])):
        raise ValueError('history must advance in both physical steps and observation sequence')
    indices, padding = [], []
    for offset in range(history_count - 1, -1, -1):
        target = steps[-1] - offset * stride
        index = bisect.bisect_right(steps, target) - 1
        is_pad = index < 0
        if is_pad:
            index = 0
            if steps[0] != 0:
                raise ValueError('startup padding requires the measured reset observation')
        elif target - steps[index] > max_history_lag_steps:
            raise ValueError('missing interior history observation')
        indices.append(index)
        padding.append(is_pad)
    states = {key: torch.tensor([validated[i][0][key] for i in indices], dtype=torch.float32)
              for key in validated[-1][0]}
    images = {camera: torch.from_numpy(np.stack([validated[i][1][camera].transpose(2, 0, 1)
                                               for i in indices])) for camera in CAMERAS}
    dimensions = {'left_arm': 7, 'left_gripper': 1, 'right_arm': 7,
                  'right_gripper': 1, 'lower_body': 7}
    data = {'images': images, 'state': states, 'task': task, 'embodiment': 'behavior_r1pro',
            'frequency': float(control_hz), 'idx': 0,
            'action': {key: torch.zeros(action_horizon, width) for key, width in dimensions.items()},
            'action_is_pad': torch.ones(action_horizon, dtype=torch.bool),
            'state_is_pad': torch.tensor(padding, dtype=torch.bool),
            'image_is_pad': torch.tensor(padding, dtype=torch.bool)}
    binding = {'episode_id': episode, 'sequence': sequences[-1], 'control_step': steps[-1],
               'history_steps': [steps[i] for i in indices], 'history_is_pad': padding,
               'native_control_hz': float(control_hz), 'history_stride_steps': stride}
    return data, binding
