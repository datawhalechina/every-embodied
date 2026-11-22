"""Summarize candidate end-effector poses using local robot kinematics.

The target is the current tracked point. These are kinematic target estimates,
not predicted physical object motion, collisions, contact or grasp success.
"""
import math

from bridge import state_from_proprio
from g05_chunk_20261106 import project_step


def current_eef_geometry(binding, arm, target):
    """Return current end-effector position and kinematic state in the robot base frame."""
    from tools.util.geometry.grasp_kinematics_local import LocalRobotState, eef_pose

    q = binding['proprioception']
    parts = state_from_proprio(q)
    robot = {'base_pose': {'pos': [0., 0., 0.], 'quat': [0., 0., 0., 1.]},
             'trunk_qpos': parts['trunk_qpos'],
             'arm_left_qpos': q[3:11], 'arm_right_qpos': q[30:38],
             'gripper_left_qpos': parts['left_gripper'], 'gripper_right_qpos': parts['right_gripper']}
    state = LocalRobotState.from_capture(robot, arm_dof=8)
    initial, _ = eef_pose(state, arm, robot['arm_' + arm + '_qpos'])
    return initial, state, robot


def current_target_distance(binding, arm, target):
    """Measure the current end-effector distance to a tracked robot-frame target."""
    initial, _, _ = current_eef_geometry(binding, arm, target)
    return math.dist(initial, target)


def summarize_geometry(proposal, arrays, binding, stage, limit=8):
    from tools.util.geometry.grasp_kinematics_local import LocalRobotState, eef_pose

    arm = stage['arm']
    tracking = stage['entry']['tracking']
    if tracking['binding']['episode_id'] != binding['episode_id'] or tracking['binding']['sequence'] != binding['sequence']:
        raise ValueError('Geometry target is not from the candidate observation')
    target = tracking['position_m']
    initial, state, robot = current_eef_geometry(binding, arm, target)
    steps = []
    for index in range(min(limit, len(arrays['left_arm']))):
        try:
            step, _ = project_step(proposal, arrays, index, binding)
            position, _ = eef_pose(state, arm, step['action'][arm + '_arm'] + [0.0])
            steps.append({'index': index, 'eef_position_m': position.tolist(),
                          'distance_to_current_target_m': math.dist(position, target)})
        except ValueError as error:
            steps.append({'index': index, 'unavailable': str(error)})
    # A later independently valid estimate cannot extend a prefix that has
    # already failed its initial-state projection.
    available = []
    for step in steps:
        if 'eef_position_m' not in step:
            break
        available.append(step)
    initial_distance = math.dist(initial, target)
    final = available[-1] if available else None
    return {'frame': 'robot_base', 'target_track_id': tracking['track_id'],
            'current_target_position_m': target, 'initial_eef_position_m': initial.tolist(),
            'initial_distance_m': initial_distance, 'steps': steps,
            'initial_state_valid_prefix_steps': len(available),
            'last_valid_step_index': final['index'] if final else None,
            'net_eef_displacement_m': math.dist(initial, final['eef_position_m']) if final else None,
            'target_distance_reduction_m': (initial_distance-final['distance_to_current_target_m']) if final else None,
            'meaning': 'initial-state-valid prefix kinematics with current target fixed; not physical object predictions'}
