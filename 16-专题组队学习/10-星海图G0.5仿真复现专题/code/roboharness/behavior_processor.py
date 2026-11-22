"""Preserve transform-derived BEHAVIOR state through grouped model padding.

The raw schema has separate trunk_qpos and base_qvel fields. The official
per-key transform merges them into lower_body before normalization. The
merger's inverse must restore that merged state before the official inverse
transform can add current torso positions to relative action predictions.
"""
from copy import deepcopy

from g05.data_processor.transforms.action_state_merger import GroupedPaddingMerger


class BehaviorGroupedPaddingMerger(GroupedPaddingMerger):
    """Use transformed state metadata for the inverse grouped tensor split."""

    def set_shape_meta(self, shape_meta):
        derived = deepcopy(shape_meta)
        keys = {entry['key'] for entry in derived['state']}
        if not {'trunk_qpos', 'base_qvel'} <= keys or 'lower_body' in keys:
            raise ValueError('BEHAVIOR merger requires separate raw trunk and base state')
        derived['state'] = [entry for entry in derived['state']
                            if entry['key'] not in {'trunk_qpos', 'base_qvel'}]
        derived['state'].append({'key': 'lower_body', 'raw_shape': 7, 'shape': 7})
        super().set_shape_meta(derived)


def processor_configuration(base_config, camera_shapes):
    """Build a candidate processor using actual reset camera dimensions.

    Base model architecture, history, image transforms, grouped layout, and
    normalization modes come from the saved checkpoint configuration. This
    config selects the published BEHAVIOR transform and statistics family.
    Physical qualification and actual model-output checks are separate.
    """
    if set(camera_shapes) != {'head_rgb', 'left_wrist_rgb', 'right_wrist_rgb'}:
        raise ValueError('three actual camera dimensions are required')
    source = deepcopy(base_config['data']['processors']['galaxea_r1pro'])
    shared = deepcopy(base_config['model']['processor'])
    source.update(shared)
    actions = [('left_arm', 7), ('left_gripper', 1), ('right_arm', 7),
               ('right_gripper', 1), ('lower_body', 7)]
    states = [('left_arm', 7), ('left_gripper', 2), ('right_arm', 7),
              ('right_gripper', 2), ('trunk_qpos', 4), ('base_qvel', 3)]
    images = deepcopy(base_config['data']['processors']['galaxea_r1pro']['shape_meta']['images'])
    for entry in images:
        shape = list(camera_shapes[entry['key']])
        if len(shape) != 3 or shape[0] != 3 or any(isinstance(x, bool) or not isinstance(x, int) or x <= 0 for x in shape):
            raise ValueError('camera dimensions must be integer CHW RGB shapes')
        entry['raw_shape'] = shape
        entry.pop('shape', None)
    source['shape_meta'] = {
        'images': images,
        'action': [{'key': key, 'raw_shape': width, 'shape': width} for key, width in actions],
        'state': [{'key': key, 'raw_shape': width, 'shape': 1 if 'gripper' in key else width}
                  for key, width in states]}
    source['embodiment_type'] = 'behavior_r1pro'
    source['action_state_transforms'] = [{'_target_': 'g05.data_processor.transforms.relative_action.BehaviorPerKeyTransform'}]
    source['action_state_merger']['_target_'] = 'behavior_processor.BehaviorGroupedPaddingMerger'
    source['action_filter'] = {'_target_': 'g05.data_processor.transforms.action_filter.BaseActionFilter'}
    # These raw fields are consumed by the transform before normalization.
    source.setdefault('norm_exception_mode', {}).setdefault('state', {}).update(
        {'trunk_qpos': 'dummy', 'base_qvel': 'dummy'})
    source['samples_builder'] = {'_target_': 'g05.data_processor.processor.samples_builder.BBoxSubtaskCoTBuilder',
                                 '_partial_': True}
    return source
