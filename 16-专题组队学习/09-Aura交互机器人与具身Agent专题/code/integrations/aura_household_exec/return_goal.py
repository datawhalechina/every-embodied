"""Evaluator-only return acceptance. The policy receives a boolean, no poses."""
import json,math
from pathlib import Path

def goal_status(runtime):
    runtime=Path(runtime).resolve();status=json.loads((runtime/'driver-status.json').read_text())
    frame=json.loads((runtime/'captures/hri-full/frames.jsonl').read_text().splitlines()[-1])
    metadata=json.loads((runtime/'latest-native-checkpoint.json').read_text());source=Path(metadata['path']).resolve()
    if not source.is_relative_to(runtime/'native-checkpoints'):raise ValueError('Native snapshot escapes current runtime')
    objects=json.loads(source.read_text())['state']['registry']['object_registry']
    robot=frame['robot_world_xyz_observer_only'];aura=objects['aura_interface']['root_link']['pos'];visitor=objects['aura_visitor']['root_link']['pos']
    if not all(math.isfinite(v) for p in [robot,aura,visitor] for v in p):raise ValueError('Non-finite return geometry')
    da=math.dist(robot[:2],aura[:2]);dh=math.dist(robot[:2],visitor[:2])
    fresh=status.get('connected') is True and 0<=status['ticks']-frame['wrapper_tick']<=3
    return {'passed':fresh and 1.0<=da<=2.4 and dh>=1.0,'distance_to_aura_m':da,'distance_to_visitor_m':dh,
        'episode_id':status['episode_id'],'native_tick':status['ticks'],'fresh':fresh,
        'source':'evaluator camera-row robot world pose; scene root positions only for static Aura/visitor; policy receives boolean only'}
