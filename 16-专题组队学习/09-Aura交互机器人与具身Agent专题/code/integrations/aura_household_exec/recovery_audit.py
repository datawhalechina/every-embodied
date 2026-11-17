"""Compare like-for-like serialized native frames, not navigation reference poses."""
import hashlib,json,math
from pathlib import Path
from .native_checkpoint import inspect_snapshot
def compare_saved_states(source,source_hash,current,current_hash):
    a,ar=inspect_snapshot(source,source_hash);b,br=inspect_snapshot(current,current_hash)
    before=a['state']['registry']['object_registry'];after=b['state']['registry']['object_registry']
    if set(before)!=set(after):raise ValueError('Restored object identities differ')
    errors={name:math.dist(state['root_link']['pos'],after[name]['root_link']['pos'])
        for name,state in before.items() if 'pos' in state.get('root_link',{})}
    def orientation_error(a,b):
        if len(a)!=4 or len(b)!=4:raise ValueError('Invalid saved quaternion layout')
        norm=math.sqrt(sum(x*x for x in a)*sum(y*y for y in b))
        if norm<1e-12:raise ValueError('Invalid zero saved quaternion')
        return 2*math.acos(min(1.,abs(sum(x*y for x,y in zip(a,b)))/norm))
    rotations={name:orientation_error(state['root_link']['ori'],after[name]['root_link']['ori'])
        for name,state in before.items() if 'ori' in state.get('root_link',{})}
    robot=ar['robot']
    if len(before[robot]['joint_pos'])!=len(after[robot]['joint_pos']):raise ValueError('Restored joint layout differs')
    qerr=max(abs(x-y) for x,y in zip(before[robot]['joint_pos'],after[robot]['joint_pos']))
    result={'passed':max(errors.values())<.05 and max(rotations.values())<.05 and qerr<.03 and ar['held_objects']==br['held_objects'],
        'object_count':len(before),'maximum_serialized_root_position_error_m':max(errors.values()),
        'maximum_serialized_root_orientation_error_rad':max(rotations.values()),
        'robot_serialized_root_error_m':errors[robot],'joint_error_max':qerr,
        'expected_held':ar['held_objects'],'actual_held':br['held_objects'],
        'source_sha256':source_hash,'actual_sha256':current_hash,
        'position_frame':'same Scene.save root_link pose fields for both scenes',
        'errors_largest':sorted(errors.items(),key=lambda item:-item[1])[:5]}
    if not result['passed']:raise RuntimeError('Native serialized restore mismatch: '+json.dumps(result))
    return result
