"""Inspect native Scene.save files and explicitly restore a compatible own scene.

Restoring is a new simulation branch. It never replays a policy command or
reuses a plan/image/session identity. Native restore still requires runtime QA.
"""
import copy,hashlib,json,math
from pathlib import Path

def inspect_snapshot(path,expected_sha256):
    raw=Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest()!=expected_sha256:raise ValueError('Native checkpoint digest mismatch')
    def invalid(value):raise ValueError('Non-finite checkpoint JSON')
    data=json.loads(raw,parse_constant=invalid)
    def finite(node):
        if isinstance(node,float) and not math.isfinite(node):raise ValueError('Non-finite checkpoint number')
        if isinstance(node,dict):
            for value in node.values():finite(value)
        elif isinstance(node,list):
            for value in node:finite(value)
    finite(data)
    if not {'versions','metadata','state','init_info','objects_info'}<=data.keys():raise ValueError('Not a native Scene.save file')
    if data['versions']['omnigibson']['version']!='3.9.1':raise ValueError('Unsupported native checkpoint release')
    if data['init_info']['args']['scene_model']!='house_double_floor_lower':raise ValueError('Checkpoint belongs to another house')
    objects=data['state']['registry']['object_registry'];definitions=data['objects_info']['init_info']
    if set(objects)!=set(definitions):raise ValueError('Checkpoint state and scene identities differ')
    required={'aura_interface','aura_visitor','can_of_soda_113','can_of_soda_114','can_of_soda_115','trash_can_116'}
    if not required<=set(objects):raise ValueError('Incomplete shared household checkpoint')
    robots=[name for name,state in objects.items() if 'controller_groups' in state]
    if len(robots)!=1:raise ValueError('Expected exactly one controlled R1')
    robot=objects[robots[0]]
    if 'ag_obj_constraint_params' not in robot:raise ValueError('Assisted-grasp state missing')
    held={arm:row['target_obj'] for arm,row in robot['ag_obj_constraint_params'].items() if row}
    if any(name not in objects for name in held.values()):raise ValueError('Grasp refers to an absent object')
    for name,state in objects.items():
        pose=state.get('root_link',{}).get('pos')
        if pose is not None and (len(pose)!=3 or not all(isinstance(x,(float,int)) and math.isfinite(x) for x in pose)):
            raise ValueError('Invalid checkpoint root pose: '+name)
    return data,{'object_count':len(objects),'robot':robots[0],'held_objects':held,
                 'native_version':data['versions']['omnigibson'],'restore_validated':False}

def restore_native_scene(env,path,expected_sha256,*,advance=True):
    """Call before a fresh policy observation, never from an Agent tool."""
    data,report=inspect_snapshot(path,expected_sha256)
    import omnigibson as og
    from omnigibson.utils.python_utils import recursively_convert_to_torch
    expected=set(data['state']['registry']['object_registry'])
    if {o.name for o in env.scene.objects}!=expected:raise ValueError('Current native scene identities differ')
    if not og.sim.is_playing():raise RuntimeError('Native restore requires an initialized playing simulator')
    env.scene.load_state(recursively_convert_to_torch(copy.deepcopy(data['state'])),serialized=False)
    if advance:og.sim.step() # A recovery wrapper may supply a pinned maintenance step instead.
    after=env.scene.dump_state(serialized=False)['registry']['object_registry']
    robot=report['robot'];saved=data['state']['registry']['object_registry'][robot]
    q=after[robot]['joint_pos'].detach().cpu().tolist()
    if len(q)!=len(saved['joint_pos']):raise RuntimeError('Native joint layout changed')
    error=float(max(abs(a-b) for a,b in zip(q,saved['joint_pos'])))
    held={arm:row['target_obj'] for arm,row in after[robot]['ag_obj_constraint_params'].items() if row}
    if error>.03 or held!=report['held_objects']:raise RuntimeError('Native joint/grasp restoration did not match checkpoint')
    return {**report,'native_state_loaded':True,'joint_coordinate_error_max':error,
            'joint_units':'native per-joint positions: metres for prismatic, radians for revolute','restored_grasp_targets':held,
            'native_predicates_recomputed':False,'new_camera_session_required':True,
            'policy_commands_replayed':0,'restore_validated':False}
