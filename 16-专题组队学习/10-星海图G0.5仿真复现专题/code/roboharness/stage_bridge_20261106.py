"""Record measured stage entry, handoff checks and reusable transition evidence.

The experience store is experiment data, not a user-memory service. Retrieved
transitions are suggestions; the caller must recheck the current observations.
"""
import hashlib
import json
import math
from pathlib import Path
import re

from proposal_exchange import encoded, exclusive, sha
from candidate_geometry_20261106 import current_target_distance
from tracked_progress_20261106 import _track
from workspace_agent_host import atomic


SKILLS = {'pick-up-object', 'open-doors-and-drawers', 'place-object-in-container'}
NEXT_SKILLS = SKILLS | {'behavior-v2-baseline'}
KEYS = ('episode_id', 'session_id', 'sequence', 'head_image_id')


def _binding(value):
    return {key: value[key] for key in KEYS}


def _bounded_text(value, name, limit=512):
    if not isinstance(value, str) or not value.strip() or len(value) > limit:
        raise ValueError(name + ' must be nonempty bounded text')
    return value.strip()


def conditioned_instruction(task, target, arm, goal):
    """Bind a local model request to the selected target and manipulating arm."""
    prefix = (f'Pick up {target} with the {arm} arm. Subtask: ' if goal['type']=='grasp_stow'
              else f'Open the drawer or door using {target} with the {arm} arm. Subtask: ')
    return prefix + task[:8192-len(prefix)]


def _call(host, call_id):
    if not isinstance(call_id, str) or not re.fullmatch(r'[A-Za-z0-9_.-]{1,180}', call_id):
        raise ValueError('A valid source call identifier is required')
    path = host.evidence / 'calls' / (call_id + '.json')
    record = json.loads(path.read_bytes())
    if record.get('status') != 'completed':
        raise ValueError('Source call did not complete')
    receipt = record.get('receipt') or {}
    response = receipt.get('response') or {}
    if receipt.get('error') or response.get('error'):
        raise ValueError('Source call has an error')
    return record, path


def _read_tracking(host, call_id, binding, point):
    record, path = _call(host, call_id)
    tracking = (record.get('receipt') or {}).get('persistent_tracking')
    item, xyz = _track(binding, tracking, point, 0.5)
    return {'call_id': call_id, 'receipt_sha256': sha(path), 'point': point,
            'track_id': item['track_id'], 'position_m': xyz,
            'confidence': item['confidence'], 'binding': _binding(binding)}


def _active_skill(host, *, allow_unset=False):
    skill = json.loads((host.workspace / 'skills/active.json').read_bytes())
    active, revision = skill.get('active_skill'), skill.get('revision')
    if type(revision) is not int or revision < 0:
        raise ValueError('A valid workspace skill revision is required')
    if active is None and allow_unset:
        return None, revision
    if active not in SKILLS or revision < 1:
        raise ValueError('A supported active skill revision is required')
    return active, revision


class StageBridge:
    """Own one bounded stage contract and append only verified transition records."""

    def __init__(self, host, experience_root=None, experience_digest=None):
        self.host = host
        self.root = host.evidence / 'stage-bridge'
        self.root.mkdir(parents=True, exist_ok=True)
        self.experience = self.root / 'experience'
        self.experience.mkdir(parents=True, exist_ok=True)
        self.reference = Path(experience_root) if experience_root else None
        self.reference_digest = experience_digest
        if self.reference is not None:
            if (not self.reference.is_dir() or not isinstance(experience_digest, str)
                    or self.reference.resolve() == self.experience.resolve()):
                raise ValueError('A frozen transition reference and its digest are required')
            self._check_reference()
        elif experience_digest is not None:
            raise ValueError('An experience digest requires a reference directory')

    def _check_reference(self):
        files = sorted(self.reference.glob('*.json'))
        contents = [{'name':path.name, 'sha256':sha(path)} for path in files]
        digest = hashlib.sha256(encoded(contents)).hexdigest()
        if digest != self.reference_digest:
            raise ValueError('Frozen transition reference changed')
        return digest

    def _active(self):
        path = self.root / 'active.json'
        return json.loads(path.read_bytes()) if path.exists() else None

    def _save(self, row):
        atomic(self.root / 'active.json', row)

    def record_reach(self, call_id, binding):
        """Remember the measured post-approach state for later target reacquisition."""
        record, path = _call(self.host, call_id)
        if record.get('tool') != 'move_to_reach_point':
            raise ValueError('A reach receipt is required')
        q = binding.get('proprioception')
        if (not isinstance(q, list) or len(q) != 65 or
                any(type(v) not in (int, float) or not math.isfinite(v) for v in q)):
            raise ValueError('A complete measured robot state is required')
        anchor = {'call_id': call_id, 'binding': _binding(binding), 'proprioception': q,
                  'reach_receipt_sha256': sha(path)}
        destination = self.root / ('reach-' + hashlib.sha256(call_id.encode()).hexdigest() + '.json')
        try:
            exclusive(destination, encoded(anchor))
        except FileExistsError:
            if destination.read_bytes() != encoded(anchor):
                raise ValueError('Reach observation changed after it was recorded')
        return anchor

    def plan(self, call_id, args, binding):
        expected = {'task', 'target_label', 'point', 'arm', 'next_skill', 'next_target_label', 'goal', 'seeds', 'maximum_steps', 'gripper_policy'}
        if not isinstance(args, dict) or set(args) not in (expected, expected | {'max_eef_distance_m'}):
            raise ValueError('A complete stage contract is required')
        max_eef_distance = args.get('max_eef_distance_m', 0.25)
        if (type(max_eef_distance) not in (int, float) or not math.isfinite(max_eef_distance)
                or not 0.05 <= max_eef_distance <= 1.0):
            raise ValueError('End-effector entry distance must be between 0.05 and 1.0 m')
        task = _bounded_text(args['task'], 'task', 8192)
        target = _bounded_text(args['target_label'], 'target label')
        next_target = _bounded_text(args['next_target_label'], 'next target label')
        point = _bounded_text(args['point'], 'point name')
        if args['arm'] not in {'left', 'right'} or args['next_skill'] not in NEXT_SKILLS:
            raise ValueError('A supported arm and next skill are required')
        if args['gripper_policy'] not in {'policy', 'hold'}:
            raise ValueError('Gripper policy must be policy or hold')
        workspace_skill, revision = _active_skill(self.host, allow_unset=True)
        goal = args['goal']
        if not isinstance(goal, dict) or goal.get('type') not in {'grasp_stow', 'tracked_displacement'}:
            raise ValueError('Supported goals are grasp_stow and tracked_displacement')
        skill = workspace_skill
        if skill is None:
            skill = 'pick-up-object' if goal['type'] == 'grasp_stow' else 'open-doors-and-drawers'
        if goal['type'] == 'grasp_stow':
            if set(goal) != {'type'} or skill != 'pick-up-object':
                raise ValueError('Grasp and stow requires the pickup skill')
        else:
            if (set(goal) not in ({'type', 'direction', 'target_m'},
                                  {'type', 'direction', 'target_m', 'stall_window', 'min_progress_m'})
                    or skill != 'open-doors-and-drawers'):
                raise ValueError('Tracked displacement requires an open-drawer skill')
            direction, distance = goal['direction'], goal['target_m']
            if (not isinstance(direction, list) or len(direction) != 3 or
                    any(type(v) not in (int, float) or not math.isfinite(v) for v in direction) or
                    sum(v*v for v in direction) < 1e-12 or type(distance) not in (int, float) or
                    not math.isfinite(distance) or not 0 < distance <= 1):
                raise ValueError('A bounded direction and displacement are required')
            if 'stall_window' in goal and (type(goal['stall_window']) is not int or not 2 <= goal['stall_window'] <= 8
                    or type(goal['min_progress_m']) not in (int, float) or not math.isfinite(goal['min_progress_m'])
                    or not 0 < goal['min_progress_m'] <= 0.1):
                raise ValueError('A bounded stall window and progress threshold are required')
        seeds = args['seeds']
        if (not isinstance(seeds, list) or not 1 <= len(seeds) <= 3
                or any(type(x) is not int or not 0 <= x < 2**32 for x in seeds)
                or len(set(seeds)) != len(seeds)):
            raise ValueError('One to three unique seeds are required')
        if type(args['maximum_steps']) is not int or not 1 <= args['maximum_steps'] <= 8:
            raise ValueError('A bounded chunk length is required')
        previous = self._active()
        if previous and previous['status'] not in {'completed', 'abandoned'}:
            raise ValueError('Finish or abandon the existing stage first')
        identity = {'call_id': call_id, 'episode_id': binding['episode_id'], 'session_id': binding['session_id']}
        identifier = 's' + hashlib.sha256(encoded(identity)).hexdigest()[:40]
        row = {'stage_id': identifier, 'status': 'planned', 'task': task,
               'model_task': conditioned_instruction(task, target, args['arm'], goal), 'target_label': target,
               'point': point, 'arm': args['arm'], 'skill': skill, 'workspace_skill': workspace_skill, 'skill_revision': revision,
               'next_skill': args['next_skill'], 'next_target_label': next_target, 'goal': goal, 'seeds': seeds,
               'maximum_steps': args['maximum_steps'], 'gripper_policy': args['gripper_policy'],
               'max_eef_distance_m': max_eef_distance,
               'planning_binding': _binding(binding)}
        exclusive(self.root / (identifier + '.plan.json'), encoded(row))
        self._save(row)
        return row

    def begin(self, binding, reach_call_id, tracking_call_id):
        row = self._active()
        if not row or row['status'] != 'planned':
            raise ValueError('A planned stage is required')
        if _active_skill(self.host, allow_unset=True) != (row.get('workspace_skill', row['skill']), row['skill_revision']):
            raise ValueError('The active skill revision changed')
        record, reach_path = _call(self.host, reach_call_id)
        receipt = record['receipt']; response = receipt.get('response') or {}
        distance = response.get('shoulder_distance_estimate') or {}
        minimum, threshold = distance.get('minimum_m'), distance.get('threshold_m')
        if (record.get('tool') != 'move_to_reach_point' or
                (receipt.get('task_skill_state') or {}).get('active_skill') != row.get('workspace_skill', row['skill']) or
                type(minimum) not in (int, float) or type(threshold) not in (int, float) or
                not 0 <= minimum <= threshold <= 2):
            raise ValueError('Current reach evidence is missing or outside bounds')
        anchor_path = self.root / ('reach-' + hashlib.sha256(reach_call_id.encode()).hexdigest() + '.json')
        drawer_grips = []
        torso_delta = None
        if anchor_path.exists():
            anchor = json.loads(anchor_path.read_bytes())
            before = anchor['binding']; old_q = anchor['proprioception']; q = binding.get('proprioception')
            if (anchor['reach_receipt_sha256'] != sha(reach_path) or
                    before['episode_id'] != binding['episode_id'] or before['session_id'] != binding['session_id'] or
                    type(binding['sequence']) is not int or binding['sequence'] < before['sequence'] or
                    not isinstance(q, list) or len(q) != 65 or
                    any(type(v) not in (int, float) or not math.isfinite(v) for v in q)):
                raise ValueError('Recorded reach does not match the current measured state')
            torso_delta = max(abs(q[i] - old_q[i]) for i in range(57, 61))
            arm_motion = False
            safe_approach = {'set_arm_to_grasp_position', 'move_tracked_point', 'exec_plan_pose',
                             'adjust_left_eef_pose_in_head_frame', 'adjust_right_eef_pose_in_head_frame',
                             'adjust_left_eef_pose_in_wrist_frame', 'adjust_right_eef_pose_in_wrist_frame'}
            # These observation tools advance the simulator with hold actions only.
            hold_observation = {'track_object_distance', 'plan_grasp_point_filter_rgbd_lite'}
            for p in (self.host.evidence / 'calls').glob('*.json'):
                call = json.loads(p.read_bytes()); b = call.get('binding') or {}
                if (b.get('episode_id') == binding['episode_id'] and type(b.get('sequence')) is int and
                        before['sequence'] <= b['sequence'] < binding['sequence'] and
                        (call.get('receipt') or {}).get('joint_commands_sent', 0)):
                    tool=call.get('tool'); call_receipt=call.get('receipt') or {}
                    response=call_receipt.get('response') or {}; arguments=call.get('arguments') or {}
                    if call.get('status') != 'completed':
                        raise ValueError('An unqualified movement occurred after the recorded reach')
                    if tool == 'close_gripper' and row['goal']['type'] == 'tracked_displacement':
                        drawer_grips.append((b['sequence'],p.stem,
                            arguments.get('arm') == row['arm'] and response.get('arm') == row['arm'] and
                            response.get('grasp_confirmed') is True and response.get('timed_out') is False and
                            not response.get('error') and not call_receipt.get('error'),'close'))
                    elif tool == 'open_gripper' and row['goal']['type'] == 'tracked_displacement':
                        if arguments.get('arm') != row['arm'] or response.get('arm') != row['arm']:
                            raise ValueError('The gripper arm differs from the planned stage')
                        drawer_grips.append((b['sequence'],p.stem,False,'open'))
                    elif tool == 'move_tracked_point' and response.get('execution') == 'plan_only_no_robot_motion':
                        continue
                    elif tool in safe_approach:
                        if tool.startswith('adjust_left_'):used_arm='left'
                        elif tool.startswith('adjust_right_'):used_arm='right'
                        else:used_arm=response.get('arm') or arguments.get('arm')
                        if used_arm != row['arm']:
                            raise ValueError('The approach arm differs from the planned stage; abandon and replan with the actual arm')
                        if (row['goal']['type']=='tracked_displacement' and tool=='set_arm_to_grasp_position'
                                and arguments.get('gripper')=='open'):
                            drawer_grips.append((b['sequence'],p.stem,False,'open'))
                        if response.get('error') or call_receipt.get('error') or response.get('timed_out') is True:
                            continue
                        arm_motion = True
                    elif tool not in hold_observation:
                        raise ValueError('An unqualified movement occurred after the recorded reach')
            if len({item[0] for item in drawer_grips}) != len(drawer_grips):
                raise ValueError('Drawer grasp attempts have an ambiguous observation order')
            drawer_closes=[item for item in drawer_grips if item[3]=='close']
            if drawer_closes and not max(drawer_grips,key=lambda item:item[0])[2]:
                raise ValueError('The latest drawer attempt requires a confirmed handle grasp with the planned arm')
            arm_delta = max(abs(q[i] - old_q[i]) for i in [*range(3, 11), *range(30, 38)])
            if arm_delta > 0.02 and not arm_motion:
                raise ValueError('Robot arm posture changed without a confirmed approach receipt')
            if torso_delta > 0.02 and (not arm_motion or arm_delta <= 0.02):
                raise ValueError('Torso changed after reach; confirm a fresh arm approach and retrack the target')
            other = range(30, 38) if row['arm']=='left' else range(3, 11)
            if max(abs(q[i]-old_q[i]) for i in other) > 0.02:
                raise ValueError('The non-selected arm changed after the recorded reach; reacquire and replan')
        elif receipt.get('latest_clickable_image_id') != binding['head_image_id']:
            raise ValueError('Current reach evidence does not match the observation')
        track = _read_tracking(self.host, tracking_call_id, binding, row['point'])
        q = binding.get('proprioception')
        if not isinstance(q, list) or len(q) != 65 or any(type(v) not in (int, float) or not math.isfinite(v) for v in q):
            raise ValueError('Current measured robot state is required')
        distance_m = current_target_distance(binding, row['arm'], track['position_m'])
        if not math.isfinite(distance_m) or distance_m > row.get('max_eef_distance_m', 0.25):
            raise ValueError(f'End effector is {distance_m:.3f} m from the target; use confirmed arm geometry to approach within {row.get("max_eef_distance_m", 0.25):.3f} m, then retrack and enter')
        row.update(status='entered', entry={'binding': _binding(binding), 'tracking': track,
            'eef_target_distance_m': distance_m,
            'reach_torso_delta_rad': torso_delta,
            'reach_torso_revalidated': torso_delta is not None and torso_delta > 0.02,
            'drawer_grasp_evidence': ({'status':'confirmed_after_reach','call_id':max(drawer_grips,key=lambda item:item[0])[1]}
                if any(item[3]=='close' for item in drawer_grips) else {'status':'not_observed_after_reach','call_id':None})
                if row['goal']['type']=='tracked_displacement' else None,
            'reach_call_id': reach_call_id, 'reach_receipt_sha256': sha(reach_path),
            'reach_minimum_m': minimum, 'reach_threshold_m': threshold,
            'proprioception': q})
        self._save(row)
        return row

    def attach_batch(self, stage_id, batch_id):
        row = self._active()
        if not row or row['stage_id'] != stage_id or row['status'] != 'entered':
            raise ValueError('Only an entered stage may attach a batch')
        row.update(status='batch_ready', batch_id=batch_id)
        self._save(row)
        return row

    def abandon(self, reason, binding):
        row = self._active()
        if not row or row['status'] in {'completed', 'abandoned'}:
            raise ValueError('No active stage to abandon')
        row.update(status='abandoned', abandonment={'reason': _bounded_text(reason, 'reason'),
                                                   'binding': _binding(binding)})
        snapshot=self.root/(row['stage_id']+'.abandoned.json')
        try:
            exclusive(snapshot,encoded(row))
        except FileExistsError:
            if snapshot.read_bytes()!=encoded(row):
                raise ValueError('An incompatible abandoned stage record already exists')
        self._save(row)
        return row

    def reanchor_geometry(self, binding, tracking_call_id, direction):
        """Record a fresh drawer reference before a later geometric pull."""
        row=self._active()
        if (not row or row.get('status')!='batch_ready' or
                (row.get('goal') or {}).get('type')!='tracked_displacement'):
            raise ValueError('An entered drawer stage is required')
        if _active_skill(self.host,allow_unset=True)!=(row.get('workspace_skill',row['skill']),row['skill_revision']):
            raise ValueError('Stage skill changed before reanchoring')
        entry=row['entry']['binding']
        if (binding.get('episode_id')!=entry['episode_id'] or binding.get('session_id')!=entry['session_id'] or
                type(binding.get('sequence')) is not int or binding['sequence']<entry['sequence']):
            raise ValueError('Reanchoring requires a current observation in the same episode and session')
        if (not isinstance(direction,list) or len(direction)!=3 or
                any(type(v) not in (int,float) or not math.isfinite(v) for v in direction)):
            raise ValueError('A finite nonzero current-frame pull direction is required')
        norm=math.hypot(*direction)
        if not math.isfinite(norm) or not 1e-6<=norm<=1e6:
            raise ValueError('A finite nonzero current-frame pull direction is required')
        batch_path=self.host.evidence/'g05-candidate-batches'/(row['batch_id']+'.json')
        choice_path=self.host.evidence/'g05-candidate-batches'/(row['batch_id']+'.choice.json')
        batch=json.loads(batch_path.read_bytes());choice=json.loads(choice_path.read_bytes())
        if (batch.get('batch_id')!=row['batch_id'] or batch.get('binding')!=entry or
                choice.get('binding')!=entry or choice.get('batch_sha256')!=sha(batch_path) or
                (choice.get('choice') or {}).get('route')!='geometry'):
            raise ValueError('Reanchoring requires the recorded geometry choice')
        track=_read_tracking(self.host,tracking_call_id,binding,row['point'])
        if track['track_id']!=row['entry']['tracking']['track_id']:
            raise ValueError('Reanchoring requires the original drawer track')
        reference={'binding':_binding(binding),'tracking':track,'direction':[v/norm for v in direction],
                   'direction_source':'planner_assertion_in_current_robot_base',
                   'original_entry_binding':entry,'choice_sha256':sha(choice_path)}
        row.setdefault('geometry_reanchors',[]).append(reference)
        self._save(row)
        return {'stage_id':row['stage_id'],'geometry_reference':reference,
                'use':'fresh_reference_before_intended_pull; no physical command'}

    def finish(self, binding, args):
        """Verify a stage's measured exit and explicitly reported next-stage view."""
        expected = {'next_tracking_call_id', 'next_point', 'visual_reason', 'exit_tracking_call_id', 'geometry_call_id',
                    'grasp_call_id', 'stow_call_id', 'reset_call_id', 'stow_kind'}
        if not isinstance(args, dict) or set(args) != expected:
            raise ValueError('A complete handoff evidence record is required')
        row = self._active()
        if not row or row['status'] != 'batch_ready':
            raise ValueError('An entered stage with a candidate batch is required')
        if _active_skill(self.host, allow_unset=True) != (row.get('workspace_skill', row['skill']), row['skill_revision']):
            raise ValueError('Stage skill changed before handoff')
        deferred = args['next_tracking_call_id'] is None and args['next_point'] is None
        if deferred:
            if row['skill'] != 'open-doors-and-drawers' or row['next_skill'] != 'pick-up-object':
                raise ValueError('Deferred next-target observation is supported for drawer-to-pickup handoffs')
            next_track = None
        else:
            if args['next_tracking_call_id'] is None or args['next_point'] is None:
                raise ValueError('Provide both next tracking fields or defer both')
            next_point = _bounded_text(args['next_point'], 'next point')
            if row['skill'] == 'open-doors-and-drawers' and row['next_skill'] == 'pick-up-object' and next_point == row['point']:
                raise ValueError('Track the next pickup target separately from the drawer handle')
            next_track = _read_tracking(self.host, args['next_tracking_call_id'], binding, next_point)
        handoff_q = binding.get('proprioception')
        if (not isinstance(handoff_q, list) or len(handoff_q) != 65 or
                any(type(v) not in (int, float) or not math.isfinite(v) for v in handoff_q)):
            raise ValueError('The measured handoff robot state is required')
        visual = {'reason': _bounded_text(args['visual_reason'], 'visual reason'),
                  'binding': _binding(binding), 'images': binding.get('images', []),
                  'next_target_label': row['next_target_label'], 'source': 'planner_assertion_from_current_views'}
        choice_path = self.host.evidence / 'g05-candidate-batches' / (row['batch_id'] + '.choice.json')
        decision = json.loads(choice_path.read_bytes())
        batch_path = choice_path.with_name(row['batch_id'] + '.json')
        batch = json.loads(batch_path.read_bytes())
        if (decision.get('batch_sha256') != sha(batch_path) or batch.get('batch_id') != row['batch_id']
                or batch.get('task') != row.get('model_task', row['task']) or batch.get('seeds') != row['seeds']
                or batch.get('binding') != row['entry']['binding'] or decision.get('binding') != batch['binding']):
            raise ValueError('Controller choice does not match the stage candidate batch')
        route = decision.get('choice', {}).get('route')
        if route not in {'g05', 'geometry'}:
            raise ValueError('The stage requires a recorded controller choice')
        route_evidence = {'route': route, 'choice_file': str(choice_path), 'choice_sha256': sha(choice_path),
                          'g05_steps_executed': 0}
        chunk = None; chunk_path = None
        if route == 'g05':
            selected = next((item for item in batch.get('candidates', [])
                             if item.get('candidate_index') == decision['choice'].get('candidate_index')), None)
            if not selected or selected.get('status') != 'candidate' or selected.get('receipt', {}).get('response') != decision.get('selected_response'):
                raise ValueError('The selected candidate does not match its saved batch')
            identifier = decision['selected_response']['request_id']
            chunk_path = self.host.evidence / 'g05-chunks' / (identifier + '.json')
            chunk = json.loads(chunk_path.read_bytes())
            if (chunk.get('status') not in {'completed', 'stop_success', 'stop_replan', 'stopped_by_boundary'}
                    or not chunk.get('steps') or chunk.get('initial_sequence') != row['entry']['binding']['sequence']):
                raise ValueError('The chosen G0.5 chunk has no confirmed completed step')
            route_evidence.update(g05_steps_executed=len(chunk['steps']), chunk_file=str(chunk_path),
                                  chunk_sha256=sha(chunk_path))
        if row['goal']['type'] == 'tracked_displacement':
            if any(args[key] is not None for key in ('grasp_call_id', 'stow_call_id', 'reset_call_id', 'stow_kind')):
                raise ValueError('Displacement stage must not claim grasp or stow evidence')
            if route == 'g05':
                if args['exit_tracking_call_id'] is not None or args['geometry_call_id'] is not None:
                    raise ValueError('The G0.5 chunk provides its own displacement observation')
                if (chunk.get('status') != 'stop_success' or not chunk.get('steps') or
                        chunk.get('initial_sequence') != row['entry']['binding']['sequence']):
                    raise ValueError('A completed displacement observation is required')
                measured = chunk['steps'][-1].get('monitor') or {}
                evidence = measured.get('evidence') or {}
                direction = row['goal']['direction']; norm = math.sqrt(sum(x*x for x in direction))
                expected_dir = [x / norm for x in direction]
                if (measured.get('decision') != 'stop_success' or
                        evidence.get('track_id') != row['entry']['tracking']['track_id'] or
                        evidence.get('initial_position_m') != row['entry']['tracking']['position_m'] or
                        evidence.get('direction') != expected_dir or evidence.get('target_m') != row['goal']['target_m'] or
                        type(evidence.get('directed_displacement_m')) not in (int, float) or
                        evidence['directed_displacement_m'] < row['goal']['target_m']):
                    raise ValueError('Displacement evidence does not match the stage contract')
                exit_evidence = {'kind': 'tracked_displacement', 'chunk_file': str(chunk_path),
                                 'chunk_sha256': sha(chunk_path), 'measurement': evidence}
            else:
                exit_evidence = self._verify_geometry_displacement(row, binding, args)
        else:
            if args['exit_tracking_call_id'] is not None or args['geometry_call_id'] is not None:
                raise ValueError('Pickup handoff does not use drawer displacement evidence')
            exit_evidence = self._verify_grasp_stow(row, binding, args)
        row.update(status='completed', handoff={'exit_evidence': exit_evidence, 'controller': route_evidence,
            'next_tracking': next_track, 'next_observation_status': 'deferred' if deferred else 'observed',
            'planner_view': visual, 'next_skill': row['next_skill'],
            'binding': _binding(binding), 'proprioception': list(handoff_q)})
        experience = self.experience / (row['stage_id'] + '.json')
        try:
            exclusive(experience, encoded(row))
        except FileExistsError:
            if experience.read_bytes() != encoded(row):
                raise ValueError('An incompatible transition record already exists')
        self._save(row)
        return row

    def _verify_geometry_displacement(self, row, binding, args):
        """Check one measured straight base motion and the same tracked handle."""
        reference=(row.get('geometry_reanchors') or [row['entry']])[-1]
        reference_binding=reference['binding']
        record, path = _call(self.host, args['geometry_call_id'])
        response = record['receipt'].get('response') or {}
        actual = response.get('actual') or {}; command = record.get('arguments') or {}
        forward, lateral, spin = actual.get('forward_m'), actual.get('translation_m'), actual.get('spin_deg')
        cross, drift = response.get('linear_cross_track_m'), response.get('linear_yaw_drift_deg')
        if (record.get('tool') != 'adjust_chassis' or record['binding']['episode_id'] != binding['episode_id'] or
                record['binding']['session_id'] != binding['session_id'] or
                reference_binding['episode_id'] != binding['episode_id'] or
                reference_binding['session_id'] != binding['session_id'] or
                not reference_binding['sequence'] <= record['binding']['sequence'] <= binding['sequence'] or
                command.get('translation', 0) != 0 or command.get('spin', 0) != 0 or
                any(type(v) not in (int, float) or not math.isfinite(v) for v in (forward, lateral, spin, cross, drift)) or
                abs(spin) > 1 or abs(drift) > 1 or abs(cross) > .01 or abs(lateral) > .01 or
                response.get('linear_target_reached') is not True or response.get('timed_out') is not False):
            raise ValueError('One confirmed straight base movement with bounded drift is required')
        for p in (self.host.evidence/'calls').glob('*.json'):
            call = json.loads(p.read_bytes()); cb = call.get('binding') or {}
            if (p != path and p.stem != row['entry']['reach_call_id'] and cb.get('episode_id') == binding['episode_id'] and
                    type(cb.get('sequence')) is int and
                    reference_binding['sequence'] <= cb['sequence'] <= binding['sequence'] and
                    call.get('tool') in {'adjust_chassis', 'move_to_reach_point', 'spin_to_facing_point',
                                         'move_chassis_to_floor_point', 'move_chassis_to_directly_facing_surface'}):
                raise ValueError('More than one base movement prevents this displacement calculation')
        final = _read_tracking(self.host, args['exit_tracking_call_id'], binding, row['point'])
        initial = reference['tracking']
        if final['track_id'] != initial['track_id']:
            raise ValueError('The final drawer track is not the entry track')
        start, end = initial['position_m'], final['position_m']
        delta = [end[0] + forward - start[0], end[1] + lateral - start[1], end[2] - start[2]]
        direction = reference.get('direction',row['goal']['direction']); norm = math.sqrt(sum(v*v for v in direction))
        progress = sum(delta[i] * direction[i] / norm for i in range(3))
        if progress < row['goal']['target_m']:
            raise ValueError('Measured drawer displacement is below the stage goal')
        return {'kind':'tracked_displacement_with_measured_base_translation', 'start_tracking':initial,
                'end_tracking':final, 'base_call_id':args['geometry_call_id'], 'base_receipt_sha256':sha(path),
                'base_forward_m':forward, 'base_lateral_m':lateral, 'base_spin_deg':spin, 'cross_track_m':cross,
                'estimated_displacement_m':delta, 'directed_displacement_m':progress,
                'direction':direction,'reference_binding':reference_binding,
                'assumptions':'single straight base translation; bounded yaw and lateral drift'}

    def _verify_grasp_stow(self, row, binding, args):
        if args['stow_kind'] not in {'small', 'large'}:
            raise ValueError('A small or large-object stow is required')
        grasp, grasp_path = _call(self.host, args['grasp_call_id'])
        response = grasp['receipt'].get('response') or {}
        if (grasp.get('tool') != 'close_gripper' or response.get('arm') != row['arm'] or
                response.get('grasp_confirmed') is not True or
                grasp['binding']['episode_id'] != binding['episode_id']):
            raise ValueError('A confirmed close-gripper receipt for the selected arm is required')
        first_seq = grasp['binding']['sequence']
        reset, reset_path = _call(self.host, args['reset_call_id'])
        reset_response = reset['receipt'].get('response') or {}
        reset_seq = reset['binding']['sequence']
        if (reset.get('tool') != 'reset_body' or reset['binding']['episode_id'] != binding['episode_id']
                or reset_seq <= first_seq or reset_seq > binding['sequence']
                or reset_response.get('timed_out') is not False):
            raise ValueError('A subsequent completed body-stow receipt is required')
        evidence = {'kind': 'grasp_stow', 'grasp_call_id': args['grasp_call_id'],
                    'grasp_receipt_sha256': sha(grasp_path), 'reset_call_id': args['reset_call_id'],
                    'reset_receipt_sha256': sha(reset_path), 'stow_kind': args['stow_kind']}
        if args['stow_kind'] == 'small':
            stow, stow_path = _call(self.host, args['stow_call_id'])
            stow_args = stow.get('arguments') or {}; stow_response = stow['receipt'].get('response') or {}
            if (stow.get('tool') != 'set_arm_to_grasp_position' or stow_args.get('arm') != row['arm']
                    or stow_args.get('gripper') != 'keep' or stow_response.get('converged') is not True
                    or stow['binding']['episode_id'] != binding['episode_id']
                    or not first_seq < stow['binding']['sequence'] < reset_seq
                    or reset_response.get('keep_ori_arm') != 'none'):
                raise ValueError('A successful holding-arm stow before body reset is required')
            evidence.update(stow_call_id=args['stow_call_id'], stow_receipt_sha256=sha(stow_path))
        else:
            if (args['stow_call_id'] is not None or reset_response.get('keep_ori_arm') != row['arm']
                    or reset_response.get('keep_orientation_satisfied') is not True):
                raise ValueError('Large-object stow must preserve the holding arm orientation')
        for path in (self.host.evidence / 'calls').glob('*.json'):
            call = json.loads(path.read_bytes()); cb = call.get('binding') or {}
            if (call.get('status') not in {'completed', 'tool_error'} or cb.get('episode_id') != binding['episode_id'] or
                    type(cb.get('sequence')) is not int or not first_seq < cb['sequence'] <= binding['sequence']):
                continue
            ca = call.get('arguments') or {}
            if ((call.get('tool') == 'open_gripper' and ca.get('arm') in {row['arm'], 'both'}) or
                    (call.get('tool') == 'set_arm_to_grasp_position' and ca.get('arm') in {row['arm'], 'both'}
                     and ca.get('gripper') == 'open')):
                raise ValueError('The holding gripper was opened after the grasp')
        return evidence

    def matches(self, binding, tracking_call_id, point, skill, target_label, arm, tolerance_m=0.5, joint_tolerance_rad=1.0):
        if (skill not in SKILLS or arm not in {'left', 'right'} or type(tolerance_m) not in (int, float)
                or not 0 < tolerance_m <= 0.5 or type(joint_tolerance_rad) not in (int, float)
                or not 0 < joint_tolerance_rad <= 1):
            raise ValueError('Supported skill, arm and bounded tolerances required')
        current = _read_tracking(self.host, tracking_call_id, binding, point)
        target = _bounded_text(target_label, 'target label')
        q = binding.get('proprioception')
        if not isinstance(q, list) or len(q) != 65 or any(type(v) not in (int, float) or not math.isfinite(v) for v in q):
            raise ValueError('Current measured robot state is required')
        start = 3 if arm == 'left' else 30
        results = []
        if self.reference is not None:
            self._check_reference()
        paths = list(self.experience.glob('*.json'))
        if self.reference is not None:
            paths += list(self.reference.glob('*.json'))
        for path in sorted(paths):
            item = json.loads(path.read_bytes())
            if item.get('status') != 'completed':
                continue
            handoff = item.get('handoff') or {}
            next_track = handoff.get('next_tracking') or {}
            if (item.get('next_skill') == skill and item.get('next_target_label') == target and
                    handoff.get('next_observation_status') == 'deferred' and
                    (handoff.get('binding') or {}).get('episode_id') == binding['episode_id']):
                results.append({'stage_id':item['stage_id'], 'match_kind':'previous_stage_handoff_deferred_target',
                    'entry_position_distance_m':None, 'entry_arm_max_delta_rad':None,
                    'readiness':'target_observed_recheck_entry', 'task':item['task'],
                    'next_skill':item['next_skill'], 'goal':item['goal'],
                    'source_episode':binding['episode_id'], 'record_sha256':sha(path),
                    'current_target_position_m':current['position_m'], 'suggested_entry':None})
            if (item.get('next_skill') == skill and item.get('next_target_label') == target and
                    isinstance(next_track.get('position_m'), list) and len(next_track['position_m']) == 3 and
                    all(type(v) in (int, float) and math.isfinite(v) for v in next_track['position_m'])):
                origin = next_track['position_m']
                distance = math.dist(origin, current['position_m'])
                old_q = handoff.get('proprioception')
                valid_q = (isinstance(old_q, list) and len(old_q) == 65 and
                           all(type(v) in (int, float) and math.isfinite(v) for v in old_q))
                joint_delta = max(abs(q[i] - old_q[i]) for i in range(start, start + 7)) if valid_q else None
                if distance <= tolerance_m and (joint_delta is None or joint_delta <= joint_tolerance_rad):
                    results.append({'stage_id': item['stage_id'], 'match_kind': 'previous_stage_handoff',
                        'entry_position_distance_m': distance, 'entry_arm_max_delta_rad': joint_delta,
                        'readiness': ('similar_entry' if joint_delta is not None and distance <= 0.15 and joint_delta <= 0.35
                                      else 'transition_suggested'),
                        'task': item['task'], 'next_skill': item['next_skill'], 'goal': item['goal'],
                        'source_episode': (handoff.get('binding') or {}).get('episode_id'), 'record_sha256': sha(path),
                        'suggested_entry': {'target_in_robot_base_m': origin,
                            'arm_joint_positions_rad': old_q[start:start + 7] if valid_q else None,
                            'torso_joint_positions_rad': old_q[57:61] if valid_q else None,
                            'approximate_base_forward_left_m': [current['position_m'][0] - origin[0],
                                                                current['position_m'][1] - origin[1]],
                            'assumptions': 'stationary target and same base heading; recheck clearance and current views'}})
            if item.get('skill') != skill or item.get('target_label') != target or item.get('arm') != arm:
                continue
            origin = item['entry']['tracking']['position_m']
            distance = math.dist(origin, current['position_m'])
            old_q = item['entry']['proprioception']
            joint_delta = max(abs(q[i] - old_q[i]) for i in range(start, start + 7))
            if distance <= tolerance_m and joint_delta <= joint_tolerance_rad:
                results.append({'stage_id': item['stage_id'], 'entry_position_distance_m': distance,
                    'entry_arm_max_delta_rad': joint_delta,
                    'readiness': 'similar_entry' if distance <= 0.15 and joint_delta <= 0.35 else 'transition_suggested',
                    'task': item['task'], 'next_skill': item['next_skill'], 'goal': item['goal'],
                    'source_episode': item['entry']['binding']['episode_id'], 'record_sha256': sha(path),
                    'suggested_entry': {'target_in_robot_base_m': origin,
                        'arm_joint_positions_rad': old_q[start:start + 7], 'torso_joint_positions_rad': old_q[57:61],
                        'approximate_base_forward_left_m': [current['position_m'][0] - origin[0],
                                                            current['position_m'][1] - origin[1]],
                        'assumptions': 'stationary target and same base heading; recheck clearance and current views'}})
        return {'current_tracking': current, 'reference_digest': self.reference_digest,
                'matches': sorted(results, key=lambda x: (x['entry_position_distance_m'] is None,
                                                           x['entry_position_distance_m'] or 0.0)),
                'use': 'suggestions_only_recheck_current_observation'}
