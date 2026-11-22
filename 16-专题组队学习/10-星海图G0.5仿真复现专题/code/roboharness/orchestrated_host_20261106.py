"""Expose candidate preparation and supervised choice to a workspace agent."""
import hashlib
import json
import math
import os
from pathlib import Path
import re
from contextlib import contextmanager

from candidate_orchestration_20261106 import execute_choice, propose_batch, record_choice, refresh_batch, resume_batch
from g05_proposal_host import G05ProposalHost
from proposal_exchange import sha
from tracked_progress_20261106 import TrackedProgress, TrackedGuard, _track
from stage_bridge_20261106 import StageBridge


HERE = Path(__file__).resolve().parent
DECLARATIONS = {'g05_refresh_batch': 'workspace_g05_refresh_batch.py',
                'g05_stage': 'workspace_g05_stage.py',
                'g05_select_chunk': 'workspace_g05_select_chunk.py'}


def geometry_stage_advice(stage, name, receipt, binding, evidence):
    """Suggest recovery when a geometry handoff's measured reference is stale."""
    if (not stage or stage.get('status')!='batch_ready' or
            (stage.get('goal') or {}).get('type')!='tracked_displacement'):
        return None
    batch_id=stage.get('batch_id')
    if not isinstance(batch_id,str) or not batch_id.isalnum():return None
    choice=Path(evidence)/'g05-candidate-batches'/(batch_id+'.choice.json')
    try:decision=json.loads(choice.read_bytes())
    except (OSError,ValueError):return None
    if not isinstance(decision,dict):return None
    if (decision.get('choice') or {}).get('route')!='geometry':return None
    response=receipt.get('response') or {}
    if not isinstance(response,dict):return None
    if name=='close_gripper' and response.get('grasp_confirmed') is not True:
        return ('The drawer grasp is not confirmed. Inspect the current observation; if another approach or '
                'base movement is needed, reanchor after reacquiring the handle and before a further intended pull, '
                'or abandon and plan a new stage.')
    moves={'adjust_chassis','move_to_reach_point','spin_to_facing_point',
           'move_chassis_to_floor_point','move_chassis_to_directly_facing_surface'}
    reference=(stage.get('geometry_reanchors') or [stage.get('entry') or {}])[-1]
    entry=reference.get('binding') or {}
    if name not in moves or type(entry.get('sequence')) is not int:return None
    count=0
    for path in (Path(evidence)/'calls').glob('*.json'):
        try:call=json.loads(path.read_bytes())
        except (OSError,ValueError):continue
        if not isinstance(call,dict):continue
        cb=call.get('binding') or {}
        if not isinstance(cb,dict):continue
        if (path.stem!=(stage.get('entry') or {}).get('reach_call_id') and
                call.get('tool') in moves and cb.get('episode_id')==binding.get('episode_id') and
                type(cb.get('sequence')) is int and entry['sequence']<=cb['sequence']<=binding['sequence']):
            count+=1
    if count>1:
        return ('Multiple base movements invalidate this stage\'s single-movement displacement reference. '
                'Re-observe and reanchor before a further intended pull, or abandon without claiming measured displacement.')
    return None


def tracking_drift_diagnostic(workspace, receipt, arguments, binding):
    """Extract measured tracking or base disturbance from the completed tool record."""
    response=receipt.get('response') or {}
    if not isinstance(response,dict) or response.get('error') not in {
            'trajectory execution failed: off_hand_points_moved','trajectory execution failed: base_disturbance'}:
        return None
    try:
        workspace=Path(workspace).resolve();history=(workspace/'tool_call_history').resolve()
        record=Path(receipt['record']);record=(workspace/record).resolve()
        if record.name!='result.json' or record.parent.parent!=history or not re.fullmatch(r'\d+_move_tracked_point',record.parent.name):
            return None
        original=json.loads(record.read_bytes());inputs=json.loads((record.parent/'input.json').read_bytes())
        session=json.loads((workspace/'.session.json').read_bytes())
        if (inputs.get('tool')!='move_tracked_point' or inputs.get('arguments')!=arguments or
                inputs.get('observation_sequence')!=binding.get('sequence') or
                any(inputs.get(key)!=binding.get(key) or session.get(key)!=binding.get(key)
                    for key in ('episode_id','session_id')) or
                original.get('tool_name')!='move_tracked_point' or
                (original.get('response') or {}).get('error')!=response['error']):
            return None
        trajectory=Path(session['trajectory_dir']).resolve();raw=(trajectory/'calls'/record.parent.name/'output.json').resolve()
        if raw.parent!=(trajectory/'calls'/record.parent.name).resolve():return None
        operation=json.loads(raw.read_bytes()).get('operation_result') or {}
        if operation.get('error')!=response['error'] or operation.get('plan_id')!=response.get('plan_id'):return None
        if response['error'].endswith('base_disturbance'):
            motion=operation.get('motion') or {};safety=motion.get('safety') or {}
            keys=('max_base_linear_speed_mps','max_base_yaw_rate_rad_s',
                  'max_base_translation_m','max_base_yaw_drift_rad')
            if (motion.get('reason')!='base_disturbance' or
                    any(type(safety.get(key)) not in (int,float) or not math.isfinite(safety[key]) or safety[key]<0
                        for key in keys)):
                return None
            return {'kind':'base_disturbance',**{key:safety[key] for key in keys},
                    'source_file':str(raw),'source_sha256':sha(raw),
                    'interpretation':'Measured base disturbance during the trajectory; inspect current observations before replanning.'}
        reference=(operation.get('post_execution_track_validation') or {}).get('off_hand_reference') or {}
        values=(reference.get('max_drift_m'),reference.get('tolerance_m'))
        points=reference.get('point_drift_m')
        if (reference.get('reason')!='off_hand_points_moved' or
                any(type(value) not in (int,float) or not math.isfinite(value) or value<0 for value in values) or
                not isinstance(points,dict) or not 1<=len(points)<=20 or
                any(not isinstance(key,str) or type(value) not in (int,float) or not math.isfinite(value) or value<0
                    for key,value in points.items())):
            return None
        return {'kind':'off_hand_points_moved','max_drift_m':values[0],'tolerance_m':values[1],'point_drift_m':points,
                'tracking_outlier_suppressed':reference.get('tracking_outlier_suppressed'),
                'source_file':str(raw),'source_sha256':sha(raw),
                'interpretation':'Measured tracked-point drift; inspect current observations before replanning.'}
    except (OSError,ValueError,TypeError,KeyError,AttributeError,OverflowError):return None


class OrchestratedHost(G05ProposalHost):
    """Keep the original host's physical checks and add two virtual tools."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        experience = self.exchange.config.get('transition_experience_root')
        if experience is not None:
            experience = Path(experience).resolve()
            allowed = Path(os.environ['G05_TRANSITION_DATA_ROOT']).resolve()
            if not experience.is_relative_to(allowed) or experience.is_relative_to(self.workspace):
                raise ValueError('Transition data must stay in the experiment output tree')
        self.stage_bridge = StageBridge(self, experience_root=experience,
                                        experience_digest=self.exchange.config.get('transition_experience_digest'))
        self._internal_depth = 0

    @contextmanager
    def _internal(self):
        self._internal_depth += 1
        try:
            yield
        finally:
            self._internal_depth -= 1

    def _check_orchestration(self, name, binding, allow_nonphysical_recovery=False):
        if not self.allow_execution:
            raise PermissionError('Execution gate is closed')
        if name not in self.tools or sha(self.tools[name]) != sha(HERE / DECLARATIONS[name]):
            raise ValueError('Orchestration tool declaration changed')
        self.validate_sources()
        self.same_observation(binding)
        unresolved = self.unresolved()
        if unresolved:
            if not allow_nonphysical_recovery:
                raise RuntimeError('A previous physical or proposal intent is unresolved')
            for filename in unresolved:
                record = json.loads((self.evidence/'calls'/filename).read_bytes())
                if record.get('tool') != 'g05_propose' or record.get('physical_commands_sent') != 0:
                    raise RuntimeError('A previous physical intent is unresolved')

    def _tracking_source(self, call_id, binding, point):
        if not isinstance(call_id, str) or not re.fullmatch(r'[A-Za-z0-9_.-]{1,180}', call_id):
            raise ValueError('A completed tracking source call is required')
        path = self.evidence / 'calls' / (call_id + '.json')
        record = json.loads(path.read_bytes())
        if record.get('status') != 'completed':
            raise ValueError('Tracking source call did not complete')
        tracking = (record.get('receipt') or {}).get('persistent_tracking')
        # Constructor validates the actual point's observation and identity.
        return tracking

    def _tracking_sources(self, binding):
        sources = []
        for path in (self.evidence / 'calls').glob('*.json'):
            record = json.loads(path.read_bytes())
            tracking = (record.get('receipt') or {}).get('persistent_tracking') or {}
            for name, point in (tracking.get('tracks') or {}).items():
                if (record.get('status') == 'completed' and point.get('status') == 'observed'
                        and point.get('episode_id') == binding['episode_id']
                        and point.get('observation_sequence') == binding['sequence']):
                    sources.append({'call_id': path.stem, 'point': name, 'track_id': point.get('track_id'),
                                    'confidence': point.get('confidence')})
        return sources[-20:]

    def execute(self, call_id, name, arguments, binding):
        if name not in DECLARATIONS:
            if not self._internal_depth:
                stage = self.stage_bridge._active()
                if stage and stage['status'] in {'entered', 'batch_ready'}:
                    batch_id = stage.get('batch_id')
                    choice = (self.evidence/'g05-candidate-batches'/(batch_id + '.choice.json')
                              if batch_id else None)
                    if choice is None or not choice.exists():
                        raise ValueError('Choose the prepared G0.5 candidate, select geometry, or abandon the stage before other tools change the observation')
            if name in {'g05_propose', 'g05_step'} and not self._internal_depth:
                raise PermissionError('Use the stage orchestration tools for G0.5 execution')
            if name == 'deactivate_skill':
                stage = self.stage_bridge._active()
                if (stage and stage['status'] not in {'completed', 'abandoned'} and
                        arguments.get('name') == stage['skill']):
                    raise ValueError('Finish or abandon the active stage before deactivating its skill')
            receipt = super().execute(call_id, name, arguments, binding)
            receipt = {**receipt, 'orchestration_call_id': call_id}
            if not self._internal_depth:
                if name=='move_tracked_point':
                    diagnostic=tracking_drift_diagnostic(self.workspace,receipt,arguments,binding)
                    if diagnostic:
                        key=('motion_guard_diagnostic' if diagnostic['kind']=='base_disturbance' else 'g05_tracking_diagnostic')
                        receipt={**receipt,key:diagnostic}
                advice=geometry_stage_advice(self.stage_bridge._active(),name,receipt,binding,self.evidence)
                if advice:receipt={**receipt,'g05_stage_prompt':advice}
            if (name == 'move_to_reach_point' and not receipt.get('error')
                    and not (receipt.get('response') or {}).get('error')):
                state = receipt.get('task_skill_state') or {}
                distance = (receipt.get('response') or {}).get('shoulder_distance_estimate') or {}
                minimum, threshold = distance.get('minimum_m'), distance.get('threshold_m')
                if (state.get('active_skill') == 'pick-up-object' and
                        type(minimum) in (int, float) and type(threshold) in (int, float) and
                        0 <= minimum <= threshold <= 2):
                    receipt = {**receipt, 'g05_stage_prompt':
                        'Pickup approach is in reach. Ground the current target and enter the planned G0.5 stage.'}
            if (name == 'move_to_reach_point' and not receipt.get('error')
                    and not (receipt.get('response') or {}).get('error')):
                stage = self.stage_bridge._active()
                if stage and stage['status'] == 'planned':
                    current, _ = self.frame()
                    self.stage_bridge.record_reach(call_id, current)
                    sources = [x for x in self._tracking_sources(current) if x['point'] == stage['point']]
                    if sources:
                        auto_id = 'auto-' + hashlib.sha256((stage['stage_id'] + call_id).encode()).hexdigest()[:40]
                        try:
                            result = self.execute(auto_id, 'g05_stage', {'action': 'enter', 'parameters': {
                                'reach_call_id': call_id, 'tracking_call_id': sources[-1]['call_id']}}, current)
                            receipt = {**receipt, 'automatic_g05_stage': result['response']}
                        except ValueError as error:
                            receipt = {**receipt, 'g05_stage_prompt': str(error)}
                    else:
                        receipt = {**receipt, 'g05_stage_prompt': {
                            'instruction': 'Reacquire the planned tracked point, then enter the stage.',
                            'reach_call_id': call_id, 'point': stage['point']}}
            return receipt
        if not isinstance(arguments, dict) or not re.fullmatch(r'[A-Za-z0-9_.-]{1,180}', call_id):
            raise ValueError('Invalid orchestration call')
        recovery = ((name == 'g05_stage' and arguments.get('action') == 'resume') or name == 'g05_refresh_batch')
        self._check_orchestration(name, binding, recovery)
        if name == 'g05_stage':
            if set(arguments) != {'action', 'parameters'} or not isinstance(arguments['parameters'], dict):
                raise ValueError('A stage action and parameters are required')
            action, params = arguments['action'], arguments['parameters']
            if action == 'plan':
                stage = self.stage_bridge.plan(call_id, params, binding)
                sources = [x for x in self._tracking_sources(binding) if x['point'] == stage['point']]
                result = {'stage': stage, 'tracking_sources': sources, 'experience': None}
                if sources:
                    result['experience'] = self.stage_bridge.matches(binding, sources[-1]['call_id'],
                        stage['point'], stage['skill'], stage['target_label'], stage['arm'])
            elif action == 'enter':
                if set(params) != {'reach_call_id', 'tracking_call_id'}:
                    raise ValueError('Reach and tracking evidence are required')
                stage = self.stage_bridge.begin(binding, params['reach_call_id'], params['tracking_call_id'])
                batch_id = 'b' + hashlib.sha256(stage['stage_id'].encode()).hexdigest()[:40]
                with self._internal():
                    batch = propose_batch(self, batch_id, stage.get('model_task', stage['task']), stage['seeds'], binding)
                if batch['status'] != 'ready_for_planner':
                    raise ValueError('Stage observation changed while preparing candidates')
                self.stage_bridge.attach_batch(stage['stage_id'], batch_id)
                result = {'stage_id': stage['stage_id'], 'batch_id': batch_id, 'status': batch['status'],
                          'candidates': batch['candidates'], 'tracking_sources': self._tracking_sources(binding)}
            elif action == 'resume':
                if params:
                    raise ValueError('Stage recovery takes no parameters')
                stage = self.stage_bridge._active()
                if not stage or stage['status'] not in {'entered', 'batch_ready'}:
                    raise ValueError('There is no entered stage to recover')
                if stage['entry']['binding'] != {key: binding[key] for key in ('episode_id','session_id','sequence','head_image_id')}:
                    raise ValueError('Stage recovery requires the original observation')
                batch_id = 'b' + hashlib.sha256(stage['stage_id'].encode()).hexdigest()[:40]
                batch_path = self.evidence/'g05-candidate-batches'/(batch_id + '.json')
                with self._internal():
                    batch = (resume_batch(self, batch_id, binding) if batch_path.exists()
                             else propose_batch(self, batch_id, stage.get('model_task', stage['task']), stage['seeds'], binding))
                if batch['status'] != 'ready_for_planner':
                    raise ValueError('The candidate batch is not ready')
                if stage['status'] == 'entered':
                    self.stage_bridge.attach_batch(stage['stage_id'], batch_id)
                result = {'stage_id': stage['stage_id'], 'batch_id': batch_id, 'status': batch['status'],
                          'candidates': batch['candidates'], 'tracking_sources': self._tracking_sources(binding)}
            elif action == 'finish':
                result = self.stage_bridge.finish(binding, params)
            elif action == 'reanchor':
                if set(params) != {'tracking_call_id', 'direction'}:
                    raise ValueError('Current tracking and pull direction are required')
                result = self.stage_bridge.reanchor_geometry(binding, **params)
            elif action == 'abandon':
                if set(params) != {'reason'}:
                    raise ValueError('An abandonment reason is required')
                result = self.stage_bridge.abandon(params['reason'], binding)
            elif action == 'matches':
                if set(params) != {'tracking_call_id', 'point', 'skill', 'target_label', 'arm'}:
                    raise ValueError('Current tracking, skill and target label are required')
                result = self.stage_bridge.matches(binding, **params)
            else:
                raise ValueError('Unknown stage action')
            return {'tool_name': name, 'response': result, 'joint_commands_sent': 0, 'images': []}
        if name == 'g05_refresh_batch':
            if set(arguments) != {'batch_id'}:
                raise ValueError('A batch identifier is required')
            stage = self.stage_bridge._active()
            if not stage or stage['status'] != 'batch_ready' or arguments['batch_id'] != stage['batch_id']:
                raise ValueError('Only the active stage batch can be refreshed')
            refresh_id = 'r' + hashlib.sha256(call_id.encode()).hexdigest()[:40]
            with self._internal():
                batch = refresh_batch(self, arguments['batch_id'], refresh_id, binding)
            return {'tool_name': name, 'response': {'batch_id': batch['batch_id'],
                'status': batch['status'], 'binding': batch['binding'], 'candidates': batch['candidates'],
                'tracking_sources': self._tracking_sources(binding)}, 'joint_commands_sent': 0, 'images': []}
        expected = {'batch_id', 'route', 'candidate_index', 'reason', 'tracking_call_id',
                    'point', 'direction', 'target_m', 'maximum_steps'}
        if set(arguments) != expected or type(arguments['maximum_steps']) is not int or not 1 <= arguments['maximum_steps'] <= 8:
            raise ValueError('A complete bounded selection is required')
        stage = self.stage_bridge._active()
        if (not stage or stage['status'] != 'batch_ready' or arguments['batch_id'] != stage['batch_id']
                or arguments['maximum_steps'] > stage['maximum_steps']):
            raise ValueError('A matching planned stage and its chunk limit are required')
        if arguments['route'] == 'g05':
            tracking = self._tracking_source(arguments['tracking_call_id'], binding, arguments['point'])
            if arguments['point'] != stage['point']:
                raise ValueError('Selection must monitor the planned tracked point')
            point, xyz = _track(binding, tracking, arguments['point'], 0.5)
            if (point['track_id'] != stage['entry']['tracking']['track_id'] or
                    xyz != stage['entry']['tracking']['position_m']):
                raise ValueError('Selection must use the stage entry tracking observation')
            if stage['goal']['type'] == 'grasp_stow':
                if arguments['direction'] is not None or arguments['target_m'] is not None:
                    raise ValueError('Pickup uses visibility monitoring and separate grasp verification')
                monitor = TrackedGuard(binding, tracking, arguments['point'])
            else:
                if arguments['direction'] != stage['goal']['direction'] or arguments['target_m'] != stage['goal']['target_m']:
                    raise ValueError('Displacement monitor must match the stage goal')
                monitor = TrackedProgress(binding, tracking, arguments['point'], arguments['direction'], arguments['target_m'],
                    stall_window=stage['goal'].get('stall_window', 0),
                    min_progress_m=stage['goal'].get('min_progress_m', 0.0))
        elif arguments['route'] == 'geometry':
            if any(arguments[key] is not None for key in ('tracking_call_id', 'point', 'direction', 'target_m')):
                raise ValueError('Geometry handoff does not accept tracking parameters')
            monitor = lambda *unused: None
        else:
            raise ValueError('Unknown route')
        choice = {key: arguments[key] for key in ('route', 'candidate_index', 'reason')}
        record_choice(self, arguments['batch_id'], choice, binding)
        with self._internal():
            result = execute_choice(self, arguments['batch_id'], binding, monitor, arguments['maximum_steps'],
                                    active_arm=stage['arm'], hold_gripper=stage['gripper_policy'] == 'hold')
        return {'tool_name': name, 'response': result,
                'joint_commands_sent': len(result.get('steps', [])), 'images': []}
