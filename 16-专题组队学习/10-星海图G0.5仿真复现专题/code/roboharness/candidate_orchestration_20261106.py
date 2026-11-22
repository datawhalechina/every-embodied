"""Build observation-bound candidate summaries for an explicit planner choice.

Generation is nonphysical. Summaries describe joint targets against the
captured state, not predicted object motion or future execution safety.
"""
import hashlib
from pathlib import Path

import numpy as np

from bridge import Calibration
from g05_chunk_20261106 import execute_chunk, load_sequence, project_step
from proposal_exchange import encoded, exclusive
from workspace_agent_host import atomic


BINDING_KEYS = ('episode_id', 'session_id', 'sequence', 'head_image_id')


def gripper_summary(action, stage):
    """Expose model and executed-scope gripper intentions using bridge thresholds."""
    calibration=Calibration();result={}
    for side in ('left','right'):
        value=action[side+'_gripper'][0]
        effective=(0.0 if stage and (side!=stage['arm'] or stage.get('gripper_policy')=='hold') else value)
        intent=('close' if effective<=calibration.close_threshold else
                'open' if effective>=calibration.open_threshold else 'retain_effort')
        result[side]={'model_normalized':value,'effective_normalized':effective,'effective_intent':intent}
    return result


def binding_id(binding):
    return {key: binding[key] for key in BINDING_KEYS}


def summarize_candidate(host, response, binding, limit=8):
    """Describe targets using one captured joint state and existing bounds."""
    if type(limit) is not int or not 1 <= limit <= 8:
        raise ValueError('Summary limit must be between one and eight')
    proposal, arrays, raw_hash = load_sequence(host, response, binding)
    steps = []
    q = binding['proprioception']
    stage_bridge = getattr(host, 'stage_bridge', None)
    stage = stage_bridge._active() if stage_bridge is not None else None
    for index in range(min(limit, len(arrays['left_arm']))):
        try:
            item, projection = project_step(proposal, arrays, index, binding)
            max_delta = {side: float(np.max(np.abs(
                np.asarray(item['action'][side + '_arm']) - np.asarray(q[start:start + 7]))))
                for side, start in (('left', 3), ('right', 30))}
            steps.append({'index': index, 'initial_state_bounds': 'pass',
                          'max_joint_delta_rad': max_delta, 'grippers':gripper_summary(item['action'],stage),
                          'projection': projection})
        except ValueError as error:
            steps.append({'index': index, 'initial_state_bounds': 'fail', 'reason': str(error)})
    result = {'request_id': response['request_id'], 'proposal_file': response['proposal_file'],
            'proposal_sha256': response['proposal_sha256'], 'raw_sha256': raw_hash,
            'available_steps': len(arrays['left_arm']), 'steps': steps,
            'reference': 'captured initial state only',
            'object_motion_prediction': None}
    if stage and stage.get('entry'):
        from candidate_geometry_20261106 import summarize_geometry
        result['kinematic_geometry'] = summarize_geometry(proposal, arrays, binding, stage, limit)
    return result


def propose_batch(host, batch_id, task, seeds, binding):
    """Request up to three candidates for one task and one captured frame."""
    if not isinstance(batch_id, str) or not batch_id.isalnum() or len(batch_id) > 80:
        raise ValueError('Batch identifier must be alphanumeric')
    if not isinstance(task, str) or not task.strip() or len(task) > 8192:
        raise ValueError('A bounded current subtask is required')
    if (not isinstance(seeds, list) or not 1 <= len(seeds) <= 3 or
            any(type(seed) is not int or not 0 <= seed < 2**32 for seed in seeds) or
            len(set(seeds)) != len(seeds)):
        raise ValueError('One to three distinct 32-bit seeds are required')
    if binding_id(host.frame()[0]) != binding_id(binding):
        raise ValueError('Candidate batch is bound to a stale observation')
    folder = Path(host.evidence) / 'g05-candidate-batches'
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / (batch_id + '.json')
    row = {'status': 'intent_recorded', 'batch_id': batch_id, 'task': task,
           'binding': binding_id(binding), 'seeds': seeds, 'candidates': []}
    try:
        exclusive(path, encoded(row))
    except FileExistsError as error:
        raise RuntimeError('Batch already exists; inspect its recorded intent') from error
    return resume_batch(host, batch_id, binding)


def resume_batch(host, batch_id, binding):
    """Reconcile nonphysical proposal intents without reserving another request."""
    import json
    if not isinstance(batch_id, str) or not batch_id.isalnum() or len(batch_id) > 80:
        raise ValueError('A valid batch identifier is required')
    path = Path(host.evidence) / 'g05-candidate-batches' / (batch_id + '.json')
    row = json.loads(path.read_bytes())
    if row.get('batch_id') != batch_id or binding_id(binding) != row.get('binding'):
        raise ValueError('Batch recovery requires its original observation')
    if row.get('status') == 'ready_for_planner':
        return row
    if row.get('status') not in {'intent_recorded', 'proposal_intent', 'collecting'}:
        raise ValueError('Batch state cannot be recovered automatically')
    for index, seed in enumerate(row['seeds']):
        if index < len(row['candidates']):
            if row['candidates'][index]['candidate_index'] != index or row['candidates'][index]['seed'] != seed:
                raise ValueError('Persisted batch candidate order changed')
            continue
        if binding_id(host.frame()[0]) != row['binding']:
            row['status'] = 'stale_observation'; atomic(path, row)
            return row
        call_id = 'batch-' + hashlib.sha256(f'{batch_id}:{index}'.encode()).hexdigest()[:40]
        row.update(status='proposal_intent', next_index=index, next_call_id=call_id)
        atomic(path, row)
        receipt = host.execute(call_id, 'g05_propose',
                               {'task': row['task'], 'seed': seed, 'candidate_index': index}, binding)
        response = receipt.get('response') or {}
        status = response.get('status')
        candidate = {'candidate_index': index, 'seed': seed, 'status': status,
                     'receipt': receipt}
        if status == 'candidate':
            candidate['summary'] = summarize_candidate(host, response, binding)
        elif status not in {'pending', 'candidate_rejected'}:
            raise ValueError('Unexpected proposal receipt; inspect recorded intent')
        row['candidates'].append(candidate)
        row['status'] = 'collecting'; atomic(path, row)
    row['status'] = 'ready_for_planner'; atomic(path, row)
    return row


def refresh_batch(host, batch_id, refresh_id, binding):
    """Collect pending requests against the original observation without reinference."""
    import json
    if (not isinstance(batch_id, str) or not batch_id.isalnum() or len(batch_id) > 80
            or not isinstance(refresh_id, str) or not refresh_id.isalnum() or len(refresh_id) > 80):
        raise ValueError('Valid batch and refresh identifiers are required')
    path = Path(host.evidence) / 'g05-candidate-batches' / (batch_id + '.json')
    row = json.loads(path.read_bytes())
    if row.get('status') != 'ready_for_planner' or binding_id(binding) != row['binding']:
        raise ValueError('Refresh requires the batch observation and a ready batch')
    if binding_id(host.frame()[0]) != row['binding']:
        raise ValueError('Cannot refresh after the observation changes')
    if (path.parent / (batch_id + '.choice.json')).exists():
        raise ValueError('Cannot refresh after the planner has selected a route')
    receipt_path = path.parent / (batch_id + '.refresh.' + refresh_id + '.json')
    refresh = {'status': 'intent_recorded', 'binding': row['binding'], 'candidates': []}
    try:
        exclusive(receipt_path, encoded(refresh))
    except FileExistsError:
        refresh = json.loads(receipt_path.read_bytes())
        if refresh.get('binding') != row['binding'] or refresh.get('status') == 'completed':
            raise RuntimeError('Refresh already exists; inspect recorded requests')
        if refresh.get('status') not in {'intent_recorded', 'proposal_intent', 'collecting'}:
            raise ValueError('Refresh state cannot be recovered automatically')
    for item in row['candidates']:
        completed = {entry['candidate_index'] for entry in refresh['candidates']}
        if item['candidate_index'] in completed or item['status'] != 'pending':
            continue
        index, seed = item['candidate_index'], item['seed']
        call_id = 'refresh-' + hashlib.sha256(f'{batch_id}:{refresh_id}:{index}'.encode()).hexdigest()[:40]
        refresh.update(status='proposal_intent', next_index=index, next_call_id=call_id)
        atomic(receipt_path, refresh)
        receipt = host.execute(call_id, 'g05_propose',
            {'task': row['task'], 'seed': seed, 'candidate_index': index}, binding)
        response = receipt.get('response') or {}
        status = response.get('status')
        if status not in {'candidate', 'pending', 'candidate_rejected'}:
            raise ValueError('Unexpected proposal receipt; inspect refresh intent')
        if status == 'candidate':
            item['summary'] = summarize_candidate(host, response, binding)
        item.update(status=status, receipt=receipt)
        refresh['candidates'].append({'candidate_index': index, 'call_id': call_id, 'status': status})
        refresh['status'] = 'collecting'
        atomic(path, row)
        atomic(receipt_path, refresh)
    refresh['status'] = 'completed'; atomic(receipt_path, refresh)
    return row


def record_choice(host, batch_id, choice, binding):
    """Record a planner's candidate choice or an explicit geometric fallback."""
    folder = Path(host.evidence) / 'g05-candidate-batches'
    if not isinstance(batch_id, str) or not batch_id.isalnum():
        raise ValueError('Invalid batch identifier')
    import json
    path = folder / (batch_id + '.json')
    batch = json.loads(path.read_bytes())
    if batch.get('status') != 'ready_for_planner' or binding_id(binding) != batch['binding']:
        raise ValueError('Choice requires a ready batch and its captured observation')
    if binding_id(host.frame()[0]) != batch['binding']:
        raise ValueError('Cannot choose after the observation changes')
    if not isinstance(choice, dict) or set(choice) != {'route', 'candidate_index', 'reason'}:
        raise ValueError('Choice requires route, candidate index and reason')
    if not isinstance(choice['reason'], str) or not choice['reason'].strip():
        raise ValueError('A decision reason is required')
    if choice['route'] == 'geometry':
        if choice['candidate_index'] is not None:
            raise ValueError('Geometric fallback cannot select a candidate')
        selected = None
    elif choice['route'] == 'g05':
        selected = next((item for item in batch['candidates']
                         if type(choice['candidate_index']) is int and
                         item['candidate_index'] == choice['candidate_index']), None)
        if selected is None or selected['status'] != 'candidate':
            raise ValueError('Selected candidate is not available')
        if selected['summary']['steps'][0]['initial_state_bounds'] != 'pass':
            raise ValueError('Selected candidate failed its initial bound')
        stage_bridge = getattr(host, 'stage_bridge', None)
        stage = stage_bridge._active() if stage_bridge is not None else None
        geometry = selected['summary'].get('kinematic_geometry') or {}
        progress = geometry.get('target_distance_reduction_m')
        if (stage and (stage.get('goal') or {}).get('type') == 'grasp_stow'
                and isinstance(progress, (int, float)) and progress < -0.001):
            raise ValueError('Candidate retreats from the tracked grasp target; choose another candidate or geometry')
    else:
        raise ValueError('Unknown route')
    decision = {'batch_id': batch_id, 'binding': batch['binding'], 'choice': choice,
                'selected_response': selected['receipt']['response'] if selected else None,
                'batch_sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
    exclusive(folder / (batch_id + '.choice.json'), encoded(decision))
    return decision


def execute_choice(host, batch_id, binding, monitor, maximum_steps=8, active_arm=None, hold_gripper=False):
    """Execute a recorded choice only with per-step measured supervision."""
    import json
    if not isinstance(batch_id, str) or not batch_id.isalnum() or not callable(monitor):
        raise ValueError('A valid batch and per-step monitor are required')
    folder = Path(host.evidence) / 'g05-candidate-batches'
    path = folder / (batch_id + '.json')
    decision = json.loads((folder / (batch_id + '.choice.json')).read_bytes())
    if decision['batch_sha256'] != hashlib.sha256(path.read_bytes()).hexdigest():
        raise ValueError('Candidate batch changed after selection')
    if decision['binding'] != binding_id(binding) or binding_id(host.frame()[0]) != decision['binding']:
        raise ValueError('Selected candidate belongs to a stale observation')
    if decision['choice']['route'] == 'geometry':
        return {'status': 'geometry_handoff', 'physical_commands_sent': 0}
    batch = json.loads(path.read_bytes())
    index = decision['choice']['candidate_index']
    selected = next(item for item in batch['candidates'] if item['candidate_index'] == index)
    response = decision['selected_response']
    if response != selected['receipt']['response']:
        raise ValueError('Recorded choice differs from the candidate batch')
    current_summary = summarize_candidate(host, response, binding)
    if current_summary != selected['summary']:
        raise ValueError('Candidate differs from the summary shown to the planner')
    return execute_chunk(host, response, binding, maximum_steps=maximum_steps, monitor=monitor,
                         active_arm=active_arm, hold_gripper=hold_gripper)
