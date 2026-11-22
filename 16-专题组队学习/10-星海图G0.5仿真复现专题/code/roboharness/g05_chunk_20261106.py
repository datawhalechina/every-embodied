"""Execute a short, observation-checked prefix of one G0.5 action sequence.

Input is the completed local worker result and its published first candidate.
Every physical step has a distinct proposal bound to the latest observation.
The caller must use a real qualified host and separately qualify this adapter.
"""
import hashlib
import json
from pathlib import Path

import numpy as np

from action_domain import gripper_domain
from bridge import PART_WIDTHS, joint_tool_arguments
from joint_domain import arm_position_domain
from proposal_exchange import encoded, exclusive, sha
from workspace_agent_host import atomic


def load_sequence(host, response, binding):
    """Bind the worker's completed action array to its published candidate."""
    if response.get('status') != 'candidate':
        raise ValueError('A completed executable candidate is required')
    identifier = response['request_id']
    if not identifier.startswith('proposal_') or not identifier[9:].isalnum():
        raise ValueError('Invalid candidate request identifier')
    proposal_path = Path(response['proposal_file']).resolve()
    if not proposal_path.is_relative_to(host.workspace / 'perception/g05_proposals'):
        raise ValueError('Candidate is outside this workspace')
    if sha(proposal_path) != response['proposal_sha256']:
        raise ValueError('Published candidate changed')
    proposal = json.loads(proposal_path.read_bytes())
    if (proposal.get('proposal_id') != identifier or proposal.get('episode_id') != binding['episode_id']
            or proposal.get('sequence') != binding['sequence']):
        raise ValueError('Candidate is bound to another observation')
    worker = host.exchange.worker
    ledger = json.loads((worker / 'calls' / (identifier + '.json')).read_bytes())
    result_path = (worker / ledger['result_file']).resolve()
    if not result_path.is_relative_to(worker) or sha(result_path) != ledger.get('result_sha256'):
        raise ValueError('Worker result is missing or changed')
    result = json.loads(result_path.read_bytes())
    if (ledger.get('status') != 'completed' or result.get('request_id') != identifier
            or result.get('status') != 'candidate' or result.get('proposal') != proposal):
        raise ValueError('Worker result and candidate do not match')
    raw_path = (worker / result['raw_action_file']).resolve()
    if not raw_path.is_relative_to(worker) or not raw_path.is_file():
        raise ValueError('Worker action sequence is missing')
    raw_hash = sha(raw_path)
    if raw_hash != result.get('raw_action_sha256'):
        raise ValueError('Worker action sequence changed or lacks a recorded digest')
    with np.load(raw_path, allow_pickle=False) as archive:
        arrays = {key: np.array(archive[key], dtype=float, copy=True) for key in PART_WIDTHS}
    if (any(x.ndim != 2 or x.shape[1] != PART_WIDTHS[key] or not np.isfinite(x).all()
            for key, x in arrays.items()) or len({len(x) for x in arrays.values()}) != 1
            or not len(arrays['left_arm'])):
        raise ValueError('Action sequence has invalid dimensions or values')
    return proposal, arrays, raw_hash


def project_step(proposal, arrays, index, binding):
    """Apply the qualified action domains to one fresh observed state."""
    q = binding['proprioception']
    parts = {key: values[index].tolist() for key, values in arrays.items()}
    parts, grip_changes = gripper_domain(parts)
    parts, joint_changes = arm_position_domain(parts, q)
    parts['lower_body'] = [0., 0., 0., *q[57:61]]
    item = {**proposal, 'proposal_id': proposal['proposal_id'] + '-step-' + str(index),
            'sequence': binding['sequence'], 'action': parts}
    joint_tool_arguments(item, q, binding['episode_id'], binding['sequence'])
    return item, {'gripper_domain_projection': grip_changes, 'joint_domain_projection': joint_changes}


def execute_chunk(host, response, binding, maximum_steps=8, monitor=None, active_arm=None, hold_gripper=False):
    """Execute a bounded prefix and optionally check measured progress after each step.

    A monitor receives the before and after bindings and physical receipt, and
    returns a decision and evidence. An invalid or failed monitor stops the
    chunk; a saved physical intent is never replayed.
    """
    if type(maximum_steps) is not int or not 1 <= maximum_steps <= 8:
        raise ValueError('Chunk length must be between one and eight')
    if active_arm not in {None, 'left', 'right'} or type(hold_gripper) is not bool:
        raise ValueError('A valid arm scope and gripper hold policy are required')
    proposal, arrays, raw_hash = load_sequence(host, response, binding)
    first, _ = project_step(proposal, arrays, 0, binding)
    if first['action'] != proposal['action']:
        raise ValueError('First projected action differs from the qualified candidate')
    identifier = proposal['proposal_id']
    folder = host.evidence / 'g05-chunks'
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / (identifier + '.json')
    row = {'status': 'intent_recorded', 'request_id': identifier, 'raw_sha256': raw_hash,
           'available_steps': len(arrays['left_arm']), 'requested_steps': maximum_steps,
           'initial_sequence': binding['sequence'], 'active_arm': active_arm,
           'hold_gripper': hold_gripper, 'steps': []}
    try:
        exclusive(path, encoded(row))
    except FileExistsError as error:
        raise RuntimeError('Chunk already exists; inspect evidence and never replay') from error
    for index in range(min(maximum_steps, len(arrays['left_arm']))):
        current, _ = host.frame()
        if current['episode_id'] != binding['episode_id'] or current['sequence'] != binding['sequence'] + index:
            raise RuntimeError('Current observation changed during the action chunk')
        try:
            step, projection = project_step(proposal, arrays, index, current)
            if active_arm is not None:
                other = 'right' if active_arm == 'left' else 'left'
                start = 30 if other == 'right' else 3
                step['action'][other + '_arm'] = list(current['proprioception'][start:start + 7])
                step['action'][other + '_gripper'] = [0.0]
                if hold_gripper:
                    step['action'][active_arm + '_gripper'] = [0.0]
                joint_tool_arguments(step, current['proprioception'], current['episode_id'], current['sequence'])
                projection['execution_scope'] = {'active_arm': active_arm, 'other_arm': 'hold_measured',
                    'other_gripper': 'retain_effort', 'active_gripper': 'retain_effort' if hold_gripper else 'policy'}
        except ValueError as error:
            row.update(status='stopped_by_boundary', stop_reason=str(error))
            atomic(path, row)
            return row
        step_path = host.workspace / 'perception/g05_proposals' / (step['proposal_id'] + '.json')
        exclusive(step_path, encoded(step))
        step_id = 'chunk-' + hashlib.sha256(step['proposal_id'].encode()).hexdigest()[:40]
        row.update(status='step_intent', next_step=index, next_call_id=step_id)
        atomic(path, row)
        receipt = host.execute(step_id, 'g05_step', {'proposal_file': str(step_path)}, current)
        if receipt.get('error') or receipt.get('joint_commands_sent') != 1 or host.unresolved():
            raise RuntimeError('Step lacks an unambiguous one-command receipt')
        after, _ = host.frame()
        if after['episode_id'] != current['episode_id'] or after['sequence'] != current['sequence'] + 1:
            raise RuntimeError('Step did not produce the expected next observation')
        row['steps'].append({'index': index, 'sequence_before': current['sequence'],
                             'sequence_after': after['sequence'], 'projection': projection,
                             'receipt': receipt})
        row['status'] = 'in_progress'
        atomic(path, row)
        if monitor is not None:
            try:
                verdict = monitor(current, after, receipt)
                if (not isinstance(verdict, dict) or verdict.get('decision') not in
                        {'continue', 'stop_success', 'stop_replan'} or
                        not isinstance(verdict.get('evidence'), dict)):
                    raise ValueError('Monitor must return a decision and evidence')
                # Ensure the verdict is persistable before allowing another command.
                encoded(verdict)
            except Exception as error:
                row.update(status='monitor_error', stop_reason=str(error))
                atomic(path, row)
                raise
            row['steps'][-1]['monitor'] = verdict
            if verdict['decision'] != 'continue':
                row.update(status=verdict['decision'], stop_reason='monitor_decision')
                atomic(path, row)
                return row
            atomic(path, row)
    row['status'] = 'completed'
    atomic(path, row)
    return row
