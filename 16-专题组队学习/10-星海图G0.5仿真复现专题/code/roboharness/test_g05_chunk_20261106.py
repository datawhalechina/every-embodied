"""Check actual multi-step dispatch contracts without a simulator or model."""
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

import numpy as np

from bridge import SEMANTICS
from g05_chunk_20261106 import execute_chunk
from proposal_exchange import encoded, sha


class Host:
    def __init__(self, root):
        self.workspace = root / 'workspace'
        (self.workspace / 'perception/g05_proposals').mkdir(parents=True)
        self.evidence = root / 'host'; self.evidence.mkdir()
        self.worker = root / 'worker'; (self.worker / 'calls').mkdir(parents=True)
        self.exchange = SimpleNamespace(worker=self.worker)
        self.sequence = 43
        self.q = [0.0] * 65
        self.q[6] = self.q[33] = -2.0
        self.fail_at = None

    def frame(self):
        return ({'episode_id': 'e', 'session_id': 's', 'sequence': self.sequence,
                 'head_image_id': 'img' + str(self.sequence), 'proprioception': list(self.q)}, [])

    def execute(self, call_id, tool, args, binding):
        if self.sequence == self.fail_at:
            return {'tool_name': tool, 'joint_commands_sent': 0}
        proposal = json.loads(Path(args['proposal_file']).read_text())
        for side, start in [('left', 3), ('right', 30)]:
            self.q[start:start+7] = proposal['action'][side+'_arm']
        self.sequence += 1
        return {'tool_name': tool, 'joint_commands_sent': 1}

    def unresolved(self): return []


def setup_candidate(host, jump_step=None):
    binding, _ = host.frame()
    arrays = {'left_arm': np.tile(host.q[3:10], (27, 1)),
              'right_arm': np.tile(host.q[30:37], (27, 1)),
              'left_gripper': np.zeros((27, 1)), 'right_gripper': np.zeros((27, 1)),
              'lower_body': np.zeros((27, 7))}
    arrays['left_arm'][:, 0] = np.arange(27) * .001
    if jump_step is not None: arrays['left_arm'][jump_step, 0] = 1.0
    identifier = 'proposal_abcdef123456'
    raw = host.worker / 'actions.npz'; np.savez_compressed(raw, **arrays)
    parts = {key: values[0].tolist() for key, values in arrays.items()}
    parts['lower_body'] = [0., 0., 0., *host.q[57:61]]
    proposal = {'semantics': SEMANTICS, 'policy_fingerprint': 'a'*64,
                'proposal_id': identifier, 'episode_id': 'e', 'sequence': 43, 'action': parts}
    p = host.workspace / 'perception/g05_proposals' / (identifier + '.json')
    p.write_bytes(encoded(proposal))
    result = {'status': 'candidate', 'request_id': identifier,
              'proposal': proposal, 'raw_action_file': raw.name, 'raw_action_sha256': sha(raw)}
    result_path = host.worker/'result.json'; result_path.write_bytes(encoded(result))
    (host.worker/'calls'/(identifier+'.json')).write_bytes(encoded(
        {'status': 'completed', 'result_file': result_path.name, 'result_sha256': sha(result_path)}))
    return {'status': 'candidate', 'request_id': identifier,
            'proposal_file': str(p), 'proposal_sha256': sha(p)}, binding


class ChunkTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.host = Host(Path(self.tmp.name))

    def tearDown(self): self.tmp.cleanup()

    def test_eight_distinct_steps_use_successive_observations(self):
        response, binding = setup_candidate(self.host)
        row = execute_chunk(self.host, response, binding)
        self.assertEqual(row['status'], 'completed')
        self.assertEqual(len(row['steps']), 8)
        self.assertEqual([x['sequence_after'] for x in row['steps']], list(range(44, 52)))
        with self.assertRaisesRegex(RuntimeError, 'never replay'):
            execute_chunk(self.host, response, binding)

    def test_out_of_range_stops_before_unsafe_step(self):
        response, binding = setup_candidate(self.host, jump_step=3)
        row = execute_chunk(self.host, response, binding)
        self.assertEqual(row['status'], 'stopped_by_boundary')
        self.assertEqual(len(row['steps']), 3)
        self.assertEqual(self.host.sequence, 46)

    def test_unconfirmed_step_remains_unresolved_and_cannot_replay(self):
        response, binding = setup_candidate(self.host)
        self.host.fail_at = 44
        with self.assertRaisesRegex(RuntimeError, 'unambiguous'):
            execute_chunk(self.host, response, binding)
        path = next((self.host.evidence/'g05-chunks').glob('*.json'))
        self.assertEqual(json.loads(path.read_text())['status'], 'step_intent')
        with self.assertRaisesRegex(RuntimeError, 'never replay'):
            execute_chunk(self.host, response, binding)

    def test_active_arm_scope_preserves_other_arm_and_gripper_efforts(self):
        response, binding = setup_candidate(self.host)
        raw = self.host.worker/'actions.npz'
        with np.load(raw, allow_pickle=False) as archive:
            arrays = {key: archive[key].copy() for key in archive.files}
        arrays['right_arm'][1, 0] = .02
        arrays['right_gripper'][1, 0] = 1.0
        arrays['left_gripper'][1, 0] = 1.0
        np.savez_compressed(raw, **arrays)
        result_path = self.host.worker/'result.json'
        result = json.loads(result_path.read_text()); result['raw_action_sha256'] = sha(raw)
        result_path.write_bytes(encoded(result))
        ledger_path = self.host.worker/'calls'/'proposal_abcdef123456.json'
        ledger = json.loads(ledger_path.read_text()); ledger['result_sha256'] = sha(result_path)
        ledger_path.write_bytes(encoded(ledger))
        proposals = []
        original = self.host.execute
        def record(call_id, tool, args, current):
            proposals.append(json.loads(Path(args['proposal_file']).read_text()))
            return original(call_id, tool, args, current)
        self.host.execute = record
        execute_chunk(self.host, response, binding, maximum_steps=2, active_arm='left', hold_gripper=True)
        self.assertEqual(proposals[1]['action']['right_arm'], binding['proprioception'][30:37])
        self.assertEqual(proposals[1]['action']['right_gripper'], [0.0])
        self.assertEqual(proposals[1]['action']['left_gripper'], [0.0])


if __name__ == '__main__': unittest.main()
