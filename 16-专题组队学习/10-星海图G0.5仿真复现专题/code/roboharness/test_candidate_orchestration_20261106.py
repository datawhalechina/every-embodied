"""Exercise candidate identity, selection and measured-stop contracts."""
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace

from candidate_orchestration_20261106 import execute_choice, gripper_summary, propose_batch, record_choice, refresh_batch, resume_batch, summarize_candidate
from g05_chunk_20261106 import execute_chunk
from g05_chunk_20261106 import load_sequence
from candidate_geometry_20261106 import summarize_geometry
from test_g05_chunk_20261106 import Host, setup_candidate


class BatchHost(Host):
    def __init__(self, root):
        super().__init__(root)
        self.calls = []

    def execute(self, call_id, tool, args, binding):
        self.calls.append((call_id, tool, args, binding['sequence']))
        return {'response': {'status': 'candidate', 'request_id': str(args['candidate_index'])}}


class OrchestrationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.host = BatchHost(Path(self.tmp.name))

    @staticmethod
    def summary(host, response, binding):
        return {'steps': [{'initial_state_bounds': 'pass'}], 'request_id': response['request_id']}

    def test_three_candidates_share_frame_and_preserve_task_and_seed(self):
        binding = self.host.frame()[0]
        with patch('candidate_orchestration_20261106.summarize_candidate', self.summary):
            batch = propose_batch(self.host, 'batch01', 'pick up the cup', [4, 9, 12], binding)
        self.assertEqual(batch['status'], 'ready_for_planner')
        self.assertEqual([item[2] for item in self.host.calls], [
            {'task': 'pick up the cup', 'seed': seed, 'candidate_index': index}
            for index, seed in enumerate([4, 9, 12])])
        self.assertEqual([item[3] for item in self.host.calls], [43, 43, 43])
        selected = record_choice(self.host, 'batch01',
            {'route': 'g05', 'candidate_index': 1, 'reason': 'second candidate'}, binding)
        self.assertEqual(selected['selected_response']['request_id'], '1')
        with self.assertRaises(FileExistsError):
            record_choice(self.host, 'batch01',
                {'route': 'geometry', 'candidate_index': None, 'reason': 'fallback'}, binding)
        with self.assertRaisesRegex(RuntimeError, 'already exists'):
            propose_batch(self.host, 'batch01', 'pick up the cup', [4], binding)

    def test_choice_rejects_changed_frame(self):
        binding = self.host.frame()[0]
        with patch('candidate_orchestration_20261106.summarize_candidate', self.summary):
            propose_batch(self.host, 'batch02', 'pick up the cup', [4], binding)
        self.host.sequence += 1
        with self.assertRaisesRegex(ValueError, 'observation changes'):
            record_choice(self.host, 'batch02',
                {'route': 'geometry', 'candidate_index': None, 'reason': 'fallback'}, binding)

    def test_pending_candidate_can_be_collected_without_new_identity(self):
        binding = self.host.frame()[0]
        calls = []
        def execute(call_id, tool, args, observed):
            calls.append((call_id, dict(args)))
            return {'response': {'status': 'pending' if len(calls) == 1 else 'candidate',
                                 'request_id': 'same-request'}}
        self.host.execute = execute
        with patch('candidate_orchestration_20261106.summarize_candidate', self.summary):
            first = propose_batch(self.host, 'pending1', 'pick up the cup', [14], binding)
            second = refresh_batch(self.host, 'pending1', 'refresh1', binding)
        self.assertEqual(first['candidates'][0]['status'], 'pending')
        self.assertEqual(second['candidates'][0]['status'], 'candidate')
        self.assertEqual(calls[0][1], calls[1][1])
        self.assertNotEqual(calls[0][0], calls[1][0])
        with self.assertRaisesRegex(RuntimeError, 'already exists'):
            refresh_batch(self.host, 'pending1', 'refresh1', binding)

    def test_interrupted_nonphysical_batch_resumes_same_call_identity(self):
        binding = self.host.frame()[0]; calls = []; fault = [True]
        def execute(call_id, tool, args, observed):
            calls.append((call_id, args['candidate_index']))
            if args['candidate_index'] == 1 and fault[0]:
                fault[0] = False
                raise RuntimeError('interrupted request')
            return {'response': {'status': 'candidate', 'request_id': str(args['candidate_index'])}}
        self.host.execute = execute
        with patch('candidate_orchestration_20261106.summarize_candidate', self.summary):
            with self.assertRaisesRegex(RuntimeError, 'interrupted'):
                propose_batch(self.host, 'resume1', 'pick the object', [3, 4], binding)
            batch = resume_batch(self.host, 'resume1', binding)
        self.assertEqual(batch['status'], 'ready_for_planner')
        self.assertEqual([index for _, index in calls], [0, 1, 1])
        self.assertEqual(calls[1][0], calls[2][0])

    def test_interrupted_refresh_reconciles_same_nonphysical_call(self):
        binding = self.host.frame()[0]; calls = []; fault = [False]
        def execute(call_id, tool, args, observed):
            calls.append(call_id)
            if len(calls) == 1:
                return {'response': {'status': 'pending', 'request_id': 'same'}}
            if not fault[0]:
                fault[0] = True
                raise RuntimeError('refresh interrupted')
            return {'response': {'status': 'candidate', 'request_id': 'same'}}
        self.host.execute = execute
        with patch('candidate_orchestration_20261106.summarize_candidate', self.summary):
            propose_batch(self.host, 'refresh2', 'pick the object', [3], binding)
            with self.assertRaisesRegex(RuntimeError, 'interrupted'):
                refresh_batch(self.host, 'refresh2', 'retry1', binding)
            batch = refresh_batch(self.host, 'refresh2', 'retry1', binding)
        self.assertEqual(batch['candidates'][0]['status'], 'candidate')
        self.assertEqual(calls[1], calls[2])

    def test_summary_does_not_claim_predicted_object_motion(self):
        response, binding = setup_candidate(self.host)
        result = summarize_candidate(self.host, response, binding, limit=4)
        self.assertEqual(result['reference'], 'captured initial state only')
        self.assertIsNone(result['object_motion_prediction'])
        self.assertEqual(len(result['steps']), 4)
        self.assertEqual(result['steps'][0]['initial_state_bounds'], 'pass')
        self.assertEqual(result['steps'][0]['grippers']['left']['effective_intent'],'retain_effort')

    def test_gripper_summary_matches_active_arm_and_hold_policy(self):
        action={'left_gripper':[-.9],'right_gripper':[.9]}
        scoped=gripper_summary(action,{'arm':'right','gripper_policy':'policy'})
        self.assertEqual(scoped['left']['effective_intent'],'retain_effort')
        self.assertEqual(scoped['right']['effective_intent'],'open')
        self.assertEqual(scoped['left']['model_normalized'],-.9)
        held=gripper_summary(action,{'arm':'right','gripper_policy':'hold'})
        self.assertEqual(held['right']['effective_intent'],'retain_effort')

    def test_changed_action_sequence_is_rejected_before_execution(self):
        response, binding = setup_candidate(self.host)
        with (self.host.worker / 'actions.npz').open('ab') as stream:
            stream.write(b'changed')
        with self.assertRaisesRegex(ValueError, 'sequence changed'):
            summarize_candidate(self.host, response, binding)
        self.assertEqual(self.host.sequence, 43)

    def test_real_kinematics_summarize_held_target_without_object_prediction(self):
        response, binding = setup_candidate(self.host)
        stage = {'arm': 'left', 'entry': {'tracking': {'binding': binding, 'track_id': 'handle1',
                                                       'position_m': [0.5, 0.0, 0.5]}}}
        self.host.stage_bridge = SimpleNamespace(_active=lambda: stage)
        result = summarize_candidate(self.host, response, binding, limit=2)
        geometry = result['kinematic_geometry']
        self.assertEqual(geometry['target_track_id'], 'handle1')
        self.assertEqual(geometry['frame'], 'robot_base')
        self.assertAlmostEqual(geometry['initial_distance_m'], geometry['steps'][0]['distance_to_current_target_m'])
        self.assertEqual(geometry['last_valid_step_index'],1)
        self.assertAlmostEqual(geometry['net_eef_displacement_m'],
            math.dist(geometry['initial_eef_position_m'],geometry['steps'][1]['eef_position_m']))
        self.assertAlmostEqual(geometry['target_distance_reduction_m'],
            geometry['initial_distance_m']-geometry['steps'][1]['distance_to_current_target_m'])
        self.assertIsNone(result['object_motion_prediction'])

    def test_geometry_summary_marks_all_rejected_steps_unavailable(self):
        response,binding=setup_candidate(self.host);proposal,arrays,_=load_sequence(self.host,response,binding)
        stage={'arm':'left','entry':{'tracking':{'binding':binding,'track_id':'handle1',
                                                'position_m':[0.5,0.,0.5]}}}
        with patch('candidate_geometry_20261106.project_step',side_effect=ValueError('outside limits')):
            geometry=summarize_geometry(proposal,arrays,binding,stage,limit=2)
        self.assertIsNone(geometry['last_valid_step_index'])
        self.assertIsNone(geometry['net_eef_displacement_m'])
        self.assertIsNone(geometry['target_distance_reduction_m'])
        self.assertEqual(len(geometry['steps']),2)

    def test_geometry_prefix_stops_before_intermediate_projection_failure(self):
        from g05_chunk_20261106 import project_step
        response,binding=setup_candidate(self.host);proposal,arrays,_=load_sequence(self.host,response,binding)
        stage={'arm':'left','entry':{'tracking':{'binding':binding,'track_id':'handle1',
                                                'position_m':[0.5,0.,0.5]}}}
        def projection(proposal,arrays,index,binding):
            if index==1:raise ValueError('outside limits')
            return project_step(proposal,arrays,index,binding)
        with patch('candidate_geometry_20261106.project_step',side_effect=projection):
            geometry=summarize_geometry(proposal,arrays,binding,stage,limit=3)
        self.assertIn('unavailable',geometry['steps'][1])
        self.assertIn('eef_position_m',geometry['steps'][2])
        self.assertEqual(geometry['initial_state_valid_prefix_steps'],1)
        self.assertEqual(geometry['last_valid_step_index'],0)

    def test_selected_candidate_is_executed_only_until_measured_stop(self):
        response, binding = setup_candidate(self.host)
        def execute(call_id, tool, args, frame):
            if tool == 'g05_propose':
                return {'response': response}
            return Host.execute(self.host, call_id, tool, args, frame)
        self.host.execute = execute
        propose_batch(self.host, 'batch03', 'pick up the cup', [17], binding)
        record_choice(self.host, 'batch03',
            {'route': 'g05', 'candidate_index': 0, 'reason': 'planner selected'}, binding)
        result = execute_choice(self.host, 'batch03', binding,
            lambda before, after, receipt: {'decision': 'stop_replan',
                                             'evidence': {'sequence': after['sequence']}})
        self.assertEqual(result['status'], 'stop_replan')
        self.assertEqual(len(result['steps']), 1)
        self.assertEqual(self.host.sequence, 44)

    def test_geometric_choice_does_not_send_a_command(self):
        binding = self.host.frame()[0]
        with patch('candidate_orchestration_20261106.summarize_candidate', self.summary):
            propose_batch(self.host, 'batch04', 'pick up the cup', [4], binding)
        record_choice(self.host, 'batch04',
            {'route': 'geometry', 'candidate_index': None, 'reason': 'planner rejected candidates'}, binding)
        result = execute_choice(self.host, 'batch04', binding, lambda *args: {})
        self.assertEqual(result['status'], 'geometry_handoff')
        self.assertEqual(self.host.sequence, 43)

    def test_monitor_stops_before_next_action_and_records_observation(self):
        response, binding = setup_candidate(self.host)
        # Use the physical-test host implementation for actual simulated receipts.
        self.host.execute = Host.execute.__get__(self.host)
        result = execute_chunk(self.host, response, binding, monitor=lambda before, after, receipt: {
            'decision': 'stop_replan' if after['sequence'] == 45 else 'continue',
            'evidence': {'observed_sequence': after['sequence']}})
        self.assertEqual(result['status'], 'stop_replan')
        self.assertEqual(len(result['steps']), 2)
        self.assertEqual(self.host.sequence, 45)
        self.assertEqual(result['steps'][-1]['monitor']['evidence']['observed_sequence'], 45)

    def test_invalid_monitor_result_stops_and_cannot_replay(self):
        response, binding = setup_candidate(self.host)
        self.host.execute = Host.execute.__get__(self.host)
        with self.assertRaisesRegex(ValueError, 'Monitor must return'):
            execute_chunk(self.host, response, binding, monitor=lambda *args: {})
        ledger = next((self.host.evidence / 'g05-chunks').glob('*.json'))
        self.assertEqual(json.loads(ledger.read_text())['status'], 'monitor_error')
        self.assertEqual(self.host.sequence, 44)
        with self.assertRaisesRegex(RuntimeError, 'never replay'):
            execute_chunk(self.host, response, binding)


if __name__ == '__main__':
    unittest.main()
