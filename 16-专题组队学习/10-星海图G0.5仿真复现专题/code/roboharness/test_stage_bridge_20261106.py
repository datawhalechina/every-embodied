"""Exercise observation-backed entry, handoff and transition retrieval."""
import json
import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from proposal_exchange import encoded, sha
from stage_bridge_20261106 import StageBridge
from orchestrated_host_20261106 import OrchestratedHost


def binding(seq):
    return {'episode_id': 'episode', 'session_id': 'session', 'sequence': seq,
            'head_image_id': 'image' + str(seq), 'proprioception': [0.0] * 65,
            'images': [{'role': 'left_wrist', 'sha256': 'a' * 64}]}


def tracking(seq, x=0.5, name='handle'):
    return {'available': True, 'xyz_in_robot_base_coord_frame': 'current_robot_base',
            'tracks': {name: {'episode_id': 'episode', 'observation_sequence': seq,
                'status': 'observed', 'confidence': 0.9, 'track_id': 'track1',
                'xyz_in_robot_base_coord_m': [x, 0.0, 0.0]}}}


class Host:
    def __init__(self, root, skill):
        self.evidence = root / 'host'; (self.evidence / 'calls').mkdir(parents=True)
        self.workspace = root / 'workspace'; (self.workspace / 'skills').mkdir(parents=True)
        (self.workspace / 'skills/active.json').write_text(json.dumps({'active_skill': skill, 'revision': 1}))

    def call(self, call_id, tool, seq, response=None, args=None, track=None, head=None):
        receipt = {'tool_name': tool, 'response': response or {},
                   'task_skill_state': {'active_skill': json.loads((self.workspace/'skills/active.json').read_text())['active_skill']},
                   'latest_clickable_image_id': head or 'image' + str(seq), 'persistent_tracking': track}
        row = {'status': 'completed', 'tool': tool, 'arguments': args or {},
               'binding': {k: binding(seq)[k] for k in ('episode_id', 'session_id', 'sequence', 'head_image_id')},
               'receipt': receipt}
        (self.evidence/'calls'/(call_id+'.json')).write_bytes(encoded(row))


def bridge_task(host):
    return json.loads((host.evidence/'stage-bridge/active.json').read_bytes())['model_task']


class StageTests(unittest.TestCase):
    def setUp(self):
        geometry = patch('stage_bridge_20261106.current_target_distance', return_value=.1)
        geometry.start(); self.addCleanup(geometry.stop)
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def prepare(self, skill='open-doors-and-drawers'):
        host = Host(self.root, skill); bridge = StageBridge(host)
        goal = ({'type': 'tracked_displacement', 'direction': [1, 0, 0], 'target_m': .05}
                if skill == 'open-doors-and-drawers' else {'type': 'grasp_stow'})
        stage = bridge.plan('p1', {'task': 'perform the current subtask', 'target_label': 'drawer',
            'point': 'handle', 'arm': 'left', 'next_skill': 'pick-up-object', 'next_target_label':'toy', 'goal': goal,
            'seeds': [0, 1], 'maximum_steps': 8, 'gripper_policy': 'policy'}, binding(9))
        self.assertIn('drawer',stage['model_task'])
        self.assertIn('left arm',stage['model_task'])
        host.call('reach', 'move_to_reach_point', 10,
                  {'shoulder_distance_estimate': {'minimum_m': .4, 'threshold_m': .7}}, track=tracking(10))
        bridge.begin(binding(10), 'reach', 'reach')
        bridge.attach_batch(stage['stage_id'], 'batch1')
        return host, bridge, stage

    def drawer_artifacts(self, host):
        batches = host.evidence/'g05-candidate-batches'; batches.mkdir()
        response = {'request_id': 'proposal_abc'}
        frame_id = {k: binding(10)[k] for k in ('episode_id','session_id','sequence','head_image_id')}
        batch = {'batch_id':'batch1', 'task':bridge_task(host), 'seeds':[0,1], 'binding':frame_id,
                 'candidates':[{'candidate_index':0, 'status':'candidate', 'receipt':{'response':response}}]}
        path = batches/'batch1.json'; path.write_bytes(encoded(batch))
        (batches/'batch1.choice.json').write_bytes(encoded({'choice': {'route': 'g05', 'candidate_index':0},
            'selected_response': response, 'binding':frame_id, 'batch_sha256':sha(path)}))
        chunks = host.evidence/'g05-chunks'; chunks.mkdir()
        path = chunks/'proposal_abc.json'
        path.write_bytes(encoded({'status': 'stop_success', 'initial_sequence': 10, 'steps': [
            {'monitor': {'decision': 'stop_success', 'evidence': {'track_id': 'track1',
                'initial_position_m': [.5, 0., 0.], 'direction': [1., 0., 0.], 'target_m': .05,
                'directed_displacement_m': .06}}}]}))
        return path

    def test_drawer_handoff_and_geometry_matched_retrieval(self):
        host, bridge, _ = self.prepare(); self.drawer_artifacts(host)
        host.call('next', 'track_object_distance', 12, track=tracking(12, .56, 'toy'))
        row = bridge.finish(binding(12), {'next_tracking_call_id': 'next', 'next_point': 'toy',
            'visual_reason': 'opening exposes the intended object', 'grasp_call_id': None,
            'stow_call_id': None, 'reset_call_id': None, 'stow_kind': None,
            'exit_tracking_call_id':None, 'geometry_call_id':None})
        self.assertEqual(row['status'], 'completed')
        self.assertEqual(row['handoff']['planner_view']['source'], 'planner_assertion_from_current_views')
        host.call('matching', 'track_object_distance', 12, track=tracking(12, .56))
        matches = bridge.matches(binding(12), 'matching', 'handle', 'open-doors-and-drawers', 'drawer', 'left')
        self.assertEqual(len(matches['matches']), 1)
        self.assertEqual(matches['use'], 'suggestions_only_recheck_current_observation')

        # The drawer's recorded next target is retrievable by the pickup stage.
        pickup = bridge.matches(binding(12), 'next', 'toy', 'pick-up-object', 'toy', 'left')
        self.assertEqual(len(pickup['matches']), 1)
        transition=pickup['matches'][0]
        self.assertEqual(transition['match_kind'],'previous_stage_handoff')
        self.assertEqual(transition['suggested_entry']['target_in_robot_base_m'],[.56,0.,0.])
        self.assertEqual(transition['entry_arm_max_delta_rad'],0.)
        self.assertEqual(bridge.matches(binding(12),'next','toy','pick-up-object','other toy','left')['matches'],[])

        # Older transition records lack a saved handoff joint state.
        record=next(bridge.experience.glob('*.json'))
        legacy=json.loads(record.read_bytes());del legacy['handoff']['proprioception']
        record.write_bytes(encoded(legacy))
        old=bridge.matches(binding(12),'next','toy','pick-up-object','toy','left')['matches'][0]
        self.assertEqual(old['readiness'],'transition_suggested')
        self.assertIsNone(old['suggested_entry']['arm_joint_positions_rad'])

    def test_drawer_does_not_accept_wrong_target_or_no_measurement(self):
        host, bridge, _ = self.prepare(); path = self.drawer_artifacts(host)
        data = json.loads(path.read_text()); data['steps'][0]['monitor']['evidence']['track_id'] = 'wrong'
        path.write_bytes(encoded(data)); host.call('next', 'track_object_distance', 12, track=tracking(12, name='toy'))
        args = {'next_tracking_call_id': 'next', 'next_point': 'toy', 'visual_reason': 'visible',
                'grasp_call_id': None, 'stow_call_id': None, 'reset_call_id': None, 'stow_kind': None,
                'exit_tracking_call_id':None, 'geometry_call_id':None}
        with self.assertRaisesRegex(ValueError, 'does not match'):
            bridge.finish(binding(12), args)
        self.assertEqual(bridge._active()['status'], 'batch_ready')
        self.assertEqual(list(bridge.experience.glob('*.json')), [])

    def test_next_stage_planning_receives_prior_drawer_handoff(self):
        host, bridge, _ = self.prepare();self.drawer_artifacts(host)
        host.call('next','track_object_distance',12,track=tracking(12,.56,'toy'))
        bridge.finish(binding(12),{'next_tracking_call_id':'next','next_point':'toy',
            'visual_reason':'toy visible','grasp_call_id':None,'stow_call_id':None,
            'reset_call_id':None,'stow_kind':None,'exit_tracking_call_id':None,'geometry_call_id':None})
        (host.workspace/'skills/active.json').write_text(json.dumps({'active_skill':'pick-up-object','revision':2}))
        orchestrator=OrchestratedHost.__new__(OrchestratedHost)
        orchestrator.evidence=host.evidence;orchestrator.stage_bridge=bridge
        orchestrator._internal_depth=0;orchestrator._check_orchestration=lambda *unused:None
        params={'task':'pick the toy','target_label':'toy','point':'toy','arm':'left',
                'next_skill':'place-object-in-container','next_target_label':'box',
                'goal':{'type':'grasp_stow'},'seeds':[0],'maximum_steps':4,'gripper_policy':'policy'}
        result=orchestrator.execute('pickup-plan','g05_stage',{'action':'plan','parameters':params},binding(12))
        matches=result['response']['experience']['matches']
        self.assertEqual(len(matches),1)
        self.assertEqual(matches[0]['match_kind'],'previous_stage_handoff')
        self.assertEqual(matches[0]['suggested_entry']['target_in_robot_base_m'],[.56,0.,0.])

    def test_geometry_handoff_corrects_tracked_position_for_base_motion(self):
        host, bridge, _ = self.prepare()
        batches = host.evidence/'g05-candidate-batches'; batches.mkdir()
        frame_id = {k: binding(10)[k] for k in ('episode_id','session_id','sequence','head_image_id')}
        path = batches/'batch1.json'; path.write_bytes(encoded({'batch_id':'batch1',
            'task':bridge._active()['model_task'], 'seeds':[0,1], 'binding':frame_id, 'candidates':[]}))
        (batches/'batch1.choice.json').write_bytes(encoded({'choice': {'route':'geometry'},
            'binding':frame_id, 'batch_sha256':sha(path)}))
        host.call('base', 'adjust_chassis', 11, {'actual':{'forward_m':-.1,'translation_m':0.,'spin_deg':0.},
            'linear_cross_track_m':0.,'linear_yaw_drift_deg':0.,'linear_target_reached':True,'timed_out':False},
            {'forward':-.1,'translation':0.,'spin':0.})
        host.call('handle-end', 'track_object_distance', 12, track=tracking(12,.7))
        host.call('next', 'track_object_distance', 12, track=tracking(12,.6,'toy'))
        result = bridge.finish(binding(12), {'next_tracking_call_id':'next','next_point':'toy',
            'visual_reason':'toy visible in drawer', 'exit_tracking_call_id':'handle-end', 'geometry_call_id':'base',
            'grasp_call_id':None,'stow_call_id':None,'reset_call_id':None,'stow_kind':None})
        measured = result['handoff']['exit_evidence']
        self.assertEqual(measured['kind'], 'tracked_displacement_with_measured_base_translation')
        self.assertAlmostEqual(measured['directed_displacement_m'], .1)
        self.assertEqual(result['handoff']['controller']['g05_steps_executed'], 0)

    def test_geometry_handoff_rejects_rotation(self):
        host, bridge, _ = self.prepare()
        host.call('base', 'adjust_chassis', 11, {'actual':{'forward_m':-.1,'translation_m':0.,'spin_deg':8.},
            'linear_cross_track_m':0.,'linear_yaw_drift_deg':0.,'linear_target_reached':True,'timed_out':False},
            {'forward':-.1,'translation':0.,'spin':0.})
        host.call('handle-end', 'track_object_distance', 12, track=tracking(12,.7))
        with self.assertRaisesRegex(ValueError, 'straight base movement'):
            bridge._verify_geometry_displacement(bridge._active(), binding(12),
                {'geometry_call_id':'base','exit_tracking_call_id':'handle-end'})

    def test_grasp_requires_confirmed_close_stow_and_no_open(self):
        host, bridge, _ = self.prepare('pick-up-object')
        batches = host.evidence/'g05-candidate-batches'; batches.mkdir()
        frame_id = {k: binding(10)[k] for k in ('episode_id','session_id','sequence','head_image_id')}
        path = batches/'batch1.json'; path.write_bytes(encoded({'batch_id':'batch1',
            'task':bridge._active()['model_task'], 'seeds':[0,1], 'binding':frame_id, 'candidates':[]}))
        (batches/'batch1.choice.json').write_bytes(encoded({'choice': {'route': 'geometry'},
            'binding':frame_id, 'batch_sha256':sha(path)}))
        host.call('close', 'close_gripper', 12, {'arm': 'left', 'grasp_confirmed': True})
        host.call('stow', 'set_arm_to_grasp_position', 14, {'converged': True},
                  {'arm': 'left', 'gripper': 'keep'})
        host.call('reset', 'reset_body', 16, {'timed_out': False, 'keep_ori_arm': 'none'})
        host.call('next', 'track_object_distance', 20, track=tracking(20))
        args = {'next_tracking_call_id': 'next', 'next_point': 'handle', 'visual_reason': 'object visible between fingers',
                'grasp_call_id': 'close', 'stow_call_id': 'stow', 'reset_call_id': 'reset', 'stow_kind': 'small',
                'exit_tracking_call_id':None, 'geometry_call_id':None}
        host.call('open', 'open_gripper', 18, args={'arm': 'left'})
        with self.assertRaisesRegex(ValueError, 'opened'):
            bridge.finish(binding(20), args)
        (host.evidence/'calls/open.json').unlink()
        self.assertEqual(bridge.finish(binding(20), args)['status'], 'completed')

    def test_entry_requires_current_reach_and_target(self):
        host, bridge, _ = self.prepare()
        bridge.abandon('inspect', binding(10))
        goal = {'type': 'tracked_displacement', 'direction': [1, 0, 0], 'target_m': .05}
        bridge.plan('p2', {'task':'open drawer', 'target_label':'drawer', 'point':'handle',
            'arm':'left', 'next_skill':'pick-up-object', 'next_target_label':'toy', 'goal':goal, 'seeds':[0],
            'maximum_steps':4, 'gripper_policy':'hold'}, binding(10))
        with self.assertRaisesRegex(ValueError, 'reach evidence'):
            bridge.begin(binding(11), 'reach', 'reach')

    def test_tracking_reacquisition_after_reach_uses_recorded_posture(self):
        host = Host(self.root, 'pick-up-object'); bridge = StageBridge(host)
        bridge.plan('p1', {'task':'pick the toy', 'target_label':'toy', 'point':'handle', 'arm':'left',
            'next_skill':'place-object-in-container', 'next_target_label':'box', 'goal':{'type':'grasp_stow'}, 'seeds':[0],
            'maximum_steps':4, 'gripper_policy':'policy'}, binding(9))
        host.call('reach', 'move_to_reach_point', 10,
                  {'shoulder_distance_estimate': {'minimum_m': .4, 'threshold_m': .7}})
        bridge.record_reach('reach', binding(10))
        host.call('retrack', 'track_object_distance', 10, track=tracking(11))
        self.assertEqual(bridge.begin(binding(11), 'reach', 'retrack')['status'], 'entered')

    def test_reacquisition_rejects_intervening_movement(self):
        host = Host(self.root, 'pick-up-object'); bridge = StageBridge(host)
        bridge.plan('p1', {'task':'pick the toy', 'target_label':'toy', 'point':'handle', 'arm':'left',
            'next_skill':'place-object-in-container', 'next_target_label':'box', 'goal':{'type':'grasp_stow'}, 'seeds':[0],
            'maximum_steps':4, 'gripper_policy':'policy'}, binding(9))
        host.call('reach', 'move_to_reach_point', 10,
                  {'shoulder_distance_estimate': {'minimum_m': .4, 'threshold_m': .7}})
        bridge.record_reach('reach', binding(10))
        host.call('retrack', 'track_object_distance', 10, track=tracking(11))
        changed = binding(11); changed['proprioception'][3] = .1
        with self.assertRaisesRegex(ValueError, 'posture changed'):
            bridge.begin(changed, 'reach', 'retrack')

    def test_frozen_reference_changes_are_rejected(self):
        host = Host(self.root, 'pick-up-object')
        reference = self.root/'reference'; reference.mkdir()
        digest = hashlib.sha256(encoded([])).hexdigest()
        bridge = StageBridge(host, experience_root=reference, experience_digest=digest)
        self.assertNotEqual(bridge.experience, reference)
        (reference/'changed.json').write_text('{}')
        with self.assertRaisesRegex(ValueError, 'reference changed'):
            bridge._check_reference()


if __name__ == '__main__': unittest.main()
