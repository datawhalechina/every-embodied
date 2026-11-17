"""Offline preparation contracts only; fixtures are never rollout evidence."""
import copy,hashlib,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch

from .agent import require_execution,close_with_native_result
from .inbox import receive,verify_native_result
from .prepare import prepare
from .orchestrator import launch

HERE=Path(__file__).resolve().parent


class WorldTests(unittest.TestCase):
    def setUp(self):self.config=json.loads((HERE/'world.json').read_text(encoding='utf-8'))

    def test_request_is_preparation_not_execution(self):
        with tempfile.TemporaryDirectory() as root:
            with patch('subprocess.Popen',side_effect=AssertionError('No process during preparation')):
                request=receive('帮我捡垃圾',root)
            self.assertEqual(request['phase'],'prepared_not_started')
            self.assertFalse(request['execution_started']);self.assertEqual(request['model_calls'],0)

    def test_cancel_not_new_household(self):
        with tempfile.TemporaryDirectory() as root:
            with self.assertRaises(ValueError):receive('不要捡垃圾',root)

    def test_other_task_not_silently_routed(self):
        with tempfile.TemporaryDirectory() as root:
            with self.assertRaises(ValueError):receive('帮我收盘子',root)

    def test_no_asr_claim_for_typed_input(self):
        with tempfile.TemporaryDirectory() as root:
            with self.assertRaises(ValueError):receive('捡垃圾',root,source='microphone')

    def test_external_command_stays_data(self):
        with tempfile.TemporaryDirectory() as root:
            text='捡垃圾; rm -rf /never'
            self.assertEqual(receive(text,root)['command'],text)

    def test_launch_gate_precedes_resources_and_processes(self):
        with patch('subprocess.Popen') as popen,patch('subprocess.check_output') as read:
            with self.assertRaises(PermissionError):launch(self.config,{},explicit_run=True)
            popen.assert_not_called();read.assert_not_called()

    def test_explicit_flag_alone_does_not_run(self):
        with self.assertRaises(PermissionError):require_execution(self.config,explicit_run=True)

    def test_layout_check_is_distinct(self):
        self.config['execution_enabled']=True
        with self.assertRaises(PermissionError):require_execution(self.config,explicit_run=True)

    def test_fixture_ready_gate(self):
        self.config['execution_enabled']=True;self.config['placement']['clearance_verified']=True
        require_execution(self.config,explicit_run=True) # Contract only, launches nothing.

    def test_model_done_does_not_override_native_false(self):
        result=verify_native_result({'task':'picking_up_trash','instance_id':301,'success':False,'q_score':{'final':1.}})
        self.assertEqual(result['phase'],'incomplete')

    def test_native_wrong_episode_rejected(self):
        with self.assertRaises(ValueError):verify_native_result({'task':'picking_up_trash','instance_id':304,'success':True,'q_score':{'final':1.}})

    def test_native_invalid_score_rejected(self):
        with self.assertRaises(ValueError):verify_native_result({'task':'picking_up_trash','instance_id':301,'success':True,'q_score':{'final':float('nan')}})

    def test_native_verified_success_publishes_source(self):
        with tempfile.TemporaryDirectory() as folder:
            base=Path(folder);source=base/'native-output/native-fixture.json';source.parent.mkdir()
            source.write_text(json.dumps({'task':'picking_up_trash','instance_id':301,'success':True,'q_score':{'final':1.}}))
            result=close_with_native_result(base/'agent',source)
            state=json.loads((base/'agent/aura-state.json').read_text())
            self.assertEqual(result['phase'],'completed');self.assertTrue(state['native_success'])
            self.assertTrue((base/'agent/public-events/000000.json').is_file())

    def test_old_result_outside_episode_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            base=Path(folder);source=base/'old-result.json';source.write_text('{}')
            with self.assertRaises(ValueError):close_with_native_result(base/'agent',source)

    def test_full_preparation_never_imports_simulator(self):
        import sys
        before=set(sys.modules)
        with tempfile.TemporaryDirectory() as folder,patch('subprocess.Popen',side_effect=AssertionError('No launch')):
            output=Path(folder)/'new';recipe=prepare(output)
            self.assertFalse(recipe['execution_enabled']);self.assertEqual(recipe['model_calls'],0)
            self.assertEqual(recipe['views'],['observer','r1_head','aura_head'])
            with self.assertRaises(FileExistsError):prepare(output)
        self.assertNotIn('isaacsim',set(sys.modules)-before);self.assertNotIn('omnigibson',set(sys.modules)-before)

    def test_reference_code_unchanged(self):
        expected={'astra_workspace_client.py':'74036ac1b24a4e9760c2600a3aa5b9f327a121f72ffafe02cada10eaa163b685',
                  'workspace_agent_host.py':'b0e9673d7701cf3c28a6b92c09e4cbad24224c1977680beafeb1078d392d2cb9',
                  'wire_payload_guard.py':'069294a71657103664ba4ddf10d454a892187f9afca05ff970e40a0f070ee8f0',
                  'tool_transport.py':'bacf3cc98f5e9bcaa40ec085a7b0a08808188f0c7de4f9f164000861cb70d715'}
        for name,digest in expected.items():self.assertEqual(hashlib.sha256((HERE/'reference'/name).read_bytes()).hexdigest(),digest)

    def test_same_world_is_explicit(self):
        self.assertEqual(self.config['scene'],'house_double_floor_lower')
        self.assertEqual(self.config['capture']['views'],['observer','r1_head','aura_head'])
        self.assertFalse(self.config['physical_robot_access'])

    def test_no_video_without_actual_camera_frames(self):
        from .replay import verify_frames
        with tempfile.TemporaryDirectory() as folder:
            (Path(folder)/'frames.jsonl').write_text('')
            with self.assertRaises(ValueError):verify_frames(folder)

    def test_video_gaps_are_rejected(self):
        from .replay import verify_frames
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            for name in ['a.png','b.png','c.png']:(root/name).write_bytes(b'fixture')
            rows=[{'native_tick':tick,'views':{'observer':'a.png','r1_head':'b.png','aura_head':'c.png'}} for tick in (0,6)]
            (root/'frames.jsonl').write_text('\n'.join(json.dumps(r) for r in rows))
            with self.assertRaises(ValueError):verify_frames(root)


if __name__=='__main__':unittest.main()
