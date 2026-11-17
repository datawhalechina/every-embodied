"""A rejected finish must produce another decision, without waiting for a user."""
import json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from .controller import run

class FinishGuardTests(unittest.TestCase):
    def test_rejected_finish_continues_same_bound_scene_automatically(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);workspace=root/'robot';evidence=root/'host'
            (workspace/'perception').mkdir(parents=True);evidence.mkdir()
            (workspace/'skills').mkdir();(workspace/'skills/active.json').write_text(json.dumps({'baseline':'base','active_skill':None}))
            (workspace/'perception/latest.json').write_text(json.dumps({'images':{role:{'path':'frame.png'} for role in ['head_rgb','left_wrist_rgb','right_wrist_rgb']}}))
            executable=root/'fake-codex';executable.write_text('fixture');qualification=root/'qualification.json';qualification.write_text('{"passed":true}')
            binding={'session_id':'same','episode_id':'same','sequence':3,'head_image_id':'current','native_ticks':20,'proprioception':{}}
            class Host:
                tools={}
                def unresolved(self):return []
                def frame(self):return dict(binding),[]
                def execute(self,*args):raise AssertionError('Finish rejection must not replay a robot command')
            host=Host();host.workspace=workspace;host.evidence=evidence;prompts=[]
            class FakeCLI:
                returncode=0
                def __init__(self,command,**kwargs):self.output=Path(command[command.index('--output-last-message')+1])
                def communicate(self,body,timeout):
                    prompts.append(json.loads(body))
                    self.output.write_text(json.dumps({'summary':'submit return','expected_check':'goal accepted','tool':'none','arguments_json':'{}','sequence':3,'head_image_id':'current','done':True,'memory':'three cans complete; return pending'}))
                    return '',None
            outcomes=iter([False,True])
            with patch('integrations.aura_household_exec.controller.subprocess.Popen',FakeCLI):
                result=run(host,'return to Aura',str(executable),qualification=qualification,max_turns=2,goal_guard=lambda:next(outcomes))
            self.assertTrue(result['external_goal_guard_passed']);self.assertEqual(result['turns'],2)
            self.assertIn('不等待用户确认',prompts[1]['previous_public_memory'])
            self.assertFalse(json.loads((evidence/'exec-turns/0001/goal-guard.json').read_text())['accepted'])
            self.assertEqual(prompts[0]['current_binding'],prompts[1]['current_binding'])
