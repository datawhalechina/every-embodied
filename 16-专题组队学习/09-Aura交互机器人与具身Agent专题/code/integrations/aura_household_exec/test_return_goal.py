import json,tempfile,unittest
from pathlib import Path
from .return_goal import goal_status

class ReturnGoalTests(unittest.TestCase):
    def setup_case(self,robot=(0,0,0),aura=(0,2,0),visitor=(2,0,0),tick=100):
        temp=tempfile.TemporaryDirectory();self.addCleanup(temp.cleanup);root=Path(temp.name)
        (root/'captures/hri-full').mkdir(parents=True);(root/'native-checkpoints').mkdir()
        (root/'driver-status.json').write_text(json.dumps({'connected':True,'episode_id':'same','ticks':tick}))
        (root/'captures/hri-full/frames.jsonl').write_text(json.dumps({'wrapper_tick':100,'robot_world_xyz_observer_only':robot})+'\n')
        scene=root/'native-checkpoints/00000100.json';scene.write_text(json.dumps({'state':{'registry':{'object_registry':{
            'robot_r1':{'root_link':{'pos':[100,100,0]}},'aura_interface':{'root_link':{'pos':aura}},'aura_visitor':{'root_link':{'pos':visitor}}}}}}))
        (root/'latest-native-checkpoint.json').write_text(json.dumps({'path':str(scene)}));return root
    def test_correct_robot_world_frame_not_articulation_reference(self):self.assertTrue(goal_status(self.setup_case())['passed'])
    def test_visible_but_far_is_not_nearby(self):self.assertFalse(goal_status(self.setup_case(aura=(0,3.5,0)))['passed'])
    def test_too_close_to_visitor_rejected(self):self.assertFalse(goal_status(self.setup_case(visitor=(.5,0,0)))['passed'])
    def test_stale_camera_row_does_not_authorize_completion(self):self.assertFalse(goal_status(self.setup_case(tick=110))['passed'])
