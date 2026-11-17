import unittest
from .controller import decode_decision,public_cli_events,measured_receipt,native_completion
import json,tempfile
from pathlib import Path
class ExecTests(unittest.TestCase):
    def setUp(self):self.binding={'sequence':3,'head_image_id':'img'};self.value={'summary':'接近罐子','expected_check':'新相機','tool':'move','arguments_json':'{}','sequence':3,'head_image_id':'img','done':False,'memory':'尚未抓取'}
    def test_valid(self):self.assertEqual(decode_decision(self.value,self.binding,{'move':None})['arguments'],{})
    def test_stale(self):
        with self.assertRaises(ValueError):decode_decision({**self.value,'sequence':2},self.binding,{'move':None})
    def test_unknown_tool(self):
        with self.assertRaises(ValueError):decode_decision(self.value,self.binding,{})
    def test_done_with_action(self):
        with self.assertRaises(ValueError):decode_decision({**self.value,'done':True},self.binding,{'move':None})
    def test_bad_json(self):
        with self.assertRaises(ValueError):decode_decision({**self.value,'arguments_json':'{"x":NaN}'},self.binding,{'move':None})
    def test_discards_private_events(self):self.assertEqual(public_cli_events('{"type":"item.completed","item":{"type":"reasoning","text":"discard"}}'),[])
    def test_unexpected_model_command_rejected(self):
        with self.assertRaises(RuntimeError):public_cli_events('{"type":"item.started","item":{"type":"command_execution"}}')
    def test_public_receipt_keeps_measured_outcomes(self):
        value={'tool_name':'adjust_chassis','response':{'actual':{'x':.22},'linear_target_reached':False,'obstacle_limited':True,'rgb':'data:image/png;base64,blob'},'record':{'reasoning':'private'}}
        row=measured_receipt(value)
        self.assertEqual(row['response']['actual']['x'],.22)
        self.assertTrue(row['response']['obstacle_limited'])
        self.assertNotIn('record',row)
        self.assertNotIn('blob',str(row))
    def test_skill_receipt_retains_authoritative_lifecycle(self):
        row=measured_receipt({'tool_name':'activate_skill','task_skill_state':{'active_skill':'pick-up-object','revision':3},'instructions':'full source available separately'})
        self.assertEqual(row['task_skill_state']['active_skill'],'pick-up-object')
        self.assertNotIn('instructions',row)
    def test_tracking_geometry_reaches_model_with_provenance(self):
        tracking={'tracks':{'rim':{'status':'observed','observation_sequence':9,'xyz_in_robot_base_coord_m':[.4,.1,.27]}}}
        row=measured_receipt({'tool_name':'track_object_distance','response':{'tracked_names':['rim']},'persistent_tracking':tracking})
        self.assertEqual(row['persistent_tracking'],tracking)
    def test_large_result_does_not_discard_tracking_or_failure_status(self):
        tracking={'tracks':{'rim':{'xyz_in_robot_base_coord_m':[.4,.1,.27]}}}
        row=measured_receipt({'tool_name':'macro','persistent_tracking':tracking,'response':{'ok':False,'stopped_at':'stow','many':[{str(i):'x'*700 for i in range(35)}]}})
        self.assertEqual(row['persistent_tracking'],tracking);self.assertEqual(row['response_status']['stopped_at'],'stow')
    def test_native_completion_needs_released_stable_same_episode(self):
        with tempfile.TemporaryDirectory() as root:
            path=Path(root)/'proof.json';binding={'episode_id':'e','native_ticks':30}
            proof={'passed':True,'scope':'one_can','episode_id':'e','native_tick':20,'released':True,'inside':['can'],'stable_native_observer_samples':2}
            path.write_text(json.dumps(proof));self.assertTrue(native_completion(path,binding))
            for change in [{'released':False},{'stable_native_observer_samples':1},{'episode_id':'other'},{'native_tick':31},{'inside':[]}]:
                path.write_text(json.dumps({**proof,**change}));self.assertIsNone(native_completion(path,binding))
if __name__=='__main__':unittest.main()
