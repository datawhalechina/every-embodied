import json,tempfile,unittest
from pathlib import Path
from .macros import MacroHost,install_specs

class Host:
    def __init__(self,root,fail=None):
        self.workspace=Path(root);self.evidence=self.workspace/'evidence';self.evidence.mkdir()
        self.fail=fail;self.calls=[];self.sequence=1
    def frame(self):return {'session_id':'s','episode_id':'e','sequence':self.sequence,'head_image_id':'image-'+str(self.sequence),'native_ticks':self.sequence},[]
    def unresolved(self):return []
    def execute(self,identity,name,arguments,binding):
        self.calls.append((identity,name,arguments,binding));self.sequence+=1
        return {'tool_name':name,'response':{'grasp_confirmed':name!='close_gripper' or self.fail!='close_gripper','timed_out':name==self.fail}}

class MacroTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.host=Host(self.temp.name);self.macro=MacroHost(self.host)
    def run_grasp(self):return self.macro.execute('call-1','grasp_plan_and_stow',{'plan_id':'p1','arm':'left'},self.host.frame()[0])
    def test_success_four_measured_tools_one_macro(self):
        self.assertTrue(self.run_grasp()['response']['ok'])
        self.assertEqual([r[1] for r in self.host.calls],['exec_plan_pose','close_gripper','set_arm_to_grasp_position','reset_body'])
        self.assertEqual([r[3]['sequence'] for r in self.host.calls],[1,2,3,4])
    def test_failed_grasp_never_stows_or_stands(self):
        self.host.fail='close_gripper';self.assertFalse(self.run_grasp()['response']['ok'])
        self.assertEqual(len(self.host.calls),2)
    def test_failed_plan_never_closes_or_releases(self):
        self.host.fail='exec_plan_pose';self.assertFalse(self.run_grasp()['response']['ok'])
        self.assertEqual(len(self.host.calls),1)
    def test_explicit_negative_plan_result_stops_before_gripper(self):
        original=self.host.execute
        def run(*args):
            result=original(*args);result['response']['ok']=False;return result
        self.host.execute=run
        self.assertFalse(self.run_grasp()['response']['ok']);self.assertEqual(len(self.host.calls),1)
    def test_stale_binding_does_not_execute(self):
        binding=self.host.frame()[0];self.host.sequence+=1
        with self.assertRaises(ValueError):self.macro.execute('c','grasp_plan_and_stow',{'plan_id':'p','arm':'left'},binding)
        self.assertFalse(self.host.calls)
    def test_incomplete_macro_is_not_replayed(self):
        folder=self.host.evidence/'macro-intents';folder.mkdir();(folder/'old.json').write_text('{"status":"intent_recorded"}')
        with self.assertRaises(RuntimeError):self.run_grasp()
        self.assertFalse(self.host.calls)
    def test_drop_never_claims_native_success(self):
        result=self.macro.execute('drop','place_plan_and_release',{'plan_id':'p1','arm':'right'},self.host.frame()[0])
        self.assertFalse(result['response']['native_success_verified']);self.assertEqual(self.host.calls[1][1],'open_gripper')

if __name__=='__main__':unittest.main()
