import json,tempfile,unittest
from pathlib import Path
from .placement_contract import PlacementHost
class ContractTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);root=Path(self.temp.name)
        class Host:
            workspace=root/'workspace';evidence=root/'host';tools={'exec_plan_pose':1,'place_plan_and_release':2,'open_gripper':3};calls=[]
            def execute(self,*args):self.calls.append(args);return {'tool_name':args[1]}
        self.raw=Host();self.raw.calls=[];self.raw.workspace.mkdir();self.raw.evidence.mkdir();self.host=PlacementHost(self.raw)
        directory=self.raw.workspace/'tool_call_history/0001_move_tracked_point';directory.mkdir(parents=True)
        (directory/'result.json').write_text(json.dumps({'response':{'plan_id':'geo','execution_mode':'plan'}}))
    def test_carry_geometry_is_rejected_before_native_command(self):
        result=self.host.execute('a','exec_plan_pose',{'plan_id':'geo'},{});self.assertTrue(result['input_rejected']);self.assertEqual(self.raw.calls,[])
    def test_grasp_execution_and_explicit_release_remain_available(self):
        self.host.execute('a','exec_plan_pose',{'plan_id':'grasp'},{});self.host.execute('b','open_gripper',{'arm':'right'},{});self.assertEqual(len(self.raw.calls),2)
    def test_incorrect_place_macro_hidden_and_rejected(self):
        self.assertNotIn('place_plan_and_release',self.host.tools)
        self.assertTrue(self.host.execute('a','place_plan_and_release',{'plan_id':'geo'}, {})['input_rejected']);self.assertEqual(self.raw.calls,[])
    def test_active_placement_rejects_grasp_executor_even_without_plan_history(self):
        folder=self.raw.workspace/'skills';folder.mkdir()
        (folder/'active.json').write_text(json.dumps({'active_skill':'place-object-in-container'}))
        result=self.host.execute('a','exec_plan_pose',{'plan_id':'unknown'}, {})
        self.assertTrue(result['input_rejected']);self.assertEqual(self.raw.calls,[])
    def test_macro_wrapper_cannot_reexpose_hidden_place_macro(self):
        from .macros import MacroHost
        host=MacroHost(self.host)
        result=host.execute('a','place_plan_and_release',{'plan_id':'grasp','arm':'right'}, {})
        self.assertTrue(result['input_rejected']);self.assertEqual(self.raw.calls,[])
        self.assertFalse((self.raw.evidence/'macro-intents').exists())
if __name__=='__main__':unittest.main()
