import unittest
from .durable import DurableHost,mutates
class DurableTests(unittest.TestCase):
    def test_plan_and_read_do_not_force_steps(self):
        self.assertFalse(mutates('move_tracked_point',{'execution_mode':'plan'}))
        self.assertFalse(mutates('activate_skill',{}));self.assertFalse(mutates('capture_head_camera',{}))
        self.assertTrue(mutates('move_tracked_point',{'execution_mode':'exec'}))
    def host(self,rejected=False,saved=True):
        class Host:
            tools={'open_gripper':1,'save_native_checkpoint':2};calls=[]
            def execute(self,identity,name,args,binding):
                self.calls.append((identity,name))
                return {'tool_name':name,'input_rejected':rejected,'response':{'saved':saved}} if name=='save_native_checkpoint' else {'tool_name':name,'input_rejected':rejected}
            def frame(self):return {'sequence':2},[]
        value=Host();value.calls=[];return value
    def test_checkpoint_follows_motion_without_model_turn(self):
        raw=self.host();host=DurableHost(raw)
        host.execute('one','open_gripper',{}, {})
        self.assertEqual(raw.calls,[('one','open_gripper'),('one-durable','save_native_checkpoint')])
        self.assertNotIn('save_native_checkpoint',host.tools);self.assertTrue(host.last_checkpoint['saved'])
    def test_rejected_motion_has_no_fence(self):
        raw=self.host(rejected=True);DurableHost(raw).execute('one','open_gripper',{}, {})
        self.assertEqual(len(raw.calls),1)
    def test_unconfirmed_save_stops_next_decision(self):
        with self.assertRaises(RuntimeError):DurableHost(self.host(saved=False)).execute('one','open_gripper',{}, {})
