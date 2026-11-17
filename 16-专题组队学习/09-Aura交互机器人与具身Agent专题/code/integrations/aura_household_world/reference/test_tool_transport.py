"""Reject ambiguous or incomplete stored receipts; exercise actual planner output."""
import unittest,json,tempfile
from pathlib import Path
from tool_transport import stored_receipt

class TransportTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name)
        self.binding={'episode_id':'e','session_id':'s','sequence':10};self.args={'u':42}
        (self.root/'.session.json').write_text(json.dumps({**self.binding,'trajectory_dir':str(self.root/'trajectory')}))
        self.path=self.root/'tool_call_history/000001_grasp';self.path.mkdir(parents=True)
        self.write(self.path)
    def write(self,p):
        (p/'input.json').write_text(json.dumps({'tool':'grasp','arguments':self.args,**self.binding,'observation_sequence':10}))
        (p/'result.json').write_text(json.dumps({'tool_name':'grasp','record':str(p/'result.json'),'joint_commands_sent':2,'rollout_budget':{'used_ticks':11}}))
        c=self.root/'trajectory/calls'/p.name;c.mkdir(parents=True,exist_ok=True)
        (c/'joint_commands.jsonl').write_text('\n'.join(json.dumps({'sequence_before':a,'sequence_after':a+1}) for a in [10,11]))
    def get(self,before=set()):return stored_receipt(self.root,before,'grasp',self.args,self.binding,lambda r,w:r)
    def test_original_completed_receipt(self):self.assertEqual(self.get()[0]['joint_commands_sent'],2)
    def test_old_receipt_cannot_be_reused(self):
        with self.assertRaises(ValueError):self.get({self.path.name})
    def test_ambiguous_receipts_rejected(self):
        p=self.path.parent/'000002_grasp';p.mkdir();self.write(p)
        with self.assertRaises(ValueError):self.get()
    def test_wrong_input_rejected(self):
        self.args={'u':99}
        with self.assertRaises(ValueError):self.get()
    def test_incomplete_physical_ack_rejected(self):
        (self.root/'trajectory/calls'/self.path.name/'joint_commands.jsonl').write_text('')
        with self.assertRaises(ValueError):self.get()
    def test_wrong_native_counter_rejected(self):
        p=self.path/'result.json';r=json.loads(p.read_text());r['rollout_budget']['used_ticks']=12;p.write_text(json.dumps(r))
        with self.assertRaises(ValueError):self.get()

if __name__=='__main__':unittest.main()
