import hashlib,json,tempfile,unittest
from pathlib import Path
from .resume_bundle import select_latest,inspect_pair
from .test_native_checkpoint import SOURCE

class ResumeBundleTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name)
        self.archive=self.root/'agent-checkpoints';self.archive.mkdir()
        self.native=self.root/'native.json';self.native.write_bytes(SOURCE.read_bytes())
        self.app=self.root/'app.json';self.app.write_text(json.dumps({'wrapper_tick':90,'hri_tick':120,'phase':'working','pose':{},'yaw':1.2}))
        self.row={'binding':{'session_id':'old','episode_id':'source','sequence':91},'last_completed_turn':4,'memory':'First can accepted; second held.',
            'active_skill_state':{'active_skill':'place-object-in-container'},'history':[{'plan_id':'STALE'}],
            'native_checkpoint':{'saved':True,'native_snapshot':str(self.native),'native_sha256':hashlib.sha256(self.native.read_bytes()).hexdigest(),
                'application_snapshot':str(self.app),'application_sha256':hashlib.sha256(self.app.read_bytes()).hexdigest(),'saved_native_tick':90}}
        self.path=self.archive/'0004.json';self.path.write_text(json.dumps(self.row))
    def test_partial_latest_falls_back_without_replaying_intent(self):
        (self.archive/'0005.json').write_text('{"unfinished":')
        r=select_latest(self.root)
        self.assertEqual(r['completed_turn'],4);self.assertEqual(len(r['newer_incomplete_candidates']),1)
        self.assertEqual(r['commands_to_replay'],[]);self.assertNotIn('STALE',json.dumps(r));self.assertFalse(r['physical_restore_validated'])
    def test_modified_application_blocks_pair(self):
        self.app.write_text('{}')
        with self.assertRaisesRegex(ValueError,'digest'):inspect_pair(self.path)
    def test_mismatched_physical_application_clock_rejected(self):
        self.row['native_checkpoint']['saved_native_tick']=91;self.path.write_text(json.dumps(self.row))
        with self.assertRaisesRegex(ValueError,'clocks'):inspect_pair(self.path)
    def test_readonly_newer_turn_keeps_completed_native_fence(self):
        r=select_latest(self.root)
        self.assertEqual(r['decision_context']['public_memory'],self.row['memory'])
        self.assertEqual(r['physical_checkpoint']['saved_native_tick'],90)
