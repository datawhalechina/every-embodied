import json,tempfile,unittest
from pathlib import Path
from .four_view_delivery import validate,VIEWS,policy_events

class FourViewTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name)
        self.rows=[]
        for i in range(2):
            views={name:name+'/'+str(i)+'.png' for name in VIEWS}
            for path in views.values():
                file=self.root/path;file.parent.mkdir(exist_ok=True);file.write_bytes(b'fixture-not-image-decoding-test')
            self.rows.append({'capture_tick':i*3,'wrapper_tick':113+i*3,'hri_tick':1+i*3,'views':views})
    def check(self):
        (self.root/'frames.jsonl').write_text('\n'.join(json.dumps(r) for r in self.rows));return validate(self.root)
    def test_continuous_four_views(self):self.assertEqual(len(self.check()),2)
    def test_missing_view_rejected(self):
        self.rows[1]['views'].pop('aura_head')
        with self.assertRaises(ValueError):self.check()
    def test_native_clock_gap_rejected(self):
        self.rows[1]['wrapper_tick']+=1
        with self.assertRaises(ValueError):self.check()
    def test_missing_camera_file_rejected(self):
        self.rows[1]['views']['r1_head']='missing.png'
        with self.assertRaises(ValueError):self.check()
    def test_other_episode_policy_cannot_enter_current_video(self):
        root=self.root/'pipeline-new';meta=root/'host/exec-turns/0001/input-metadata.json';meta.parent.mkdir(parents=True)
        meta.write_text(json.dumps({'binding':{'episode_id':'old'}}))
        with self.assertRaisesRegex(ValueError,'another native episode'):policy_events([root],'current')
    def test_only_explicit_current_trial_events_are_collected(self):
        for name in ['pipeline-old','pipeline-new']:
            directory=self.root/name/'host/exec-events';directory.mkdir(parents=True)
            (directory/'000001.json').write_text(json.dumps({'kind':'think','summary':name,'native_tick':3}))
        events=policy_events([self.root/'pipeline-new'],'current')
        self.assertEqual(len(events),1);self.assertEqual(events[0]['controller_trial'],'pipeline-new')

if __name__=='__main__':unittest.main()
