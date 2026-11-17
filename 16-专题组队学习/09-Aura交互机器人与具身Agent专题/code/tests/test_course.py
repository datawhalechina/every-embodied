"""Failure-oriented contracts for the public course, using only the standard library."""
import hashlib,json,sys,tempfile,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'code/tools'))
import course_lab as lab

class CourseTests(unittest.TestCase):
    def test_actual_native_proofs_and_media(self):self.assertTrue(lab.audit(ROOT)['passed'])
    def test_packaged_model_resources(self):self.assertTrue(lab.twin(ROOT/'code')['passed'])
    def test_invalid_depth_rejected(self):
        for z in [0,-1,float('nan'),float('inf')]:
            with self.assertRaises(ValueError):lab.unproject(1,1,z,2,2,0,0)
    def test_ambiguity_resets_hand_confirmation(self):
        self.assertEqual(lab.confirmed_edges([True,True,None,True,True,True,True],3),[5])
    def test_partial_completion_continues_without_first_can_replay(self):
        with tempfile.TemporaryDirectory() as d:
            r=lab.checkpoint_exercise(Path(d));self.assertTrue(r['completed']);self.assertEqual(r['rejected_done'],1);self.assertFalse(r['first_can_replayed'])
    def test_tampered_checkpoint_is_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'state.json';digest=lab.save_checkpoint(p,{'sequence':4});p.write_text('{"sequence":3}')
            with self.assertRaisesRegex(ValueError,'hash mismatch'):lab.load_checkpoint(p,digest)
if __name__=='__main__':unittest.main()
