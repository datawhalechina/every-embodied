import hashlib,json,tempfile,unittest
from pathlib import Path
from .native_checkpoint import inspect_snapshot
SOURCE=Path(__file__).resolve().parents[2]/'tests/fixtures/native-state-sanitized.json'
class NativeSnapshotTests(unittest.TestCase):
    def test_real_native_snapshot_retains_controllers_and_identity(self):
        data,report=inspect_snapshot(SOURCE,hashlib.sha256(SOURCE.read_bytes()).hexdigest())
        self.assertEqual(report['object_count'],149)
        self.assertEqual(report['robot'],'robot_r1')
        self.assertIn('controller_groups',data['state']['registry']['object_registry']['robot_r1'])
        self.assertFalse(report['restore_validated'])
    def test_changed_checkpoint_is_rejected(self):
        with self.assertRaisesRegex(ValueError,'digest'):inspect_snapshot(SOURCE,'0'*64)
    def test_missing_assisted_grasp_state_blocks_restore_plan(self):
        data=json.loads(SOURCE.read_text())
        del data['state']['registry']['object_registry']['robot_r1']['ag_obj_constraint_params']
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder)/'state.json';p.write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError,'grasp state'):inspect_snapshot(p,hashlib.sha256(p.read_bytes()).hexdigest())
if __name__=='__main__':unittest.main()
