import copy,hashlib,json,tempfile,unittest
from pathlib import Path
from .recovery_audit import compare_saved_states
from .test_native_checkpoint import SOURCE

class RecoveryAuditTests(unittest.TestCase):
    def check(self,mutate):
        data=json.loads(SOURCE.read_text());mutate(data['state']['registry']['object_registry'])
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder)/'restored.json';p.write_text(json.dumps(data))
            return compare_saved_states(SOURCE,hashlib.sha256(SOURCE.read_bytes()).hexdigest(),p,hashlib.sha256(p.read_bytes()).hexdigest())
    def test_quaternion_sign_is_same_orientation(self):
        def mutate(objects):objects['can_of_soda_114']['root_link']['ori']=[-v for v in objects['can_of_soda_114']['root_link']['ori']]
        self.assertTrue(self.check(mutate)['passed'])
    def test_wrong_object_orientation_blocks_restore(self):
        def mutate(objects):objects['can_of_soda_114']['root_link']['ori']=[0,0,0,1]
        with self.assertRaisesRegex(RuntimeError,'restore mismatch'):self.check(mutate)
    def test_joint_layout_change_not_silently_truncated_by_zip(self):
        def mutate(objects):objects['robot_r1']['joint_pos'].pop()
        with self.assertRaisesRegex(ValueError,'layout'):self.check(mutate)
    def test_wrong_grasp_state_blocks_restore(self):
        def mutate(objects):objects['robot_r1']['ag_obj_constraint_params']['right']={'target_obj':'can_of_soda_115'}
        with self.assertRaisesRegex(RuntimeError,'restore mismatch'):self.check(mutate)
