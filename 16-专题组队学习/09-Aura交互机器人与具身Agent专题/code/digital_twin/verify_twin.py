"""Evidence checks for geometry, native display bytes, and real MuJoCo rollout."""
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import unittest
import mujoco
import numpy as np
from PIL import Image
from run_mujoco import ROOT, load_model, set_screen
from model_parameters import P


class TwinTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest=json.loads((ROOT/'model_manifest.json').read_text(encoding='utf-8'))
        cls.model=load_model()
        path=Path(os.environ.get('AURA_TWIN_ROLLOUT',str(ROOT/'outputs/my-first-run/trajectory.json')))
        cls.rollout=json.loads(path.read_text(encoding='utf-8')) if path.is_file() else None

    def test_closed_positive_cad_parts(self):
        self.assertEqual(self.manifest['solids'],72)
        self.assertTrue(all(x['cad_valid'] and x['volume_mm3']>0 for x in self.manifest['components']))

    def test_nominal_envelope_only_not_metrology(self):
        self.assertTrue(np.allclose(self.manifest['bbox_mm'],[200,210,200],atol=10,rtol=0))
        self.assertIn('not_metrology',self.manifest['parameters']['fidelity'])

    def test_shared_mesh_files(self):
        self.assertTrue(all((ROOT/item['mesh']).is_file() for item in self.manifest['components']))
        self.assertEqual(self.manifest['mesh_units'],'m')

    def test_wheel_symmetry(self):
        left=self.model.body('wheel_left').pos
        right=self.model.body('wheel_right').pos
        np.testing.assert_allclose(left[[0,2]],right[[0,2]])
        self.assertAlmostEqual(left[1],-right[1])

    def test_actuators_and_sensors(self):
        self.assertEqual(self.model.nu,2)
        self.assertEqual(self.model.nsensordata,6)

    def test_virtual_cameras_face_outwards(self):
        data=mujoco.MjData(self.model); mujoco.mj_forward(self.model,data)
        camera=self.model.camera('rgb_left').id
        optical_axis=-data.cam_xmat[camera].reshape(3,3)[:,2]
        expected=np.array([math.cos(P.tilt),0,math.sin(P.tilt)])
        self.assertGreater(float(np.dot(optical_axis,expected)),.999)

    def test_authored_screen_fixture_hash(self):
        meta=json.loads((ROOT/'assets/screen/provenance.json').read_text(encoding='utf-8'))
        self.assertEqual(hashlib.sha256((ROOT/'assets/screen/hello.png').read_bytes()).hexdigest(),meta['source_sha256'])
        self.assertEqual(meta['frames'],len(list((ROOT/'assets/screen/sequence').glob('*.png'))))
        self.assertEqual(meta['sequence_source'],'course_custom_states')

    def test_original_png_pixels_preserved(self):
        expected=np.asarray(Image.open(ROOT/'assets/screen/screen.png').convert('RGB'))
        np.testing.assert_array_equal(self.model.tex_data.reshape(expected.shape),expected)

    def test_display_dynamic_upload_without_flip(self):
        with mujoco.Renderer(self.model,height=64,width=64) as renderer:
            path=ROOT/'assets/screen/hello.png'
            set_screen(self.model,renderer,path)
            expected=np.asarray(Image.open(path).convert('RGB')).reshape(-1)
            np.testing.assert_array_equal(self.model.tex_data,expected)
            set_screen(self.model,renderer,ROOT/'assets/screen/screen.png')

    def test_contact_rollout_finite_no_warning(self):
        if self.rollout is None:self.skipTest('Run the finite rollout or set AURA_TWIN_ROLLOUT first')
        self.assertTrue(self.rollout['finite'])
        self.assertEqual(sum(self.rollout['warning_counts']),0)
        self.assertGreater(self.rollout['forward_displacement_m'],.02)
        self.assertFalse(self.rollout['physical_robot_access'])

    def test_turning_and_screen_visible(self):
        if self.rollout is None:self.skipTest('Run the finite rollout or set AURA_TWIN_ROLLOUT first')
        w,x,y,z=self.rollout['frames'][-1]['qpos'][3:7]
        yaw=math.atan2(2*(w*z+x*y),1-2*(y*y+z*z))
        self.assertGreater(yaw,.2)
        self.assertGreater(self.rollout['display_changed_pixels'],100)


if __name__=='__main__':
    unittest.main()
