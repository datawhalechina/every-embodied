import copy,unittest
from .recovery_delivery import join_plan

class RecoveryDeliveryTests(unittest.TestCase):
    def args(self):
        old={'episode_id':'old','fps':10,'frames':4,'events':[{'video_s':.05,'kind':'think'},{'video_s':.35,'kind':'action'}]}
        new={'episode_id':'new','fps':10,'frames':5,'duration_s':.5,'events':[{'video_s':0.,'kind':'think'}],'first_wrapper_tick':61}
        rows=[{'wrapper_tick':113+3*i} for i in range(4)]
        restore={'passed':True,'source':{'source_episode_id':'old','source_tick':120},'new_session':{'episode_id':'new'},'rollback_sim_seconds':.0667}
        return old,new,rows,restore
    def test_no_old_tail_replayed_and_new_events_offset(self):
        r=join_plan(*self.args());self.assertEqual(r['prefix_frames'],3);self.assertEqual(r['frames'],23)
        self.assertFalse(r['single_native_episode']);self.assertEqual(r['old_commands_replayed'],0)
        self.assertEqual(r['events'][-1]['video_s'],1.8);self.assertFalse(any(e['kind']=='action' for e in r['events']))
    def test_other_episode_media_rejected(self):
        a=self.args();a[1]['episode_id']='unrelated'
        with self.assertRaisesRegex(ValueError,'episode'):join_plan(*a)
    def test_unverified_restore_cannot_be_presented_as_continuation(self):
        a=self.args();a[3]['passed']=False
        with self.assertRaisesRegex(ValueError,'accepted'):join_plan(*a)
    def test_camera_gap_at_boundary_rejected(self):
        a=self.args();a[3]['source']['source_tick']=140
        with self.assertRaisesRegex(ValueError,'boundary'):join_plan(*a)
