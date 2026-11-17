import unittest
from .attention import Attention,Camera,Person,Speech


class AttentionTests(unittest.TestCase):
    def speech(self,text='看向我',stamp=0.,identity='u1'):
        return Speech(identity,stamp,text,'a'*64,.95)
    def frame(self,time,x=None,identity='visitor',image=None):
        return Camera(image or str(time),time,() if x is None else (Person(identity,x,.95),))
    def test_wake_starts_search_and_right_person_turns_right(self):
        a=Attention();r=a.update(0,self.frame(0),self.speech());self.assertGreater(r.angular_z,0)
        r=a.update(.1,self.frame(.1,.8));self.assertLess(r.angular_z,0);self.assertEqual(r.linear_x,0)
    def test_no_motion_without_wake(self):
        self.assertEqual(Attention().update(0,self.frame(0,.8)).angular_z,0)
    def test_no_handoff_until_same_target_stably_centered(self):
        a=Attention();a.update(0,self.frame(0,.8),self.speech('帮我把垃圾收拾一下'))
        self.assertIsNone(a.update(.1,self.frame(.1,.5)).task)
        self.assertIsNone(a.update(.2,self.frame(.2,.5)).task)
        r=a.update(.3,self.frame(.3,.5));self.assertEqual(r.task,'pick_up_trash')
        self.assertIsNone(a.update(.4,self.frame(.4,.5)).task)
    def test_repeated_frame_does_not_confirm(self):
        a=Attention();a.update(0,self.frame(0,.5),self.speech())
        for t in [.1,.2,.3]:self.assertEqual(a.update(t,self.frame(0,.5)).reason,'camera_not_advanced')
        self.assertFalse(a.greeted)
    def test_stale_or_failed_camera_stops_rotation(self):
        a=Attention();a.update(0,self.frame(0),self.speech())
        self.assertEqual(a.update(1,self.frame(0)).angular_z,0)
        self.assertEqual(a.update(1.1,Camera('new',1.1,(),False)).angular_z,0)
    def test_controller_polling_faster_than_camera_can_still_confirm(self):
        a=Attention();a.update(0,self.frame(0,.5),self.speech())
        a.update(.02,self.frame(0,.5));a.update(.1,self.frame(.1,.5))
        a.update(.12,self.frame(.1,.5))
        self.assertIsNotNone(a.update(.2,self.frame(.2,.5)).say)
    def test_target_loss_reacquires_without_switching_to_other_person(self):
        a=Attention();a.update(0,self.frame(0,.5),self.speech())
        r=a.update(.1,self.frame(.1,.5,identity='someone_else'))
        self.assertEqual(r.state,'searching');self.assertEqual(a.stable,0)
    def test_multiple_people_require_unique_wave(self):
        a=Attention();people=(Person('a',.4,.9),Person('b',.7,.9))
        self.assertEqual(a.update(0,Camera('1',0,people),self.speech()).angular_z,0)
        people=(people[0],Person('b',.7,.9,True))
        self.assertLess(a.update(.1,Camera('2',.1,people)).angular_z,0)
    def test_lost_track_can_be_reidentified_by_unique_wave(self):
        a=Attention();a.update(0,self.frame(0,.5),self.speech())
        frame=Camera('new',2,(Person('new-track',.5,.95,True),))
        a.update(2,frame);self.assertEqual(a.target,'new-track');self.assertEqual(a.stable,1)
    def test_timeout_and_cancel_stop_search(self):
        a=Attention(search_timeout=2);a.update(0,self.frame(0),self.speech())
        self.assertEqual(a.update(2,self.frame(2)).state,'timed_out')
        b=Attention();b.update(0,self.frame(0),self.speech())
        r=b.update(.1,self.frame(.1),self.speech('停止',.1,'u2'));self.assertEqual(r.state,'idle');self.assertEqual(r.angular_z,0)
    def test_greeting_once_and_duplicate_utterance_not_redelegated(self):
        a=Attention();a.update(0,self.frame(0,.5),self.speech())
        a.update(.1,self.frame(.1,.5));self.assertIsNotNone(a.update(.2,self.frame(.2,.5)).say)
        self.assertIsNone(a.update(.3,self.frame(.3,.5)).say)
        request=self.speech('捡垃圾',.4,'u2');self.assertEqual(a.update(.4,self.frame(.4,.5),request).task,'pick_up_trash')
        self.assertEqual(a.update(.5,self.frame(.5,.5),request).state,'delegated')
    def test_invalid_speech_provenance_and_clock(self):
        a=Attention();bad=Speech('u',0,'看向我','',.95,'typed')
        self.assertEqual(a.update(0,self.frame(0),bad).state,'idle')
        with self.assertRaises(ValueError):a.update(-1,self.frame(-1))

if __name__=='__main__':unittest.main()
