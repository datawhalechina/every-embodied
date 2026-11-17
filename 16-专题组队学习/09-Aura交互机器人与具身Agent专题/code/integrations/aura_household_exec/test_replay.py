import unittest
from .replay import calibrated_offset,event_time
class ReplayClockTests(unittest.TestCase):
    def test_actual_counter_alignment_and_capture_origin(self):
        shift=calibrated_offset([{'native_ticks':48,'wrapper_tick':48},{'native_ticks':978,'wrapper_tick':978}])
        self.assertEqual(event_time({'source':'runtime','native_tick':1435},1405,shift,100),1.)
    def test_clock_drift_blocks_false_synchronization(self):
        with self.assertRaises(ValueError):calibrated_offset([{'native_ticks':48,'wrapper_tick':48},{'native_ticks':979,'wrapper_tick':978}])
    def test_unrecorded_and_fixture_events_are_rejected(self):
        for row in [{'source':'fixture','native_tick':1406},{'source':'runtime','native_tick':1390},{'source':'runtime','native_tick':2000}]:
            with self.assertRaises(ValueError):event_time(row,1405,0,100)
if __name__=='__main__':unittest.main()
