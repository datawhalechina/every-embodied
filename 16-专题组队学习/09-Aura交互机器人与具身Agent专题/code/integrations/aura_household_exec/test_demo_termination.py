import unittest
from .demo_termination import defer_success
class DemoTerminationTests(unittest.TestCase):
    def test_only_defers_native_success_shutdown(self):
        self.assertEqual(defer_success(True,False,{'done':{'success':True}},False),(False,False,True))
        self.assertEqual(defer_success(True,False,{'done':{'success':False}},False),(True,False,False))
        self.assertEqual(defer_success(True,True,{'done':{'success':True}},False),(True,True,False))
        self.assertEqual(defer_success(True,False,{'done':{'success':True}},True),(True,False,False))
if __name__=='__main__':unittest.main()
