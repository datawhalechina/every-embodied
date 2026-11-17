import copy,json,tempfile,unittest
from pathlib import Path
from .three_can_verifier import TARGETS,verify
from .controller import native_completion

class ThreeCanTests(unittest.TestCase):
    def setUp(self):
        self.status={'connected':True,'episode_id':'current','ticks':100}
        self.observer={'wrapper_tick':99,'inside_bin_observer_only':{n:True for n in TARGETS},
                       'stable_released_samples':{n:2 for n in TARGETS},'assisted_grasp_targets_observer_only':{'left':None,'right':None},
                       'progress':{'ok':True,'complete':True}}
    def test_requires_three_released_stable_and_official_goal(self):
        self.assertTrue(verify(self.observer,self.status))
        for field,value in [('inside_bin_observer_only',False),('stable_released_samples',1)]:
            row=copy.deepcopy(self.observer);row[field]['can_of_soda_113']=value;self.assertIsNone(verify(row,self.status))
        row=copy.deepcopy(self.observer);row['assisted_grasp_targets_observer_only']['right']='can_of_soda_114';self.assertIsNone(verify(row,self.status))
        row=copy.deepcopy(self.observer);row['progress']['complete']=False;self.assertIsNone(verify(row,self.status))
        self.assertIsNone(verify(self.observer,{**self.status,'ticks':200}))
    def test_scope_cannot_accept_one_can_as_three(self):
        proof=verify(self.observer,self.status)
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'proof.json';binding={'episode_id':'current','native_ticks':100}
            path.write_text(json.dumps(proof));self.assertTrue(native_completion(path,binding,'three_cans'))
            self.assertIsNone(native_completion(path,binding))
            for change in [{'inside':['can_of_soda_113']},{'inside':['can_of_soda_113']*3},{'official_task_complete':False},{'episode_id':'old'}]:
                path.write_text(json.dumps({**proof,**change}));self.assertIsNone(native_completion(path,binding,'three_cans'))

if __name__=='__main__':unittest.main()
