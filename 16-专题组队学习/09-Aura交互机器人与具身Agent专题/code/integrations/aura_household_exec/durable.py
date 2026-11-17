"""A completed motion gets a native/app checkpoint without another model turn."""
import json
from pathlib import Path

def mutates(name,args):
    if name=='move_tracked_point':return args.get('execution_mode','exec')=='exec'
    return name.startswith(('adjust_','move_chassis_','spin_to_')) or name in {
        'move_to_reach_point','exec_plan_pose','open_gripper','close_gripper',
        'set_arm_to_grasp_position','reset_body','joint_control','grasp_plan_and_stow','release_and_stow'}

class DurableHost:
    def __init__(self,host):
        self.host=host;self.tools={k:v for k,v in host.tools.items() if k!='save_native_checkpoint'}
        self.last_checkpoint=None
    def __getattr__(self,name):return getattr(self.host,name)
    def execute(self,call_id,name,arguments,binding):
        result=self.host.execute(call_id,name,arguments,binding)
        if mutates(name,arguments) and not result.get('input_rejected'):
            current,_=self.host.frame()
            fence=self.host.execute(call_id+'-durable','save_native_checkpoint',{},current)
            if fence.get('error') or not fence.get('response',{}).get('saved'):
                raise RuntimeError('Motion finished but its native checkpoint is not confirmed; inspect before continuing')
            self.last_checkpoint=fence['response']
        return result
