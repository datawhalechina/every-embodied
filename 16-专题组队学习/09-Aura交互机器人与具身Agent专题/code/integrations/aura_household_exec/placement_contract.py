"""Opt-in official placement contract; reject pre-opening carry execution."""
import json
from pathlib import Path

class PlacementHost:
    def __init__(self,host):
        self.host=host
        self.tools={k:v for k,v in host.tools.items() if k!='place_plan_and_release'}
    def __getattr__(self,name):return getattr(self.host,name)
    def execute(self,call_id,name,arguments,binding):
        geometry_plan=False
        if name=='exec_plan_pose':
            skill_file=self.workspace/'skills/active.json'
            if skill_file.is_file():geometry_plan=json.loads(skill_file.read_text()).get('active_skill')=='place-object-in-container'
            for path in (self.workspace/'tool_call_history').glob('*_move_tracked_point/result.json'):
                row=json.loads(path.read_text());response=row.get('response',{})
                if response.get('plan_id')==arguments.get('plan_id') and response.get('execution_mode')=='plan':geometry_plan=True;break
        if name=='place_plan_and_release' or geometry_plan:
            receipt={'tool_name':name,'input_rejected':True,'error':'This executor opens the selected gripper before motion. Carrying placement must use the active official place-object-in-container Skill Cartesian EEF tools, or move_tracked_point execution_mode=exec; release only with explicit open_gripper after alignment.',
                     'response':{'physical_steps':0,'unsafe_preopen_prevented':True}}
            folder=self.evidence/'input-rejections';folder.mkdir(exist_ok=True)
            with (folder/(call_id+'.json')).open('x') as file:json.dump(receipt,file,indent=2)
            return receipt
        return self.host.execute(call_id,name,arguments,binding)
