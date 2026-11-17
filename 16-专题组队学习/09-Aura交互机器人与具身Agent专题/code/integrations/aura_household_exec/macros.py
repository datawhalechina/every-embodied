"""Opt-in goal macros over unchanged measured native tools, with an intent ledger."""
import json
from pathlib import Path
from .controller import measured_receipt

SPECS={
 'grasp_plan_and_stow':('Execute a current reviewed grasp plan, close with measured grasp confirmation, stow the holding arm and stand upright. Stops immediately on failed feedback; one model decision, individual native tool receipts remain visible.',
    {'type':'object','properties':{'plan_id':{'type':'string'},'arm':{'type':'string','enum':['left','right']}},'required':['plan_id','arm'],'additionalProperties':False}),
 'place_plan_and_release':('Execute a current reviewed placement plan above a visible container opening, release, then stow and stand. Use only after the plan overlay is visibly inside the opening. Tool completion does not establish inside-container success; native evaluator must verify.',
    {'type':'object','properties':{'plan_id':{'type':'string'},'arm':{'type':'string','enum':['left','right']}},'required':['plan_id','arm'],'additionalProperties':False})}

def install_specs(workspace):
    for name,(description,schema) in SPECS.items():
        path=Path(workspace)/'tools'/(name+'.py')
        with path.open('x',encoding='utf-8') as file:
            file.write(repr(description)+'\nSCHEMA = '+repr(schema)+'\n# Executed by the explicitly enabled measured MacroHost, never standalone.\n')

class MacroHost:
    def __init__(self,host):
        self.host=host
        from integrations.aura_household_director.events import EventLog
        self.log=EventLog(host.evidence/'macro-events',clock=lambda:host.frame()[0]['native_ticks'])
    def __getattr__(self,name):return getattr(self.host,name)
    def unresolved(self):
        pending=self.host.unresolved()
        for path in (self.evidence/'macro-intents').glob('*.json'):
            if json.loads(path.read_text())['status']=='intent_recorded':pending.append(path.name)
        return pending
    def execute(self,call_id,name,arguments,binding):
        available=getattr(self.host,'tools',None)
        if name in SPECS and available is not None and name not in available:
            # Preserve an inner host's rejection before recording or executing
            # any macro substep, even for a direct call outside the decoder.
            return self.host.execute(call_id,name,arguments,binding)
        if name not in SPECS:return self.host.execute(call_id,name,arguments,binding)
        if set(arguments)!={'plan_id','arm'} or arguments['arm'] not in {'left','right'} or not isinstance(arguments['plan_id'],str):raise ValueError('Invalid goal macro arguments')
        if self.unresolved():raise RuntimeError('Unresolved prior intent; never replay macro')
        current,_=self.frame()
        if any(current[k]!=binding[k] for k in ['session_id','episode_id','sequence','head_image_id']):raise ValueError('Stale macro observation')
        directory=self.evidence/'macro-intents';directory.mkdir(exist_ok=True);path=directory/(call_id+'.json')
        record={'status':'intent_recorded','name':name,'arguments':arguments,'binding':{k:binding[k] for k in ['session_id','episode_id','sequence','head_image_id']},'completed_steps':[]}
        with path.open('x') as file:json.dump(record,file)
        def save():
            temp=path.with_suffix('.tmp');temp.write_text(json.dumps(record));temp.replace(path)
        arm=arguments['arm'];grasp=name=='grasp_plan_and_stow'
        steps=[('exec_plan_pose',{'plan_id':arguments['plan_id'],'arm':arm}),
               ('close_gripper' if grasp else 'open_gripper',{'arm':arm}),
               ('set_arm_to_grasp_position',{'arm':arm,'gripper':'keep'}),('reset_body',{'keep_ori_arm':'none'})]
        for i,(tool,args) in enumerate(steps):
            current,_=self.frame();identity=call_id+'-goal-'+str(i+1)
            self.log.emit('action',call_id=identity,tool=tool,arguments=args,sequence=current['sequence'],head_image_id=current['head_image_id'])
            receipt=self.host.execute(identity,tool,args,current);compact=measured_receipt(receipt)
            self.log.emit('result',call_id=identity,receipt=compact)
            record['completed_steps'].append(compact);save()
            response=receipt.get('response') if isinstance(receipt.get('response'),dict) else {}
            failed=(bool(receipt.get('error')) or bool(response.get('error')) or response.get('timed_out') is True
                    or response.get('converged') is False or response.get('ok') is False
                    or response.get('success') is False or response.get('tracking_stalled') is True)
            if grasp and tool=='close_gripper' and response.get('grasp_confirmed') is not True:failed=True
            if failed:
                record['status']='stopped_after_confirmed_tool';save()
                return {'tool_name':name,'response':{'ok':False,'stopped_at':tool,'measured_steps':record['completed_steps'],'native_success_verified':False}}
        record['status']='completed';save()
        return {'tool_name':name,'response':{'ok':True,'measured_steps':record['completed_steps'],'native_success_verified':False}}
