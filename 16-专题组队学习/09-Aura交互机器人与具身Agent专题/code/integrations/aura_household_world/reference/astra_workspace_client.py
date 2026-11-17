"""Run a persistent Astra session with direct current-camera inputs.

Only named workspace tools are available. Each decision permits one tool
execution, then waits for a new host-supplied observation. Connection recovery
resumes the same stored model thread and never repeats an unresolved action.
"""
from collections import deque
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import selectors
import signal
import subprocess
import time

from workspace_agent_host import WorkspaceHost, RecoveryBudget, atomic

MODEL='gpt-6-astra'
CODEX=os.environ.get('CODEX_EXEC','codex')
TOOLS=[
 {'type':'function','name':'find_text','description':'Find up to20 literal text offsets in a readable robot tool or instruction file.',
  'inputSchema':{'type':'object','properties':{'path':{'type':'string'},'query':{'type':'string'}},'required':['path','query'],'additionalProperties':False}},
 {'type':'function','name':'view_image','description':'Stage one readable robot illustration, posture image or tool preview for the next directly attached image turn. End the current turn afterward.',
  'inputSchema':{'type':'object','properties':{'path':{'type':'string'}},'required':['path'],'additionalProperties':False}},
 {'type':'function','name':'read_file','description':'Read a slice of a readable robot workspace file.',
  'inputSchema':{'type':'object','properties':{'path':{'type':'string'},'offset':{'type':'integer','minimum':0}},'required':['path','offset'],'additionalProperties':False}},
 {'type':'function','name':'list_files','description':'List a readable robot workspace directory.',
  'inputSchema':{'type':'object','properties':{'path':{'type':'string'}},'required':['path'],'additionalProperties':False}},
 {'type':'function','name':'run_tool','description':'Execute one named workspace Python tool bound to the current observation. Stop this decision afterward and await new camera inputs.',
  'inputSchema':{'type':'object','properties':{'name':{'type':'string'},'arguments':{'type':'object','additionalProperties':True},
    'sequence':{'type':'integer','minimum':0},'head_image_id':{'type':'string'}},'required':['name','arguments','sequence','head_image_id'],'additionalProperties':False}}]
INSTRUCTIONS=('Control this robot through the supplied workspace tools. Read AGENTS.md, tools/README.md, '
 'the baseline skill and the relevant tool source/schema before first use. Read the joint manual before direct joint control. '
 'Use view_image to inspect joint illustrations and auxiliary tool images; it stages direct input for the next turn. '
 'Only the newest directly attached head image is clickable, using its original720x720 pixels and current head_image_id. '
 'Inspect wrist views as additional observations. Execute at most one run_tool call per model turn, then finish the turn '
 'so the host can attach new observations. Use measured outcomes and images to choose subsequent actions. '
 'Do not reset the episode or access evaluator answers. When the task is complete, return JSON with done=true; otherwise done=false.')


class ModelInactivity(TimeoutError):pass


class ServiceFault(RuntimeError):
    def __init__(self,info):super().__init__('Model service fault; inspect structured code');self.info=info


def recoverable(info):
    if not isinstance(info,dict) or len(info)!=1:return False
    key=next(iter(info))
    if key not in {'httpConnectionFailed','responseStreamConnectionFailed','responseStreamDisconnected'}:return False
    return info[key].get('httpStatusCode') in (None,408,502,504)


class AstraRpc:
    def __init__(self,root,codex=CODEX):
        self.root=Path(root);self.root.mkdir(parents=True,exist_ok=True);self.root.chmod(0o700)
        home=self.root/'codex-home';home.mkdir(mode=0o700,exist_ok=True)
        auth=home/'auth.json'
        if not auth.exists():auth.symlink_to((Path.home()/'.codex/auth.json').resolve())
        config='model = "'+MODEL+'"\n[features]\nmemories = false\n[analytics]\nenabled = false\n'
        path=home/'config.toml'
        if path.exists() and path.read_text()!=config:raise ValueError('Private client config changed')
        if not path.exists():path.write_text(config);path.chmod(0o600)
        args=[codex,'app-server','--listen','stdio://','-c','sqlite_home='+json.dumps(str(self.root/'db')),
              '-c','log_dir='+json.dumps(str(self.root/'logs'))]
        self.process=subprocess.Popen(args,cwd=self.root,env=dict(os.environ,CODEX_HOME=str(home)),stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,stderr=(self.root/'stderr.log').open('ab'),start_new_session=True)
        self.selector=selectors.DefaultSelector();self.selector.register(self.process.stdout,selectors.EVENT_READ)
        self.buffer=b'';self.counter=0;self.pending=deque()

    def write(self,row):
        self.process.stdin.write(json.dumps({'jsonrpc':'2.0',**row}).encode()+b'\n');self.process.stdin.flush()

    def read(self,deadline):
        while time.monotonic()<deadline:
            if b'\n' in self.buffer:
                line,self.buffer=self.buffer.split(b'\n',1)
                if line:return json.loads(line)
            elif self.selector.select(min(1,max(0,deadline-time.monotonic()))):
                blob=os.read(self.process.stdout.fileno(),65536)
                if not blob:raise EOFError('Model transport closed')
                self.buffer+=blob
        raise ModelInactivity('Model transport inactive')

    def request(self,method,params,timeout=180):
        self.counter+=1;identity=self.counter;self.write({'id':identity,'method':method,'params':params})
        deadline=time.monotonic()+timeout
        while True:
            row=self.read(deadline)
            if row.get('id')==identity and 'method' not in row:
                if 'error' in row:raise ServiceFault({'rpc_error':{'code':row['error'].get('code')}})
                return row['result']
            if row.get('method') in {'item/tool/call','turn/completed','item/completed','thread/tokenUsage/updated'}:
                self.pending.append(row)

    def initialize(self):
        self.request('initialize',{'clientInfo':{'name':'roboharness-g05-workspace','version':'0.1'},'capabilities':{'experimentalApi':True}})
        self.write({'method':'initialized'})

    def event(self,timeout=1800):
        if self.pending:return self.pending.popleft()
        return self.read(time.monotonic()+timeout)

    def reply(self,identity,result):self.write({'id':identity,'result':result})

    def close(self):
        if self.process.poll() is None:
            os.killpg(self.process.pid,signal.SIGTERM)
            try:self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:os.killpg(self.process.pid,signal.SIGKILL);self.process.wait()
        self.selector.close()


class WorkspaceAgent:
    def __init__(self,host,task,*,max_turns,rpc_factory=AstraRpc,effort='xhigh'):
        if type(max_turns) is not int or max_turns<1:raise ValueError('An explicit finite model-turn budget is required')
        self.host,self.task,self.max_turns,self.rpc_factory,self.effort=host,task,max_turns,rpc_factory,effort
        self.root=host.evidence;self.path=self.root/'agent-state.json';self.rpc=None
        self.recovery=RecoveryBudget(self.root/'connection-recoveries.json',3)
        contract={'task_sha256':hashlib.sha256(task.encode()).hexdigest(),'instructions_sha256':hashlib.sha256(INSTRUCTIONS.encode()).hexdigest(),
            'tool_spec_sha256':hashlib.sha256(json.dumps(TOOLS,sort_keys=True).encode()).hexdigest(),'model':MODEL,'effort':effort,'max_turns':max_turns,
            'session_id':host.session['session_id'],'episode_id':host.session['episode_id']}
        self.state=json.loads(self.path.read_text()) if self.path.exists() else {'contract':contract,'thread_id':None,'turns_started':0,'phase':'prepared'}
        if self.state['contract']!=contract:raise ValueError('Model session contract changed')
        atomic(self.path,self.state)

    def connect(self):
        self.rpc=self.rpc_factory(self.root/'.private/model-client');self.rpc.initialize()
        previous_thread=self.state['thread_id']
        if previous_thread:
            result=self.rpc.request('thread/resume',{'threadId':self.state['thread_id'],'model':MODEL,'excludeTurns':True,'environments':[]})
        else:
            result=self.rpc.request('thread/start',{'model':MODEL,'allowProviderModelFallback':False,'cwd':str(self.host.workspace),
                'approvalPolicy':'never','sandbox':'read-only','environments':[],'ephemeral':False,'experimentalRawEvents':False,
                'baseInstructions':INSTRUCTIONS,'dynamicTools':TOOLS})
        if result.get('model')!=MODEL:raise ValueError('Requested model was not preserved')
        if previous_thread and result['thread']['id']!=previous_thread:raise ValueError('Resumed model thread identity changed')
        self.state['thread_id']=result['thread']['id'];self.state['phase']='connected';atomic(self.path,self.state)

    def recover(self,reason):
        if not self.state['thread_id']:raise RuntimeError('No existing model thread to resume')
        if self.host.unresolved():raise RuntimeError('Physical confirmation is unresolved; connection recovery blocked')
        before,_=self.host.frame()
        recovery_number=self.recovery.reserve(reason,self.state['thread_id'])
        if self.rpc:self.rpc.close()
        self.state['phase']='recovering';atomic(self.path,self.state)
        self.connect()
        # A current, coherent observation must exist before another model turn.
        after,_=self.host.frame()
        if any(before[key]!=after[key] for key in ['session_id','episode_id','sequence']):
            raise RuntimeError('Owned observation changed during model-only recovery')
        atomic(self.root/'recoveries'/(str(recovery_number)+'.json'),{'reason':reason,'thread_id':self.state['thread_id'],
            'sequence_before':before['sequence'],'sequence_after':after['sequence'],
            'native_ticks_before':before.get('native_ticks'),'native_ticks_after':after.get('native_ticks'),
            'physical_commands_replayed':0})

    def decision(self):
        if self.host.unresolved():raise RuntimeError('Unresolved tool execution blocks model dispatch')
        if self.state['turns_started']>=self.max_turns:raise RuntimeError('Explicit model-turn budget exhausted')
        if self.rpc is None:self.connect()
        binding,images=self.host.frame();number=self.state['turns_started']+1
        frame_record={k:v for k,v in binding.items() if k!='proprioception'}
        atomic(self.root/'frames'/(str(number)+'.json'),frame_record)
        text={'task':self.task,'binding':{k:binding[k] for k in ['episode_id','session_id','sequence','head_image_id']},
              'measured_proprioception':binding['proprioception'],'previous_tool_receipt':self.state.get('last_receipt')}
        self.state.update(turns_started=number,phase='turn_requested');atomic(self.path,self.state)
        images+=self.host.auxiliary_inputs(self.state.get('pending_images',[]))
        raw_inputs=[{'type':'text','text':json.dumps(text)},*images]
        from wire_payload_guard import optimize_frame_inputs_if_needed, assert_inputs_within_safe_ceiling
        turn_inputs, transcoded = optimize_frame_inputs_if_needed(raw_inputs)
        assert_inputs_within_safe_ceiling(turn_inputs)
        result=self.rpc.request('turn/start',{'threadId':self.state['thread_id'],'model':MODEL,'effort':self.effort,
            'environments':[],'input':turn_inputs,
            'outputSchema':{'type':'object','properties':{'done':{'type':'boolean'}},'required':['done'],'additionalProperties':False}})
        self.state.update(turn_id=result['turn']['id'],phase='thinking');atomic(self.path,self.state)
        run_calls=set();messages=[];activity=time.monotonic();usage=None;staged_image=False;hosted_calls=0
        self.state['pending_images']=[];atomic(self.path,self.state)
        while True:
            event=self.rpc.event(max(.01,1800-(time.monotonic()-activity)))
            method=event.get('method');params=event.get('params',{})
            if params.get('threadId') not in (None,self.state['thread_id']) or params.get('turnId') not in (None,self.state['turn_id']):
                if method=='item/tool/call':raise ValueError('Tool request belongs to another model session or turn')
                continue
            if method=='turn/completed' and params.get('turn',{}).get('id')!=self.state['turn_id']:continue
            if method and (method.startswith('item/') or method=='thread/tokenUsage/updated'):activity=time.monotonic()
            if method=='item/tool/call':
                hosted_calls+=1
                if hosted_calls>64:raise RuntimeError('Read/tool request bound per decision exhausted')
                if params['threadId']!=self.state['thread_id'] or params['turnId']!=self.state['turn_id']:
                    raise ValueError('Tool call belongs to another active model turn')
                name,args,call_id=params['tool'],params['arguments'],params['callId']
                try:
                    if name=='read_file':answer=self.host.read_text(args['path'],args['offset'])
                    elif name=='list_files':answer=self.host.list_files(args['path'])
                    elif name=='find_text':answer=self.host.find_text(args['path'],args['query'])
                    elif name=='view_image':
                        if staged_image:raise ValueError('One staged image per decision')
                        staged_image=True
                        self.state['pending_images']=[self.host.image(args['path'])];atomic(self.path,self.state)
                        answer={'image_staged':True,'next_step':'Finish this turn to receive its pixels directly.'}
                    elif name=='run_tool':
                        if staged_image:raise ValueError('Inspect staged pixels in the next decision before execution')
                        run_calls.add(call_id)
                        if len(run_calls)>1:raise RuntimeError('One tool execution per decision exceeded')
                        requested={**binding,'sequence':args['sequence'],'head_image_id':args['head_image_id']}
                        answer=self.host.execute(call_id,args['name'],args['arguments'],requested)
                        self.state['last_receipt']=answer
                        self.state['pending_images']=[]
                        for media in answer.get('images',[])[:2]:
                            path=Path(media['path'])
                            if path.is_absolute():path=path.relative_to(self.host.workspace)
                            self.state['pending_images'].append(self.host.image(str(path),media.get('caption','Auxiliary tool image; inspect current head camera for clicks.')))
                        atomic(self.path,self.state)
                        if self.host.unresolved():raise RuntimeError('Tool outcome requires observation review')
                    else:raise ValueError('Unknown hosted tool')
                except (ValueError,KeyError,FileNotFoundError,PermissionError) as error:
                    if self.host.unresolved():raise RuntimeError('Tool outcome is unresolved') from error
                    answer={'input_rejected':True,'error_type':type(error).__name__}
                self.rpc.reply(event['id'],{'success':True,'contentItems':[{'type':'inputText','text':json.dumps(answer)}]})
            elif method=='thread/tokenUsage/updated':usage=params.get('tokenUsage')
            elif method=='item/completed' and params.get('item',{}).get('type')=='agentMessage':
                # Only the final completion flag is kept; analysis/reasoning is discarded.
                body=params['item'].get('text','')
                try:
                    value=json.JSONDecoder().raw_decode(body[body.index('{'):])[0]
                    if isinstance(value,dict) and type(value.get('done')) is bool:messages.append(value['done'])
                except (ValueError,json.JSONDecodeError):pass
            elif event.get('id') is not None and method:
                raise RuntimeError('Unexpected model-side execution or approval request')
            elif method=='turn/completed':
                turn=params['turn']
                if turn['status']=='failed':raise ServiceFault((turn.get('error') or {}).get('codexErrorInfo'))
                if turn['status']!='completed':raise RuntimeError('Model turn was interrupted')
                outcome={'turn_number':number,'turn_id':self.state['turn_id'],'done':messages[-1] if messages else False,
                    'tool_execution_ids':sorted(run_calls),'usage':usage,'observation_sequence':binding['sequence']}
                atomic(self.root/'turns'/(str(number)+'.json'),outcome)
                self.state.update(phase='ready',last_outcome=outcome);atomic(self.path,self.state);return outcome

    def run(self):
        with (self.root/'agent.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            try:
                while True:
                    try:
                        outcome=self.decision()
                        if outcome['done']:return outcome
                    except (EOFError,ConnectionError,BrokenPipeError):self.recover('connection_closed')
                    except ModelInactivity:self.recover('model_inactivity')
                    except ServiceFault as error:
                        if not recoverable(error.info):raise
                        self.recover('connection_closed')
            finally:
                if self.rpc:self.rpc.close()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace',type=Path,required=True)
    parser.add_argument('--evidence',type=Path,required=True)
    parser.add_argument('--task-file',type=Path,required=True)
    parser.add_argument('--max-model-turns',type=int,required=True)
    parser.add_argument('--readiness',type=Path,required=True)
    parser.add_argument('--run',action='store_true')
    args=parser.parse_args()
    readiness=json.loads(args.readiness.read_text())
    required=['licensed_data_verified','workspace_imports_passed','direct_image_service_passed','native_step_cap_enforced']
    if not all(readiness.get(key) is True for key in required):raise ValueError('Readiness gates are incomplete')
    if readiness.get('max_model_turns')!=args.max_model_turns:raise ValueError('Model budget differs from readiness record')
    if readiness.get('stage')=='development':
        if readiness.get('physical_qualification_passed') is not True:raise ValueError('Physical qualification is incomplete')
    elif readiness.get('stage')=='qualification':
        if readiness.get('native_step_cap')!=120:raise ValueError('Qualification native cap must be120')
    else:raise ValueError('Unknown evaluation stage')
    live_status=None
    if args.run:
        from tools.util.protocol import request
        config=json.loads((args.workspace/'.session.json').read_text())
        live_status=lambda:request(Path(config['socket']),{'operation':'status'})
    host=WorkspaceHost(args.workspace,args.evidence,allow_execution=args.run,live_status=live_status)
    agent=WorkspaceAgent(host,args.task_file.read_text(),max_turns=args.max_model_turns)
    if args.run:agent.run()
    else:
        binding,_=host.frame()
        print(json.dumps({'ready_for_explicit_run':True,'episode_id':binding['episode_id'],
            'sequence':binding['sequence'],'model_calls':0,'physical_commands':0}))


if __name__=='__main__':main()
