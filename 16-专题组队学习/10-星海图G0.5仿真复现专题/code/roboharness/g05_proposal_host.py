"""Add a nonphysical G0.5 proposal tool to the common direct-image host.

A native observation, subtask, candidate index, seed and qualified policy
identify one local model request. Model requests and workspace tool receipts are persisted before
publication. The common host still owns physical execution and its receipts.
"""
import fcntl
import json
from pathlib import Path
import re
import time

from proposal_exchange import ProposalExchange,sha
from workspace_agent_host import WorkspaceHost,atomic,digest

HERE=Path(__file__).resolve().parent
BINDING_KEYS=('episode_id','session_id','sequence','head_image_id')


class G05ProposalHost(WorkspaceHost):
    def __init__(self,workspace,evidence,registration,capture_rpc,**kwargs):
        super().__init__(workspace,evidence,**kwargs)
        if not callable(capture_rpc) or not callable(self.live_status):
            raise ValueError('Proposal requests require the owned control channel and live observation check')
        self.capture_rpc=capture_rpc
        self.exchange=ProposalExchange(registration,self.evidence/'g05-exchange',self.workspace)
        config=self.exchange.config
        if config.get('logical_id') not in {f'task02-{instance}-roboharness_v2_g05' for instance in (301,304,306)}:
            raise ValueError('Only the fixed combination episodes expose the proposal tool')
        self.task=config.get('g05_task_instruction')
        self.wait_seconds=config.get('candidate_wait_seconds')
        if not isinstance(self.task,str) or not self.task.strip() or len(self.task)>8192:
            raise ValueError('A frozen bounded G0.5 task instruction is required')
        if type(self.wait_seconds) is not int or not 0<=self.wait_seconds<=1800:
            raise ValueError('An explicit bounded candidate wait is required')
        if config['episode_id']!=self.session['episode_id']:
            raise ValueError('Development registration does not match this actual workspace episode')
        for name,reference in [('g05_propose','workspace_g05_propose.py'),('g05_step','workspace_g05_step.py')]:
            tool=self.tools.get(name)
            if tool is None or sha(tool)!=sha(HERE/reference):
                raise ValueError('Hybrid tool declaration differs from its registered implementation')

    def request_id(self,binding,task,seed,candidate_index):
        # Camera capture IDs can change without advancing the native sequence.
        identity={key:binding[key] for key in ('episode_id','session_id','sequence')}
        identity.update(task=task,seed=seed,candidate_index=candidate_index,policy_fingerprint=self.exchange.config['policy_fingerprint'])
        return 'proposal_'+digest(identity)[:48]

    def validate_sources(self):
        self.exchange.check_frozen()
        for relative,expected in self.manifest.items():
            if sha(self.workspace/relative)!=expected:raise ValueError('Workspace tool code changed')

    def same_observation(self,binding):
        current,_=self.frame()
        if any(current[key]!=binding[key] for key in BINDING_KEYS):
            raise ValueError('Proposal input belongs to a stale observation')
        return current

    def staged_capture(self,identifier,binding,task,seed,candidate_index):
        folder=self.exchange.input_root/'captures'/identifier
        path=folder/'request.json'
        if not folder.exists():
            result=self.capture_rpc({'operation':'capture_g05_input',
                **{key:binding[key] for key in ('episode_id','session_id','sequence')},
                'request_id':identifier,'task':task,'seed':seed,'candidate_index':candidate_index})
            if (Path(result['staged_request']).resolve()!=path.resolve()
                    or result.get('model_submitted') is not False
                    or result.get('episode_id')!=binding['episode_id']
                    or result.get('sequence')!=binding['sequence']):
                raise ValueError('Capture reply does not match the requested owned observation')
            if sha(path)!=result['request_sha256']:
                raise ValueError('Capture reply digest differs from staged input')
        if not path.is_file():return None
        try:request=json.loads(path.read_bytes())
        except json.JSONDecodeError:return None
        if (request.get('request_id')!=identifier or request.get('task')!=task
                or request.get('seed')!=seed or request.get('candidate_index')!=candidate_index):
            raise ValueError('Staged request identity or task changed')
        return path

    def proposal_response(self,identifier,binding,task,seed,candidate_index):
        self.exchange.check_frozen()
        reservations=self.exchange.evidence/'requests'
        if (not (reservations/(identifier+'.json')).exists() and
                len(list(reservations.glob('*.json'))) >= self.exchange.config['model_call_limit']):
            return {'status':'candidate_rejected','request_id':identifier,
                    'rejection':'Model-call budget exhausted; select an available candidate or use geometry',
                    'physical_execution':False}
        path=self.staged_capture(identifier,binding,task,seed,candidate_index)
        if path is None:
            return {'status':'pending','request_id':identifier,'stage':'capturing','physical_execution':False}
        self.same_observation(binding)
        try:
            self.exchange.submit(path,binding)
        except RuntimeError as error:
            if str(error)!='Development model-call budget exhausted':raise
            return {'status':'candidate_rejected','request_id':identifier,
                    'rejection':'Model-call budget exhausted; select an available candidate or use geometry',
                    'physical_execution':False}
        deadline=time.monotonic()+self.wait_seconds
        while True:
            current=self.same_observation(binding)
            response=self.exchange.collect(identifier,current,current['proprioception'])
            response['physical_execution']=False
            if response['status']!='pending' or time.monotonic()>=deadline:return response
            time.sleep(min(1,max(0,deadline-time.monotonic())))

    def execute(self,call_id,name,arguments,binding):
        if name!='g05_propose':return super().execute(call_id,name,arguments,binding)
        if not re.fullmatch(r'[A-Za-z0-9_.-]{1,180}',call_id):raise ValueError('Invalid tool call identifier')
        if not isinstance(arguments,dict) or set(arguments)!={'task','seed','candidate_index'}:
            raise ValueError('A task, seed, and candidate index are required')
        task,seed,candidate_index=arguments['task'],arguments['seed'],arguments['candidate_index']
        if not isinstance(task,str) or not task.strip() or len(task)>8192:
            raise ValueError('A bounded current subtask is required')
        if type(seed) is not int or not 0<=seed<2**32 or type(candidate_index) is not int or not 0<=candidate_index<3:
            raise ValueError('Invalid candidate seed or index')
        identity=digest({'tool':name,'arguments':arguments,
                         'binding':{key:binding[key] for key in BINDING_KEYS}})
        calls=self.evidence/'calls';calls.mkdir(exist_ok=True);path=calls/(call_id+'.json')
        with (self.evidence/'execution.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            row=json.loads(path.read_bytes()) if path.exists() else None
            if row is not None:
                if row['identity']!=identity:raise ValueError('Tool call identifier reused with different inputs')
                if row['status']=='completed':return row['receipt']
                if row.get('physical_commands_sent')!=0:
                    raise RuntimeError('Unknown physical confirmation blocks proposal reconciliation')
            unresolved=[item for item in self.unresolved() if item!=path.name]
            if unresolved:raise RuntimeError('A previous physical or proposal intent is unresolved')
            if not self.allow_execution:raise PermissionError('Model request execution gate is closed')
            self.validate_sources();self.same_observation(binding)
            identifier=self.request_id(binding,task,seed,candidate_index)
            if row is None:
                row={'identity':identity,'status':'intent_recorded','tool':name,'arguments':arguments,
                    'binding':{key:binding[key] for key in BINDING_KEYS},'request_id':identifier,
                    'physical_commands_sent':0}
                atomic(path,row)
            elif row.get('request_id')!=identifier:
                raise ValueError('Persisted proposal request identity changed')
            response=self.proposal_response(identifier,binding,task,seed,candidate_index)
            receipt={'tool_name':name,'response':{'ok':True,**response},
                     'joint_commands_sent':0,'images':[]}
            row.update(status='completed',receipt=receipt);atomic(path,row)
            return receipt
