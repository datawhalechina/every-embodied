"""Check fresh image delivery, durable action receipts, and same-thread recovery."""
import base64
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from PIL import Image

from workspace_agent_host import WorkspaceHost, RecoveryBudget
from astra_workspace_client import WorkspaceAgent, recoverable, MODEL


class HostTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name);self.workspace=self.root/'workspace'
        for name in ['tools','perception/exteroception','perception/proprioception','.venv/bin']:
            (self.workspace/name).mkdir(parents=True)
        (self.workspace/'tools/move.py').write_text('SCHEMA = {}\n')
        (self.workspace/'tools/activate_skill.py').write_text('NAME = \"activate_skill\"\n')
        self.session={'workspace':str(self.workspace),'episode_id':'episode-a','session_id':'session-a'}
        (self.workspace/'.session.json').write_text(json.dumps(self.session))
        self.calls=0;self.update(1)
        self.host=WorkspaceHost(self.workspace,self.root/'evidence',allow_execution=True,runner=self.runner)

    def tearDown(self):self.tmp.cleanup()

    def update(self,sequence):
        images={}
        for role in ['head','left_wrist','right_wrist']:
            path=self.workspace/'perception/exteroception'/(role+'.png')
            size=(720,720) if role=='head' else (16,12)
            Image.new('RGB',size,(sequence,2,3)).save(path)
            images[role+'_rgb']={'path':str(path),'width':size[0],'height':size[1]}
        joint=self.workspace/'perception/proprioception/joint_state.json'
        joint.write_text(json.dumps({'episode_id':'episode-a','sequence':sequence,'proprioception':[0.0]*65}))
        (self.workspace/'perception/latest.json').write_text(json.dumps({**self.session,'sequence':sequence,
            'head_image_id':'img'+str(sequence),'joint_state':str(joint),'images':images}))

    def runner(self,*args,**kwargs):
        self.calls+=1;self.update(2)
        return SimpleNamespace(returncode=0,stdout=json.dumps({'tool_name':'move','joint_commands_sent':1,'response':{'measured':True},'images':[]}))

    def test_current_camera_bytes_are_attached_directly_with_exact_sequence(self):
        binding,inputs=self.host.frame();self.assertEqual(binding['sequence'],1)
        pictures=[x for x in inputs if x['type']=='image'];self.assertEqual(len(pictures),3)
        self.assertEqual(base64.b64decode(pictures[0]['url'].split(',',1)[1]),(self.workspace/'perception/exteroception/head.png').read_bytes())
        self.assertEqual(binding['images'][0]['width'],720)

    def test_named_joint_control_uses_the_frozen_robot_configuration_entry(self):
        path=self.workspace/'robot_config/control.py';path.parent.mkdir();path.write_text('SCHEMA = {}\n')
        host=WorkspaceHost(self.workspace,self.root/'joint-host',allow_execution=True,
            runner=lambda *args,**kwargs:SimpleNamespace(returncode=0,stdout=json.dumps({'tool_name':'joint_control'})))
        binding,_=host.frame()
        self.assertEqual(host.tools['joint_control'],path)
        host.execute('joint-check','joint_control',{'steps':1},binding)
        path.write_text('SCHEMA = {"changed":true}\n')
        with self.assertRaisesRegex(ValueError,'code changed'):
            host.execute('joint-check-2','joint_control',{'steps':1},binding)

    def test_camera_state_sequence_mismatch_is_rejected(self):
        p=self.workspace/'perception/proprioception/joint_state.json';d=json.loads(p.read_text());d['sequence']=2;p.write_text(json.dumps(d))
        with self.assertRaises(ValueError):self.host.frame()

    def test_stale_observation_cannot_execute(self):
        binding,_=self.host.frame();self.update(2)
        with self.assertRaises(ValueError):self.host.execute('call1','move',{},binding)
        self.assertEqual(self.calls,0)

    def test_completed_call_returns_saved_receipt_without_replaying(self):
        binding,_=self.host.frame();one=self.host.execute('call1','move',{},binding)
        two=self.host.execute('call1','move',{},binding)
        self.assertEqual(one,two);self.assertEqual(self.calls,1)
        with self.assertRaises(ValueError):self.host.execute('call1','move',{'changed':True},binding)

    def test_lost_receipt_blocks_recovery_and_new_commands(self):
        binding,_=self.host.frame()
        def lost(*args,**kwargs):self.calls+=1;raise ConnectionError('fixture lost receipt')
        self.host.runner=lost
        with self.assertRaises(ConnectionError):self.host.execute('call1','move',{},binding)
        self.assertEqual(self.host.unresolved(),['call1.json'])
        with self.assertRaises(RuntimeError):self.host.execute('call1','move',{},binding)
        with self.assertRaises(RuntimeError):self.host.execute('call2','move',{},binding)
        self.assertEqual(self.calls,1)

    def test_ambiguous_rpc_failure_in_receipt_stays_unresolved(self):
        binding,_=self.host.frame()
        self.host.runner=lambda *args,**kwargs:SimpleNamespace(returncode=0,stdout=json.dumps({'tool_name':'move','error':'EOFError: RPC connection closed'}))
        self.host.execute('call1','move',{},binding)
        self.assertEqual(self.host.unresolved(),['call1.json'])

    def test_source_changes_rejected_before_execution(self):
        binding,_=self.host.frame();(self.workspace/'tools/move.py').write_text('SCHEMA = {"changed":1}\n')
        with self.assertRaises(ValueError):self.host.execute('call1','move',{},binding)
        self.assertEqual(self.calls,0)

    def test_private_and_escaping_paths_are_unreadable(self):
        for path in ['.session.json','../other','/etc/passwd','.runtime/plan.key']:
            with self.assertRaises(ValueError):self.host.readable(path)
        (self.workspace/'tools/escape.py').symlink_to('/etc/passwd')
        with self.assertRaises(ValueError):self.host.readable('tools/escape.py')

    def test_physical_gate_closed_before_command_launch(self):
        self.host.allow_execution=False;binding,_=self.host.frame()
        with self.assertRaises(PermissionError):self.host.execute('call1','move',{},binding)
        self.assertEqual(self.calls,0)

    def test_auxiliary_pixels_are_preserved_for_the_next_direct_input(self):
        meta=self.host.image('perception/exteroception/head.png');self.update(2)
        inputs=self.host.auxiliary_inputs([meta]);blob=base64.b64decode(inputs[1]['url'].split(',',1)[1])
        self.assertNotEqual(blob,(self.workspace/'perception/exteroception/head.png').read_bytes())
        self.assertEqual(meta['sha256'],__import__('hashlib').sha256(blob).hexdigest())

    def test_recovery_budget_survives_instances_and_stops_at_three(self):
        path=self.root/'recoveries.json'
        for i in range(3):self.assertEqual(RecoveryBudget(path).reserve('connection_closed','thread-one'),i+1)
        with self.assertRaises(RuntimeError):RecoveryBudget(path).reserve('model_inactivity','thread-one')
        with self.assertRaises(ValueError):RecoveryBudget(path).reserve('authentication','thread-one')

    def test_only_connection_error_codes_allow_automatic_recovery(self):
        self.assertTrue(recoverable({'responseStreamDisconnected':{'httpStatusCode':None}}))
        for code in ['unauthorized','rateLimitExceeded','cyberPolicy','other',{'httpConnectionFailed':{'httpStatusCode':401}}, {'httpConnectionFailed':{'httpStatusCode':429}}]:
            self.assertFalse(recoverable(code))

    def test_upstream_skill_receipt_without_tool_name_is_preserved(self):
        binding,_=self.host.frame()
        self.host.runner=lambda *a,**k:SimpleNamespace(returncode=0,stdout=json.dumps({'instructions':'fixture skill text','images':[]}))
        result=self.host.execute('skill1','activate_skill',{'name':'pick-up-object'},binding)
        self.assertEqual(result['tool_name'],'activate_skill')
        self.assertEqual(result['instructions'],'fixture skill text')
        self.assertEqual(self.host.unresolved(),[])

    def test_resume_cannot_silently_replace_the_model_thread(self):
        class Rpc:
            def __init__(self,root):pass
            def initialize(self):pass
            def request(self,method,params):return {'model':MODEL,'thread':{'id':'wrong-thread'}}
        agent=WorkspaceAgent(self.host,'fixture task',max_turns=2,rpc_factory=Rpc)
        agent.state['thread_id']='original-thread'
        with self.assertRaisesRegex(ValueError,'identity changed'):agent.connect()
        self.assertEqual(self.calls,0)

    def test_tool_request_from_another_model_thread_stops_without_execution(self):
        class Rpc:
            def __init__(self,root):pass
            def initialize(self):pass
            def close(self):pass
            def request(self,method,params):
                if method=='thread/start':return {'model':MODEL,'thread':{'id':'owned-thread'}}
                if method=='turn/start':return {'turn':{'id':'owned-turn'}}
                raise AssertionError(method)
            def event(self,timeout):
                return {'id':12,'method':'item/tool/call','params':{
                    'threadId':'other-thread','turnId':'owned-turn','tool':'run_tool',
                    'callId':'foreign-call','arguments':{'name':'move','arguments':{},
                    'sequence':1,'head_image_id':'img1'}}}
        agent=WorkspaceAgent(self.host,'fixture task',max_turns=1,rpc_factory=Rpc)
        with self.assertRaisesRegex(ValueError,'another model session'):agent.run()
        self.assertEqual(self.calls,0)

    def test_same_model_thread_resumes_and_completed_tool_is_not_reexecuted(self):
        instances=[];host=self.host
        class Rpc:
            def __init__(self,root):self.index=len(instances);instances.append(self);self.events=[];self.methods=[]
            def initialize(self):pass
            def close(self):pass
            def request(self,method,params):
                self.methods.append((method,params))
                if method in ['thread/start','thread/resume']:return {'model':MODEL,'thread':{'id':'thread-one'}}
                if method=='turn/start':
                    turn='turn-'+str(self.index+1)
                    self.events=[{'id':101,'method':'item/tool/call','params':{'threadId':'thread-one','turnId':turn,'tool':'run_tool','callId':'call1',
                        'arguments':{'name':'move','arguments':{},'sequence':1,'head_image_id':'img1'}}}]
                    if self.index==0:self.events.append(EOFError('fixture disconnected after tool receipt'))
                    else:self.events.extend([{'method':'item/completed','params':{'threadId':'thread-one','turnId':turn,'item':{'type':'agentMessage','text':'{"done":true}'}}},
                        {'method':'turn/completed','params':{'threadId':'thread-one','turn':{'id':turn,'status':'completed'}}}])
                    return {'turn':{'id':turn}}
                raise AssertionError(method)
            def event(self,timeout):
                row=self.events.pop(0)
                if isinstance(row,Exception):raise row
                return row
            def reply(self,*args):pass
        agent=WorkspaceAgent(host,'fixture task',max_turns=2,rpc_factory=Rpc)
        outcome=agent.run();self.assertTrue(outcome['done']);self.assertEqual(self.calls,1)
        resume=[params for name,params in instances[1].methods if name=='thread/resume']
        self.assertEqual(resume[0]['threadId'],'thread-one')
        self.assertEqual(json.loads((host.evidence/'connection-recoveries.json').read_text())['count'],1)
        self.assertEqual(agent.state['turns_started'],2)


if __name__=='__main__':unittest.main()
