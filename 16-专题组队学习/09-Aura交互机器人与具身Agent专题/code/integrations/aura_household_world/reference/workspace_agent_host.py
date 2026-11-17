"""Execute fixed workspace tools once and attach coherent camera bytes directly.

The host owns its ledger outside the model workspace. Model tools can inspect
readable robot instructions and invoke a single named Python tool, without
arbitrary shell execution or access to authentication/evaluator artifacts.
"""
import base64
import datetime
import fcntl
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import subprocess

from PIL import Image


def atomic(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix('.tmp');temporary.write_text(json.dumps(value,indent=2)+'\n');temporary.replace(path)


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


class RecoveryBudget:
    def __init__(self,path,maximum=3):self.path,self.maximum=Path(path),maximum

    def reserve(self,reason,thread_id):
        if reason not in {'connection_closed','model_inactivity'}:raise ValueError('Not a recoverable connection fault')
        self.path.parent.mkdir(parents=True,exist_ok=True)
        with self.path.with_suffix('.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            row=json.loads(self.path.read_text()) if self.path.exists() else {'count':0,'attempts':[]}
            if row['count']>=self.maximum:raise RuntimeError('Same-episode recovery budget exhausted')
            row['count']+=1;row['attempts'].append({'at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'number':row['count'],'reason':reason,'thread_id':thread_id})
            atomic(self.path,row);return row['count']


class WorkspaceHost:
    def __init__(self,workspace,evidence,*,allow_execution=False,runner=subprocess.run,live_status=None):
        self.workspace,self.evidence=Path(workspace).resolve(),Path(evidence).resolve()
        if self.evidence.is_relative_to(self.workspace):raise ValueError('Host evidence must be outside the workspace')
        self.evidence.mkdir(parents=True,exist_ok=True)
        self.allow_execution,self.runner,self.live_status=allow_execution,runner,live_status
        self.session=json.loads((self.workspace/'.session.json').read_text())
        if Path(self.session['workspace']).resolve()!=self.workspace:raise ValueError('Session workspace mismatch')
        self.python=self.workspace/'.venv/bin/python'
        self.tools={p.stem:p for p in (self.workspace/'tools').glob('*.py') if not p.name.startswith('_')}
        if (self.workspace/'robot_config/control.py').is_file():
            self.tools['joint_control']=self.workspace/'robot_config/control.py'
        manifest={str(p.relative_to(self.workspace)):hashlib.sha256(p.read_bytes()).hexdigest()
                  for p in (self.workspace/'tools').rglob('*.py')}
        for p in [self.workspace/'robot_config/control.py']:
            if p.is_file():manifest[str(p.relative_to(self.workspace))]=hashlib.sha256(p.read_bytes()).hexdigest()
        path=self.evidence/'tool-source.json'
        if path.exists() and json.loads(path.read_text())!=manifest:raise ValueError('Workspace code changed after binding')
        if not path.exists():atomic(path,manifest)
        self.manifest=manifest

    def readable(self,relative):
        if not isinstance(relative,str) or Path(relative).is_absolute():raise ValueError('Use a workspace-relative path')
        path=(self.workspace/relative).resolve()
        if not path.is_relative_to(self.workspace):raise ValueError('Path escapes the workspace')
        part=path.relative_to(self.workspace).parts
        if not part or part[0] not in {'AGENTS.md','tools','skills','robot_config','perception','tool_call_history'}:
            raise ValueError('Path is outside readable robot instructions and observations')
        return path

    def read_text(self,relative,offset=0,limit=16000):
        if type(offset) is not int or offset<0 or type(limit) is not int or not 1<=limit<=32000:
            raise ValueError('Invalid text slice')
        path=self.readable(relative)
        if path.stat().st_size>2*1024**2:raise ValueError('Text exceeds host bound')
        text=path.read_text()
        return {'path':relative,'offset':offset,'total_characters':len(text),'text':text[offset:offset+limit]}

    def find_text(self,relative,query):
        if not isinstance(query,str) or not 1<=len(query)<=200:raise ValueError('Bounded literal query required')
        path=self.readable(relative)
        if path.stat().st_size>2*1024**2:raise ValueError('Text exceeds host bound')
        text=path.read_text();offsets=[];position=0
        while len(offsets)<20:
            position=text.find(query,position)
            if position<0:break
            offsets.append(position);position+=len(query)
        return {'path':relative,'query':query,'offsets':offsets}

    def list_files(self,relative='tools'):
        folder=self.readable(relative)
        if not folder.is_dir():raise ValueError('Directory required')
        return {'path':relative,'entries':[p.name for p in sorted(folder.iterdir()) if not p.name.startswith('.')]}

    def frame(self):
        path=self.workspace/'perception/latest.json';before=path.read_bytes();meta=json.loads(before)
        for key in ['episode_id','session_id']:
            if meta[key]!=self.session[key]:raise ValueError('Perception belongs to another session')
        if type(meta['sequence']) is not int or meta['sequence']<0 or not meta.get('head_image_id'):
            raise ValueError('Latest observation binding is incomplete')
        joint_path=self.readable(str(Path(meta['joint_state']).relative_to(self.workspace)))
        joint=json.loads(joint_path.read_text())
        if joint['sequence']!=meta['sequence'] or joint['episode_id']!=meta['episode_id']:
            raise ValueError('Camera and measured state are from different observations')
        proprio=joint.get('proprioception')
        if not isinstance(proprio,list) or len(proprio)!=65 or any(type(v) not in (int,float) or not math.isfinite(v) for v in proprio):
            raise ValueError('Measured state must contain 65 finite values')
        inputs=[];images=[]
        for role in ['head','left_wrist','right_wrist']:
            item=meta['images'][role+'_rgb'];image_path=Path(item['path'])
            if image_path.is_absolute():image_path=image_path.relative_to(self.workspace)
            image_path=self.readable(str(image_path));blob=image_path.read_bytes()
            if len(blob)>16*1024**2:raise ValueError('Image exceeds host bound')
            with Image.open(io.BytesIO(blob)) as image:
                if image.size!=(item['width'],item['height']) or image.mode not in ('RGB','RGBA'):
                    raise ValueError('Image dimensions or mode disagree with manifest')
                if role=='head' and image.size!=(720,720):raise ValueError('Clickable head frame must be720x720')
                mime=Image.MIME.get(image.format)
            if mime not in ('image/png','image/jpeg'):raise ValueError('Unsupported camera encoding')
            images.append({'role':role,'sha256':hashlib.sha256(blob).hexdigest(),'width':item['width'],'height':item['height']})
            inputs.extend([{'type':'text','text':role+' camera; sequence='+str(meta['sequence'])+'; head_image_id='+meta['head_image_id']},
                {'type':'image','url':'data:'+mime+';base64,'+base64.b64encode(blob).decode()}])
        if path.read_bytes()!=before:raise ValueError('Perception changed while preparing input')
        binding={k:meta[k] for k in ['episode_id','session_id','sequence','head_image_id']}
        binding.update(proprioception=proprio,images=images)
        if self.live_status is not None:
            status=self.live_status()
            if not status.get('connected') or any(status.get(key)!=binding[key] for key in ['episode_id','session_id','sequence']):
                raise ValueError('Published camera bundle is stale relative to the owned driver')
            binding['native_ticks']=status['ticks']
        return binding,inputs

    def image(self,relative,caption='Auxiliary tool image; only the current head camera is clickable.'):
        path=self.readable(relative);blob=path.read_bytes()
        if len(blob)>16*1024**2:raise ValueError('Auxiliary image exceeds bound')
        with Image.open(io.BytesIO(blob)) as image:
            mime=Image.MIME.get(image.format)
            if mime not in ('image/png','image/jpeg') or image.width*image.height>16000000:
                raise ValueError('Unsupported auxiliary image')
        sha=hashlib.sha256(blob).hexdigest();folder=self.evidence/'auxiliary';folder.mkdir(exist_ok=True)
        destination=folder/(sha+('.png' if mime=='image/png' else '.jpg'))
        if not destination.exists():destination.write_bytes(blob)
        return {'sha256':sha,'file':str(destination.relative_to(self.evidence)),'mime':mime,'caption':caption}

    def auxiliary_inputs(self,rows):
        inputs=[]
        for row in rows:
            path=(self.evidence/row['file']).resolve()
            if not path.is_relative_to(self.evidence/'auxiliary'):raise ValueError('Invalid auxiliary artifact')
            blob=path.read_bytes()
            if hashlib.sha256(blob).hexdigest()!=row['sha256']:raise ValueError('Auxiliary artifact changed')
            inputs.extend([{'type':'text','text':row['caption']},
                {'type':'image','url':'data:'+row['mime']+';base64,'+base64.b64encode(blob).decode()}])
        return inputs

    def unresolved(self):
        return [p.name for p in (self.evidence/'calls').glob('*.json')
                if json.loads(p.read_text())['status'] not in {'completed','tool_error'}]

    def execute(self,call_id,name,arguments,binding):
        if not re.fullmatch(r'[A-Za-z0-9_.-]{1,180}',call_id):raise ValueError('Invalid tool call identifier')
        if name not in self.tools or not isinstance(arguments,dict):raise ValueError('Unknown named tool or invalid arguments')
        identity=digest({'tool':name,'arguments':arguments,'binding':{k:binding[k] for k in ['episode_id','session_id','sequence','head_image_id']}})
        calls=self.evidence/'calls';calls.mkdir(exist_ok=True);path=calls/(call_id+'.json')
        with (self.evidence/'execution.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            if path.exists():
                row=json.loads(path.read_text())
                if row['identity']!=identity:raise ValueError('Tool call identifier was reused with different inputs')
                if row['status'] in {'completed','tool_error'}:return row['receipt']
                raise RuntimeError('Unresolved physical intent requires inspection; never replayed')
            if self.unresolved():raise RuntimeError('A previous physical intent is unresolved')
            if not self.allow_execution:raise PermissionError('Physical execution gate is closed')
            for relative,expected in self.manifest.items():
                if hashlib.sha256((self.workspace/relative).read_bytes()).hexdigest()!=expected:
                    raise ValueError('Workspace tool code changed')
            current,_=self.frame()
            if any(current[k]!=binding[k] for k in ['episode_id','session_id','sequence','head_image_id']):
                raise ValueError('Tool input was prepared from a stale observation')
            row={'identity':identity,'status':'intent_recorded','tool':name,'arguments':arguments,'binding':{k:binding[k] for k in ['episode_id','session_id','sequence','head_image_id']}}
            atomic(path,row)
            env=dict(os.environ,ROBOT_WORKSPACE=str(self.workspace),PYTHONDONTWRITEBYTECODE='1')
            env.pop('PYTHONPATH',None);env.pop('PYTHONHOME',None)
            history=self.workspace/'tool_call_history'
            history_before={p.name for p in history.iterdir()} if history.exists() else set()
            result=self.runner([str(self.python),str(self.tools[name]),json.dumps(arguments,allow_nan=False)],
                cwd=self.workspace,env=env,capture_output=True,text=True)
            transport=self.evidence/'tool-transport'/call_id
            transport.mkdir(parents=True,exist_ok=False)
            (transport/'stdout.txt').write_text(result.stdout or "")
            (transport/'stderr.txt').write_text(getattr(result,"stderr","") or "")
            atomic(transport/'process-result.json',{'returncode':result.returncode})
            if result.returncode!=0:
                raise RuntimeError('Tool process exited without a confirmed execution receipt')
            try:receipt=json.loads(result.stdout)
            except json.JSONDecodeError:
                from tool_transport import stored_receipt
                from tools.util.receipt import model_receipt
                receipt,original=stored_receipt(self.workspace,history_before,name,
                    arguments,binding,model_receipt)
                atomic(transport/'atomic-receipt-binding.json',{
                    'original_receipt':str(original),
                    'original_sha256':hashlib.sha256(original.read_bytes()).hexdigest(),
                    'physical_commands_replayed':0})
            if isinstance(receipt,dict) and name in {'activate_skill','deactivate_skill'} and 'tool_name' not in receipt:
                receipt['tool_name']=name
            if not isinstance(receipt,dict) or receipt.get('tool_name')!=name:
                raise ValueError('Invalid tool receipt; inspect current observation')
            if result.returncode!=0:raise RuntimeError('Tool process exited without a confirmed execution receipt')
            response=receipt.get('response') if isinstance(receipt.get('response'),dict) else {}
            errors=str(receipt.get('error') or '')+' '+str(response.get('error') or '')
            ambiguous=any(term in errors.lower() for term in ['eoferror','timeouterror','connectionerror','connectionreseterror','connectionabortederror','brokenpipeerror','rpc connection','connection closed','connection reset','broken pipe','evaluator disconnected','no evaluator observation','observation failed'])
            row.update(status='physical_confirmation_unknown' if ambiguous else 'tool_error' if receipt.get('error') else 'completed',receipt=receipt)
            atomic(path,row);return receipt
