"""Allowlisted presentation records: public summaries, actions and receipts only."""
import datetime, json, math
from pathlib import Path

KINDS={
 'think': {'summary','expected_check','sequence','head_image_id'},
 'action': {'call_id','tool','arguments','sequence','head_image_id'},
 'result': {'call_id','receipt'},
 'interaction': {'speaker','text','input_source','state'},
 'camera': {'view','path','tick','sequence','head_image_id'},
 'trajectory': {'actor','position_m','tick','provenance'},
 'verification': {'native_success','q_score','source','sha256'},
}

def validate(kind,payload):
    if kind not in KINDS or set(payload)-KINDS[kind]:raise ValueError('Unknown public event field')
    # Bounded serialization also rejects NaN. Unknown raw model events never enter.
    if len(json.dumps(payload,ensure_ascii=False,allow_nan=False).encode())>32768:raise ValueError('Public event too large')
    return payload

class EventLog:
    def __init__(self,root,*,source='runtime',clock=None):
        if source not in {'runtime','fixture'}:raise ValueError('Explicit provenance required')
        self.root=Path(root);self.root.mkdir(parents=True,exist_ok=True);self.source=source
        self.counter=len(list(self.root.glob('*.json')));self.clock=clock
    def emit(self,kind,**payload):
        validate(kind,payload)
        row={'kind':kind,'source':self.source,'at':datetime.datetime.now(datetime.timezone.utc).isoformat(),**payload}
        if self.clock is not None:
            tick=self.clock()
            if type(tick) is not int or tick<0:raise ValueError('Current native tick required for synchronized runtime events')
            row['native_tick']=tick
        with (self.root/f'{self.counter:06d}.json').open('x',encoding='utf-8') as f:json.dump(row,f,ensure_ascii=False,indent=2,allow_nan=False)
        self.counter+=1
        return row

def public_receipt(value):
    """No arbitrary stderr, prompts, auth or private model text in presentation."""
    keys={'status','execution_id','sequence','head_image_id','input_rejected','error_type','returncode','ticks','completed','success','sha256','tool_name'}
    return {k:v for k,v in value.items() if k in keys and isinstance(v,(str,int,float,bool,type(None)))}

def measured_pose(actor,position,tick):
    if actor not in {'R1','Aura','Visitor'} or len(position)!=3 or type(tick) is not int or tick<0:raise ValueError('Invalid pose')
    if any(type(x) not in (int,float) or not math.isfinite(x) for x in position):raise ValueError('Non-finite pose')
    return {'actor':actor,'position_m':position,'tick':tick,'provenance':'simulator_observer_only_not_model_input'}
