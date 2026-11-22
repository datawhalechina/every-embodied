#!/usr/bin/env python3
"""Build an isolated, validated replay manifest from the archived baseline."""
import argparse
import hashlib
import json
import math
from pathlib import Path


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()

def build(run, out):
    out.mkdir(parents=True,exist_ok=False)
    commands=[];command_sources=[]
    for file in (run/'trajectory/calls').glob('*/joint_commands.jsonl'):
        command_sources.append({'file':str(file),'sha256':digest(file)})
        for line in file.read_text().splitlines():
            row=json.loads(line)
            if type(row.get('sequence_before')) is not int or row.get('sequence_after')!=row['sequence_before']+1:
                raise ValueError('Every command must advance one sequence')
            a=row.get('action')
            if not isinstance(a,list) or len(a)!=27 or any(type(x) not in (int,float) or not math.isfinite(x) for x in a):
                raise ValueError('Action must contain 27 finite numeric values')
            commands.append({'sequence_before':row['sequence_before'],'sequence_after':row['sequence_after'],'action':a})
    commands.sort(key=lambda x:x['sequence_before'])
    if not commands or commands[0]['sequence_before']!=43 or commands[-1]['sequence_after']!=16785:
        raise ValueError('Unexpected run boundaries')
    if any(a['sequence_after']!=b['sequence_before'] for a,b in zip(commands,commands[1:])):
        raise ValueError('Recorded action stream has gaps or duplicates')
    events=[]
    for file in (run/'agent/host/calls').glob('exec-*.json'):
        d=json.loads(file.read_text());binding=d.get('binding') or {};seq=binding.get('sequence')
        if type(seq) is not int:continue
        a=d.get('arguments') or {};response=(d.get('receipt') or {}).get('response') or {}
        if not isinstance(response,dict):response={}
        event={'sequence':seq,'image_id':binding.get('head_image_id'),'tool':d.get('tool'),
               'reason':a.get('reason') if isinstance(a.get('reason'),str) else None,
               'source':str(file),'arguments':a,
               'result':{k:response[k] for k in ('target_reached','ok','failure_reason','error','action_steps','pos_err_mm') if k in response}}
        events.append(event)
    events.sort(key=lambda x:(x['sequence'],x['tool'] or ''))
    checkpoints=[]
    for file in (run/'trajectory/calls').glob('*/observation/manifest.json'):
        d=json.loads(file.read_text()); joints=file.parent/'joint_state.json';head=d.get('head_image_id')
        if not joints.exists():continue
        j=json.loads(joints.read_text()); p=j.get('proprioception')
        if not isinstance(p,list) or len(p)!=65 or j.get('sequence')!=d.get('sequence'):continue
        meta={'sequence':d['sequence'],'head_image_id':head,'joint_state':str(joints),'manifest':str(file)}
        archive=d.get('archive',{}).get('head_rgb',{}); hp=archive.get('path')
        if hp and Path(hp).is_file():meta['head_image']=hp
        checkpoints.append(meta)
    checkpoints.sort(key=lambda x:x['sequence'])
    rows=out/'actions.jsonl'
    with rows.open('x') as f:
        for row in commands:f.write(json.dumps(row,separators=(',',':'))+'\n')
    (out/'events.json').write_text(json.dumps(events,ensure_ascii=False,indent=2)+'\n')
    (out/'checkpoints.json').write_text(json.dumps(checkpoints,ensure_ascii=False,indent=2)+'\n')
    report={'source_run':str(run),'usage':'visual replay only; not a scored evaluation',
            'action_count':len(commands),'first_sequence':commands[0]['sequence_before'],'last_sequence':commands[-1]['sequence_after'],
            'source_files':len(command_sources),'action_stream_sha256':digest(rows),'checkpoint_count':len(checkpoints),
            'checkpoint_head_images':sum('head_image' in x for x in checkpoints),'events':len(events),
            'explicit_reasons':sum(bool(e['reason']) for e in events),
            'source_hashes':command_sources,'model_calls':0,'simulator_commands_sent':0}
    (out/'manifest.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    return {k:v for k,v in report.items() if k!='source_hashes'}

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path,required=True);parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();print(json.dumps(build(args.run,args.out),ensure_ascii=False,indent=2))
