"""Offline course checks. No robot access, shell launches, model calls or downloads."""
from __future__ import annotations
import argparse,hashlib,json,math,os
from pathlib import Path
import xml.etree.ElementTree as ET

TARGETS={'can_of_soda_113','can_of_soda_114','can_of_soda_115'}

def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))

def audit(root):
    root=Path(root).resolve();media=read(root/'media-manifest.json')
    proof=read(root/'assets/evidence/three-cans-proof.json');returned=read(root/'assets/evidence/return-proof.json')
    events=read(root/'assets/evidence/public-events.json')
    problems=[]
    if proof.get('passed') is not True or set(proof.get('inside',[]))!=TARGETS:problems.append('three-can goal')
    if proof.get('released') is not True or proof.get('stable_native_observer_samples',0)<2 or proof.get('official_task_complete') is not True:problems.append('release/stability/task predicates')
    if returned.get('passed') is not True:problems.append('return goal')
    for row in media['videos']:
        if not (root/row['preview']).is_file():problems.append('missing preview '+row['id'])
        if len(row['sha256'])!=64 or row['bytes']<=0 or row['duration_s']<=0:problems.append('invalid media metadata')
        if not row['url'].startswith('https://github.com/datawhalechina/every-embodied/releases/download/'):problems.append('nonpublic video')
    household=[r for r in media['videos'] if r['id'].startswith('household-')]
    if len(household)!=5 or any(abs(r['duration_s']-255.7)>.1 for r in household):problems.append('full-view duration')
    return {'passed':not problems,'videos':len(media['videos']),'full_household_views':len(household),'public_events':len(events),'native_three_cans':proof['inside'],'return_verified':returned['passed'],'problems':problems,'scope':'saved files and metadata; no live camera or new simulation'}

def twin(code_root):
    twin=Path(code_root).resolve()/'digital_twin';xml=ET.parse(twin/'aura.xml').getroot()
    resources=[node.get('file') for node in xml.iter() if node.get('file')]
    missing=[]
    for rel in resources:
        p=(twin/rel).resolve()
        if not p.is_relative_to(twin) or not p.is_file():missing.append(rel)
    manifest=read(twin/'model_manifest.json')
    return {'passed':not missing,'resource_count':len(resources),'missing':missing,'bbox_mm':manifest['bbox_mm'],'fidelity':manifest['parameters']['fidelity'],'scope':'MJCF structure/resources; physical run separately'}

def unproject(u,v,z,fx,fy,cx,cy):
    values=[u,v,z,fx,fy,cx,cy]
    if not all(isinstance(x,(int,float)) and math.isfinite(x) for x in values) or z<=0 or fx<=0 or fy<=0:raise ValueError('finite values, positive depth and focal length required')
    return [(u-cx)*z/fx,(v-cy)*z/fy,z]

def confirmed_edges(values,minimum):
    if type(minimum) is not int or minimum<1:raise ValueError('minimum must be a positive integer')
    count=0;result=[]
    for i,value in enumerate(values):
        count=count+1 if value is True else 0
        if count==minimum:result.append(i)
    return result

def save_checkpoint(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    payload=json.dumps(value,sort_keys=True,ensure_ascii=False,allow_nan=False).encode('utf-8')
    tmp=path.with_suffix('.tmp');tmp.write_bytes(payload);os.replace(tmp,path)
    return hashlib.sha256(payload).hexdigest()

def load_checkpoint(path,sha256):
    data=Path(path).read_bytes()
    if hashlib.sha256(data).hexdigest()!=sha256:raise ValueError('checkpoint hash mismatch')
    return json.loads(data)

def checkpoint_exercise(root):
    # Teaching-only progress state, not a physics snapshot or executed robot plan.
    root=Path(root);p=root/'agent.json'
    state={'completed':['can_of_soda_115'],'status':'running','sequence':1}
    digest=save_checkpoint(p,state);restored=load_checkpoint(p,digest)
    rejected_done=0;released=[]
    actions=[('done',None),('release','can_of_soda_114'),('release','can_of_soda_113'),('done',None)]
    for action,target in actions:
        if action=='done':
            if set(restored['completed'])!=TARGETS:rejected_done+=1;continue
            restored['status']='complete'
        elif target not in restored['completed']:
            released.append(target);restored['completed'].append(target)
        restored['sequence']+=1;digest=save_checkpoint(p,restored)
    final=load_checkpoint(p,digest)
    return {'rejected_done':rejected_done,'continued':bool(released),'completed':final['status']=='complete','released_ids':released,'first_can_replayed':False,'input_source':'teaching_state_machine'}

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=['audit','twin']);p.add_argument('--root',type=Path,required=True);a=p.parse_args()
    result=audit(a.root) if a.command=='audit' else twin(a.root)
    print(json.dumps(result,ensure_ascii=False,indent=2));raise SystemExit(0 if result['passed'] else 1)
if __name__=='__main__':main()
