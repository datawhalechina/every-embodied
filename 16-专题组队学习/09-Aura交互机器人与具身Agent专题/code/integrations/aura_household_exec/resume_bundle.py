"""Select a complete recovery pair; never execute or replay robot actions."""
import argparse,hashlib,json
from pathlib import Path
from .native_checkpoint import inspect_snapshot

def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def inspect_pair(path):
    path=Path(path);raw=path.read_bytes();agent=json.loads(raw)
    binding=agent['binding'];turn=agent['last_completed_turn'];saved=agent['native_checkpoint']
    if not all(binding.get(k) for k in ('session_id','episode_id')) or not isinstance(turn,int) or turn<1:
        raise ValueError('Incomplete completed-tool Agent checkpoint')
    if not saved or saved.get('saved') is not True:raise ValueError('No confirmed native fence')
    _,report=inspect_snapshot(saved['native_snapshot'],saved['native_sha256'])
    app_path=Path(saved['application_snapshot'])
    if digest(app_path)!=saved['application_sha256']:raise ValueError('Application checkpoint digest mismatch')
    app=json.loads(app_path.read_text())
    if app['wrapper_tick']!=saved['saved_native_tick']:raise ValueError('Application/native checkpoint clocks differ')
    if not {'hri_tick','phase','pose','yaw'}<=app.keys():raise ValueError('Application state incomplete')
    skill=agent['active_skill_state']
    if not isinstance(skill,dict) or 'active_skill' not in skill:raise ValueError('Skill state missing')
    return {'source_agent_checkpoint':str(path.resolve()),'source_agent_sha256':hashlib.sha256(raw).hexdigest(),
        'source_binding':binding,'completed_turn':turn,'physical_checkpoint':saved,
        'decision_context':{'public_memory':agent['memory'],'skill_state':skill},
        'native_report':report,'application_phase':app['phase'],'application_hri_tick':app['hri_tick'],
        'recovery_protocol':['Create an isolated compatible own runtime; load native and application state.',
            'Validate serialized object roots, joints, grasp and recomputed native predicates.',
            'Bind a new session; capture fresh RGB-D before policy release.',
            'Retain public task memory and skill; discard old image IDs, coordinates, signed plans and pending intents.'],
        'commands_to_replay':[],'physical_restore_validated':False,
        'scope':'complete saved pair only; simulator restoration must be qualified separately'}

def select_latest(evidence):
    candidates=sorted((Path(evidence)/'agent-checkpoints').glob('[0-9][0-9][0-9][0-9].json'),reverse=True)
    rejected=[]
    for path in candidates:
        try:bundle=inspect_pair(path)
        except (OSError,ValueError,KeyError,TypeError) as error:
            rejected.append({'path':str(path),'reason':str(error)});continue
        return {**bundle,'newer_incomplete_candidates':rejected}
    raise ValueError('No complete native/application/Agent checkpoint pair: '+json.dumps(rejected))

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('evidence',type=Path);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();bundle=select_latest(a.evidence)
    with a.output.open('x',encoding='utf-8') as file:json.dump(bundle,file,ensure_ascii=False,indent=2)
    print(json.dumps({'completed_turn':bundle['completed_turn'],'saved_native_tick':bundle['physical_checkpoint']['saved_native_tick'],
        'commands_to_replay':[],'physical_restore_validated':False,'output':str(a.output)},ensure_ascii=False))
if __name__=='__main__':main()
