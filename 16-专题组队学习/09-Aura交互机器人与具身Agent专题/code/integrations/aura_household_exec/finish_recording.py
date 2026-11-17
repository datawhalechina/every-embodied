"""Bounded stationary finale after the policy exits; native proof gates speech."""
import argparse,fcntl,json,math,subprocess,sys
from pathlib import Path

def finish(workspace,runtime,interaction,pipeline,output,*,scope='one_can',audio_manifest=None):
    workspace,runtime,interaction,pipeline,output=map(Path,[workspace,runtime,interaction,pipeline,output])
    if output.exists() or (interaction/'audio-timeline-with-report.json').exists():raise FileExistsError('Preserve previous finale')
    process=pipeline/'r1-agent-process-result.json'
    if not process.is_file():raise RuntimeError('Policy process completion must be recorded before the finale')
    if json.loads(process.read_text())['returncode']!=0:raise RuntimeError('Policy failed; inspect before reporting completion')
    for path in (pipeline/'host/calls').glob('*.json'):
        if json.loads(path.read_text())['status'] not in {'completed','tool_error'}:raise RuntimeError('Unresolved native intent')
    for path in (pipeline/'host/macro-intents').glob('*.json'):
        if json.loads(path.read_text())['status']=='intent_recorded':raise RuntimeError('Unresolved macro')
    sys.path.insert(0,str(workspace))
    from tools.util.protocol import request
    from .controller import native_completion
    config=json.loads((workspace/'.session.json').read_text());sock=Path(config['socket']);session={k:config[k] for k in ['session_id','episode_id']}
    def status():return request(sock,{'operation':'status'})
    def hold():
        current=status()
        if any(current[k]!=session[k] for k in session) or current['pending_action'] or current['inflight_action']:raise RuntimeError('Native writer/session changed')
        action=request(sock,{'operation':'joints','method':'hold_action','args':[],'kwargs':{},**session})
        return request(sock,{'operation':'step','sequence':current['sequence'],'action':action.tolist(),**session})
    with (pipeline/'host/execution.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        for _ in range(35):hold()
        current=status();proof=native_completion(runtime/('three-cans-accepted.json' if scope=='three_cans' else 'one-can-accepted.json'),{'episode_id':current['episode_id'],'native_ticks':current['ticks']},scope)
        observer=json.loads((runtime/'native-observer-live.json').read_text());held=set(observer['assisted_grasp_targets_observer_only'].values())
        passed=bool(proof and all(observer['inside_bin_observer_only'].get(n) and observer['stable_released_samples'].get(n,0)>=2 and n not in held for n in proof['inside']))
        voice=('success_three' if scope=='three_cans' else 'success') if passed else 'incomplete'
        lines=json.loads(Path(audio_manifest).read_text())['lines'];line=lines[voice]
        stamp=json.loads((runtime/'hri-status.json').read_text())['hri_tick']/30
        duration=float(subprocess.check_output(['ffprobe','-v','error','-show_entries','format=duration','-of','default=noprint_wrappers=1:nokey=1',line['file']],text=True))
        temp=runtime/'hri-control.tmp';temp.write_text(json.dumps({'start':True,'phase':'completed' if passed else 'incomplete','angular_z':0.,'valid_until_tick':current['ticks']+3}));temp.replace(runtime/'hri-control.json')
        clips=json.loads((interaction/'audio-timeline.json').read_text());clips.append({'name':voice,'start_hri_s':stamp,'duration_s':duration,'file':line['file'],'speaker':'aura','synthetic_voice':True,'basis':'current released stable native Inside verification'})
        with (interaction/'audio-timeline-with-report.json').open('x') as file:json.dump(clips,file,ensure_ascii=False,indent=2)
        for _ in range(math.ceil((duration+1)*30)):hold()
        (runtime/'checkpoint-request.json').write_text('{"reason":"finale after measured result, stationary maintenance hold"}');hold()
        result={'scope':scope,'single_can_success':passed,'full_three_can_task_success':passed and scope=='three_cans',
            'speech_camera_handoff':json.loads((interaction/'result.json').read_text())['passed'],'native_proof':proof,
            'final_driver':status(),'final_hri_status':json.loads((runtime/'hri-status.json').read_text()),
            'final_native_observer':json.loads((runtime/'native-observer-live.json').read_text()),'voice_report':line['authored_text'],'new_media_verified':False}
        with output.open('x') as file:json.dump(result,file,ensure_ascii=False,indent=2)
        return result

if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ['workspace','runtime','interaction','pipeline','output','audio-manifest']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--scope',choices=['one_can','three_cans'],default='one_can');a=p.parse_args()
    result=finish(a.workspace,a.runtime,a.interaction,a.pipeline,a.output,scope=a.scope,audio_manifest=a.audio_manifest)
    print(json.dumps({k:result[k] for k in ['scope','single_can_success','full_three_can_task_success','voice_report']},ensure_ascii=False),flush=True)
