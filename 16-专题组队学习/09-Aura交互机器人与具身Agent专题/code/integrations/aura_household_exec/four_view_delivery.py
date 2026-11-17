import os
"""Continuous native four-camera delivery with audio and public execution trace."""
import argparse,hashlib,html,json,shutil,subprocess
from pathlib import Path

VIEWS={'observer','aura_head','r1_head','global'}

def policy_events(pipelines,episode):
    events=[]
    for root in map(Path,pipelines):
        for meta in (root/'host/exec-turns').glob('*/input-metadata.json'):
            if json.loads(meta.read_text())['binding']['episode_id']!=episode:raise ValueError('Policy trace belongs to another native episode')
        for folder in ['exec-events','macro-events']:
            for file in sorted((root/'host'/folder).glob('*.json')):events.append({**json.loads(file.read_text()),'controller_trial':root.name})
    return events

def validate(capture):
    capture=Path(capture).resolve()
    rows=[json.loads(line) for line in (capture/'frames.jsonl').read_text().splitlines() if line.strip()]
    if not rows:raise ValueError('No native camera footage')
    offset=rows[0]['wrapper_tick']-rows[0]['hri_tick']
    for i,row in enumerate(rows):
        if row['capture_tick']!=i*3 or set(row['views'])!=VIEWS:raise ValueError('Four-camera frame gap')
        if row['wrapper_tick']!=rows[0]['wrapper_tick']+3*i or row['wrapper_tick']-row['hri_tick']!=offset:raise ValueError('Native recording clock discontinuity')
        for name in VIEWS:
            path=(capture/row['views'][name]).resolve()
            if not path.is_relative_to(capture) or not path.is_file():raise ValueError('Missing or escaping native frame')
    return rows

def encode_images(paths,destination,*,size=None,transform=None):
    from PIL import Image
    if size:
        cmd=['ffmpeg','-hide_banner','-loglevel','error','-n','-f','rawvideo','-pix_fmt','rgb24','-video_size',str(size[0])+'x'+str(size[1]),'-framerate','10','-i','-']
    else:cmd=['ffmpeg','-hide_banner','-loglevel','error','-n','-f','image2pipe','-framerate','10','-vcodec','png','-i','-']
    cmd+=['-an','-c:v','libx264','-threads','2','-preset','fast','-crf','20','-pix_fmt','yuv420p','-movflags','+faststart',str(destination)]
    with destination.with_suffix('.ffmpeg.log').open('x') as log:
        process=subprocess.Popen(['nice','-n','15',*cmd],stdin=subprocess.PIPE,stdout=subprocess.DEVNULL,stderr=log)
        try:
            for index,path in enumerate(paths):
                data=transform(index,Image.open(path).convert('RGB')).tobytes() if transform else Path(path).read_bytes()
                process.stdin.write(data)
            process.stdin.close()
            if process.wait(timeout=600):raise RuntimeError('Movie encoder failed')
        finally:
            if process.poll() is None:process.terminate();process.wait(timeout=20)

def build(base,runtime,interaction,pipeline,output,*,extra_pipelines=(),final_result=None,media_port=8799):
    from PIL import Image,ImageDraw,ImageFont,ImageOps
    base=Path(base);runtime=Path(runtime);interaction=Path(interaction);pipeline=Path(pipeline);output=Path(output)
    output.mkdir(parents=True,exist_ok=False);capture=runtime/'captures/hri-full';rows=validate(capture);duration=len(rows)/10.
    fontpath=Path(os.environ['AURA_VIDEO_FONT'])
    font=ImageFont.truetype(str(fontpath),28)
    titles={'observer':'完整第三视角 · 人 / Aura / R1','aura_head':'Aura 自身相机 · 寻找与面向访客','r1_head':'R1 Pro 头部相机 · 完整过程','global':'全屋俯视 · 同一个家庭场景'}
    def decorate(label,padded=True):
        def transform(index,image):
            w,h=image.size
            if padded:
                panel=Image.new('RGB',(w,h+76),(6,15,24));panel.paste(image,(0,76));image=panel
            draw=ImageDraw.Draw(image)
            draw.rounded_rectangle((12,12,w-12,63),radius=15,fill=(13,42,55))
            draw.text((27,21),label,font=font,fill='white')
            return image
        return transform
    for view in sorted(VIEWS):
        paths=[capture/row['views'][view] for row in rows]
        with Image.open(paths[0]) as image:size=image.size
        encode_images(paths,output/(view+'-silent.mp4'),size=(size[0],size[1]+76),transform=decorate(titles[view]))
    selected=[]
    for row in rows:selected.append(capture/row['views']['aura_head' if row['phase'] in {'interaction','handoff'} else 'r1_head'])
    def robot(index,image):
        panel=Image.new('RGB',(960,720),(6,15,24));picture=ImageOps.contain(image,(960,644))
        panel.paste(picture,((960-picture.width)//2,76+(644-picture.height)//2))
        label='机器人视角 · Aura 找人' if rows[index]['phase'] in {'interaction','handoff'} else '机器人视角 · R1 清理'
        return decorate(label,padded=False)(index,panel)
    encode_images(selected,output/'robot-silent.mp4',size=(960,720),transform=robot)
    timeline=interaction/'audio-timeline-with-report.json'
    if not timeline.is_file():timeline=interaction/'audio-timeline.json'
    clips=json.loads(timeline.read_text());first_hri=rows[0]['hri_tick']/30
    for clip in clips:
        clip['video_s']=clip['start_hri_s']-first_hri
        if clip['video_s']<0 or clip['video_s']+clip['duration_s']>duration+.15:raise ValueError('Speech extends beyond complete footage')
    cmd=['ffmpeg','-hide_banner','-loglevel','error','-n','-f','lavfi','-i','anullsrc=channel_layout=stereo:sample_rate=48000']
    for clip in clips:cmd+=['-i',clip['file']]
    filters=[]
    for i,clip in enumerate(clips,1):filters.append('['+str(i)+':a]adelay='+str(round(clip['video_s']*1000))+':all=1[a'+str(i)+']')
    filters.append('[0:a]'+''.join('[a'+str(i)+']' for i in range(1,len(clips)+1))+'amix=inputs='+str(len(clips)+1)+':duration=first:normalize=0,alimiter=limit=0.95[audio]')
    cmd+=['-filter_complex',';'.join(filters),'-map','[audio]','-t',str(duration),'-c:a','aac','-b:a','160k',str(output/'dialogue.m4a')]
    subprocess.run(['nice','-n','15',*cmd],check=True,timeout=300)
    media={}
    for view in [*sorted(VIEWS),'robot']:
        movie=output/(view+'-full.mp4')
        subprocess.run(['ffmpeg','-hide_banner','-loglevel','error','-n','-i',str(output/(view+'-silent.mp4')),'-i',str(output/'dialogue.m4a'),'-map','0:v','-map','1:a','-c','copy','-movflags','+faststart',str(movie)],check=True,timeout=120)
        probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-show_entries','format=duration:stream=codec_type,width,height,nb_frames','-of','json',str(movie)],text=True))
        videos=[s for s in probe['streams'] if s['codec_type']=='video'];audio=[s for s in probe['streams'] if s['codec_type']=='audio']
        if len(videos)!=1 or int(videos[0]['nb_frames'])!=len(rows) or not audio:raise ValueError('Incomplete encoded camera or audio stream')
        media[view]={'file':movie.name,'sha256':hashlib.sha256(movie.read_bytes()).hexdigest(),'probe':probe}
    events=[json.loads(s) for s in (interaction/'events.jsonl').read_text().splitlines() if s.strip()]
    pipeline_roots=[Path(p) for p in extra_pipelines]+[pipeline]
    episode=json.loads((runtime/'driver-status.json').read_text())['episode_id']
    events.extend(policy_events(pipeline_roots,episode))
    if runtime.name=='runtime-v7' and (base/'tracking-handoff-v1/result.json').exists():
        handoff=json.loads((base/'tracking-handoff-v1/result.json').read_text())
        events.append({'kind':'harness_revision','native_tick':handoff['native_ticks_after'],'summary':'修复回执裁剪漏传三维跟踪数据；同一现场续跑，0动作重播。','source':'operator_evidence'})
    for row in events:row['video_s']=(row['native_tick']-rows[0]['wrapper_tick'])/30.
    events.sort(key=lambda e:(e['video_s'],e.get('at','')))
    result=json.loads((pipeline/'host/controller-result.json').read_text()) if (pipeline/'host/controller-result.json').is_file() else {'native_success_verified':False}
    final_path=Path(final_result) if final_result else base/'final-native-result-v1.json'
    if final_path.exists():
        final=json.loads(final_path.read_text())
        if final['final_driver']['episode_id']!=episode:raise ValueError('Final result belongs to another native episode')
        scope=final.get('scope','one_can')
        result={**result,'native_success_verified':final['full_three_can_task_success'] if scope=='three_cans' else final['single_can_success'],'scope':scope,'full_three_can_task_success':final['full_three_can_task_success']}
    stop=json.loads((runtime/'process-result.json').read_text()) if (runtime/'process-result.json').is_file() else None
    report={'frames':len(rows),'fps':10,'duration_s':duration,'native_physics_hz':30,'source':'complete continuous same-episode native frames; no missing-frame substitution or speedup',
        'first_wrapper_tick':rows[0]['wrapper_tick'],'last_wrapper_tick':rows[-1]['wrapper_tick'],'media':media,'audio':clips,'events':events,
        'result':result,'human_source':'photoreal skinned avatar, authored101-joint kinematic walking and wave; synthetic speech with actual ASR',
        'aura_turn_source':'actual camera YOLO feedback; kinematic yaw adapter','hidden_reasoning_included':False,'font':'LXGW WenKai; website uses system font',
        'model_waits':'native physics pauses during model and CPU perception waits; video time is simulation time',
        'presentation':'handwritten rounded header outside native camera image; full original field of view retained',
        'runtime_stop':{k:stop.get(k) for k in ['phase','error_type','message','minimum_free_mib','peak_own_mib']} if stop else None,
        'episode_id':episode,'controller_trials':[p.name for p in pipeline_roots],'media_base':'http://127.0.0.1:'+str(media_port)+'/'}
    with (output/'manifest.json').open('x',encoding='utf-8') as file:json.dump(report,file,ensure_ascii=False,indent=2)
    template=Path(__file__).with_name('four_view_replay.html').read_text(encoding='utf-8')
    (output/'replay.html').write_text(template.replace('http://127.0.0.1:8799/','http://127.0.0.1:'+str(media_port)+'/'),encoding='utf-8')
    return report

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('base',type=Path);p.add_argument('runtime',type=Path);p.add_argument('interaction',type=Path);p.add_argument('pipeline',type=Path);p.add_argument('output',type=Path)
    p.add_argument('--extra-pipeline',action='append',type=Path,default=[]);p.add_argument('--final-result',type=Path);p.add_argument('--media-port',type=int,default=8799);a=p.parse_args()
    result=build(a.base,a.runtime,a.interaction,a.pipeline,a.output,extra_pipelines=a.extra_pipeline,final_result=a.final_result,media_port=a.media_port);print(json.dumps({'frames':result['frames'],'duration_s':result['duration_s'],'result':result['result']},ensure_ascii=False),flush=True)
