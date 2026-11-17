"""Join verified executed prefix and resumed cameras with a visible recovery cut."""
import argparse,hashlib,json,subprocess
from pathlib import Path

VIEWS=('observer','robot','aura_head','r1_head','global')

def join_plan(prefix,resumed,old_frames,restoration,*,card_frames=15):
    if not restoration.get('passed'):raise ValueError('No accepted physical restoration')
    source=restoration['source'];new=restoration['new_session']['episode_id']
    if prefix['episode_id']!=source['source_episode_id'] or resumed['episode_id']!=new:
        raise ValueError('Media episode does not match accepted restoration lineage')
    if prefix['fps']!=10 or resumed['fps']!=10:raise ValueError('Expected native 10fps camera sampling')
    selected=[r for r in old_frames if r['wrapper_tick']<=source['source_tick']]
    if not selected or selected!=old_frames[:len(selected)] or len(old_frames)!=prefix['frames']:
        raise ValueError('Old camera prefix is not complete')
    if source['source_tick']-selected[-1]['wrapper_tick'] not in (0,1,2):
        raise ValueError('No camera frame at the restored boundary')
    prefix_s=len(selected)/10.;card_s=card_frames/10.;offset=prefix_s+card_s
    events=[dict(e,segment='executed_prefix') for e in prefix['events'] if 0<=e['video_s']<prefix_s]
    events.append({'kind':'checkpoint_resume','video_s':prefix_s,'segment':'recovery_boundary',
        'summary':'恢复检查点，保留首罐结果；从第二罐继续。开场与已完成动作没有重播。',
        'old_episode_id':prefix['episode_id'],'new_episode_id':new,'rollback_sim_seconds':restoration['rollback_sim_seconds']})
    events.extend(dict(e,video_s=offset+e['video_s'],segment='resumed') for e in resumed['events'] if e['video_s']>=0)
    return {'prefix_frames':len(selected),'prefix_duration_s':prefix_s,'card_frames':card_frames,'card_duration_s':card_s,
        'resumed_offset_s':offset,'frames':len(selected)+card_frames+resumed['frames'],
        'duration_s':offset+resumed['duration_s'],'events':sorted(events,key=lambda e:e['video_s']),
        'episodes':[prefix['episode_id'],new],'source_checkpoint':source,'rollback_sim_seconds':restoration['rollback_sim_seconds'],
        'prefix_last_native_tick':selected[-1]['wrapper_tick'],'resumed_first_native_tick':resumed['first_wrapper_tick'],
        'source':'executed prefix trimmed at checkpoint + labelled recovery boundary + complete resumed native footage',
        'single_native_episode':False,'old_commands_replayed':0}

def build(prefix_dir,resumed_dir,old_capture,restoration_file,output,*,font_path,media_port=8803):
    from PIL import Image,ImageDraw,ImageFont
    prefix_dir,resumed_dir,old_capture,restoration_file,output=map(Path,[prefix_dir,resumed_dir,old_capture,restoration_file,output])
    prefix=json.loads((prefix_dir/'manifest.json').read_text());resumed=json.loads((resumed_dir/'manifest.json').read_text())
    rows=[json.loads(s) for s in (old_capture/'frames.jsonl').read_text().splitlines() if s.strip()]
    plan=join_plan(prefix,resumed,rows,json.loads(restoration_file.read_text()));output.mkdir(parents=True,exist_ok=False)
    media={}
    for view in VIEWS:
        before=prefix_dir/prefix['media'][view]['file'];after=resumed_dir/resumed['media'][view]['file']
        # Validate original media digests before encoding a derived presentation.
        for path,meta in [(before,prefix['media'][view]),(after,resumed['media'][view])]:
            if hashlib.sha256(path.read_bytes()).hexdigest()!=meta['sha256']:raise ValueError('Source video digest mismatch')
        old_stream=next(s for s in prefix['media'][view]['probe']['streams'] if s['codec_type']=='video')
        new_stream=next(s for s in resumed['media'][view]['probe']['streams'] if s['codec_type']=='video')
        size=(old_stream['width'],old_stream['height'])
        if size!=(new_stream['width'],new_stream['height']):raise ValueError('Camera canvas dimensions differ')
        image=Image.new('RGB',size,(8,27,38));draw=ImageDraw.Draw(image)
        draw.rounded_rectangle((20,20,size[0]-20,size[1]-20),radius=26,fill=(15,48,61))
        font=ImageFont.truetype(str(font_path),max(22,min(38,size[0]//23)))
        lines=['检查点恢复 · 接着原现场做','第一罐已在桶内，从第二罐继续','原互动和动作未重播']
        for i,line in enumerate(lines):draw.text((45,size[1]//2-80+i*57),line,font=font,fill='white')
        card=output/(view+'-recovery-card.png');image.save(card)
        p=plan['prefix_duration_s'];c=plan['card_duration_s'];r=resumed['duration_s']
        filters=[f'[0:v]trim=end_frame={plan["prefix_frames"]},setpts=PTS-STARTPTS,setsar=1,format=yuv420p[pv]',
            f'[0:a]atrim=end={p},asetpts=PTS-STARTPTS,aresample=48000[pa]',
            f'[1:v]trim=end_frame={plan["card_frames"]},setpts=PTS-STARTPTS,setsar=1,format=yuv420p[cv]',
            f'[2:a]atrim=end={c},asetpts=PTS-STARTPTS[ca]',
            f'[3:v]trim=end_frame={resumed["frames"]},setpts=PTS-STARTPTS,setsar=1,format=yuv420p[rv]',
            f'[3:a]atrim=end={r},asetpts=PTS-STARTPTS,aresample=48000[ra]',
            '[pv][pa][cv][ca][rv][ra]concat=n=3:v=1:a=1[v][a]']
        movie=output/(view+'-full.mp4')
        cmd=['nice','-n','15','ffmpeg','-hide_banner','-loglevel','error','-n','-threads','2','-i',str(before),
            '-loop','1','-framerate','10','-t',str(c),'-i',str(card),'-f','lavfi','-t',str(c),'-i','anullsrc=r=48000:cl=stereo','-threads','2','-i',str(after),
            '-filter_complex_threads','1','-filter_complex',';'.join(filters),'-map','[v]','-map','[a]',
            '-c:v','libx264','-threads','2','-preset','fast','-crf','20','-r','10','-c:a','aac','-b:a','160k','-movflags','+faststart',str(movie)]
        with (output/(view+'-encode.log')).open('x') as log:subprocess.run(cmd,stdout=log,stderr=log,check=True,timeout=1800)
        probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-show_entries','format=duration:stream=codec_type,width,height,nb_frames','-of','json',str(movie)],text=True))
        video=next(s for s in probe['streams'] if s['codec_type']=='video')
        if int(video['nb_frames'])!=plan['frames'] or abs(float(probe['format']['duration'])-plan['duration_s'])>.15:
            raise ValueError('Recovery presentation dropped/duplicated frames or altered duration')
        media[view]={'file':movie.name,'sha256':hashlib.sha256(movie.read_bytes()).hexdigest(),'probe':probe}
    audio=[dict(a) for a in prefix['audio'] if a['video_s']+a['duration_s']<=plan['prefix_duration_s']+.01]
    audio.extend(dict(a,video_s=a['video_s']+plan['resumed_offset_s']) for a in resumed['audio'])
    report={**resumed,**plan,'media':media,'audio':audio,'media_base':f'http://127.0.0.1:{media_port}/',
        'prefix_manifest':str(prefix_dir/'manifest.json'),'resumed_manifest':str(resumed_dir/'manifest.json'),
        'human_source':prefix['human_source'],'recovery_disclosure':'Two native episodes joined at the validated checkpoint; boundary card is presentation only.'}
    (output/'manifest.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
    template=Path(__file__).with_name('four_view_replay.html').read_text(encoding='utf-8')
    template=template.replace('http://127.0.0.1:8799/',f'http://127.0.0.1:{media_port}/')
    template=template.replace('<h1>人来招呼，Aura 找人，R1 收拾垃圾</h1>','<h1>从原现场接续 · 人、Aura 与 R1 的三罐清理</h1><p><strong>本片保留原互动与已执行前缀，在明确的恢复提示后接续。两个原生运行的完整原片另行保留。</strong></p>')
    (output/'replay.html').write_text(template,encoding='utf-8')
    return report
