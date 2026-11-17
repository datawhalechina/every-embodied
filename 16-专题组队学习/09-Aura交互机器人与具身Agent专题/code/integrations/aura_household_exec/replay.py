"""Complete two-view native footage with explicitly calibrated public trace."""
import hashlib,html,json,subprocess
from pathlib import Path

def calibrated_offset(samples):
    if len(samples)<2:raise ValueError('At least two independently observed clock pairs required')
    offsets={s['native_ticks']-s['wrapper_tick'] for s in samples}
    if len(offsets)!=1:raise ValueError('Camera and driver clocks diverged')
    return offsets.pop()

def event_time(event,first_wrapper_tick,offset,frame_count):
    if event.get('source')!='runtime' or type(event.get('native_tick')) is not int:raise ValueError('Unclocked or fixture event')
    seconds=(event['native_tick']-offset-first_wrapper_tick)/30.
    if not 0<=seconds<frame_count/10.:raise ValueError('Event lies outside recorded native footage')
    return seconds

def build(capture,events_roots,calibration,output):
    capture=Path(capture).resolve();output=Path(output)
    if output.exists():raise FileExistsError('Never overwrite a prior replay')
    rows=[json.loads(s) for s in (capture/'frames.jsonl').read_text().splitlines() if s.strip()]
    if not rows:raise ValueError('Actual native camera frames required')
    offset=calibrated_offset(json.loads(Path(calibration).read_text())['samples'])
    for i,row in enumerate(rows):
        if row['capture_tick']!=i*3 or set(row['views'])!={'observer','r1_head'}:raise ValueError('Incomplete native dual-view stream')
        if i and row['wrapper_tick']!=rows[i-1]['wrapper_tick']+3:raise ValueError('Native camera gap')
        for relative in row['views'].values():
            p=(capture/relative).resolve()
            if not p.is_relative_to(capture) or not p.is_file():raise ValueError('Missing or escaping camera frame')
    output.mkdir(parents=True)
    media={}
    for view in ['observer','r1_head']:
        listing=output/(view+'-ordered-frames.txt');lines=[]
        for row in rows:
            path=str((capture/row['views'][view]).resolve())
            if "'" in path or '\n' in path:raise ValueError('Unsafe concat path')
            lines.extend(["file '"+path+"'",'duration 0.1'])
        listing.write_text('\n'.join(lines)+'\n')
        dest=output/(view+'-full.mp4')
        subprocess.run(['ionice','-c','2','-n','7','nice','-n','15','ffmpeg','-hide_banner','-loglevel','error','-n','-safe','0','-f','concat','-i',str(listing),'-r','10','-frames:v',str(len(rows)),'-c:v','libx264','-threads','2','-preset','fast','-crf','20','-pix_fmt','yuv420p','-movflags','+faststart',str(dest)],check=True,timeout=600)
        probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-show_entries','format=duration:stream=width,height,nb_frames','-of','json',str(dest)],text=True))
        if int(probe['streams'][0]['nb_frames'])!=len(rows):raise ValueError('Encoded frame count mismatch')
        media[view]={'sha256':hashlib.sha256(dest.read_bytes()).hexdigest(),'probe':probe}
    events=[]
    for root in events_roots:
        root=Path(root)
        for p in sorted((root/'exec-events').glob('*.json')):
            value=json.loads(p.read_text());value['controller_trial']=root.parent.name
            value['video_s']=event_time(value,rows[0]['wrapper_tick'],offset,len(rows));events.append(value)
    events.sort(key=lambda e:e['at'])
    report={'fps':10,'native_physics_hz':30,'frames':len(rows),'capture_first_wrapper_tick':rows[0]['wrapper_tick'],
            'clock_offset':offset,'media':media,'events':events,'source':'complete actual same-episode native camera frames',
            'audio':'not recorded in this raw trial','private_reasoning_included':False,'completion':'See separate verified task result; footage alone is not a success flag'}
    (output/'manifest.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    page='''<!doctype html><html lang="zh-Hant"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Aura × R1 Pro · 原生完整雙視角</title><style>body{font:17px/1.7 system-ui;color:#1a3c4a;background:#edf3f5;margin:24px}main{max-width:1520px;margin:auto}.views{display:grid;grid-template-columns:1.2fr 1fr;gap:20px}article,aside{background:white;padding:20px;border-radius:18px}video{width:100%;border-radius:16px}pre{white-space:pre-wrap;overflow-wrap:anywhere}button{padding:12px;border-radius:10px}.card{border-left:4px solid #187c88;padding:12px;margin:16px 0}@media(max-width:850px){.views{display:block}}</style><main><h1>Aura × R1 Pro · 完整相機與公開執行回放</h1><p>同一個原生仿真的完整第三俯視與R1頭部相機，10fps仿真採樣；模型決策期間物理暫停。此原始版沒有錄製聲音。Think 是公開決策摘要；影片結果不自動代表任務完成。</p><button id="play">同步播放／暫停</button><div class="views"><article><h2>完整第三視角</h2><video controls src="observer-full.mp4"></video></article><article><h2>完整 R1 相機</h2><video controls src="r1_head-full.mp4"></video></article></div><aside><h2>Decision · Action · Result</h2><p>時序依原生控制tick與相機tick的實測配對校準。唯讀工具可在同一物理tick發生，完整原始順序與牆鐘時間保留於紀錄。</p><div id="trace"></div><details><summary>完整公開執行紀錄</summary><pre id="all"></pre></details><a href="manifest.json">影片幀數、雜湊與對時證據</a></aside></main><script>const vs=[...document.querySelectorAll('video')];let es=[];fetch('manifest.json').then(r=>r.json()).then(m=>{es=m.events;document.querySelector('#all').textContent=JSON.stringify(es,null,2)});document.querySelector('#play').onclick=()=>{const play=vs[0].paused;vs[1].currentTime=vs[0].currentTime;vs.forEach(v=>play?v.play().catch(()=>{}):v.pause())};vs[0].addEventListener('timeupdate',()=>{const t=vs[0].currentTime;if(Math.abs(vs[1].currentTime-t)>.2)vs[1].currentTime=t;const b=document.querySelector('#trace');b.replaceChildren();es.filter(e=>e.video_s<=t).slice(-6).forEach(e=>{const a=document.createElement('div');a.className='card';const p=document.createElement('pre');p.textContent=e.kind.toUpperCase()+' · '+e.video_s.toFixed(2)+'s\\n'+JSON.stringify(e,null,2);a.append(p);b.append(a)})})</script></html>'''
    (output/'index.html').write_text(page,encoding='utf-8');return report
