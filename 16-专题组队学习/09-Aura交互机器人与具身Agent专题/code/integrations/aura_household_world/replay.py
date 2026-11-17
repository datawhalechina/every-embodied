"""Build synchronized full videos only from complete actual same-world frames."""
import argparse,json,shutil,subprocess
from pathlib import Path


def verify_frames(root):
    root=Path(root)
    rows=[json.loads(s) for s in (root/'frames.jsonl').read_text().splitlines() if s.strip()]
    if not rows:raise ValueError('No actual frames; cannot create a task replay')
    for index,row in enumerate(rows):
        tick=row['native_tick']
        if index and tick-rows[index-1]['native_tick']!=3:raise ValueError('Frame gap; no interpolation or frozen replacement')
        if set(row['views'])!={'observer','r1_head','aura_head'}:raise ValueError('Missing same-tick camera')
        for name,value in row['views'].items():
            path=(root/value).resolve()
            if not path.is_relative_to(root.resolve()) or not path.is_file():raise ValueError('Missing or invalid source image')
    return rows


def build(root,output):
    root=Path(root);output=Path(output)
    rows=verify_frames(root/'camera-streams')
    if output.exists():raise FileExistsError('Keep previous videos unchanged')
    if not shutil.which('ffmpeg'):raise RuntimeError('Existing ffmpeg executable required')
    output.mkdir(parents=True)
    for name in ('observer','r1_head','aura_head'):
        subprocess.run(['ffmpeg','-hide_banner','-loglevel','error','-n','-framerate','10',
            '-pattern_type','glob','-i',str(root/'camera-streams'/name/'*.png'),
            '-c:v','libx264','-pix_fmt','yuv420p','-crf','19','-movflags','+faststart',str(output/(name+'-full.mp4'))],check=True)
    events=[]
    for path in sorted((root/'agent/public-events').glob('*.json')):events.append(json.loads(path.read_text()))
    manifest={'source':'continuous actual same-world frames','frames':len(rows),'fps':10,
        'first_native_tick':rows[0]['native_tick'],'last_native_tick':rows[-1]['native_tick'],
        'audio':'not_recorded','events':events,'interpolation_used':False}
    (output/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    text='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>Aura 家务闭环回放</title>
<style>body{font:17px/1.8 system-ui;max-width:1300px;margin:24px auto;padding:20px;background:#f3f6f7;color:#203b47}.grid{display:grid;grid-template-columns:repeat(3,1fr);gap:16px}video{width:100%;background:#122a35;border-radius:16px}article{background:white;padding:16px;border-radius:16px}button{padding:10px;margin:8px}</style>
<h1>同一家庭 · 人、Aura 与 R1 Pro</h1><p>三路来自同一 evaluator 的原生时钟；无补帧、无替代画面。此版未录声音。</p>
<button id="play">同步播放／暂停</button><button id="reset">从头播放</button><div class="grid">
<article>完整第三视角<video controls src="observer-full.mp4"></video></article>
<article>完整 R1 Pro 相机<video controls muted src="r1_head-full.mp4"></video></article>
<article>完整 Aura 相机<video controls muted src="aura_head-full.mp4"></video></article></div><pre id="events"></pre>
<script>const v=[...document.querySelectorAll('video')];let sync=false;
function align(t){v.forEach(x=>{if(Math.abs(x.currentTime-t)>.15)x.currentTime=t})}
document.querySelector('#play').onclick=()=>{align(v[0].currentTime);if(v[0].paused)v.forEach(x=>x.play().catch(()=>{}));else v.forEach(x=>x.pause())};
document.querySelector('#reset').onclick=()=>{align(0);v.forEach(x=>x.play().catch(()=>{}))};
v.forEach(x=>x.addEventListener('seeking',()=>{if(sync)return;sync=true;align(x.currentTime);sync=false}));
fetch('manifest.json').then(x=>x.json()).then(x=>document.querySelector('#events').textContent=JSON.stringify(x.events,null,2));</script></html>'''
    (output/'index.html').write_text(text,encoding='utf-8')
    return manifest

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    print(json.dumps(build(a.run_root,a.output),ensure_ascii=False))
