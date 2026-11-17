"""Offline physical rollout and dynamic display rendering; never connects to Aura."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess
import time
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
from PIL import Image
from model_parameters import P

ROOT=Path(__file__).resolve().parent


def load_model(filename='aura.xml'):
    # MuJoCo's Windows native path loader cannot open this Unicode workspace.
    # Keep the files in place; provide bytes through the official in-memory VFS.
    model_path=(ROOT/filename).resolve()
    if not model_path.is_relative_to(ROOT):
        raise ValueError('Model escaped the twin directory')
    xml=model_path.read_text(encoding='utf-8')
    assets={}
    for node in ET.fromstring(xml).iter():
        if node.get('file'):
            relative=node.get('file')
            path=(ROOT/relative).resolve()
            if not path.is_relative_to(ROOT):
                raise ValueError('Asset escaped the twin directory')
            assets[relative]=path.read_bytes()
    return mujoco.MjModel.from_xml_string(xml,assets=assets)


def set_screen(model, renderer, path):
    tid=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_TEXTURE,'screen_texture')
    width,height=int(model.tex_width[tid]),int(model.tex_height[tid])
    pixels=np.asarray(Image.open(path).convert('RGB').resize((width,height)))
    start=int(model.tex_adr[tid]); size=width*height*int(model.tex_nchannel[tid])
    # Match MuJoCo's PNG loader ordering; OBJ texture V is handled by MuJoCo.
    model.tex_data[start:start+size]=pixels.reshape(-1)
    renderer._gl_context.make_current()
    mujoco.mjr_uploadTexture(model,renderer._mjr_context,tid)


def run(output, video=True, model_file='aura.xml'):
    output.mkdir(parents=True,exist_ok=False)
    model=load_model(model_file)
    data=mujoco.MjData(model)
    renderer=mujoco.Renderer(model,height=720,width=960)
    source=ROOT/'assets/screen/sequence'
    sequence=sorted(source.glob('*.png'))
    screen_info=json.loads((ROOT/'assets/screen/provenance.json').read_text(encoding='utf-8'))
    if not sequence:
        raise RuntimeError('Prepare screen assets first')
    frames=[]
    ff=None
    if video:
        ff=subprocess.Popen([shutil.which('ffmpeg') or 'ffmpeg','-loglevel','error','-y',
            '-f','rawvideo','-pix_fmt','rgb24','-s','960x720','-r','12','-i','-',
            '-an','-c:v','libx264','-pix_fmt','yuv420p',str(output/'mujoco-rollout.mp4')],stdin=subprocess.PIPE)
    started=time.monotonic()
    try:
        for tick in range(2501):
            t=float(data.time)
            if t<.5 or t>=4.2:
                v,w=0.,0.
            elif t<2.5:
                v,w=.04,0.
            else:
                v,w=0.,.45
            # Differential wheel kinematics; actual motion is produced by contacts.
            data.ctrl[:]=[(v-w*P.wheel_y*.001)/(P.wheel_radius*.001),
                          (v+w*P.wheel_y*.001)/(P.wheel_radius*.001)]
            mujoco.mj_step(model,data)
            if not np.all(np.isfinite(data.qpos)) or not np.all(np.isfinite(data.qvel)):
                raise RuntimeError('Nonfinite dynamics')
            if tick % 42 == 0:
                index=len(frames)
                screen_index=int(data.time*screen_info['fps'])%len(sequence)
                set_screen(model,renderer,sequence[screen_index])
                renderer.update_scene(data,camera='lab_overview' if model_file=='lab.xml' else 'hero')
                image=renderer.render().copy()
                if index==6:
                    Image.fromarray(image).save(output/'mujoco-hero.png')
                if ff:
                    ff.stdin.write(image.tobytes())
                frames.append({'frame':index+1,'time':float(data.time),'qpos':data.qpos.tolist(),
                               'qvel':data.qvel.tolist(),'ctrl':data.ctrl.tolist(),
                               'screen_frame':screen_index+1})
        for camera in ('front','side','rgb_left','rgb_right'):
            renderer.update_scene(data,camera=camera)
            Image.fromarray(renderer.render()).save(output/(camera+'.png'))
        # Freeze the exact same pose/camera to verify the display can change pixels.
        renderer.update_scene(data,camera='front')
        set_screen(model,renderer,ROOT/'assets/screen/screen.png')
        before=renderer.render().copy()
        Image.fromarray(before).save(output/'screen-original.png')
        set_screen(model,renderer,ROOT/'assets/screen/hello.png')
        after=renderer.render().copy()
        Image.fromarray(after).save(output/'screen-custom.png')
        changed=int(np.count_nonzero(np.any(before!=after,axis=2)))
        start=np.array(frames[6]['qpos'][:3]); finish=np.array(frames[-1]['qpos'][:3])
        first_turn=min(frames,key=lambda x:abs(x['time']-2.5))
        forward=float(first_turn['qpos'][0]-frames[6]['qpos'][0])
        result={'engine':'MuJoCo '+mujoco.__version__,'steps':2501,'simulated_s':float(data.time),
            'wall_s':time.monotonic()-started,'finite':True,'frames':frames,
            'forward_displacement_m':forward,'final_displacement_m':float(np.linalg.norm(finish[:2]-start[:2])),
            'display_changed_pixels':changed,'warning_counts':data.warning.number.tolist(),
            'mass_and_contact_parameters':'assumed_not_calibrated',
            'physical_robot_access':False,'fidelity':P.record()['fidelity']}
        result['model_file']=model_file
        (output/'trajectory.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
        if changed<100:
            raise RuntimeError('Screen update not visibly effective')
        if forward<.02:
            raise RuntimeError('Wheel-drive forward movement failed')
        print(json.dumps({k:v for k,v in result.items() if k!='frames'},ensure_ascii=False))
    finally:
        renderer.close()
        if ff:
            ff.stdin.close()
            if ff.wait(timeout=20)!=0:
                raise RuntimeError('Video encoding failed')


def viewer(smoke_seconds=None):
    import mujoco.viewer
    model=load_model()
    data=mujoco.MjData(model)
    state={'v':0.,'w':0.,'screen':'nebula','reset':False}
    def keypress(key):
        if key==ord('W'): state.update(v=.04,w=0.)
        elif key==ord('S'): state.update(v=-.04,w=0.)
        elif key==ord('A'): state.update(v=0.,w=.45)
        elif key==ord('D'): state.update(v=0.,w=-.45)
        elif key==32: state.update(v=0.,w=0.)
        elif key==ord('R'): state.update(reset=True,v=0.,w=0.)
        elif ord('1')<=key<=ord('5'):
            state['screen']=['nebula','hello','listening','thinking','battery_unknown'][key-ord('1')]
    paths=sorted((ROOT/'assets/screen/sequence').glob('*.png'))
    pixels=[np.asarray(Image.open(p).convert('RGB')).reshape(-1) for p in paths]
    custom={name:np.asarray(Image.open(ROOT/f'assets/screen/{name}.png').convert('RGB')).reshape(-1)
            for name in ('hello','listening','thinking','battery_unknown')}
    meta=json.loads((ROOT/'assets/screen/provenance.json').read_text(encoding='utf-8'))
    tid=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_TEXTURE,'screen_texture')
    start=int(model.tex_adr[tid]); last=None
    with mujoco.viewer.launch_passive(model,data,key_callback=keypress) as window:
        viewer_started=time.monotonic()
        window.cam.lookat[:]=[0,0,.11]; window.cam.distance=.65
        window.cam.azimuth=135; window.cam.elevation=-20
        while window.is_running():
            if smoke_seconds and time.monotonic()-viewer_started>smoke_seconds:
                break
            started=time.monotonic()
            with window.lock():
                if state['reset']:
                    mujoco.mj_resetData(model,data); state['reset']=False
                v,w=state['v'],state['w']
                data.ctrl[:]=[(v-w*.095)/.072,(v+w*.095)/.072]
                mujoco.mj_step(model,data)
                index=int(data.time*meta['fps'])%len(pixels)
                screen=(state['screen'],index if state['screen']=='nebula' else 0)
                if screen!=last:
                    rgb=pixels[index] if screen[0]=='nebula' else custom[screen[0]]
                    model.tex_data[start:start+len(rgb)]=rgb
            if screen!=last:
                window.update_texture(tid); last=screen
            window.sync()
            time.sleep(max(0,model.opt.timestep-(time.monotonic()-started)))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'outputs/mujoco')
    parser.add_argument('--no-video',action='store_true')
    parser.add_argument('--viewer',action='store_true')
    parser.add_argument('--viewer-smoke',action='store_true')
    parser.add_argument('--lab',action='store_true',help='Render the virtual room (finite rollout only)')
    args=parser.parse_args()
    if args.viewer_smoke:
        viewer(1.5)
        print('Viewer context and dynamic texture upload completed; keyboard interaction remains manual.')
    elif args.viewer:
        viewer()
    else:
        run(args.output,not args.no_video,'lab.xml' if args.lab else 'aura.xml')
