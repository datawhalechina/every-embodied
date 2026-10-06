"""Render recorded states with geometry-ID mattes for scenic video compositing."""
import argparse
import hashlib
import json
import math
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--session', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--speed', type=float, default=1.3)
    parser.add_argument('--fps', type=int, default=25)
    parser.add_argument('--lead-seconds', type=float, default=.5)
    args = parser.parse_args()
    if not .5 <= args.speed <= 2 or not 1 <= args.fps <= 60 or args.lead_seconds < 0:
        raise ValueError('Invalid presentation timing')
    import imageio.v2 as imageio
    import numpy as np
    from dm_control import mujoco

    report = json.loads((args.session/'report.json').read_text())
    if not report['playback_completed'] or report['safety_stop']:
        raise ValueError('Session must complete without a safety stop')
    events = json.loads((args.session/'events.json').read_text())
    print('event schema',str(events)[:250],flush=True)
    onsets = [x['time_seconds'] for x in events if x['type']=='NoteOn']
    start = max(0., min(onsets)-args.lead_seconds*args.speed)
    duration = (report['executed_seconds']-start)/args.speed
    args.out.mkdir(parents=True,exist_ok=False)
    physics = mujoco.Physics.from_xml_path(str(args.session/'scene/scene.xml'))
    with np.load(args.session/'trajectory.npz',allow_pickle=False) as tr:
        states, initial = tr['qpos'], tr['initial_qpos']
        offsets = tr['piano_offsets'].copy() if 'piano_offsets' in tr else None
    mover = None
    if report.get('mode') == 'g05_mid_phrase_piano_relocation' and offsets is None:
        raise ValueError('Relocation session requires recorded piano translation')
    if offsets is not None:
        from piano_relocation import PianoTranslation
        if offsets.shape != (len(states), 3):
            raise ValueError('Invalid recorded piano translation')
        mover = PianoTranslation(physics.model.ptr)
    dt = report['executed_seconds']/len(states)
    wanted = np.array([i for i in range(physics.model.ngeom)
        if str(physics.model.id2name(i,'geom')).startswith(('piano/','piano_leg_','galaxea_r1pro/'))])
    frames = int(math.floor(duration*args.fps))
    rgb = imageio.get_writer(str(args.out/'rgb.mp4'),fps=args.fps,codec='libx264',
        macro_block_size=1,output_params=['-crf','16','-preset','fast'])
    matte = imageio.get_writer(str(args.out/'mask.mkv'),fps=args.fps,codec='ffv1',
        pixelformat='gray',macro_block_size=1)
    bounds=[]
    try:
        for i in range(frames):
            t = start + i*args.speed/args.fps
            index = min(len(states)-1,int(t/dt)-1)
            if mover is not None:
                mover.apply(np.zeros(3) if index < 0 else offsets[index])
            physics.data.qpos[:] = initial if index < 0 else states[index]
            physics.data.time=t
            physics.forward()
            frame = physics.render(width=960,height=720,camera_id='whole_robot')
            seg = physics.render(width=960,height=720,camera_id='whole_robot',segmentation=True)
            mask = ((seg[:,:,1]==5)&np.isin(seg[:,:,0],wanted)).astype('uint8')*255
            rgb.append_data(frame)
            matte.append_data(np.repeat(mask[:,:,None],3,axis=2))
            if i % args.fps == 0:
                yy,xx=np.where(mask>0)
                bounds.append([int(xx.min()),int(yy.min()),int(xx.max()),int(yy.max())])
            if i in (0, args.fps*5):
                rgba=np.dstack((frame,mask))
                imageio.imwrite(args.out/f'layer-{i:04d}.png',rgba)
            if i % (args.fps*5)==0: print('layers',i,'/',frames,flush=True)
    finally:
        rgb.close();matte.close()
    manifest=dict(session=str(args.session),source_video=str(args.session/'demo.mp4'),
        source_scene_sha256=hashlib.sha256((args.session/'scene/scene.xml').read_bytes()).hexdigest(),
        source_trajectory_sha256=hashlib.sha256((args.session/'trajectory.npz').read_bytes()).hexdigest(),
        start_seconds=start,speed=args.speed,fps=args.fps,frames=frames,duration=frames/args.fps,
        first_note_seconds=(min(onsets)-start)/args.speed,
        geometry_ids=wanted.tolist(),bounds=bounds,
        audio_source='Original measured.wav trimmed by start_seconds, atempo=speed',
        state_render='Recorded physical joint positions, no generated or edited robot motions',
        source_report=report)
    (args.out/'layers.json').write_text(json.dumps(manifest,indent=2))


if __name__=='__main__':
    main()
